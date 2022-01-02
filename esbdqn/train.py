"""
A pick-up agent and a drop-off agent alternatively training on ODySSEUS.

Both agents are implemented as epsilon-greedy Rainbow agents, from the paper
"Rainbow: Combining Improvements in Deep Reinforcement Learning"
https://arxiv.org/abs/1710.02298.

This agent combines:
*   Double Q-learning
*   Prioritized experience replay
*   Dueling networks
*   Multi-step learning
*   Distributional RL (C51)
*   Noisy networks
"""


import collections
import copy
import datetime
import dm_env
import filecmp
import haiku as hk
import importlib
import itertools as it
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathlib
import psutil
import shutil
import sys
import typing as t

from absl import app
from absl import flags
from absl import logging

from jax.config import config

# Custom imports
from dqn_zoo import parts
from dqn_zoo import replay

from dqn_zoo.rainbow import agent

from esbdqn.escooter_simulator import EscooterSimulator

from esbdqn.utils.checkpoint import PickleCheckpoint
from esbdqn.utils.environments import ConstrainedEnvironment

from esbdqn.utils.parts import generate_statistics
from esbdqn.utils.parts import make_odysseus_trackers
from esbdqn.utils.parts import processor
from esbdqn.utils.parts import rainbow_odysseus_network
from esbdqn.utils.parts import run_loop

from odysseus.simulator.simulation_input.sim_config_grid import EFFCS_SimConfGrid


DEFAULT_conf_filename = 'sim_conf'
DEFAULT_n_lives = 50
DEFAULT_resu_filename = 'sim_stats.csv'
DEFAULT_sim_scenario_name = 'escooter_mobility'


FLAGS = flags.FLAGS

# Mandatory arguments
flags.DEFINE_string('exp_name', None,
                    'Name of the experiment')

# Optional arguments
flags.DEFINE_string('conf_filename', DEFAULT_conf_filename,
                    'Name of the configuration file')

flags.DEFINE_integer('n_cpus', psutil.cpu_count(),
                     'Max number of CPUs to be used')
flags.DEFINE_integer('n_lives', DEFAULT_n_lives,
                     'Max number of lives')

# JAX parameters
flags.DEFINE_bool('checkpoint', True, '')
flags.DEFINE_bool('compress_state', True, '')
flags.DEFINE_bool('normalize_weights', True, '')

# # Austin parameters
# flags.DEFINE_integer('max_steps_per_episode', int(3e4), '')
# flags.DEFINE_integer('num_eval_frames', int(6e4), '')
# flags.DEFINE_integer('num_iterations', 3, '')
# flags.DEFINE_integer('num_train_frames', int(1.2e5), '')
# flags.DEFINE_integer('replay_capacity', int(1.2e5), '')
# flags.DEFINE_integer('target_network_update_period', int(3.8e3), '')

# Louisville parameters
flags.DEFINE_integer('max_steps_per_episode', int(1.3e3), '')
flags.DEFINE_integer('num_eval_frames', int(2.6e3), '')
flags.DEFINE_integer('num_iterations', 48, '')
flags.DEFINE_integer('num_train_frames', int(5.2e3), '')
flags.DEFINE_integer('replay_capacity', int(5.2e3), '')
flags.DEFINE_integer('target_network_update_period', int(1.6e2), '')

flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('learn_period', 16, '')
flags.DEFINE_integer('checkpoint_period', 3, '')
flags.DEFINE_integer('n_steps', 3, '')
flags.DEFINE_integer('num_atoms', 51, '')
flags.DEFINE_integer('seed', 1, '') # GPU may introduce non-determinism.

flags.DEFINE_float('importance_sampling_exponent_begin_value', 0.4, '')
flags.DEFINE_float('importance_sampling_exponent_end_value', 1., '')
flags.DEFINE_float('learning_rate', 0.00025 / 4, '')
flags.DEFINE_float('max_abs_reward', 100., '')
flags.DEFINE_float('max_global_grad_norm', 10., '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.02, '')
flags.DEFINE_float('noisy_weight_init', 0.1, '')
flags.DEFINE_float('optimizer_epsilon', 0.005 / 32, '')
flags.DEFINE_float('priority_exponent', 0.5, '')
flags.DEFINE_float('uniform_sample_probability', 1e-3, '')
flags.DEFINE_float('vmax', 10., '')


def main(argv):
    """
    Train pick-up and drop-off Rainbow agents on ODySSEUS.
    """
    del argv # Unused arguments

    # Metadata configuration
    parent_dir = pathlib.Path(__file__).parent.absolute()

    sim_input_conf_dir = parent_dir / 'configs' / DEFAULT_sim_scenario_name

    # Load configuration
    sim_conf = importlib.import_module('esbdqn.configs.{}.{}'
                                       .format(DEFAULT_sim_scenario_name,
                                               FLAGS.conf_filename))

    # Extract a single conf pair
    sim_general_conf  = EFFCS_SimConfGrid(sim_conf.General)       \
                                          .conf_list[0]
    sim_scenario_conf = EFFCS_SimConfGrid(sim_conf.Multiple_runs) \
                                          .conf_list[0]

    experiment_dir = parent_dir                     \
                        / 'experiments'             \
                        / DEFAULT_sim_scenario_name \
                        / FLAGS.exp_name            \
                        / sim_general_conf['city']

    if pathlib.Path.exists(experiment_dir):
        # Ensure configuration has not changed
        if not filecmp.cmp(str(sim_input_conf_dir
                               / FLAGS.conf_filename)   + '.py',
                           str(experiment_dir
                               / DEFAULT_conf_filename) + ".py",
                           shallow=False):
            raise IOError('Configuration changed at: {}'
                          .format(str(experiment_dir)))
    else:
        pathlib.Path.mkdir(experiment_dir, parents=True,
                           exist_ok=True)

        # Copy configuration files
        shutil.rmtree(experiment_dir)
        shutil.copytree(sim_input_conf_dir, experiment_dir)

        # Rename to the default name
        conf_filepath = experiment_dir / (FLAGS.conf_filename + ".py")
        conf_filepath.rename(experiment_dir
                             / (DEFAULT_conf_filename + ".py"))

        # Delete all other potential conf files
        for filename in experiment_dir.glob(
                DEFAULT_conf_filename + "_*.py"):
            filename.unlink()

    # Create results files
    results_dir = experiment_dir / 'results'

    pathlib.Path.mkdir(results_dir, parents=True,
                       exist_ok=True)

    results_filepath = results_dir / DEFAULT_resu_filename

    logging.info('Rainbow agents on ODySSEUS running on %s.',
                 jax.lib.xla_bridge.get_backend().platform.upper())

    if FLAGS.checkpoint:
        checkpoint = PickleCheckpoint(
            experiment_dir / 'models',
            'ODySSEUS-' + sim_general_conf['city'])
    else:
        checkpoint = parts.NullCheckpoint()

    checkpoint_restored = False

    if FLAGS.checkpoint:
        if checkpoint.can_be_restored():
            logging.info('Restoring checkpoint...')

            checkpoint.restore()
            checkpoint_restored = True

    # Generate RNG key
    rng_state = np.random.RandomState(FLAGS.seed)

    if checkpoint_restored:
        rng_state.set_state(checkpoint.state
                                      .rng_state)

    rng_key   = jax.random.PRNGKey(
        rng_state.randint(-sys.maxsize - 1,
                          sys.maxsize + 1,
                          dtype=np.int64))

    # Generate results file writer
    if sim_general_conf['save_history']:
        writer = parts.CsvWriter(str(results_filepath))

        if checkpoint_restored:
            writer.set_state(checkpoint.state
                                       .writer)
    else:
        writer = parts.NullWriter()

    def environment_builder() -> ConstrainedEnvironment:
        """
        Create the ODySSEUS environment.
        """
        return EscooterSimulator(
                        (sim_general_conf,
                         sim_scenario_conf),
                    FLAGS.n_lives)

    def preprocessor_builder():
        """
        Create the ODySSEUS input preprocessor.
        """
        return processor(
            max_abs_reward=FLAGS.max_abs_reward,
            zero_discount_on_life_loss=True
        )

    env = environment_builder()

    logging.info('Environment: %s', FLAGS.exp_name)
    logging.info('Action spec: %s', env.action_spec())
    logging.info('Observation spec: %s', env.observation_spec())

    # Take [0] as both Rainbow have
    # the same number of actions
    num_actions = env.action_spec()[0].num_values
    support = jnp.linspace(-FLAGS.vmax, FLAGS.vmax,
                           FLAGS.num_atoms)

    network = hk.transform(rainbow_odysseus_network(
                           num_actions, support,
                           FLAGS.noisy_weight_init))

    # Create sample network input from reset.
    sample_processed_timestep = preprocessor_builder()(env.reset())
    sample_processed_timestep = t.cast(dm_env.TimeStep,
                                       sample_processed_timestep)

    sample_processed_network_input = sample_processed_timestep.observation

    # Note the t in the replay is not exactly
    # aligned with the Rainbow agents t.
    importance_sampling_exponent_schedule = parts.LinearSchedule(
        begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity),
        end_t=(FLAGS.num_iterations * FLAGS.num_train_frames),
        begin_value=FLAGS.importance_sampling_exponent_begin_value,
        end_value=FLAGS.importance_sampling_exponent_end_value)

    if FLAGS.compress_state:
        def encoder(transition):
            return transition._replace(
                s_tm1=replay.compress_array(transition.s_tm1),
                s_t=replay.compress_array(transition.s_t))

        def decoder(transition):
            return transition._replace(
                s_tm1=replay.uncompress_array(transition.s_tm1),
                s_t=replay.uncompress_array(transition.s_t))
    else:
        encoder = None
        decoder = None

    replay_struct = replay.Transition(
        s_tm1=None,
        a_tm1=None,
        r_t=None,
        discount_t=None,
        s_t=None,
    )

    transition_accumulator = replay.NStepTransitionAccumulator(FLAGS.n_steps)

    transition_replay = replay.PrioritizedTransitionReplay(
        FLAGS.replay_capacity, replay_struct,
        FLAGS.priority_exponent,
        importance_sampling_exponent_schedule,
        FLAGS.uniform_sample_probability,
        FLAGS.normalize_weights,
        rng_state, encoder, decoder)

    optimizer = optax.adam(
        learning_rate=FLAGS.learning_rate,
        eps=FLAGS.optimizer_epsilon)

    if FLAGS.max_global_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(
                FLAGS.max_global_grad_norm),
            optimizer)

    train_rng_key, eval_rng_key = jax.random.split(rng_key)

    # Create pick-up/drop-off agents
    P_train_agent = agent.Rainbow(
        preprocessor=preprocessor_builder(),
        sample_network_input=copy.deepcopy(sample_processed_network_input),
        network=copy.deepcopy(network),
        support=copy.deepcopy(support),
        optimizer=copy.deepcopy(optimizer),
        transition_accumulator=copy.deepcopy(transition_accumulator),
        replay=copy.deepcopy(transition_replay),
        batch_size=FLAGS.batch_size,
        min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
        learn_period=FLAGS.learn_period,
        target_network_update_period=FLAGS.target_network_update_period,
        rng_key=train_rng_key,
    )

    D_train_agent = agent.Rainbow(
        preprocessor=preprocessor_builder(),
        sample_network_input=copy.deepcopy(sample_processed_network_input),
        network=copy.deepcopy(network),
        support=copy.deepcopy(support),
        optimizer=copy.deepcopy(optimizer),
        transition_accumulator=copy.deepcopy(transition_accumulator),
        replay=copy.deepcopy(transition_replay),
        batch_size=FLAGS.batch_size,
        min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
        learn_period=FLAGS.learn_period,
        target_network_update_period=FLAGS.target_network_update_period,
        rng_key=train_rng_key,
    )

    P_eval_agent = parts.EpsilonGreedyActor(
        preprocessor=preprocessor_builder(),
        network=copy.deepcopy(network),
        exploration_epsilon=0,
        rng_key=eval_rng_key,
    )

    D_eval_agent = parts.EpsilonGreedyActor(
        preprocessor=preprocessor_builder(),
        network=copy.deepcopy(network),
        exploration_epsilon=0,
        rng_key=eval_rng_key,
    )

    if checkpoint_restored:
        P_train_agent.set_state(checkpoint.state.P_agent['train'])
        D_train_agent.set_state(checkpoint.state.D_agent['train'])

        P_eval_agent.set_state(checkpoint.state.P_agent['eval'])
        D_eval_agent.set_state(checkpoint.state.D_agent['eval'])

    state = checkpoint.state

    if not checkpoint_restored:
        state.iteration = 0

    state.P_agent = {}
    state.D_agent = {}

    state.rng_state = rng_state
    state.writer = writer

    state.P_agent['train'] = P_train_agent
    state.D_agent['train'] = D_train_agent

    state.P_agent['eval'] = P_eval_agent
    state.D_agent['eval'] = D_eval_agent

    while state.iteration < FLAGS.num_iterations:
        # Create a new environment at each new iteration
        # to allow for determinism if preempted.
        env = environment_builder()

        # Leave some spacing
        print('\n')

        logging.info('Training iteration: %d', state.iteration)

        train_trackers = make_odysseus_trackers(FLAGS.max_abs_reward)
        eval_trackers  = make_odysseus_trackers(FLAGS.max_abs_reward)

        train_seq = run_loop(P_train_agent, D_train_agent,
                             env, FLAGS.max_steps_per_episode)

        num_train_frames = 0        \
            if state.iteration == 0 \
            else FLAGS.num_train_frames

        train_seq_truncated = it.islice(train_seq, num_train_frames)

        train_stats = generate_statistics(train_trackers,
                                          train_seq_truncated)

        logging.info('Evaluation iteration: %d', state.iteration)

        # Synchronize network parameters
        P_eval_agent.network_params = P_train_agent.online_params
        D_eval_agent.network_params = P_train_agent.online_params

        eval_seq = run_loop(P_eval_agent, D_eval_agent,
                            env, FLAGS.max_steps_per_episode)

        eval_seq_truncated = it.islice(eval_seq, FLAGS.num_eval_frames)

        eval_stats = generate_statistics(eval_trackers,
                                         eval_seq_truncated)

        # Logging and checkpointing
        L = [
            # Simulation metadata
            ('iteration', state.iteration, '%3d'),

            # ODySSEUS metadata
            ('n_charging_workers', sim_scenario_conf['n_workers'], '%3d'),
            ('n_relocation_workers', sim_scenario_conf['n_relocation_workers'], '%3d'),
            ('n_vehicles', sim_scenario_conf['n_vehicles'], '%3d'),
            ('pct_incentive_willingness', sim_scenario_conf['incentive_willingness'], '%2.2f'),
            ('zone_side_m', sim_general_conf['bin_side_length'], '%3d'),

            # Validation agents
            ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),

            ('eval_P_episode_return', eval_stats['episode_return'][0], '%2.2f'),
            ('eval_D_episode_return', eval_stats['episode_return'][1], '%2.2f'),

            ('eval_min_n_accepted_incentives',
             np.min(eval_stats['episodes_n_accepted_incentives']), '%2.2f'),
            ('eval_avg_n_accepted_incentives',
             np.mean(eval_stats['episodes_n_accepted_incentives']), '%2.2f'),
            ('eval_max_n_accepted_incentives',
             np.max(eval_stats['episodes_n_accepted_incentives']), '%2.2f'),

            ('eval_min_n_lives',
             np.min(eval_stats['episodes_n_lives']), '%2.2f'),
            ('eval_avg_n_lives',
             np.mean(eval_stats['episodes_n_lives']), '%2.2f'),
            ('eval_max_n_lives',
             np.max(eval_stats['episodes_n_lives']), '%2.2f'),

            ('eval_min_pct_satisfied_demand',
             np.min(eval_stats['pct_satisfied_demands']), '%2.2f'),
            ('eval_avg_pct_satisfied_demand',
             np.mean(eval_stats['pct_satisfied_demands']), '%2.2f'),
            ('eval_max_pct_satisfied_demand',
             np.max(eval_stats['pct_satisfied_demands']), '%2.2f'),

            # Training agents
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),

            ('train_P_episode_return', train_stats['episode_return'][0], '%2.2f'),
            ('train_D_episode_return', train_stats['episode_return'][1], '%2.2f'),

            ('train_min_n_accepted_incentives',
             np.min(train_stats['episodes_n_accepted_incentives']), '%2.2f'),
            ('train_avg_n_accepted_incentives',
             np.mean(train_stats['episodes_n_accepted_incentives']), '%2.2f'),
            ('train_max_n_accepted_incentives',
             np.max(train_stats['episodes_n_accepted_incentives']), '%2.2f'),

            ('train_min_n_lives',
             np.min(train_stats['episodes_n_lives']), '%2.2f'),
            ('train_avg_n_lives',
             np.mean(train_stats['episodes_n_lives']), '%2.2f'),
            ('train_mac_n_lives',
             np.max(train_stats['episodes_n_lives']), '%2.2f'),

            ('train_min_pct_satisfied_demand',
             np.min(train_stats['pct_satisfied_demands']), '%2.2f'),
            ('train_avg_pct_satisfied_demand',
             np.mean(train_stats['pct_satisfied_demands']), '%2.2f'),
            ('train_max_pct_satisfied_demand',
             np.max(train_stats['pct_satisfied_demands']), '%2.2f'),

            ('P_importance_sampling_exponent',
             P_train_agent.importance_sampling_exponent, '%.3f'),
            ('D_importance_sampling_exponent',
             D_train_agent.importance_sampling_exponent, '%.3f'),

            ('P_max_seen_priority', P_train_agent.max_seen_priority, '%.3f'),
            ('D_max_seen_priority', D_train_agent.max_seen_priority, '%.3f'),
        ]

        L_str = '\n'.join(('%s: ' + f) % (n, v) for n, v, f in L)

        logging.info(L_str)

        if state.iteration == \
                FLAGS.num_iterations - 1:
            print('\n')

        writer.write(collections.OrderedDict(
            (n, v) for n, v, _ in L))

        state.iteration += 1

        if state.iteration \
                % FLAGS.checkpoint_period == 0:
            checkpoint.save()

    writer.close()


if __name__ == '__main__':
    config.update('jax_platform_name', 'gpu') # Default to GPU.
    config.update('jax_numpy_rank_promotion', 'raise')

    config.config_with_absl()

    flags.mark_flags_as_required([
        'exp_name',
    ])

    _t = datetime.datetime.now()

    # app.run(main)

    try:
        app.run(main)
    except TypeError:
        pass

    print('\nTotal training time:',
          datetime.datetime.now() - _t)
