"""
Testing real-time ESBDQN API with trained Rainbow pick-up/drop-off agents.
"""


import copy
import haiku as hk
import importlib
import jax
import jax.numpy as jnp
import numpy as np
import psutil
import sys
import typing as t

from absl import app
from absl import flags
from absl import logging

from jax.config import config

# Custom imports
from dqn_zoo import parts

from esbdqn.escooter_simulator import EscooterSimulator

from esbdqn.utils.checkpoint import PickleCheckpoint

from esbdqn.utils.parts import processor
from esbdqn.utils.parts import rainbow_odysseus_network

from odysseus.simulator.simulation_input.sim_config_grid import EFFCS_SimConfGrid


DEFAULT_conf_filename = 'sim_conf'
DEFAULT_sim_scenario_name = 'escooter_mobility'


FLAGS = flags.FLAGS

# Mandatory arguments
flags.DEFINE_string('checkpoint_dirpath', None,
                    'Path to the directory with the trained Rainbow agents')

# Optional arguments
flags.DEFINE_string('conf_filename', DEFAULT_conf_filename,
                    'Name of the configuration file')

flags.DEFINE_integer('n_cpus', psutil.cpu_count(),
                     'Max number of CPUs to be used')

# JAX parameters
flags.DEFINE_integer('num_atoms', 51, '')
flags.DEFINE_integer('seed', 1, '') # GPU may introduce nondeterminism.

flags.DEFINE_float('max_abs_reward', 100., '')
flags.DEFINE_float('noisy_weight_init', 0.1, '')
flags.DEFINE_float('vmax', 10., '')


def main(argv):
    """
    Test pick-up and drop-off Rainbow agents on ESBDQN API.
    """
    del argv # Unused arguments

    # Load configuration
    sim_conf = importlib.import_module('api.configs.{}.{}'
                                       .format(DEFAULT_sim_scenario_name,
                                               FLAGS.conf_filename))

    # Extract a single conf pair
    sim_general_conf  = EFFCS_SimConfGrid(sim_conf.General)       \
                                          .conf_list[0]
    sim_scenario_conf = EFFCS_SimConfGrid(sim_conf.Multiple_runs) \
                                          .conf_list[0]

    logging.info('Rainbow agents on ODySSEUS running on %s.',
                 jax.lib.xla_bridge.get_backend().platform.upper())

    checkpoint = PickleCheckpoint(
        FLAGS.checkpoint_dirpath,
        'ODySSEUS-' + sim_general_conf['city'])

    if not checkpoint.can_be_restored():
        raise IOError('Cannot load the trained Rainbow agents.')

    logging.info('Restoring checkpoint...')

    checkpoint.restore()

    # Generate RNG key
    rng_state = np.random.RandomState(FLAGS.seed)
    rng_state.set_state(checkpoint.state
                                  .rng_state)

    rng_key   = jax.random.PRNGKey(
        rng_state.randint(-sys.maxsize - 1,
                          sys.maxsize + 1,
                          dtype=np.int64))

    def environment_builder() -> EscooterSimulator:
        """
        Create the ODySSEUS environment.
        """
        return EscooterSimulator(
                        (sim_general_conf,
                         sim_scenario_conf),
                        None, rt=True)

    def preprocessor_builder():
        """
        Create the ODySSEUS input preprocessor.
        """
        return processor(
            max_abs_reward=FLAGS.max_abs_reward,
            zero_discount_on_life_loss=True
        )

    env = environment_builder()

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

    _, eval_rng_key = jax.random.split(rng_key)

    # Create pick-up/drop-off agents
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

    P_eval_agent.set_state(checkpoint.state.P_agent['eval'])
    D_eval_agent.set_state(checkpoint.state.D_agent['eval'])

    env.run(P_eval_agent, D_eval_agent)


if __name__ == '__main__':
    config.update('jax_platform_name', 'gpu') # Default to GPU.
    config.update('jax_numpy_rank_promotion', 'raise')

    config.config_with_absl()

    flags.mark_flags_as_required([
        'checkpoint_dirpath',
    ])

    try:
        app.run(main)
    except TypeError:
        pass
