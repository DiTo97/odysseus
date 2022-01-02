import collections
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import typing as t

# Custom imports
from dqn_zoo.networks import C51NetworkOutputs
from dqn_zoo.networks import NetworkFn
from dqn_zoo.networks import linear
from dqn_zoo.networks import noisy_linear

from dqn_zoo.parts import Agent

from dqn_zoo.processors import Processor
from dqn_zoo.processors import Sequential
from dqn_zoo.processors import ApplyToNamedTupleField

from esbdqn.escooter_trackers import EpisodeTracker
from esbdqn.escooter_trackers import SimulationTracker

from esbdqn.utils.environments import ConstrainedEnvironment
from esbdqn.utils.linalg import one_hot


# Custom type definitions
Action = t.Tuple[np.int16, np.int16]


# Custom lambda definitions
Identity = lambda v: v


def run_loop(
        P_agent: Agent,
        D_agent: Agent,
        environment: ConstrainedEnvironment,
        max_steps_per_episode: int = 0,
        yield_before_reset: bool = False,
) -> t.Iterable[
         t.Tuple[dm_env.Environment,
                 t.Optional[dm_env.TimeStep],
                 t.Tuple[Agent, Agent],
                 t.Optional[Action]]]:
    """
    Repeatedly alternate step calls on environment and agents.

    At time `t`, `t + 1` environment timesteps and `t + 1` agent steps have been
    seen in the current episode. `t` resets to `0` for the next episode after
    `T` steps have been run or a terminal state have been reached.

    Parameters
    ----------
    P_agent: Agent
        Pick-up agent to be run. It has methods `step(timestep)` and `reset()`.

    D_agent: Agent
        Drop-off agent to be run. It has methods `step(timestep)` and `reset()`.

    environment: dm_env.Environment
        Environment to run. It has methods `step(action)` and `reset()`.

    max_steps_per_episode: int
        Maximum steps per episode, `T`. If positive, when time t reaches
        this value within an episode, the episode is truncated.

    yield_before_reset: bool
        Whether to additionally yield `(environment, None, agents, None)`
        before the agents and environment are reset at the start
        of each episode. The default is False.

    Yields
    ------
    t.Tuple
        Tuple `(environment, timestep_t, agents, a_t)` where
        `a_t` is the action selected at time `t`.
    """
    while True: # For each episode.
        if yield_before_reset:
            yield environment, None, (P_agent, D_agent), None

        _t = 0

        P_agent.reset()
        D_agent.reset()

        timestep_t = environment.reset()

        # Necessary when willingness is set to 0
        # and Rainbow agents are disabled
        if timestep_t.last():
            yield environment, None, (P_agent, D_agent), None

            # Stop as there are no
            # Rainbow agents to train
            break

        while True: # For each step in the current episode.
            discount = timestep_t.discount \
                if timestep_t.discount     \
                else (None, None)

            reward   = timestep_t.reward   \
                if timestep_t.reward       \
                else (None, None)

            # Update agents
            P_a_t = P_agent.step(timestep_t._replace(
                discount=discount[0], reward=reward[0]))

            D_a_t = D_agent.step(timestep_t._replace(
                discount=discount[1], reward=reward[1]))

            a_t = (P_a_t, D_a_t)

            yield environment, timestep_t, \
                  (P_agent, D_agent),      \
                  environment.valid_action(a_t)

            # Move step counter by 1
            _t += 1
            a_tm1 = a_t

            # Update environment
            timestep_t = environment.step(a_tm1)

            if 0 < max_steps_per_episode <= _t:
                timestep_t = timestep_t._replace(
                    step_type=dm_env.StepType.LAST)

            if timestep_t.last():
                # Extra agents step - No actions
                _ = P_agent.step(timestep_t._replace(
                    discount=discount[0], reward=reward[0]))
                _ = D_agent.step(timestep_t._replace(
                    discount=discount[1], reward=reward[1]))

                yield environment, timestep_t, \
                      (P_agent, D_agent), None

                # Start a new episode
                break


def processor(
        max_abs_reward: t.Optional[float] = None,
        zero_discount_on_life_loss: bool = False,
) -> Processor[[dm_env.TimeStep], dm_env.TimeStep]:
    """
    Rainbow preprocessing on ODySSEUS.

    This processor does the following to a timestep:
        1. Zeroes the discount on life loss.
        2. Clips the reward
        3. Enlarges the dimension by 1 with a two-hot encoded vector
           with the pick-up/drop-off zone Ids pair as the only 1s.

    Parameters
    ----------
    max_abs_reward : t.Optional[float]
        Clip reward, if out of bounds.
        The default is None.

    zero_discount_on_life_loss : bool
        Whether to zero the discount on life loss.
        The default is False.
    """
    return Sequential(
        # Zero the discount on life loss
        ZeroDiscountOnLifeLoss()          \
            if zero_discount_on_life_loss \
            else Identity,

        # Clip the reward
        ApplyToNamedTupleField(
            'reward',
            clip_reward(max_abs_reward) \
                if max_abs_reward       \
                else Identity,
        ),

        # Enlarge the dimension by 1 with a two-hot encoded vector
        # with the pick-up/drop-off zone Ids pair as the only 1s
        encode_observation
    )


class ZeroDiscountOnLifeLoss:
    """
    Zero the discount on timestep on life loss.

    This processor assumes observations to be tuples whose last entry is a
    scalar int indicating the remaining # of lives.
    """

    def __init__(self):
        self._num_lives_on_prev_step = None

    def reset(self) -> None:
        self._num_lives_on_prev_step = None

    def __call__(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
        # A life loss is registered when the timestep is a regular transition
        # and lives have decreased since the previous timestep.
        num_lives = timestep.observation[-1]
        life_lost = timestep.mid() and (num_lives <
                                        self._num_lives_on_prev_step)

        self._num_lives_on_prev_step = num_lives

        return timestep._replace(discount=0.) \
               if life_lost else timestep


def clip_reward(bound: float) \
               -> Processor[[t.Optional[float]],
                             t.Optional[float]]:
    """
    Return a function that clips non-`None` inputs to (`-bound`, `bound`).
    """

    def clip_reward_fn(reward):
        return None if reward is None \
            else np.clip(reward, -bound, bound)

    return clip_reward_fn


def encode_observation(timestep: dm_env.TimeStep) \
                      -> dm_env.TimeStep:
    """
    Concat the pick-up/drop-off zone Ids as a
    two-hot encoded vector to the observed state X.
    """
    X, p_zone_idx, d_zone_idx, _ = timestep.observation

    N = X.shape[0]

    if p_zone_idx is not None \
            and d_zone_idx is not None:
        p_one_hot_vec = one_hot(p_zone_idx, N)
        d_one_hot_vec = one_hot(d_zone_idx, N)

        v = p_one_hot_vec + d_one_hot_vec
        v = v.reshape(N, -1)
    else:
        v = np.zeros((N, 1))

    return timestep._replace(observation=np.hstack((X, v)))


def rainbow_odysseus_network(
        num_actions: int,
        support: jnp.ndarray,
        noisy_weight_init: float
    ) -> NetworkFn:
    """Rainbow network, expects `uint8` input."""

    if support.ndim != 1:
        raise ValueError('Support should be 1D.')

    num_atoms = len(support)
    support = support[None, None, :]

    def net_fn(inputs):
        """
        Function representing Rainbow Q-network.
        """
        inputs = jnp.asarray(inputs)
        inputs = dqn_dense_torso()(inputs)

        # Advantage head.
        advantage = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
        advantage = jax.nn.relu(advantage)
        advantage = noisy_linear(
            num_actions * num_atoms, noisy_weight_init,
            with_bias=False)(advantage)
        advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))

        # Value head.
        value = noisy_linear(512, noisy_weight_init,
                             with_bias=True)(inputs)
        value = jax.nn.relu(value)
        value = noisy_linear(num_atoms, noisy_weight_init,
                             with_bias=False)(value)
        value = jnp.reshape(value, (-1, 1, num_atoms))

        # Q-distribution and values.
        q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)

        assert q_logits.shape[1:] == (num_actions, num_atoms)

        q_dist = jax.nn.softmax(q_logits)
        q_values = jnp.sum(q_dist * support, axis=2)
        q_values = jax.lax.stop_gradient(q_values)

        return C51NetworkOutputs(q_logits=q_logits,
                                 q_values=q_values)

    return net_fn


def dqn_dense_torso() -> NetworkFn:
    """
    DQN dense torso for ODySSEUS.
    """

    def net_fn(inputs):
        """
        Function representing dense torso for a DQN Q-network.
        """
        network = hk.Sequential([
            # Standardization
            z_norm(),
            hk.Flatten(),

            # Latent space construction
            linear(512, with_bias=True),
            jax.nn.relu,
            linear(512, with_bias=True),
            jax.nn.relu,
            linear(256, with_bias=True),
            jax.nn.relu,
            hk.Flatten(),
        ])
        return network(inputs)

    return net_fn


def z_norm() -> NetworkFn:
    """
    Z-normalization for the state dimens X.
    """
    def net_fn(inputs):
        X = inputs[:, :, :2]
        X = (X - jnp.mean(X))/jnp.std(X)

        return inputs.at[:, :, :2].set(X)

    return net_fn


def make_odysseus_trackers(
        max_abs_reward: t.Optional[float] = None):
    """
    Make statistics trackers on ODySSEUS.
    """
    return [
        EpisodeTracker(max_abs_reward),
        SimulationTracker()
    ]


def generate_statistics(
        trackers: t.Iterable[t.Any],
        episode_timesteps: t.Iterable[t.Tuple[dm_env.Environment,
                                      t.Optional[dm_env.TimeStep],
                                      t.Tuple[Agent, Agent],
                                      t.Optional[Action]]]) \
        -> t.Mapping[t.Text, t.Any]:
    """
    Generate statistics from a sequence of timesteps and actions.
    """
    # Reset at the start only, not between episodes.
    for tracker in trackers:
        tracker.reset()

    for environment, timestep_t, agents, a_t in episode_timesteps:
        for tracker in trackers:
            tracker.step(environment, timestep_t, agents, a_t)

    # Merge all statistics dictionaries into one
    statistics_dicts = (tracker.get() for
                        tracker in trackers)

    return dict(collections.ChainMap(*statistics_dicts))
