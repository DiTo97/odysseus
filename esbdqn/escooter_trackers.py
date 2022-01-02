import dm_env
import numpy as np
import typing as t

# Custom imports
from dqn_zoo.parts import Agent

from esbdqn.escooter_simulator import EscooterSimulator

from esbdqn.utils.commons import sum_tuple_vals
from esbdqn.utils.environments import ConstrainedEnvironment


# Custom type definitions
Action = t.Tuple[np.int16, np.int16]


class EpisodeTracker:
    """
    Track episode return and other statistics on ODySSEUS.
    """
    def __init__(self, max_abs_reward: t.Optional[float]):
        self._max_abs_reward = max_abs_reward

        self._num_steps_since_reset = None
        self._num_steps_over_episodes = None
        self._episode_returns = None
        self._current_episode_rewards = None
        self._current_episode_step = None

    def step(
            self,
            environment: t.Optional[ConstrainedEnvironment],
            timestep_t: t.Optional[dm_env.TimeStep],
            agents: t.Optional[t.Tuple[Agent, Agent]],
            a_t: t.Optional[Action],
    ) -> None:
        """
        Accumulate statistics from a timestep.
        """
        # Unused arguments
        del (environment, agents, a_t)

        if timestep_t is None:
            return

        # First reward is invalid,
        # all other rewards are appended.
        if timestep_t.first():
            if self._current_episode_rewards:
                raise ValueError('Current episode reward list should be empty.')
            if self._current_episode_step != 0:
                raise ValueError('Current episode step should be zero.')
        else:
            if self._max_abs_reward is None:
                self._current_episode_rewards \
                    .append(timestep_t.reward)
            else:
                clipped_reward = tuple(np.clip(timestep_t.reward,
                                               -self._max_abs_reward,
                                               self._max_abs_reward))

                self._current_episode_rewards \
                    .append(clipped_reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if timestep_t.last():
            # Sum the individual reward for both
            # pick-up/drop-off Rainbow agents
            self._episode_returns.append(
                sum_tuple_vals(
                    self._current_episode_rewards))
            self._current_episode_rewards = []

            self._num_steps_over_episodes += self._current_episode_step
            self._current_episode_step = 0

    def reset(self) -> None:
        """
        Reset all gathered statistics.
        Should not be called between episodes.
        """
        self._num_steps_since_reset = 0
        self._num_steps_over_episodes = 0
        self._episode_returns = []
        self._current_episode_step = 0
        self._current_episode_rewards = []

    def get(self) -> t.Mapping[t.Text,
                               t.Union[int, float, None]]:
        """
        Aggregate statistics from all episodes.

        The convention is `episode_return` is set to `current_episode_return`
        if a full episode has not been encountered. Otherwise it is set to
        `mean_episode_return` which is the mean return of complete episodes only.

        If no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns
        -------
        t.Mapping[t.Text, t.Union[int, float, None]]
            Dictionary of aggregated statistics.
        """
        if self._episode_returns:
            current_episode_return = sum_tuple_vals(
                self._current_episode_rewards)

            mean_episode_return    = tuple(
                np.array(self._episode_returns)
                  .mean(axis=0))

            if not current_episode_return:
                current_episode_return = tuple([np.nan, np.nan])

            episode_return = mean_episode_return
        else:
            current_episode_return = tuple([np.nan, np.nan])
            mean_episode_return    = tuple([np.nan, np.nan])

            if self._num_steps_since_reset > 0:
                current_episode_return = sum_tuple_vals(
                    self._current_episode_rewards)

            if not current_episode_return:
                current_episode_return = tuple([np.nan, np.nan])

            episode_return = current_episode_return

        return {
            # Last episode data
            'current_episode_return': current_episode_return,
            'current_episode_step': self._current_episode_step,

            # Aggregated data
            'episode_return': episode_return,
            'mean_episode_return': mean_episode_return,

            # Training metadata
            'num_episodes': len(self._episode_returns),

            'num_steps_over_episodes': self._num_steps_over_episodes,
            'num_steps_since_reset': self._num_steps_since_reset,
        }


class SimulationTracker:
    """
    Track simulation statistics on ODySSEUS.
    """

    def __init__(self):
        self._episodes_n_accepted_incentives = None
        self._episodes_n_lives = None
        self._pct_satisfied_demands = None

    def step(
            self,
            environment: t.Optional[EscooterSimulator],
            timestep_t: t.Optional[dm_env.TimeStep],
            agents: t.Optional[t.Tuple[Agent, Agent]],
            a_t: t.Optional[Action],
    ) -> None:
        """
        Accumulate statistics from a timestep.
        """
        # Unused arguments
        del (agents, a_t)

        if timestep_t is None \
                or timestep_t.last() :
            # Get statistics from a whole episode
            self._episodes_n_accepted_incentives.append(
                environment.get_n_accepted_incentives())

            self._episodes_n_lives.append(
                environment.get_n_lives())

            self._pct_satisfied_demands.append(
                environment.pct_satisfied_demand())

    def reset(self) -> None:
        """
        Reset all gathered statistics.
        Should not be called between episodes.
        """
        self._episodes_n_accepted_incentives = []
        self._episodes_n_lives = []
        self._pct_satisfied_demands = []

    def get(self) -> t.Mapping[t.Text,
                               t.Union[int, float, t.List, None]]:
        """
        Aggregate statistics from all episodes.

        The convention is `pct_satisfied_demands` is set to an empty list `[]`
        if a full episode has not been encountered. Otherwise it is left as
        a list of satisfied demand % with one entry per episode.

        If no steps have been taken at all, `pct_satisfied_demands` is set to None.

        Returns
        -------
        t.Mapping[t.Text, t.Union[int, float, None]]
            Dictionary of aggregated statistics.
        """
        episodes_n_accepted_incentives = [np.nan]
        episodes_n_lives = [np.nan]
        pct_satisfied_demands = [np.nan]

        if self._episodes_n_accepted_incentives:
            episodes_n_accepted_incentives = \
                self._episodes_n_accepted_incentives

        if self._episodes_n_lives:
            episodes_n_lives = self._episodes_n_lives

        if self._pct_satisfied_demands:
            pct_satisfied_demands = \
                self._pct_satisfied_demands

        return {
            # Simulation metadata
            'episodes_n_accepted_incentives': episodes_n_accepted_incentives,
            'episodes_n_lives': episodes_n_lives,
            'pct_satisfied_demands': pct_satisfied_demands,
        }
