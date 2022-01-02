import dm_env
import numpy as np
import typing as t

from dm_env import specs

# Custom imports
import esbdqn.escooter_typing as et

import odysseus.simulator.simulation.trace_driven_simulator as tdsim
import odysseus.simulator.simulation_input.sim_input as sim_input

from dqn_zoo.parts import EpsilonGreedyActor

from esbdqn.utils.environments import ConstrainedEnvironment


# Null action value
NOP: np.int8 = 0


class EscooterSimulator(ConstrainedEnvironment):
    _sim_input: sim_input.SimInput
    _sim: t.Optional[tdsim.TraceDrivenSim]
    _start_of_episode: bool

    """
    E-scooter ODySSEUS simulator with a `dm_env.Environment` interface.
    """
    def __init__(self, conf_tuple: t.Tuple[et.Conf, et.Conf],
                 n_lives: t.Optional[int],
                 rt: bool = False):
        """
        Parameters
        ----------
        conf_tuple : t.Tuple[et.Conf, et.Conf]
            Pair of general and scenario parameters

        n_lives : t.Optional[int]
            Maximum # of invalid actions per simulation

        rt : bool
            Whether the SimPy env should be real-time.
        """
        self._sim_input = sim_input.SimInput(conf_tuple)
        self._sim_input.init()

        self._start_of_episode = True

        self._sim = tdsim.TraceDrivenSim(
            self._sim_input, n_lives, rt)

    def reset(self) -> dm_env.TimeStep:
        """
        Reset the environment and start a new episode.
        """
        observation = self._sim.reset(self._sim_input, False)

        # Necessary when willingness is set to 0
        # and Rainbow agents are disabled
        if self.sim_finished():
            self._start_of_episode = True
            return dm_env.termination(None, observation)

        self._start_of_episode = False
        return dm_env.restart(observation)

    def step(self, action: t.Tuple[np.int8, np.int8]) \
            -> dm_env.TimeStep:
        """
        Update the environment given an action.

        Returns
        -------
        dm_env.TimeStep
            Timestep resulting from the action.
        """
        # If the previous timestep was LAST then call reset() on the Simulator
        # environment to launch a new episode, otherwise step().
        if self._start_of_episode:
            observation = self._sim.reset(self._sim_input, False)

            discount = None
            finished = False
            reward = None
            step_type = dm_env.StepType.FIRST
        else:
            observation, reward, finished, _ = self._sim.step(action)

            step_type = dm_env.StepType.LAST \
                if finished                  \
                else dm_env.StepType.MID

            # TODO: Choose discount value
            discount = tuple([float(step_type ==
                                    dm_env.StepType.MID)]*2)

        self._start_of_episode = finished

        return dm_env.TimeStep(
            discount=discount,
            observation=observation,
            reward=reward,
            step_type=step_type
        )

    def observation_spec(self) -> t.Tuple[specs.Array,
                                          specs.DiscreteArray,
                                          specs.DiscreteArray,
                                          t.Optional[specs.DiscreteArray]]:
        N = len(self._sim.zone_dict)
        N_lives = self._sim.max_n_lives

        return (
            specs.Array(shape=(N, 2), dtype=np.int16, name='Zones'),

            specs.DiscreteArray(num_values=N,
                                dtype=np.int16,
                                name='P-zone'),

            specs.DiscreteArray(num_values=N,
                                dtype=np.int16,
                                name='D-zone'),

            specs.DiscreteArray(num_values=N_lives + 1,
                                dtype=np.int16,
                                name='Lives') \
                                if N_lives else None
        )

    def action_spec(self) -> t.Tuple[specs.DiscreteArray,
                                     specs.DiscreteArray]:
        N = 9 # 1-hop neighbourhood

        return (
            specs.DiscreteArray(num_values=N,
                                dtype=np.int8,
                                name='P-action'),

            specs.DiscreteArray(num_values=N,
                                dtype=np.int8,
                                name='D-action'),
        )

    def reward_spec(self) -> t.Tuple[specs.Array,
                                     specs.Array]:
        return (
            specs.Array(shape=(), dtype=float, name='P-reward'),
            specs.Array(shape=(), dtype=float, name='D-reward')
        )

    def discount_spec(self) -> t.Tuple[specs.BoundedArray,
                                       specs.BoundedArray]:
        return (
            specs.BoundedArray(shape=(), dtype=float,
                               minimum=0.,
                               maximum=1.,
                               name='P-discount'),

            specs.BoundedArray(shape=(), dtype=float,
                               minimum=0.,
                               maximum=1.,
                               name='D-discount')
        )

    def close(self):
        del self._sim
        self._sim = None

    def valid_action(self, action: t.Tuple[np.int8, np.int8]) \
                    -> t.Tuple[np.int8, np.int8]:
        """
        Return the action if valid, otherwise return a No OPeration (NOP).
        """
        P_status, D_status = self._sim.valid_action(action)

        return action[0] if P_status else NOP, \
               action[1] if D_status else NOP

    def sim_finished(self):
        return self._sim.booking_request is None \
               and self._sim.booking_request_idx > 0

    def run(self, P_agent: EpsilonGreedyActor,
            D_agent: EpsilonGreedyActor):
        """
        Run simulation waiting for API calls.
        """
        self._sim.run_server(P_agent, D_agent)

    #
    # Simulation statistics
    #

    def pct_satisfied_demand(self):
        n_satisfied_trips = self._sim.n_same_zone_trips \
                          + self._sim.n_not_same_zone_trips

        n_unsatisfied_trips = self._sim.n_deaths \
                            + self._sim.n_no_close_vehicles

        return n_satisfied_trips / \
               (n_satisfied_trips + n_unsatisfied_trips)

    def get_n_lives(self):
        return self._sim.n_lives

    def get_n_accepted_incentives(self):
        return self._sim.n_incentives
