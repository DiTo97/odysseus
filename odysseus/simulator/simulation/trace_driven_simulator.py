import copy
import datetime
import dm_env
import json
import numpy as np
import typing as t
import zmq

# Custom imports
from api.api_request import ApiRequest
from api.api_request import Type
from api.server import _Server

from dqn_zoo.parts import EpsilonGreedyActor

from esbdqn.utils.balancing import compute_delta

from esbdqn.utils.reward import choose_hour
from esbdqn.utils.reward import get_zone_id_from_action
from esbdqn.utils.reward import compute_pick_up_reward
from esbdqn.utils.reward import compute_drop_off_reward

from odysseus.simulator.simulation.simulator import SharedMobilitySim
from odysseus.simulator.simulation_input.sim_input import SimInput
from odysseus.utils.time_utils import update_req_time_info
from odysseus.utils.geospatial_utils import get_od_distance


# Convenience alias for infinity
Infinity = float('inf')


NOP = 0


class TraceDrivenSim(SharedMobilitySim):
    def __init__(self, sim_input,
                 n_lives: t.Optional[int],
                 rt: bool = False):
        super().__init__(sim_input, rt)

        self.max_n_lives = n_lives

        self.booking_request_idx = -1

        self.booking_lock = None
        self.n_incentives = -1
        self.n_lives = -1
        self.rng_state = None
        self.valid_zones = None

    def init_data_structures(self):
        self.booking_request_idx = 0
        self.n_incentives = 0

        self.n_lives = self.max_n_lives

        # Fix incentives RNG state
        self.rng_state = np.random.RandomState(seed=46)

        self.valid_zones = copy.deepcopy(
            self.simInput.valid_zones)
        
        self.booking_lock = self.env.event()

    def update_time_info(self):
        self.current_datetime = self.start + datetime \
                                    .timedelta(seconds=self.env.now)

        self.current_hour = self.current_datetime.hour
        self.current_weekday = self.current_datetime.weekday()

        if self.current_weekday in [5, 6]: # Saturday and Sunday
            self.current_daytype = "weekend"
        else:
            self.current_daytype = "weekday"

    def mobility_requests_generator(self):
        self.init_data_structures()

        self.env.process(self.workers_clock())

        for booking_request in self.simInput.booking_requests_list:
            if self._valid_request(booking_request):
                # Wait for the inter-arrival timeout until the next request
                #
                # The yield before the booking processing ensures that the very 1st
                # booking will happen only at its exact firing time, as SimPy timeout
                # behaves differently depending on the parameter type:
                #     - It waits for that exact time, if datetime;
                #     - It waits for that amount of time, if timedelta.
                yield self.env.timeout(booking_request["ia_timeout"])

                self.update_time_info()
                self.booking_request = \
                    update_req_time_info(booking_request)

                # Process the trip request
                self.process_booking_request(self.booking_request)

    def workers_clock(self):
        """
        Clock workers hourly scheduling operations.
        """
        if 'relocation_workers_working_hours' in self.simInput.demand_model_config:
            working_hours = self.simInput.demand_model_config[
                "relocation_workers_working_hours"].split("-")

            l, h = [int(i) for i in working_hours]

            self.simInput.demand_model_config[
                "relocation_workers_working_hours"] = (l, h)
        else:
            l, h = 0, 23

        _t = 0 # Elapsed mins
        t_Delta = 60 # min

        while True:
            _t += t_Delta
            # Wait for the next scheduling to tick
            yield self.env.timeout(60*t_Delta)

            if _t % t_Delta == 0: # 60 min
                self.hours_spent += 1
                self.update_time_info()

                self.trigger_battery_swap()
                self.trigger_relocation(l, h)

    def trigger_relocation(self, l, h):
        """
        Trigger a relocation schedule if it is within working hours.

        Parameters
        ----------
        l : int
            Starting hour in [0, 23] leq h

        h : int
            Closing hour in [0, 23]
        """
        # Check it is within working hours
        self.update_relocation_schedule = l <= self.current_hour <= h

        if self.update_relocation_schedule \
                and self.simInput.sim_scenario_conf["scooter_relocation"]:
            self.scooterRelocationStrategy.generate_relocation_schedule(
                                                  self.current_datetime,
                                                  self.current_daytype,
                                                  self.current_hour)

    def trigger_battery_swap(self):
        """
        Trigger a battery swap schedule.
        """
        if self.update_battery_swap_schedule \
                and self.simInput.sim_scenario_conf["charging_strategy"] == "proactive":
            self.chargingStrategy.generate_swap_schedule()

    def _next_valid_request(self):
        """
        Identify the next valid booking request in the trace.
        """
        # TODO: Drop invalid booking requests
        N_br = len(self.simInput.booking_requests_list)

        tot_ia_timeout = 0
        next_valid_idx = self.booking_request_idx

        for booking_request in self.simInput.booking_requests_list[
                               next_valid_idx:]:
            tot_ia_timeout += booking_request['ia_timeout']

            if self._valid_request(booking_request):
                if self._suggestion_accepted():
                    break

            next_valid_idx += 1

            if next_valid_idx >= N_br:
                if not self.booking_lock.triggered:
                    self.booking_lock.succeed()

                return -1, tot_ia_timeout

        return next_valid_idx, tot_ia_timeout

    def _suggestion_accepted(self):
        if 'incentive_willingness' in self.simInput.sim_scenario_conf:
            willingness = self.simInput.sim_scenario_conf['incentive_willingness']

            if self.rng_state.uniform(0, 1) <= willingness:
                self.n_incentives += 1
                return True

        return False

    def _valid_request(self, request) -> bool:
        return request["origin_id"]          \
                   in self.valid_zones       \
               and request["destination_id"] \
                   in self.valid_zones

    def _requests_generator(self):
        self.env.process(self.workers_clock())

        for booking_request in self.simInput.booking_requests_list:
            # Wait for the inter-arrival timeout until the next request
            #
            # The yield before the booking processing ensures that the very 1st
            # booking will happen only at its exact firing time, as SimPy timeout
            # behaves differently depending on the parameter type:
            #     - It waits for that exact time, if datetime;
            #     - It waits for that amount of time, if timedelta.
            yield self.env.timeout(booking_request["ia_timeout"])

            self.booking_request_idx += 1

            if self._valid_request(booking_request):
                self.update_time_info()
                booking_request = update_req_time_info(
                                  booking_request)

                self.booking_request = copy.deepcopy(booking_request)

                yield self.booking_lock

                self.process_booking_request(self.booking_request)

    def _observe(self, booking_request: t.Optional[t.Dict] = None) \
                -> t.Tuple[np.ndarray, int, int, int]:
        delta = compute_delta(self.available_vehicles_dict,
                              self.simInput.avg_out_flows_train,
                              self.simInput.avg_in_flows_train,
                              self.current_daytype,
                              self.current_hour)

        num_avail_vehicles = {
            z: len(V) for z, V
            in self.available_vehicles_dict.items()
        }

        asc_sorted_zone_ids = sorted(self.zone_dict.keys())

        # Extract the current state, X_i
        X = [[num_avail_vehicles[z], delta[z]]
            for z in asc_sorted_zone_ids]

        p_zone_idx, d_zone_idx = None, None

        if booking_request is None:
            booking_request = self.booking_request

        if booking_request is not None:
            p_zone_idx = asc_sorted_zone_ids.index(
                booking_request['origin_id'])
            d_zone_idx = asc_sorted_zone_ids.index(
                booking_request['destination_id'])

        return np.asarray(X), p_zone_idx, \
               d_zone_idx, self.n_lives   \
               if self.max_n_lives else None

    def reset(self, sim_input: SimInput, fresh: bool = True):
        """
        Reset the simulator back to the beginning.

        Parameters
        ----------
        sim_input : SimInput
            Simulation parameters

        fresh : bool
            Whether to generate a fresh input. The default is True.
        """
        # 1. Reset the simulator
        self.init(sim_input, fresh)
        self.init_data_structures()

        # 2. Generate the mobility requests
        self.env.process(
            self._requests_generator()
        )

        request_idx, ia_timeout = self._next_valid_request()
        self.env.run(until=self.env.now + ia_timeout + 0.001)

        if request_idx == -1:
            self.booking_request = None

        # 3. Return the initial state, X_0
        return self._observe()

    def get_cumulative_rewards(self, action: t.Tuple[np.int16, np.int16],
                               pick_up_valid: bool, drop_off_valid: bool):

        # Get booking request info
        original_pick_up_zone = self.booking_request['origin_id']
        original_drop_off_zone = self.booking_request['destination_id']

        date = self.booking_request['date']
        hour = choose_hour(self.booking_request,
                           self.simInput.city)

        # Get suggested and final actions
        pick_up_zone_sugg = get_zone_id_from_action(action=action[0], no_op_zone_id=original_pick_up_zone,
                                                    neighbors_dict=self.neighbors_dict)
        drop_off_zone_sugg = get_zone_id_from_action(action=action[1], no_op_zone_id=original_drop_off_zone,
                                                     neighbors_dict=self.neighbors_dict)
        if pick_up_valid:
            final_pick_up_zone = pick_up_zone_sugg
        else:
            final_pick_up_zone = original_pick_up_zone

        if drop_off_valid:
            final_drop_off_zone = drop_off_zone_sugg
        else:
            final_drop_off_zone = original_drop_off_zone

        # Deltas
        deltas_dict = compute_delta(self.available_vehicles_dict,
                                    self.simInput.avg_out_flows_train,
                                    self.simInput.avg_in_flows_train,
                                    self.current_daytype,
                                    self.current_hour)

        # Get all the bookings
        bookings_df = self.simInput.bookings.dropna()

        # Pick-up agent
        pick_up_reward = np.zeros(9)
        for i in range(9):
            current_zone_id = get_zone_id_from_action(action=i, no_op_zone_id=original_pick_up_zone,
                                                      neighbors_dict=self.neighbors_dict)
            pick_up_reward[i] = compute_pick_up_reward(bookings_df, deltas_dict, self.available_vehicles_dict,
                                                       current_zone_id, date, hour, final_pick_up_zone)

        # Enforce a worsening reward, -2*max(R),
        # if the action was invalid
        if pick_up_valid:
            pick_up_reward = np.nanmean(pick_up_reward)
        else:
            pick_up_reward = -2 * np.nanmax(np.abs(pick_up_reward))

        _, pick_up_vehicle, _ = self.find_vehicle(final_pick_up_zone)

        if pick_up_vehicle is not None:
            pick_up_vehicle = self.vehicles_list[pick_up_vehicle]
            found_vehicle = True
        else:
            found_vehicle = False

        # Drop-off agent
        drop_off_reward = np.zeros(9)
        driving_distance = self.booking_request['driving_distance']
        alpha = self.simInput.supply_model_conf["alpha"]

        for i in range(9):
            current_zone_id = get_zone_id_from_action(action=i, no_op_zone_id=original_drop_off_zone,
                                                      neighbors_dict=self.neighbors_dict)

            drop_off_reward[i] = compute_drop_off_reward(bookings_df, deltas_dict, self.available_vehicles_dict,
                                                         self.chargingStrategy, pick_up_vehicle,
                                                         driving_distance, alpha, current_zone_id, date, hour,
                                                         final_drop_off_zone)

        if drop_off_valid:
            drop_off_reward = np.nanmean(drop_off_reward)
        else:
            drop_off_reward = -2 * np.nanmax(np.abs(drop_off_reward))

        return (pick_up_reward, drop_off_reward), found_vehicle

    def step(self, action: t.Tuple[np.int16, np.int16]):
        """

        Returns
        -------
        observation, reward, done, info
        """
        pick_up_valid, drop_off_valid = self.valid_action(action)

        reward, found_vehicle = self.get_cumulative_rewards(action,
            pick_up_valid, drop_off_valid)

        pick_up_action = action[0]
        if pick_up_valid is False:
            pick_up_action = NOP

        drop_off_action = action[1]
        if drop_off_valid is False:
            drop_off_action = NOP

        valid_action = (pick_up_valid,
                        drop_off_valid)
        finished = False

        if self.max_n_lives \
                and (not all(valid_action)
                     or not found_vehicle):
            self.n_lives -= 1

            if self.n_lives == 0:
                finished = True

        self._update_booking_request(pick_up_action, drop_off_action)

        request_idx, ia_timeout = self._next_valid_request()

        # Go to the next booking request
        self.booking_lock.succeed()
        self.env.run(until=self.env.now + ia_timeout + 0.001)
        self.booking_lock = self.env.event()

        if request_idx == -1:
            self.booking_request = None
            finished = True

        observation = self._observe()

        # if self._terminal_state(observation):
        #     finished = True

        return observation, reward, finished, None

    def _terminal_state(self, observation):
        pass

    def _update_booking_request(self, pick_up_action, drop_off_action):
        update_end_time = False

        if pick_up_action != NOP:
            new_pick_up_zone = get_zone_id_from_action(
                action=pick_up_action,
                no_op_zone_id=self.booking_request['origin_id'],
                neighbors_dict=self.neighbors_dict)

            self.booking_request['origin_id'] = \
                new_pick_up_zone

            update_end_time = True

        if drop_off_action != NOP:
            new_drop_off_zone = get_zone_id_from_action(
                action=drop_off_action,
                no_op_zone_id=self.booking_request['destination_id'],
                neighbors_dict=self.neighbors_dict)

            self.booking_request['destination_id'] = \
                new_drop_off_zone

            euclidean_distance_org = self.booking_request['euclidean_distance']
            euclidean_distance_new = get_od_distance(
                self.simInput.grid,
                self.booking_request['origin_id'],
                new_drop_off_zone)

            if euclidean_distance_new > euclidean_distance_org:
                self.booking_request['euclidean_distance'] = \
                    euclidean_distance_new

                # Ratio of 1.4 taken from T. Damoulas, et al., Road Distance and Travel Time
                # for an Improved House Price Kriging Predictor, 2018
                self.booking_request['driving_distance'] =   \
                    euclidean_distance_new * 1.4

                update_end_time = True

        if update_end_time:
            # Perturbate the end time by at most 10 minutes,
            # not to introduce too much noise within
            # the distribution of ODySSEUS bookings
            self.booking_request['end_time'] += \
                datetime.timedelta(seconds=np.random.uniform(0, 600))

            self.booking_request['duration'] = \
                (self.booking_request['end_time'] -
                 self.booking_request['start_time']).total_seconds()

    def valid_action(self, action,
                     booking_request:
                         t.Optional[t.Dict] = None):
        if booking_request is None:
            booking_request = self.booking_request

        original_pick_up_zone  = booking_request['origin_id']
        original_drop_off_zone = booking_request['destination_id']

        pick_up_valid = True
        drop_off_valid = True

        # Extract zone Ids from actions
        pick_up_zone_sugg = get_zone_id_from_action(
            action=action[0], no_op_zone_id=original_pick_up_zone,
            neighbors_dict=self.neighbors_dict)

        drop_off_zone_sugg = get_zone_id_from_action(
            action=action[1], no_op_zone_id=original_drop_off_zone,
            neighbors_dict=self.neighbors_dict)

        # Any suggestion is invalid,
        # if its zone Id is None
        if pick_up_zone_sugg is None:
            pick_up_valid = False

        if drop_off_zone_sugg is None:
            drop_off_valid = False

        # If the suggested pick-up zone exists on the grid
        if pick_up_zone_sugg is not None:
            # If suggested zones are both NOP but only one of the
            # two was originally NOP, then reject the other sug
            if pick_up_zone_sugg == drop_off_zone_sugg:
                if (action[0] == NOP) \
                        and (action[1] != NOP):
                    drop_off_valid = False
                elif (action[1] == NOP) \
                        and (action[0] != NOP):
                    pick_up_valid = False
                else:
                    pick_up_valid  = False
                    drop_off_valid = False

            # If the suggested pick-up zone coincides with the original
            # drop off zone but the original pick-up zone was different
            if (pick_up_zone_sugg == original_drop_off_zone) \
                    and (action[0] != NOP):
                pick_up_valid = False

        # If the suggested drop-off zone exists on the grid
        if drop_off_zone_sugg is not None:
            # If the suggested drop-off zone coincides with the original
            # pick-up zone but the original drop-off zone was different
            if (drop_off_zone_sugg == original_pick_up_zone) \
                    and (action[1] != NOP):
                drop_off_valid = False

        # If the suggested pick-up zone is valid up to now,
        # make sure it has available vehicles with enough SoC
        # to cover the whole modified trip
        if pick_up_valid:
            vehicle_avail, _, _ = \
                self.find_vehicle(pick_up_zone_sugg,
                                  booking_request)

            if vehicle_avail is False:
                pick_up_valid = False

        return pick_up_valid, drop_off_valid

    #
    # Stuff related to ESBDQN API
    #

    def run_server(self, P_agent: EpsilonGreedyActor,
            D_agent: EpsilonGreedyActor):
        _Server['ENDPOINT'] = '{}://*:{}'.format(_Server['PROTOCOL'],
                                                 _Server['PORT'])

        Delta_t: float = 0.033

        with zmq.Context() as context:
            server = context.socket(zmq.REP)
            server.bind(_Server['ENDPOINT'])

            self.env.process(
                self.mobility_requests_generator())

            # Synchronize internal clock
            self.env.sync()

            try:
                while True:
                    try:
                        api_request = server.recv(flags=zmq.NOBLOCK)

                        if api_request is not None:
                            api_request = json.loads(api_request.decode())
                            api_request = ApiRequest.parse(api_request)

                            response = self.consume_request(
                                api_request, (P_agent, D_agent))

                            server.send(response.encode())
                    except zmq.Again:
                        pass

                    if self.env.peek() == Infinity:
                        break

                    self.env.run(until=self.env.now
                                       + Delta_t)
            except Exception as e:
                pass
            finally:
                server.close()

    def consume_request(self, _request: ApiRequest,
                        agents: t.Tuple[EpsilonGreedyActor,
                                        EpsilonGreedyActor]) \
                       -> t.Text:
        P_agent, D_agent = agents

        response = {}

        if _request.type == Type.Pick_up:
            response = \
                self.compute_P_sugg(
                    _request.data, P_agent)
        elif _request.type == Type.Drop_off:
            response = \
                self.compute_D_sugg(
                    _request.data, D_agent)
        elif _request.type == Type.Incentive:
            response = \
                self.compute_incentive(
                     _request.data)
        else:
            response['error'] = 'Unknown request'

        return json.dumps(response)

    def make_booking_request(self, data: t.Dict):
        P_zone = int(data['P_zone'])
        D_zone = int(data['D_zone'])

        booking_request = {
            'origin_id': P_zone,
            'destination_id': D_zone,
            'euclidean_distance':
                get_od_distance(
                    self.simInput.grid,
                    P_zone, D_zone)
        }

        # Ratio of 1.4 taken from T. Damoulas, et al., Road Distance and Travel Time
        # for an Improved House Price Kriging Predictor, 2018
        booking_request['driving_distance'] = \
            booking_request['euclidean_distance'] * 1.4

        return booking_request

    def compute_P_sugg(self, data: t.Dict,
                       P_agent: EpsilonGreedyActor):
        try:
            booking_request = \
                self.make_booking_request(data)
        except KeyError:
            return {
                'error': 'Invalid zone Ids'
                   }

        observation = self._observe(booking_request)

        P_a = P_agent.step(dm_env.restart(
                           observation))

        P_status, _ = \
            self.valid_action((P_a, 0),
                              booking_request)

        if not P_status:
            return {
                'error': 'No suggestion available'
                   }

        P_zone_id_sugg = \
            get_zone_id_from_action(P_a,
                booking_request['origin_id'],
                self.neighbors_dict,
            )

        status, vehicle_id, _ = \
            self.find_vehicle(P_zone_id_sugg,
                              booking_request)

        if vehicle_id is None:
            return {
                'error': 'No suggestion available'
                   }

        vehicle = self.vehicles_list[vehicle_id]

        if P_a == NOP:
            return {
                'P_zone_sugg': P_zone_id_sugg,
                'vehicle': str(vehicle),
                'info': 'No better pick-up zone Id'
                   }

        return {
            'P_zone_sugg': P_zone_id_sugg,
            'vehicle': str(vehicle)
               }

    def compute_D_sugg(self, data: t.Dict,
                       D_agent: EpsilonGreedyActor):
        try:
            booking_request = \
                self.make_booking_request(data)
        except KeyError:
            return {
                'error': 'Invalid zone Ids'
                   }

        observation = self._observe(booking_request)

        D_a = D_agent.step(dm_env.restart(
                           observation))

        _, D_status = \
            self.valid_action((0, D_a),
                              booking_request)

        if not D_status:
            return {
                'error': 'No suggestion available'
                   }

        D_zone_id_sugg = \
            get_zone_id_from_action(D_a,
                booking_request['destination_id'],
                self.neighbors_dict,
            )

        if D_a == NOP:
            return {
                'D_zone_sugg': D_zone_id_sugg,
                'info': 'No better drop-off zone Id'
                   }

        return {
            'D_zone_sugg': D_zone_id_sugg
               }

    def compute_incentive(self, data: t.Dict) \
                         -> t.Dict:
        P_incentive = None
        D_incentive = None

        min_zone_id = self.valid_zones.min()
        max_zone_id = self.valid_zones.max()

        d = max_zone_id - min_zone_id

        if 'P_zone' in data.keys():
            P_zone = int(data['P_zone'])
            P_zone_sugg = int(data['P_zone_sugg'])

            if P_zone not in self.valid_zones:
                return {
                    'error': 'Invalid pick-up zone Id'
                       }

            if P_zone_sugg not in self.valid_zones:
                return {
                    'error': 'Invalid suggested pick-up zone Id'
                       }

            P_incentive = 50.*abs(P_zone_sugg - P_zone) / d

        if 'D_zone' in data.keys():
            D_zone = int(data['D_zone'])
            D_zone_sugg = int(data['D_zone_sugg'])

            if D_zone not in self.valid_zones:
                return {
                    'error': 'Invalid drop-off zone Id'
                       }

            if D_zone_sugg not in self.valid_zones:
                return {
                    'error': 'Invalid suggested drop-off zone Id'
                       }

            D_incentive = 50.*abs(D_zone_sugg - D_zone) / d

        response = {
            'currency': 'EUR'
                   }

        if P_incentive is not None:
            response['P_incentive'] = \
                round(P_incentive, 3)

        if D_incentive is not None:
            response['D_incentive'] = \
                round(D_incentive, 3)

        return response
