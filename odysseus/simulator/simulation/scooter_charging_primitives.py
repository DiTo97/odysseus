import datetime
import numpy as np
import simpy

from enum import Enum

# Custom imports
from odysseus.simulator.simulation_data_structures.worker import Worker
from odysseus.utils.geospatial_utils import get_od_distance


class Type(str, Enum):
    SYSTEM = 'system'


class ScooterChargingPrimitive:
    def __init__(self, env, sim):
        self.env = env

        self.start_datetime = sim.start

        self.sim_input = sim.simInput
        self.vehicles_list = sim.vehicles_list

        self.available_vehicles_dict = sim.available_vehicles_dict
        self.vehicles_zones = sim.vehicles_zones

        self.zone_dict = sim.zone_dict
        self.charging_stations_dict = sim.charging_stations_dict

        if not self.sim_input.supply_model_conf["battery_swap"]:
            raise NotImplementedError()

        # Allocate the N workers resource
        N = self.sim_input.supply_model_conf["n_workers"]
        self.workers = simpy.FilterStore(env, capacity=N)

        for i in range(N):
            init_pos = self.sim_input.supply_model \
                .initial_workers_positions[i]

            self.workers.put(Worker(env, i, init_pos))

        self.scheduled_swaps = []

        self.sim_charges = []
        self.sim_unfeasible_charge_bookings = []

        self.n_charges = 0
        self.n_charged_vehicles = 0

        self.tot_charge_distance = 0
        self.tot_charge_duration = 0

        self.n_vehicles_charging_system = 0
        self.n_vehicles_charging_users = 0
        self.dead_vehicles = set()
        self.n_dead_vehicles = 0

        self.list_system_charging_bookings = []
        self.list_users_charging_bookings = []

        self.tot_charge_return_distance = 0

        self.sim_metrics = sim.sim_metrics

    def charge_scooters_multi_zones(self, swap, path, worker):
        """
        Charge a set of scooters via battery swap potentially across multiple zones.

        Parameters
        ----------
        swap : dict
            Scheduled swap info

        path : list
            Sequence of zone Ids to visit

        worker : Worker
            Worker assigned for the job
        """
        with self.workers.get(lambda w: w.id == worker.id) as worker_req:
            # Wait for the assigned worker
            yield worker_req
            worker = worker_req.value

            worker.start_working()

            tot_distance = 0
            tot_duration = 0

            charged_vehicles = []

            # Step-by-step navigation through the path
            for j in range(1, len(path)):
                step_o = path[j - 1] # Step origin
                step_d = path[j]     # Step destination

                distance = self.get_distance(step_o, step_d,
                                             False)

                tot_distance += distance
                self.sim_metrics.update_metrics(
                    'avg_charging_step_distance', distance)

                # Perturbate the duration with a std of 1 min such that 99.73% of
                # the relocation events fall into the +-3 mins range
                duration = (distance / 1000) / self.sim_input \
                    .supply_model_conf["avg_worker_truck_speed"] # t (h)
                duration = np.random.normal(duration * 60) # t (min)

                duration = duration * 60 \
                    if duration > 0 else 0 # t (sec)

                tot_duration += duration

                # Simulate navigation time
                yield self.env.timeout(duration)
                worker.update_position(step_d)

                dead_vehicles = self.get_dead_vehicles_by_zone(step_d)

                # Number of vehicles to serve
                n = min(swap[step_d], len(dead_vehicles))

                reach_dist = distance
                reach_t = duration

                service_t = self.sim_input \
                    .supply_model_conf['avg_service_time']

                for i, v in enumerate(dead_vehicles[:n]):
                    self.n_vehicles_charging_system += 1

                    yield self.env.process(
                        self.__charge_scooter(v, step_d, service_t,
                                              reach_t, reach_dist))
                    self.__make_available(v, step_d)

                    charged_vehicles.append(v)

                    # Movements between vehicles belonging to
                    # the same zone are assumed to be negligible
                    if i == 0:
                        reach_dist = 0
                        reach_t = 0

                    self.n_vehicles_charging_system -= 1

            worker.stop_working()

            # Make the resource available
            self.workers.put(worker)

        self.__update_charging_metrics(tot_distance, tot_duration,
                                       charged_vehicles)

    def get_dead_vehicles(self):
        """
        Get the vehicles whose SoC is currently under threshold.
        """
        # available_vehicles = [v for v in self.vehicles_list if v.available]
        #
        # return [v for v in available_vehicles
        #         if v.soc.level < self.sim_input.supply_model_conf["alpha"]]

        return list(self.dead_vehicles)

    def get_distance(self, origin_id, destination_id,
                       non_zero=True):
        """
        Compute a trip's haversine distance (m) within the simulated grid.
        """
        distance = get_od_distance(
            self.sim_input.grid,
            origin_id,
            destination_id
        )

        if non_zero and distance == 0:
            distance = self.sim_input \
                .demand_model_config["bin_side_length"]

        return distance

    def get_supply_param(self, key, default_val=None):
        if key in self.sim_input.supply_model_conf:
            return self.sim_input.supply_model_conf[key]

        return default_val

    def __update_charging_metrics(self, tot_distance, tot_duration,
                                  charged_vehicles):
        self.n_charges += 1

        self.tot_charge_distance += tot_distance
        self.tot_charge_duration += tot_duration

        n_charged_vehicles = len(charged_vehicles)

        self.sim_metrics.update_metrics("min_vehicles_charged", n_charged_vehicles)
        self.sim_metrics.update_metrics("max_vehicles_charged", n_charged_vehicles)
        self.sim_metrics.update_metrics("tot_vehicles_charged", n_charged_vehicles)

    def __update_charging_stats(self, v, charge_dict):
        if "save_history" in self.sim_input.demand_model_config:
            if self.sim_input.demand_model_config["save_history"]:
                # ODySSEUS-compatible metrics
                charge_dict['operator'] = Type.SYSTEM
                charge_dict['timeout_return'] = 0

                charge_dict["cr_soc_delta"] = 0
                charge_dict["cr_soc_delta_kwh"] = v.tanktowheel_energy_from_perc(
                                                     charge_dict["cr_soc_delta"])

                charge_dict["end_time"] = charge_dict["start_time"] \
                                          + datetime.timedelta(seconds=charge_dict['duration'])

                self.sim_charges += [charge_dict]

        self.n_charged_vehicles += 1

    def __make_available(self, v, zone):
        """
        Flag a vehicle as available in a given zone.
        """
        self.available_vehicles_dict[zone].append(v.plate)
        self.vehicles_zones[v.plate] = zone

        self.n_dead_vehicles -= 1
        self.dead_vehicles.remove(v)

    def __charge_scooter(self, v, zone, service_t,
                         reach_dist, reach_t):
        """
        Charge a vehicles for a given amount of time.
        """
        charge_dict = self.__get_charge_dict(v)

        # ODySSEUS-compatible metrics
        charge_dict['zone_id']  = zone
        charge_dict['duration'] = service_t

        charge_dict['timeout_outward'] = reach_t
        charge_dict['distanc_outward'] = reach_dist

        # Fully charge the vehicle
        yield self.env.timeout(service_t)
        v.charge(charge_dict['soc_delta'])

        self.__update_charging_stats(v, charge_dict)

    def get_dead_vehicles_by_zone(self, zone):
        """
        Get the vehicles whose SoC is currently under threshold in a given zone.
        """
        dead_vehicles = self.get_dead_vehicles()

        return [v for v in dead_vehicles if v.zone == zone]

    def __get_charge_dict(self, vehicle):
        beta = self.get_supply_param('beta', 100)

        current_datetime = self.start_datetime + datetime \
            .timedelta(seconds=self.env.now)

        charge_dict = {
            'plate': vehicle.plate,
            'start_time': current_datetime,
            'start_soc': vehicle.soc.level,
            'end_soc': beta,
            'date': current_datetime.date(),
            'hour': current_datetime.hour,
            'day_hour': current_datetime.replace(minute=0,
                                                 second=0,
                                                 microsecond=0)}

        # Compute derived metrics
        charge_dict['soc_delta'] = charge_dict["end_soc"] \
                                   - charge_dict["start_soc"]

        charge_dict['soc_delta_kwh'] = vehicle.tanktowheel_energy_from_perc(
                                                   charge_dict['soc_delta'])

        return charge_dict
