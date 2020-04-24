import datetime

import numpy as np

from simulator.utils.vehicle_utils import soc_to_kwh
from simulator.utils.vehicle_utils import get_soc_delta


def get_charging_time (soc_delta,
                       battery_capacity = 3.5,
                       charging_efficiency = 0.92,
                       charger_rated_power = 3.7):

    return (soc_delta * 60 * battery_capacity) / (charging_efficiency * charger_rated_power * 100)


def get_charging_soc (duration,
                    battery_capacity = 3.5,
                    charging_efficiency = 0.92,
                    charger_rated_power = 3.7):

    return (charging_efficiency * charger_rated_power * 100 * duration) / (60 * battery_capacity)


def init_charge (booking_request, vehicles_soc_dict, vehicle, beta):

    charge = {}
    charge["plate"] = vehicle
    charge["start_time"] = booking_request["end_time"]
    charge["date"] = charge["start_time"].date()
    charge["hour"] = charge["start_time"].hour
    charge["day_hour"] = \
        charge["start_time"].replace(minute=0, second=0, microsecond=0)
    charge["start_soc"] = vehicles_soc_dict[vehicle]
    charge["end_soc"] = beta
    charge["soc_delta"] = charge["end_soc"] - charge["start_soc"]
    charge["soc_delta_kwh"] = soc_to_kwh(charge["soc_delta"])
    return charge


class EFFCS_ChargingPrimitives:

    def charge_vehicle (
            self,
            charge_dict
    ):

        charge = charge_dict["charge"]
        resource = charge_dict["resource"]
        vehicle = charge_dict["vehicle"]
        operator = charge_dict["operator"]
        zone_id = charge_dict["zone_id"]
        timeout_outward = charge_dict["timeout_outward"]
        timeout_return = charge_dict["timeout_return"]
        cr_soc_delta = charge_dict["cr_soc_delta"]

        def check_queuing ():
            if self.simInput.sim_scenario_conf["queuing"]:
                return True
            else:
                if resource.count < resource.capacity:
                    return True
                else:
                    return False

        charge["operator"] = operator
        charge["zone_id"] = zone_id
        charge["timeout_outward"] = timeout_outward
        charge["timeout_return"] = timeout_return
        charge["cr_soc_delta"] = cr_soc_delta
        charge["cr_soc_delta_kwh"] = soc_to_kwh(cr_soc_delta)

        if self.simInput.sim_scenario_conf["battery_swap"]:
            if check_queuing():
                with self.workers.request() as worker_request:
                    yield worker_request
                    self.n_vehicles_charging_system += 1
                    yield self.env.timeout(charge["timeout_outward"])
                    charge["start_soc"] -= charge["cr_soc_delta"]
                    self.sim_charges += [charge]
                    yield self.env.timeout(charge["duration"])
                    self.n_vehicles_charging_system -= 1
                    yield self.env.timeout(charge["timeout_return"])
                    self.vehicles_soc_dict[vehicle] = charge["end_soc"]
                    charge["end_soc"] -= charge["cr_soc_delta"]

        elif operator == "system":
            if check_queuing():
                with self.workers.request() as worker_request:
                    yield worker_request
                    yield self.env.timeout(charge["timeout_outward"])
                    charge["start_soc"] -= charge["cr_soc_delta"]
                    self.sim_charges += [charge]
                    with resource.request() as charging_request:
                        yield charging_request
                        self.n_vehicles_charging_system += 1
                        yield self.env.timeout(charge["duration"])
                    self.n_vehicles_charging_system -= 1
                    yield self.env.timeout(charge["timeout_return"])
                    self.vehicles_soc_dict[vehicle] = charge["end_soc"]
                    charge["end_soc"] -= charge["cr_soc_delta"]

        elif operator == "users":
            if resource.count < resource.capacity:
                self.sim_charges += [charge]
                with resource.request() as charging_request:
                    yield charging_request
                    self.n_vehicles_charging_users += 1
                    yield self.env.timeout(charge["duration"])
                self.vehicles_soc_dict[vehicle] = charge["end_soc"]
                self.n_vehicles_charging_users -= 1

        charge["end_time"] = charge["start_time"] + datetime.timedelta(seconds=charge["duration"])

    def get_charge_dict(self, vehicle, charge, booking_request, operator):

        if self.simInput.sim_scenario_conf["battery_swap"]:
            charging_zone_id = booking_request["destination_id"]
            if self.simInput.sim_scenario_conf["time_estimation"]:
                timeout_outward = np.random.exponential(
                    self.simInput.sim_scenario_conf[
                        "avg_reach_time"
                    ] * 60
                )
                charge["duration"] = np.random.exponential(
                    self.simInput.sim_scenario_conf[
                        "avg_service_time"
                    ] * 60
                )
                timeout_return = 0
                cr_soc_delta = 0
            else:
                timeout_outward = 0
                charge["duration"] = 0
                timeout_return = 0
                cr_soc_delta = 0

        charge_dict = {
            "charge": charge,
            "resource": self.workers,
            "vehicle": vehicle,
            "operator": operator,
            "zone_id": charging_zone_id,
            "timeout_outward": timeout_outward,
            "timeout_return": timeout_return,
            "cr_soc_delta": cr_soc_delta
        }

        return charge_dict

    def compute_hub_charging_params (self, booking_request, vehicle):

        unfeasible_charge_flag = False

        if self.simInput.sim_scenario_conf["time_estimation"]:

            charging_zone_id = self.simInput.sim_scenario_conf["hub_zone"]
            timeout_outward = self.get_timeout(
                booking_request["destination_id"],
                charging_zone_id
            )

            if not self.simInput.sim_scenario_conf["relocation"]:
                timeout_return = 0
            elif self.simInput.sim_scenario_conf["relocation"]:
                timeout_return = timeout_outward

            cr_soc_delta = self.get_cr_soc_delta(
                booking_request["destination_id"],
                charging_zone_id
            )

            if cr_soc_delta > booking_request["end_soc"]:
                unfeasible_charge_flag = True
                self.dead_vehicles.add(vehicle)
                self.n_dead_vehicles = len(self.dead_vehicles)
                self.sim_unfeasible_charge_bookings.append(booking_request)

        else:

            charging_zone_id = -1
            timeout_outward = 0
            timeout_return = 0
            cr_soc_delta = 0

        return unfeasible_charge_flag, \
                charging_zone_id, \
                timeout_outward, \
                timeout_return, \
                cr_soc_delta

    # Compute cp charging params

    def compute_cp_charging_params (self, booking_request, vehicle):

        timeout_outward = 0
        timeout_return = 0
        cr_soc_delta = 0
        unfeasible_charge_flag = False

        zones_by_distance = self.simInput.zones_cp_distances.loc[
            booking_request["destination_id"]
        ].sort_values()

        free_pole_flag = 0
        for zone in zones_by_distance.index:
            if len(self.charging_poles_dict[zone].users) < self.charging_poles_dict[zone].capacity:
                free_pole_flag = 1
                charging_zone_id = zone
                cr_soc_delta = self.get_cr_soc_delta \
                    (booking_request["destination_id"],
                     charging_zone_id)
                if cr_soc_delta > booking_request["end_soc"]:
                    free_pole_flag = 0
                else:
                    break

        if free_pole_flag == 0:
            charging_zone_id = \
                self.simInput.closest_cp_zone \
                    [booking_request["destination_id"]]

        charging_station = \
            self.charging_poles_dict\
            [charging_zone_id]

        if self.simInput.sim_scenario_conf["time_estimation"]:

            timeout_outward = self.get_timeout\
                (booking_request["destination_id"],
                 charging_zone_id)

            if not self.simInput.sim_scenario_conf["relocation"]:
                timeout_return = 0
            elif self.simInput.sim_scenario_conf["relocation"]:
                timeout_return = timeout_outward

            cr_soc_delta = self.get_cr_soc_delta\
                (booking_request["destination_id"],
                 charging_zone_id)

            if cr_soc_delta > booking_request["end_soc"]:

                unfeasible_charge_flag = True
                self.dead_vehicles.add(vehicle)
                self.n_dead_vehicles = len(self.dead_vehicles)
                self.sim_unfeasible_charge_bookings.append(booking_request)


        else:

            timeout_outward = 0
            timeout_return = 0
            cr_soc_delta = 0

        return unfeasible_charge_flag, \
                charging_station, \
                charging_zone_id, \
                timeout_outward, \
                timeout_return, \
                cr_soc_delta

    def check_system_charge (self, booking_request, vehicle):

        if self.vehicles_soc_dict[vehicle] < self.simInput.sim_scenario_conf["alpha"]:
            charge = init_charge(
                booking_request,
                self.vehicles_soc_dict,
                vehicle,
                self.simInput.sim_scenario_conf["beta"]
            )
            return True, charge
        else:
            return False, None

    def check_user_charge (self, booking_request, vehicle):

        if booking_request["end_soc"] < self.simInput.sim_scenario_conf["beta"]:
            if booking_request["end_soc"] < self.simInput.sim_scenario_conf["alpha"]:
                destination_id = booking_request["destination_id"]
                if destination_id in self.charging_poles_dict.keys():
                    if np.random.binomial(1, self.simInput.sim_scenario_conf["willingness"]):
                        charge = init_charge(
                            booking_request,
                            self.vehicles_soc_dict,
                            vehicle,
                            self.simInput.sim_scenario_conf["beta"]
                        )
                        return True, charge
                    else:
                        return False, None
                else:
                    return False, None
            else:
                return False, None
        else:
            return False, None

    def get_timeout (self, origin_id, destination_id):
        distance = self.simInput.od_distances.loc[origin_id, destination_id] / 1000
        return distance / 15 * 3600

    def get_cr_soc_delta (self, origin_id, destination_id):
        distance = self.simInput.od_distances.loc[origin_id, destination_id] / 1000
        return get_soc_delta(distance)
