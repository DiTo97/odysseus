import copy
import datetime
import pytz
import simpy
import typing as t

from abc import ABC, abstractmethod
from calendar import monthrange

# Proactive charging
from odysseus.simulator.simulation.scooter_charging_strategies import ScooterChargingStrategy
from odysseus.simulator.simulation.scooter_relocation_strategies import ScooterRelocationStrategy
from odysseus.simulator.simulation.sim_metrics import SimMetrics
from odysseus.simulator.simulation.vehicle_relocation_strategies import VehicleRelocationStrategy
from odysseus.simulator.simulation_data_structures.vehicle import Vehicle
from odysseus.simulator.simulation_data_structures.zone import Zone
from odysseus.simulator.simulation_input.sim_input import SimInput
from odysseus.simulator.simulation_input.vehicle_conf import vehicle_conf


class SharedMobilitySim(ABC):
    def __init__(self, simInput, rt: bool = False):
        self.start = None
        self.end = None

        self.total_seconds = None
        self.hours_spent = None
        self.booking_request = None

        self.current_datetime = None
        self.current_hour = None
        self.current_weekday = None

        self.current_daytype = None

        self.update_relocation_schedule = None
        self.update_battery_swap_schedule = None

        self.available_vehicles_dict = None

        self.neighbors_dict = None
        self.closest_valid_zone = None

        self.vehicles_soc_dict = None
        self.vehicles_zones = None

        self.env = None

        self.sim_booking_requests = None
        self.sim_bookings = None

        self.sim_booking_requests_deaths = None  # No energy in nearby vehicles
        self.sim_unsatisfied_requests = None
        self.sim_no_close_vehicle_requests = None

        self.n_booking_requests = None
        self.n_same_zone_trips = None
        self.n_not_same_zone_trips = None
        self.n_no_close_vehicles = None
        self.n_deaths = None
        self.n_booked_vehicles = None

        self.current_hour_origin_count = None
        self.current_hour_destination_count = None
        self.current_hour_n_bookings = None

        self.tot_mobility_distance = None
        self.tot_mobility_duration = None

        self.list_n_vehicles_charging_system = None
        self.list_n_vehicles_charging_users = None

        self.list_n_vehicles_booked = None
        self.list_n_vehicles_available = None
        self.list_n_vehicles_dead = None

        self.list_n_scooters_relocating = None
        self.list_n_vehicles_relocating = None

        self.vehicles_zones_history = None

        self.charging_return_distance = None
        self.charging_outward_distance = None

        self.charging_stations_dict = None

        self.zone_dict = None
        self.vehicles_list = None

        self.sim_metrics = None

        self.scooterRelocationStrategy = None
        self.chargingStrategy = None

        self.simInput = None
        self.init(simInput, False, rt)

    def init(self, sim_input: SimInput,
             fresh: bool,
             rt: bool = False):
        """
        Initialize the simulator.

        Parameters
        ----------
        sim_input : SimInput
            Simulation parameters

        fresh : bool
            Whether to generate a fresh input.

        rt : bool
            Whether the SimPy env should be real-time.
        """
        self.simInput = copy.deepcopy(sim_input)

        if fresh:
            self.simInput.refresh()

        self.start = datetime.datetime(
            self.simInput.demand_model_config["year"],
            self.simInput.demand_model_config["month_start"],
            1, tzinfo=pytz.UTC
        )

        if self.simInput.demand_model_config["month_end"] == 13:
            self.end = datetime.datetime(
                self.simInput.demand_model_config["year"] + 1,
                1,
                31, tzinfo=pytz.UTC
            )
        else:
            self.end = datetime.datetime(
                self.simInput.demand_model_config["year"],
                self.simInput.demand_model_config["month_end"],
                monthrange(self.simInput.demand_model_config["year"],
                           self.simInput.demand_model_config["month_end"])[1], tzinfo=pytz.UTC
            )

        self.total_seconds = (self.end - self.start).total_seconds()
        self.hours_spent = 0

        self.current_datetime = self.start
        self.current_hour = self.current_datetime.hour
        self.current_weekday = self.current_datetime.weekday()

        if self.start.weekday() in [5, 6]:
            self.current_daytype = "weekend"
        else:
            self.current_daytype = "weekday"

        self.booking_request = None

        self.update_relocation_schedule = True
        self.update_battery_swap_schedule = True

        self.available_vehicles_dict = self.simInput.supply_model.available_vehicles_dict

        self.neighbors_dict = self.simInput.neighbors_dict
        self.closest_valid_zone = self.simInput.closest_valid_zone

        self.vehicles_soc_dict = copy.deepcopy(self.simInput.supply_model.vehicles_soc_dict)
        self.vehicles_zones    = copy.deepcopy(self.simInput.supply_model.vehicles_zones)

        self.env = simpy.Environment() if not rt \
            else simpy.rt.RealtimeEnvironment(strict=False)

        self.sim_booking_requests = []
        self.sim_bookings = []

        self.sim_booking_requests_deaths = []  # No energy in nearby vehicles
        self.sim_unsatisfied_requests = []
        self.sim_no_close_vehicle_requests = []

        self.n_booking_requests = 0
        self.n_same_zone_trips = 0
        self.n_not_same_zone_trips = 0
        self.n_no_close_vehicles = 0
        self.n_deaths = 0
        self.n_booked_vehicles = 0

        self.current_hour_origin_count = {}
        self.current_hour_destination_count = {}
        self.current_hour_n_bookings = 0

        self.tot_mobility_distance = 0
        self.tot_mobility_duration = 0

        self.list_n_vehicles_charging_system = []
        self.list_n_vehicles_charging_users = []

        self.list_n_vehicles_booked = []
        self.list_n_vehicles_available = []
        self.list_n_vehicles_dead = []

        self.list_n_scooters_relocating = []
        self.list_n_vehicles_relocating = []

        self.vehicles_zones_history = []

        self.charging_return_distance = []
        self.charging_outward_distance = []

        self.charging_stations_dict = {}

        self.zone_dict = {}

        for zone_id in self.simInput.valid_zones:
            self.zone_dict[zone_id] = Zone(self.env, zone_id, self.start, self.available_vehicles_dict[zone_id])

        self.vehicles_list = []

        for i in range(self.simInput.n_vehicles_sim):
            vehicle_object = Vehicle(
                self.env, i, self.vehicles_zones[i], self.vehicles_soc_dict[i],
                vehicle_conf, self.simInput.supply_model.energy_mix, self.simInput.supply_model_conf, self.start
            )
            self.vehicles_list.append(vehicle_object)

        if "alpha_policy" in self.simInput.supply_model_conf:
            if self.simInput.supply_model_conf["alpha_policy"] == "auto":
                self.simInput.supply_model_conf["alpha"] = self.vehicles_list[0].consumption_to_percentage(
                    self.vehicles_list[0].distance_to_consumption(
                        self.simInput.max_driving_distance / 1000
                    )
                )
            else:
                raise ValueError("Policy for alpha not recognised!")

        metrics_dict = {
            # Relocation metrics
            "min_vehicles_relocated": "min",
            "max_vehicles_relocated": "max",
            "avg_relocation_step_distance": "avg",
            "tot_vehicles_moved": "sum",

            # Charging metrics
            "min_vehicles_charged": "min",
            "max_vehicles_charged": "max",
            "avg_charging_step_distance": "avg",
            "tot_vehicles_charged": "sum",

            "cum_relo_ret_t": "sum",
        }

        self.sim_metrics = SimMetrics(metrics_dict)

        if self.simInput.supply_model_conf["battery_swap"] \
                and self.simInput.supply_model_conf["scooter_relocation"]:
            self.scooterRelocationStrategy = ScooterRelocationStrategy(self.env, self)
        elif self.simInput.supply_model_conf["vehicle_relocation"]:
            self.vehicleRelocationStrategy = VehicleRelocationStrategy(self.env, self)

        self.chargingStrategy = ScooterChargingStrategy(self.env, self)

        # Mark initial dead vehicles
        for vehicle in self.vehicles_list:
            if vehicle.soc.level < self.simInput.supply_model_conf["alpha"]:
                self.chargingStrategy.n_dead_vehicles += 1
                self.chargingStrategy.dead_vehicles.add(vehicle)

    def find_vehicle(self, zone_id,
                     booking_request:
                         t.Optional[t.Dict] = None):
        """
        Once a set of available vehicles has been established in a given zone,
        find the most appropriate vehicle in terms of maximum SoC.

        Parameters
        ----------
        zone_id : int
            Zone Id to look into for a vehicle.
            
        booking_request : t.Optional[t.Dict]
            Specific booking request.
            The default is None.
        """
        available_vehicles_soc_dict = {k: self.vehicles_list[k].soc.level for k in
                                       self.available_vehicles_dict[zone_id]}

        if not available_vehicles_soc_dict.values():
            return False, None, None

        max_soc = max(available_vehicles_soc_dict.values())
        max_soc_vehicle = max(available_vehicles_soc_dict,
                              key=available_vehicles_soc_dict.get)

        if booking_request is None:
            booking_request = self.booking_request

        # Check if the available vehicle with the highest SoC
        # has enough charge to complete the trip
        if self.vehicles_list[max_soc_vehicle].soc.level > abs(
                self.vehicles_list[max_soc_vehicle].consumption_to_percentage(
                    self.vehicles_list[max_soc_vehicle].distance_to_consumption(
                        booking_request['driving_distance'] / 1000)
                )
        ):
            return True, max_soc_vehicle, max_soc
        else:
            return False, max_soc_vehicle, max_soc

    def schedule_booking(self, vehicle, zone_id):
        """
        Schedule a trip end event, after the trip sanity has been verified

        Parameters
        ----------
        vehicle : int
            Vehicle Id

        zone_id : int
            Zone Id to look into for the vehicle.
        """
        self.tot_mobility_distance += self.booking_request["driving_distance"]
        self.tot_mobility_duration += self.booking_request["duration"]

        if self.simInput.supply_model_conf["scooter_relocation"] \
                and self.simInput.supply_model_conf["scooter_relocation_strategy"] in ["predictive"]:
            self.scooterRelocationStrategy.update_current_hour_stats(self.booking_request)

        if "save_history" in self.simInput.demand_model_config:
            if self.simInput.demand_model_config["save_history"]:
                self.sim_bookings += [self.booking_request]

        if vehicle in self.available_vehicles_dict[zone_id]:
            self.available_vehicles_dict[zone_id].remove(vehicle)

        if vehicle in self.vehicles_zones:
            del self.vehicles_zones[vehicle]

        self.booking_request["start_soc"] = self.vehicles_list[vehicle].soc.level

        self.n_booked_vehicles += 1

        self.booking_request["plate"] = vehicle

        self.zone_dict[self.booking_request["origin_id"]].remove_vehicle(self.booking_request["start_time"])

        # Wait for the trip to end and update the state
        yield self.env.process(self.vehicles_list[vehicle].booking(self.booking_request))

        self.zone_dict[self.booking_request["destination_id"]].add_vehicle(
            self.booking_request["start_time"] + datetime.timedelta(seconds=self.booking_request['duration'])
        )

        self.booking_request["end_soc"] = self.vehicles_list[vehicle].soc.level

        self.n_booked_vehicles -= 1

        # Flag the vehicle as available if it still retains
        # a supra-threshold SoC level after the trip...
        if self.booking_request["end_soc"] >= self.simInput.supply_model_conf["alpha"]:
            relocation_zone_id = self.booking_request['destination_id']

            self.available_vehicles_dict[relocation_zone_id].append(vehicle)
            self.vehicles_zones[vehicle] = relocation_zone_id
        else: # ...otherwise flag it as dead
            self.chargingStrategy.n_dead_vehicles += 1
            self.chargingStrategy.dead_vehicles.add(
                        self.vehicles_list[vehicle])

    def process_booking_request(self, booking_request):
        #
        # Update logging lists
        #
        self.booking_request = copy.deepcopy(booking_request)

        self.list_n_vehicles_booked += [self.n_booked_vehicles]
        self.list_n_vehicles_charging_system += [self.chargingStrategy.n_vehicles_charging_system]
        self.list_n_vehicles_charging_users += [self.chargingStrategy.n_vehicles_charging_users]
        self.list_n_vehicles_dead += [self.chargingStrategy.n_dead_vehicles]

        n_vehicles_charging = self.chargingStrategy.n_vehicles_charging_system \
                            + self.chargingStrategy.n_vehicles_charging_users

        self.list_n_vehicles_available += [
            self.simInput.n_vehicles_sim - n_vehicles_charging
                                         - self.n_booked_vehicles
        ]

        if self.simInput.supply_model_conf["battery_swap"] \
                and self.simInput.supply_model_conf["scooter_relocation"]:
            self.list_n_scooters_relocating += [self.scooterRelocationStrategy.n_scooters_relocating]
        elif self.simInput.supply_model_conf["vehicle_relocation"]:
            self.list_n_vehicles_relocating += [self.vehicleRelocationStrategy.n_vehicles_relocating]

        # Proactive charging
        self.charging_outward_distance = [self.chargingStrategy.tot_charge_distance]
        self.charging_return_distance = [self.chargingStrategy.tot_charge_return_distance]

        if "save_history" in self.simInput.demand_model_config:
            if self.simInput.demand_model_config["save_history"]:
                self.sim_booking_requests += [self.booking_request]

        self.n_booking_requests += 1

        available_vehicle_flag = False
        found_vehicle_flag = False
        available_vehicle_flag_same_zone = False
        available_vehicle_flag_not_same_zone = False

        # 1. Check if a vehicle is available in the request zone
        if len(self.available_vehicles_dict[self.booking_request["origin_id"]]):
            available_vehicle_flag = True
            available_vehicle_flag_same_zone = True
            found_vehicle_flag, max_soc_vehicle_origin, max_soc_origin = \
                self.find_vehicle(self.booking_request["origin_id"])

        # 2. If a vehicle with the appropriate SoC has been found,
        #    compute the emission and SoC delta and schedule the booking...
        if found_vehicle_flag:
            self.booking_request["soc_delta"] = self.vehicles_list[
                max_soc_vehicle_origin].consumption_to_percentage(
                self.vehicles_list[max_soc_vehicle_origin].distance_to_consumption(
                    self.booking_request["driving_distance"] / 1000
                )
            )
            self.booking_request["welltotank_kwh"] = self.vehicles_list[
                max_soc_vehicle_origin].welltotank_energy_from_perc(
                self.booking_request["soc_delta"]
            )
            self.booking_request["tanktowheel_kwh"] = self.vehicles_list[
                max_soc_vehicle_origin].tanktowheel_energy_from_perc(
                self.booking_request["soc_delta"]
            )
            self.booking_request["soc_delta_kwh"] = self.booking_request["welltotank_kwh"] + self.booking_request["tanktowheel_kwh"]
            self.booking_request["welltotank_emissions"] = self.vehicles_list[
                max_soc_vehicle_origin].distance_to_welltotank_emission(self.booking_request["driving_distance"] / 1000)
            self.booking_request["tanktowheel_emissions"] = self.vehicles_list[
                max_soc_vehicle_origin].distance_to_tanktowheel_emission(self.booking_request["driving_distance"] / 1000)
            self.booking_request["co2_emissions"] = self.booking_request["welltotank_emissions"] + \
                                               self.booking_request["tanktowheel_emissions"]

            # Launch a separate process that handles all
            # the trip states from start to finish
            self.env.process(
                self.schedule_booking(max_soc_vehicle_origin, self.booking_request["origin_id"])
            )
            self.n_same_zone_trips += 1
        else: # ...otherwise look for the best candidate
              # vehicle in the neighbor zones
            available_vehicle_flag = False
            found_vehicle_flag = False
            available_vehicle_flag_same_zone = False
            available_vehicle_flag_not_same_zone = False
            max_soc_vehicle_neighbors = None

            max_soc_neighbors = -1
            max_neighbor = None

            for neighbor in self.neighbors_dict[self.booking_request["origin_id"]].dropna().values:
                if neighbor in self.available_vehicles_dict:
                    if len(self.available_vehicles_dict[neighbor]) and not found_vehicle_flag:
                        available_vehicle_flag = True
                        available_vehicle_flag_not_same_zone = True
                        found_vehicle_flag, max_soc_vehicle_neighbor, max_soc_neighbor = self.find_vehicle(neighbor)
                        if max_soc_neighbors < max_soc_neighbor:
                            max_neighbor = neighbor
                            max_soc_vehicle_neighbors = max_soc_vehicle_neighbor

            # If a vehicle was found in a neighbor zone,
            # update the emissions and the SoC and schedule the booking
            if found_vehicle_flag:
                self.booking_request["soc_delta"] = self.vehicles_list[max_soc_vehicle_neighbors].consumption_to_percentage(
                    self.vehicles_list[max_soc_vehicle_neighbors].distance_to_consumption(
                        self.booking_request["driving_distance"] / 1000
                    )
                )
                # self.booking_request["avg_speed_kmh"] = (self.booking_request["driving_distance"] / 1000)/\
                #                                    (self.booking_request['duration'] / 3600)
                self.booking_request["welltotank_kwh"] = self.vehicles_list[
                    max_soc_vehicle_neighbors].welltotank_energy_from_perc(
                    self.booking_request["soc_delta"]
                )
                self.booking_request["tanktowheel_kwh"] = self.vehicles_list[
                    max_soc_vehicle_neighbors].tanktowheel_energy_from_perc(
                    self.booking_request["soc_delta"]
                )
                self.booking_request["soc_delta_kwh"] = self.booking_request["welltotank_kwh"] + self.booking_request[
                    "tanktowheel_kwh"]
                self.booking_request["welltotank_emissions"] = self.vehicles_list[
                    max_soc_vehicle_neighbors].distance_to_welltotank_emission(self.booking_request["driving_distance"] / 1000)
                self.booking_request["tanktowheel_emissions"] = self.vehicles_list[
                    max_soc_vehicle_neighbors].distance_to_tanktowheel_emission(self.booking_request["driving_distance"] / 1000)
                self.booking_request["co2_emissions"] = self.booking_request["welltotank_emissions"] + \
                                                   self.booking_request["tanktowheel_emissions"]
                self.env.process(
                    self.schedule_booking(max_soc_vehicle_neighbors, max_neighbor)
                )
                self.n_not_same_zone_trips += 1

        if not available_vehicle_flag:
            self.n_no_close_vehicles += 1

            if "save_history" in self.simInput.demand_model_config:
                if self.simInput.demand_model_config["save_history"]:
                    self.sim_unsatisfied_requests += [self.booking_request]
                    self.sim_no_close_vehicle_requests += [self.booking_request]

        # TODO: Utilize the lack of SoC retroactively to forward push
        #  the relocation/user incentives if it is over a threshold
        #  i.e., if Delta(soc) - max_soc > r
        if not found_vehicle_flag and available_vehicle_flag:
            self.n_deaths += 1

            death = copy.deepcopy(self.booking_request)
            death["hour"] = death["start_time"].hour

            if available_vehicle_flag_same_zone and available_vehicle_flag_not_same_zone:
                if max_soc_origin > max_soc_neighbor:
                    death["plate"] = max_soc_vehicle_origin
                    death["zone_id"] = self.booking_request["origin_id"]
                else:
                    death["plate"] = max_soc_vehicle_neighbor
                    death["zone_id"] = max_neighbor
            elif available_vehicle_flag_same_zone:
                death["plate"] = max_soc_vehicle_origin
                death["zone_id"] = self.booking_request["origin_id"]
            elif available_vehicle_flag_not_same_zone:
                death["plate"] = max_soc_vehicle_neighbor
                death["zone_id"] = max_neighbor

            if "save_history" in self.simInput.demand_model_config:
                if self.simInput.demand_model_config["save_history"]:
                    self.sim_booking_requests_deaths += [death]

    @abstractmethod
    def mobility_requests_generator(self):
        pass

    def run(self):
        self.env.process(self.mobility_requests_generator())
        self.env.run(until=self.total_seconds)
