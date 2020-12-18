import pandas as pd



class EFFCS_SimOutput ():

	def __init__ (self, sim):

		self.valid_zones = sim.simInput.valid_zones

		self.sim_general_conf = sim.simInput.sim_general_conf
		self.sim_scenario_conf = sim.simInput.sim_scenario_conf

		self.sim_booking_requests = pd.DataFrame(sim.sim_booking_requests)
		self.sim_bookings = self.sim_booking_requests.dropna()
		self.sim_charges = pd.DataFrame(sim.chargingStrategy.sim_charges)
		self.sim_not_enough_energy_requests = pd.DataFrame(sim.sim_booking_requests_deaths)
		self.sim_no_close_vehicle_requests = pd.DataFrame(sim.sim_no_close_vehicle_requests)
		self.sim_unsatisfied_requests = pd.DataFrame(sim.sim_unsatisfied_requests)
		self.sim_system_charges_bookings = pd.DataFrame(sim.chargingStrategy.list_system_charging_bookings)
		self.sim_users_charges_bookings = pd.DataFrame(sim.chargingStrategy.list_users_charging_bookings)
		self.sim_vehicle_relocations = pd.DataFrame(sim.VehicleRelocationStrategy.sim_vehicle_relocations)

		if self.sim_scenario_conf["battery_swap"]:
			self.sim_scooter_relocations = pd.DataFrame(sim.scooterRelocationStrategy.sim_scooter_relocations)
		else:
			self.sim_vehicle_relocations = pd.DataFrame(sim.VehicleRelocationStrategy.sim_vehicle_relocations)

		if "end_time" not in self.sim_system_charges_bookings:
			self.sim_system_charges_bookings["end_time"] = pd.Series()
		if "end_time" not in self.sim_users_charges_bookings:
			self.sim_users_charges_bookings["end_time"] = pd.Series()

		if len(self.sim_system_charges_bookings):
			self.sim_system_charges_bookings["end_hour"] = self.sim_system_charges_bookings["end_time"].apply(
				lambda d: d.hour
			)

		if len(self.sim_users_charges_bookings):
			self.sim_users_charges_bookings["end_hour"] = self.sim_users_charges_bookings["end_time"].apply(
				lambda d: d.hour
			)

		self.sim_unfeasible_charge_bookings = pd.DataFrame(
			sim.chargingStrategy.sim_unfeasible_charge_bookings
		)

		self.sim_booking_requests["n_vehicles_charging_system"] = \
			pd.Series(sim.list_n_vehicles_charging_system)

		self.sim_booking_requests["n_vehicles_charging_users"] = \
			pd.Series(sim.list_n_vehicles_charging_users).fillna(0)

		self.sim_booking_requests["n_vehicles_available"] = \
			pd.Series(sim.list_n_vehicles_available)

		self.sim_booking_requests["n_vehicles_booked"] = \
			pd.Series(sim.list_n_vehicles_booked)

		self.sim_booking_requests["n_vehicles_dead"] = \
			pd.Series(sim.list_n_vehicles_dead)


		self.sim_charge_deaths = pd.DataFrame(sim.chargingStrategy.sim_unfeasible_charge_bookings)

		self.grid = sim.simInput.grid

		# Sim Stats creation

		self.sim_stats = pd.Series(name="sim_stats")

		self.sim_stats = pd.concat([
			self.sim_stats,
			pd.Series(sim.simInput.sim_general_conf)
		])

		self.sim_stats = pd.concat([
			self.sim_stats,
			pd.Series(sim.simInput.sim_scenario_conf)
		])

		self.sim_stats.loc["n_same_zone_trips"] = \
			sim.n_same_zone_trips

		self.sim_stats.loc["n_not_same_zone_trips"] = \
			sim.n_not_same_zone_trips

		self.sim_stats.loc["n_no_close_vehicles"] = \
			sim.n_no_close_vehicles

		self.sim_stats.loc["n_deaths"] = \
			sim.n_deaths

		self.sim_stats["n_booking_reqs"] = \
			self.sim_stats["n_same_zone_trips"]\
			+ self.sim_stats["n_not_same_zone_trips"]\
			+ self.sim_stats["n_no_close_vehicles"]\
			+ self.sim_stats["n_deaths"]

		self.sim_stats["n_bookings"] = \
			self.sim_stats["n_same_zone_trips"]\
			+ self.sim_stats["n_not_same_zone_trips"]

		self.sim_stats["n_unsatisfied"] = \
			self.sim_stats["n_no_close_vehicles"]\
			+ self.sim_stats["n_deaths"]

		self.sim_stats.loc["fraction_satisfied"] = \
			self.sim_stats["n_bookings"] / self.sim_stats["n_booking_reqs"]

		self.sim_stats.loc["fraction_unsatisfied"] = \
			1.0 - self.sim_stats.loc["fraction_satisfied"]

		self.sim_stats.loc["fraction_same_zone_trips"] = \
			sim.n_same_zone_trips / self.sim_stats["n_booking_reqs"]

		self.sim_stats.loc["fraction_not_same_zone_trips"] = \
			sim.n_not_same_zone_trips / self.sim_stats["n_booking_reqs"]

		self.sim_stats.loc["fraction_no_close_vehicles"] = \
			sim.n_no_close_vehicles / self.sim_stats["n_booking_reqs"]

		self.sim_stats.loc["fraction_not_enough_energy"] = \
			sim.n_deaths / self.sim_stats["n_booking_reqs"]

		self.sim_stats.loc["fraction_same_zone_trips_satisfied"] = \
			sim.n_same_zone_trips / self.sim_stats["n_bookings"]

		self.sim_stats.loc["fraction_not_same_zone_trips_satisfied"] = \
			sim.n_not_same_zone_trips / self.sim_stats["n_bookings"]

		if self.sim_stats["n_unsatisfied"] > 0:
			self.sim_stats.loc["fraction_no_close_vehicles_unsatisfied"] = \
			sim.n_no_close_vehicles / self.sim_stats["n_unsatisfied"]
		else:
			self.sim_stats.loc["fraction_no_close_vehicles_unsatisfied"] = 0

		if self.sim_stats["n_unsatisfied"] > 0:
			self.sim_stats.loc["fraction_deaths_unsatisfied"] = \
				sim.n_deaths / self.sim_stats["n_unsatisfied"]
		else:
			self.sim_stats.loc["fraction_deaths_unsatisfied"] = 0


		self.sim_stats.loc["n_charges"] = \
			len(self.sim_charges)

		self.sim_stats.loc["n_charging_requests_system"] = len(self.sim_system_charges_bookings)

		if "system" in self.sim_charges.operator:
			self.sim_stats.loc["n_charges_system"] = self.sim_charges.groupby("operator").date.count().loc["system"]
		else:
			self.sim_stats.loc["n_charges_system"] = 0

		if "users" in self.sim_charges.operator:
			self.sim_stats.loc["n_charges_users"] = self.sim_charges.groupby("operator").date.count().loc["users"]
		else:
			self.sim_stats.loc["n_charges_users"] = 0

		self.sim_stats.loc["n_charge_deaths"] = \
			len(self.sim_charge_deaths)

		self.sim_stats.loc["fraction_charge_deaths"] = \
			len(self.sim_charge_deaths) / self.sim_stats.loc["n_charges"]

		self.sim_stats.loc["soc_avg"] = \
			self.sim_bookings.start_soc.mean()

		self.sim_stats.loc["soc_med"] = \
			self.sim_bookings.start_soc.median()

		self.sim_stats.loc["charging_time_avg"] = \
			self.sim_charges.duration.mean() / 3600

		self.sim_stats.loc["charging_time_med"] = \
			self.sim_charges.duration.median() / 3600

		self.sim_stats.loc["n_charges_by_vehicle_avg"] = \
			self.sim_charges.groupby("plate").date.count().mean()

		self.sim_stats.loc["n_charges_by_vehicle_system_avg"] = \
			self.sim_charges[self.sim_charges.operator == "system"]\
				.groupby("plate").date.count().mean()

		if len(self.sim_users_charges_bookings):
			self.sim_stats.loc["n_charges_by_vehicle_users_avg"] = \
				self.sim_charges[self.sim_charges.operator == "users"]\
					.groupby("plate").date.count().mean()
		else:
			self.sim_stats.loc["n_charges_by_vehicle_users_avg"] = 0

		self.sim_stats["sim_duration"] = (sim.end - sim.start).total_seconds()
		self.sim_stats.loc["tot_potential_mobility_distance"] = self.sim_booking_requests.driving_distance.sum()
		self.sim_stats.loc["tot_potential_mobility_duration"] = self.sim_booking_requests.duration.sum()
		self.sim_stats.loc["tot_potential_welltotank_energy"] = self.sim_booking_requests.welltotank_kwh.sum()
		self.sim_stats.loc["tot_potential_tanktowheel_energy"] = self.sim_booking_requests.tanktowheel_kwh.sum()
		self.sim_stats.loc["tot_potential_mobility_energy"] = self.sim_booking_requests.soc_delta_kwh.sum()
		self.sim_stats.loc["tot_potential_welltotank_co2_emissions"] = self.sim_booking_requests.welltotank_emissions.sum() / 1000
		self.sim_stats.loc["tot_potential_welltowheel_co2_emissions"] = self.sim_booking_requests.tanktowheel_emissions.sum() / 1000
		self.sim_stats.loc["tot_potential_co2_emissions_kg"] = self.sim_booking_requests.co2_emissions.sum() / 1000

		self.sim_stats.loc["tot_mobility_distance"] = self.sim_bookings.driving_distance.sum()
		self.sim_stats.loc["tot_mobility_duration"] = self.sim_bookings.duration.sum()
		self.sim_stats.loc["tot_welltotank_energy"] = self.sim_bookings.welltotank_kwh.sum()
		self.sim_stats.loc["tot_tanktowheel_energy"] = self.sim_bookings.tanktowheel_kwh.sum()
		self.sim_stats.loc["tot_mobility_energy"] = self.sim_bookings.soc_delta_kwh.sum()
		self.sim_stats.loc["tot_welltotank_co2_emissions"] = self.sim_bookings.welltotank_emissions.sum() / 1000
		self.sim_stats.loc["tot_tanktowheel_co2_emissions"] = self.sim_bookings.tanktowheel_emissions.sum() / 1000
		self.sim_stats.loc["tot_co2_emissions_kg"] = self.sim_bookings.co2_emissions.sum() / 1000

		self.sim_stats.loc["n_vehicles_sim"] = sim.simInput.n_vehicles_sim
		self.sim_stats.loc["tot_n_charging_poles"] = sim.simInput.tot_n_charging_poles

		self.sim_stats.loc["tot_potential_charging_energy"] = self.sim_stats.loc["sim_duration"] / 3600 * (
			self.sim_stats.loc["tot_n_charging_poles"] * 3.7
		)

		self.sim_stats.loc["tot_charging_energy"] = self.sim_charges["soc_delta_kwh"].sum()
		self.sim_stats.loc["tot_charging_outwards_distance"] =sim.chargingStrategy.charging_outward_distance

		if "system" in self.sim_charges.operator.unique():
			self.sim_stats.loc["fraction_charges_system"] = \
				self.sim_charges.groupby("operator")\
				.date.count().loc["system"]\
				/ len(self.sim_charges)
			self.sim_stats.loc["fraction_energy_system"] = \
				self.sim_charges.groupby("operator")\
				.soc_delta_kwh.sum().loc["system"]\
				/ self.sim_stats["tot_charging_energy"]
			self.sim_stats.loc["fraction_duration_system"] = \
				self.sim_charges.groupby("operator")\
				.duration.sum().loc["system"]\
				/ self.sim_charges.duration.sum()
		else:
			self.sim_stats.loc["fraction_charges_system"] = 0
			self.sim_stats.loc["fraction_energy_system"] = 0
			self.sim_stats.loc["fraction_duration_system"] = 0

		if "users" in self.sim_charges.operator.unique():
			self.sim_stats.loc["fraction_charges_users"] = \
				self.sim_charges.groupby("operator")\
				.date.count().loc["users"]\
				/ len(self.sim_charges)
			self.sim_stats.loc["fraction_energy_users"] = \
				self.sim_charges.groupby("operator")\
				.soc_delta_kwh.sum().loc["users"]\
				/ self.sim_stats["tot_charging_energy"]
			self.sim_stats.loc["fraction_duration_users"] = \
				self.sim_charges.groupby("operator")\
				.duration.sum().loc["users"]\
				/ self.sim_charges.duration.sum()
		else:
			self.sim_stats.loc["fraction_charges_users"] = 0
			self.sim_stats.loc["fraction_energy_users"] = 0
			self.sim_stats.loc["fraction_duration_users"] = 0

		self.sim_stats.loc["charging_duration_avg"] = \
			self.sim_charges.duration.mean()

		self.sim_stats.loc["charging_energy_event_avg"] = \
			self.sim_charges.soc_delta_kwh.mean()

		self.sim_stats.loc["charging_energy_event_max"] = \
			self.sim_charges.soc_delta_kwh.max()

		self.sim_stats.loc["charging_energy_event_med"] = \
			self.sim_charges.soc_delta_kwh.median()

		self.sim_charges["cr_timeout"] = \
			self.sim_charges.timeout_outward\
			+ self.sim_charges.timeout_return

		self.sim_stats.loc["cum_relo_out_t"] = \
			self.sim_charges.timeout_outward.sum() / 60 / 60

		self.sim_stats.loc["cum_relo_ret_t"] = \
			self.sim_charges.timeout_return.sum() / 60 / 60

		self.sim_stats.loc["cum_relo_t"] = \
			self.sim_stats.cum_relo_out_t + \
			self.sim_stats.cum_relo_ret_t

		self.sim_stats.loc["cum_relo_khw"] = \
			self.sim_charges.cr_soc_delta_kwh.sum()

		self.sim_stats.loc["avg_hourly_relo_t"] = \
			self.sim_charges.groupby("hour").cr_timeout.sum().mean()

		if self.sim_scenario_conf["battery_swap"]:
			self.sim_stats.loc["n_scooter_relocations"] = \
				len(self.sim_scooter_relocations)

			if len(self.sim_scooter_relocations):
				self.sim_stats.loc["tot_scooter_relocations_distance"] = \
					self.sim_scooter_relocations.distance.sum()
			else:
				self.sim_stats.loc["tot_scooter_relocations_distance"] = 0
		else:
			self.sim_stats.loc["n_vehicle_relocations"] = \
				len(self.sim_vehicle_relocations)

			if len(self.sim_vehicle_relocations):
				self.sim_stats.loc["tot_vehicle_relocations_distance"] = \
					self.sim_vehicle_relocations.distance.sum()
			else:
				self.sim_stats.loc["tot_scooter_relocations_distance"] = 0

		for key in self.sim_stats.index:
			if key.startswith("fraction"):
				self.sim_stats["percentage" + key[8:]] = self.sim_stats[key] * 100

		self.grid[
			"origin_count"
		] = self.sim_booking_requests.origin_id.value_counts()
		self.grid[
			"destination_count"
		] = self.sim_booking_requests.destination_id.value_counts()

		if len(self.sim_system_charges_bookings):
			self.grid[
				"charge_needed_system_zones_count"
			] = self.sim_system_charges_bookings.destination_id.value_counts()
		else:
			self.grid[
				"charge_needed_system_zones_count"
			] = 0
		if len(self.sim_users_charges_bookings):
			self.grid[
				"charge_needed_users_zones_count"
			] = self.sim_users_charges_bookings.destination_id.value_counts()
		else:
			self.grid[
				"charge_needed_users_zones_count"
			] = 0

		if self.sim_stats["n_unsatisfied"]:
			self.grid[
				"unsatisfied_demand_origins_fraction"
			] = self.sim_unsatisfied_requests.origin_id.value_counts() / len(self.sim_unsatisfied_requests)
		else:
			self.grid[
				"unsatisfied_demand_origins_fraction"
			] = 0

		if len(self.sim_not_enough_energy_requests):
			self.grid[
				"not_enough_energy_origins_count"
			] = self.sim_not_enough_energy_requests.origin_id.value_counts()
		if len(self.sim_charge_deaths):
			self.grid[
				"charge_deaths_origins_count"
			] = self.sim_charge_deaths.origin_id.value_counts()

		self.sim_stats.loc["hub_zone"] = sim.simInput.hub_zone
		self.sim_stats.loc["n_charging_zones"] = sim.simInput.n_charging_zones

		self.sim_stats.loc["max_driving_distance"] = self.sim_booking_requests.driving_distance.max()

		self.vehicles_history = pd.DataFrame()
		for vehicle in sim.vehicles_list:
			vehicle_df = pd.DataFrame(vehicle.status_dict_list)
			vehicle_df["plate"] = vehicle.plate
			self.vehicles_history = pd.concat([self.vehicles_history, vehicle_df], ignore_index=True)

		self.stations_history = pd.DataFrame()
		for key in sim.chargingStrategy.charging_stations_dict:
			station_df = pd.DataFrame(sim.chargingStrategy.charging_stations_dict[key].status_dict_list)
			station_df["id"] = key
			self.stations_history = pd.concat([self.stations_history, station_df], ignore_index=True)

		self.zones_history = pd.DataFrame()
		for key in sim.chargingStrategy.zone_dict:
			zone_df = pd.DataFrame(sim.chargingStrategy.zone_dict[key].status_dict_list)
			zone_df["zone_id"] = key
			self.zones_history = pd.concat([self.zones_history, zone_df], ignore_index=True)
