import numpy as np
import os
import pickle5 as pickle

from odysseus.supply_modelling.energymix_loader import EnergyMix


class SupplyModel:
	def __init__(self, supply_model_conf, year):
		"""
		Attributes
		----------
		supply_model_conf : dict
			Scenario conf dict
				+ requests rate factor update
				+ city, data source Id, # of vehicles, extra about chargin poles
		"""

		self.supply_model_conf = supply_model_conf

		self.city = self.supply_model_conf["city"]

		demand_model_path = os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			"demand_modelling",
			"demand_models",
			self.supply_model_conf["city"],
		)

		self.grid = pickle.Unpickler(open(os.path.join(demand_model_path, "grid.pickle"), "rb")).load()
		self.grid_matrix = pickle.Unpickler(open(os.path.join(demand_model_path, "grid_matrix.pickle"), "rb")).load()
		self.request_rates = pickle.Unpickler(open(os.path.join(demand_model_path, "request_rates.pickle"), "rb")).load()
		self.trip_kdes = pickle.Unpickler(open(os.path.join(demand_model_path, "trip_kdes.pickle"), "rb")).load()
		self.valid_zones = pickle.Unpickler(open(os.path.join(demand_model_path, "valid_zones.pickle"), "rb")).load()
		self.neighbors_dict = pickle.Unpickler(open(os.path.join(demand_model_path, "neighbors_dict.pickle"), "rb")).load()
		self.integers_dict = pickle.Unpickler(open(os.path.join(demand_model_path, "integers_dict.pickle"), "rb")).load()

		self.n_vehicles_original = self.integers_dict["n_vehicles_original"]
		self.avg_speed_mean = self.integers_dict["avg_speed_mean"]
		self.avg_speed_std = self.integers_dict["avg_speed_std"]
		self.avg_speed_kmh_mean = self.integers_dict["avg_speed_kmh_mean"]
		self.avg_speed_kmh_std = self.integers_dict["avg_speed_kmh_std"]
		self.max_driving_distance = self.integers_dict["max_driving_distance"]

		self.n_vehicles_sim = self.supply_model_conf["n_vehicles"]

		self.energy_mix = EnergyMix(self.city, year)

		self.initial_relocation_workers_positions = []
		self.initial_workers_positions = []

	def init_vehicles(self):
		"""
		Please note: The vehicle Id is generated sequentially from 0 to N_vehicles.

		Returns
		-------
		vehicles_soc_dict : dict[int]
			Dict of vehicle SOCs with vehicle Id as key

		vehicles_zones : dict[int]
			Dict of zone Ids assigned to each vehicle with vehicle Id as key

		available_vehicles_dict : dict[list]
			Dict of lists of vehicles Ids belonging to each zone with zone Id as key
		"""
		# Extract maximum SoC
		beta = self.supply_model_conf['beta']

		# Assign with uniform probability distribution
		# the initial state of charge (SoC) of all the vehicles
		vehicles_random_soc = list(
			np.random.uniform(beta/2, beta,
							  self.n_vehicles_sim).astype(int)
		)

		self.vehicles_soc_dict = {
			i: vehicles_random_soc[i] for i in range(self.n_vehicles_sim)
		}

		# Assign with uniform probability distribution
		# the vehicles to the Top 30 most requested zones
		top_o_zones = self.grid.zone_id_origin_count.sort_values(ascending=False).iloc[:31]

		vehicles_random_zones = list(
			np.random.uniform(0, 30, self.n_vehicles_sim).astype(int).round()
		)

		self.vehicles_zones = []
		for i in vehicles_random_zones:
			self.vehicles_zones.append(self.grid.loc[int(top_o_zones.index[i])].zone_id)

		self.vehicles_zones = {
			i: self.vehicles_zones[i]
			for i in range(self.n_vehicles_sim)
		}

		# Dict of lists:
		#
		# For each zone Id generate a list of
		# the vehicles Ids belonging to the zone
		self.available_vehicles_dict = {
			int(zone): [] for zone in self.grid.zone_id
		}

		for vehicle in range(len(self.vehicles_zones)):
			zone = self.vehicles_zones[vehicle]
			self.available_vehicles_dict[zone] += [vehicle]

		return self.vehicles_soc_dict, self.vehicles_zones, self.available_vehicles_dict

	def init_relocation(self):
		# Assign with uniform probability distribution
		# the relocation workers to the Top 30 most requested zones
		if "n_relocation_workers" in self.supply_model_conf:
			n_relocation_workers = self.supply_model_conf["n_relocation_workers"]

			top_o_zones = self.grid.zone_id_origin_count \
							  .sort_values(ascending=False).iloc[:31]

			workers_random_zones = list(
				np.random.uniform(0, 30, n_relocation_workers)
					     .astype(int).round()
			)

			self.initial_relocation_workers_positions = [
				self.grid.loc[int(top_o_zones.index[i])].zone_id
				for i in workers_random_zones ]

	def init_workers(self):
		# Assign with uniform probability distribution
		# the battery swap workers to the Top 30 most requested zones
		if "n_workers" in self.supply_model_conf:
			n_workers = self.supply_model_conf["n_workers"]

			top_o_zones = self.grid.zone_id_origin_count \
							  .sort_values(ascending=False).iloc[:31]

			workers_random_zones = list(
				np.random.uniform(0, 30, n_workers)
					     .astype(int).round()
			)

			self.initial_workers_positions = [
				self.grid.loc[int(top_o_zones.index[i])].zone_id
				for i in workers_random_zones]
