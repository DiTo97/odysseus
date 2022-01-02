import os
import pickle5 as pickle
import pandas as pd


from odysseus.supply_modelling.supply_model import SupplyModel


class SimInput:

	def __init__(self, conf_tuple):
		"""
		Initialize a Simulation Input object

		Parameters
		----------
		conf_tuple: tuple
			Tuple containing (demand_model_config, sim_scenario_conf)
		"""

		self.demand_model_config = conf_tuple[0] # General conf
		self.sim_scenario_conf = conf_tuple[1]

		self.city = self.demand_model_config["city"]

		# Get the city's demand model dir at odysseus/demand_modelling/demand_models/<city>
		demand_model_path = os.path.join(
			os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
			"demand_modelling",
			"demand_models",
			self.demand_model_config["city"],
		)

		# Load pickle files
		self.grid = pickle.Unpickler(open(os.path.join(demand_model_path, "grid.pickle"), "rb")).load()
		self.grid_matrix = pickle.Unpickler(open(os.path.join(demand_model_path, "grid_matrix.pickle"), "rb")).load()
		self.avg_out_flows_train = pickle.Unpickler(open(os.path.join(demand_model_path, "avg_out_flows_train.pickle"), "rb")).load()
		self.avg_in_flows_train = pickle.Unpickler(open(os.path.join(demand_model_path, "avg_in_flows_train.pickle"), "rb")).load()

		# All the zones in the operative area that retain more than a threshold number of trips
		# Please note: At the moment no zone is discarded
		self.valid_zones = pickle.Unpickler(open(os.path.join(demand_model_path, "valid_zones.pickle"), "rb")).load()

		self.neighbors_dict = pickle.Unpickler(open(os.path.join(demand_model_path, "neighbors_dict.pickle"), "rb")).load()
		self.integers_dict = pickle.Unpickler(open(os.path.join(demand_model_path, "integers_dict.pickle"), "rb")).load()

		self.closest_valid_zone = pickle.Unpickler(open(os.path.join(demand_model_path, "closest_valid_zone.pickle"), "rb")).load()

		self.avg_request_rate = self.integers_dict["avg_request_rate"]
		self.n_vehicles_original = self.integers_dict["n_vehicles_original"]
		self.avg_speed_mean = self.integers_dict["avg_speed_mean"]
		self.avg_speed_std = self.integers_dict["avg_speed_std"]
		self.avg_speed_kmh_mean = self.integers_dict["avg_speed_kmh_mean"]
		self.avg_speed_kmh_std = self.integers_dict["avg_speed_kmh_std"]
		self.max_driving_distance = self.integers_dict["max_driving_distance"]
		self.max_in_flow = self.integers_dict["max_in_flow"]
		self.max_out_flow = self.integers_dict["max_out_flow"]

		if self.demand_model_config["sim_technique"] == "traceB":
			self.bookings = pickle.Unpickler(open(os.path.join(demand_model_path, "bookings_test.pickle"), "rb")).load()
			self.booking_requests_list = self.get_booking_requests_list()
		elif self.demand_model_config["sim_technique"] == "eventG":
			self.request_rates = pickle.Unpickler(open(os.path.join(demand_model_path, "request_rates.pickle"), "rb")).load()
			self.trip_kdes = pickle.Unpickler(open(os.path.join(demand_model_path, "trip_kdes.pickle"), "rb")).load()

		# Number of requests per month
		if "n_requests" in self.sim_scenario_conf.keys():
			# Desired request rate per second (RRS)
			self.desired_avg_rate = self.sim_scenario_conf["n_requests"] / 30 / 24 / 3600

			# Ratio between the RRS desired by the user
			# and the average RRS computed by the demand model
			self.rate_ratio = self.desired_avg_rate / self.avg_request_rate

			self.sim_scenario_conf["requests_rate_factor"] = self.rate_ratio

		if "n_vehicles" in self.sim_scenario_conf.keys():
			self.n_vehicles_sim = self.sim_scenario_conf["n_vehicles"]
		elif "n_vehicles_factor" in self.sim_scenario_conf.keys():
			self.n_vehicles_sim = int(
				self.n_vehicles_original * self.sim_scenario_conf["n_vehicles_factor"]
			)

		if not self.sim_scenario_conf["battery_swap"]:
			raise KeyError('E-scooters must follow a battery swap policy')

		# Do NOT erase these as they're required
		# by the supply model
		self.n_charging_zones = 0
		self.tot_n_charging_poles = 0

		self.n_charging_poles_by_zone = {}

		self.vehicles_soc_dict = {}
		self.vehicles_zones = {}

		self.zones_cp_distances = pd.Series()
		self.closest_cp_zone = pd.Series()

		self.start = None

		self.supply_model_conf = dict()

		self.supply_model_conf.update(self.sim_scenario_conf)

		self.supply_model_conf.update({
			"city": self.city,
			"data_source_id": self.demand_model_config['data_source_id'],
			"n_vehicles": self.n_vehicles_sim,
			# "tot_n_charging_poles": self.tot_n_charging_poles,
			# "n_charging_zones": self.n_charging_zones,
		})

		# City and year are required by the supply model to load
		# the correct configuration from energy_mix.json
		self.supply_model = SupplyModel(self.supply_model_conf,
										self.demand_model_config["year"])

	def get_booking_requests_list(self):

		bookings_df = self.bookings[[
			"origin_id",
			"destination_id",
			"start_time",
			"end_time",
			"ia_timeout",
			"euclidean_distance",
			"driving_distance",
			"date",
			"hour",
			"duration",
		]].dropna()

		if 'month_start' in self.demand_model_config:
			if 'day_start' in self.demand_model_config:
				bookings_df = bookings_df[bookings_df['start_time'].apply
							(lambda x: (x.year == self.demand_model_config['year']) &
									   (x.month >= self.demand_model_config['month_start']) &
									   (x.month <= self.demand_model_config['month_end']) &
									   (x.day >= self.demand_model_config['day_start']) &
									   (x.day <= self.demand_model_config['day_end']))]

			else:
				bookings_df = bookings_df[bookings_df['start_time'].apply
					(lambda x: (x.year == self.demand_model_config['year']) &
							   (x.month >= self.demand_model_config['month_start']) &
								(x.month <= self.demand_model_config['month_end']))]

		return bookings_df.to_dict("records")

	def init_vehicles(self):
		return self.supply_model.init_vehicles()

	def init_charging_poles(self):
		return None

	def init_relocation(self):
		return self.supply_model.init_relocation()

	def init_workers(self):
		return self.supply_model.init_workers()

	def init(self):
		self.init_relocation()
		self.init_workers()
		self.init_vehicles()

	def refresh(self):
		self.init()
