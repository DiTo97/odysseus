sim_scenario_conf = {

	"n_vehicles": 2500,
	"requests_rate_factor": 1,
	"fleet_load_factor": 1,

	"time_estimation": True,
	"queuing": True,

	"alpha_policy": 'auto',

	"beta": 100,
	"n_poles_n_vehicles_factor": 1,

	"hub_zone_policy": "num_parkings",
	"hub": False,

	"cps_placement_policy": "num_parkings",
	"distributed_cps": False,
	"system_cps": False,
	"cps_zones_percentage": 0.2,

	"battery_swap": True,
	"avg_reach_time": 30,
	"avg_service_time": 5,

	"n_workers": 1000,
	"relocation": False,

	"user_contribution": False,
	"willingness": 0,

	"charging_strategy": "reactive",
	"charging_relocation_strategy": "closest_free", #closest_free/random/closest_queueing

	"number of workers": 1000,

	"scooter_relocation": False

}