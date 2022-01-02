sim_scenario_conf = {

	"requests_rate_factor": 1,
	"n_vehicles_factor": 1,

	"engine_type": "electric",
	"vehicle_model_name": "generic e-scooter",
	"profile_type": "single_phase_1",

	"time_estimation": True,
	"queuing": True,

	"alpha_policy": 'auto',

	"beta": 100,
	"n_poles_n_vehicles_factor": 0.06,

	"hub_zone_policy": "num_parkings",
	"hub": False,

	"cps_placement_policy": "num_parkings",
	"distributed_cps": False, # Disattiva init_charging_poles in supply_model (che tra l'altro non viene usata mai in SimInput)
	"system_cps": True, # Non usato in supply_model nè in sim_input
	"cps_zones_percentage": 0.2,

	"battery_swap": True,
	"avg_reach_time": 20,
	"avg_service_time": 0,

	"n_workers": 1000,
	"relocation": False,

	"user_contribution": False,
	"willingness": 0,

	"charging_strategy": "reactive",
	"charging_relocation_strategy": "closest_free", #closest_free/random/closest_queueing

	"scooter_relocation": False,
	"vehicle_relocation": False

}
