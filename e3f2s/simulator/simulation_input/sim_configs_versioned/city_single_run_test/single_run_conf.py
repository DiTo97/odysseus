sim_scenario_conf = {

	"requests_rate_factor": 1,
	"n_vehicles_factor": 1,
	"engine_type": "electric",
	"profile_type": "single_phase_1", # works only if engine_type = electric
	"vehicle_model_name": "Smart fortwo Electric Drive 2018",
	"country_energymix": "Italy",
	"year_energymix": "2018",

	"time_estimation": True,
	"queuing": True,

	"alpha_policy": 'auto',

	"beta": 100,
	"n_poles_n_vehicles_factor": 0.2,

	"hub_zone_policy": "num_parkings",
	"hub": False,

	"cps_placement_policy": "num_parkings",
	"distributed_cps": True,
	"system_cps": True,
	"cps_zones_percentage": 0.2,

	"battery_swap": False,
	"avg_reach_time": 20,
	"avg_service_time": 0,

	"n_workers": 12,
	"relocation": False,

	"number of workers": 1000,

	"user_contribution": False,
	"willingness": 0,

	"scooter_relocation": False,

	"charging_strategy": "reactive",
	"charging_relocation_strategy": "closest_free", #closest_free/random/closest_queueing
	"scooter_relocation": False

	"scooter_relocation": False,
	"scooter_relocation_strategy": "magic_relocation",

	"vehicle_relocation": True,
	"vehicle_relocation_strategy": "only_scheduled",

	"vehicle_relocation_scheduling": True,
	"vehicle_relocation_technique": frozenset({
			"start": "aggregation",
			# "start_demand_weight": 0.9,
			# "start_vehicles_factor": 1,
			"end": "kde_sampling",
			# "end_demand_weight": 0.9,
			# "end_vehicles_factor": 1,
		}.items()),

	"vehicle_scheduled_relocation_triggers": frozenset({
        "post_charge": False,
        "post_trip": True,
    }.items()),
	"n_relocation_workers": 12,
	"avg_relocation_speed": 20  # km/h

}
