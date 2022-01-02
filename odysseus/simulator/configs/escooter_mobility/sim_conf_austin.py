General = {

    # Run configuration
    "city": ["Austin"],
    "sim_run_mode": ["multiple_runs"],

    "data_source_id": ["city_open_data"],
    "sim_technique": ["traceB"],

    "sim_scenario_name": ["escooter_mobility"],

    "relocation_workers_working_hours": ["9-18"],

    # Fleet general parameters
    "const_load_factor": [False], # : float, False
                                  # If fixed, consider only the configurations where it matches
                                  # the requests_rate_factor/n_vehicles_factor ratio

    # Space general parameters
    "bin_side_length": [200],
    "k_zones_factor": [1],

    # Time general parameters to
    # govern the simulation duration
    "year": [2019],
    "month_start": [9],
    "month_end": [9],

    "save_history": [True]

}

Multiple_runs = {
    "requests_rate_factor": [1],
    # "n_vehicles_factor": [1], # If n_vehicles is not provided, the number of vehicles will be
    #                           # the nominal number of vehicles (from integers_dict.pickle) times this factor

    # "n_requests": ?, # Number of monthly requests
    "n_vehicles": [1000, 2000, 4000, 8350],

    "engine_type": ["electric"],
    "vehicle_model_name": ["generic e-scooter"],
    "profile_type": ["single_phase_1"],

    "time_estimation": [True],
    "queuing": [True],

    # "alpha_policy": ['auto'],
    "alpha": [30], # Between 0 and beta
    "beta": [100], # E-scooter battery capacity

    # <Unused parameters>
    "n_poles_n_vehicles_factor": [0.06],

    "hub_zone_policy": ["num_parkings"],
    "hub": [False],

    "cps_placement_policy": ["num_parkings"],
    "distributed_cps": [False],
    "system_cps": [False],
    "cps_zones_percentage": [0.2],

    "user_contribution": [False],
    "willingness": [0],

    "charging_relocation_strategy": ["closest_free"], # Post-charge relocation strategy:
                                                      # closest_free/random/closest_queueing

    "relocation": [False],
    "vehicle_relocation": [False],
    # <\Unused parameters>

    "avg_worker_truck_speed": [20], # km/h
    "worker_truck_consumption": [7], # l/100km

    "battery_swap": [True],

    "avg_reach_time": [30],
    "avg_service_time": [5],

    "n_workers": [12],
    "battery_swap_capacity": [30],

    "charging_strategy": ["proactive"],

    "scooter_relocation": [False],

    "n_relocation_workers": [6],
    "relocation_capacity": [30],

    "profitability_check": [True],
    "diesel_price": [0.65], # $/l (USA)
    "unlock_fee": [1], # $
    "rent_fee": [0.15], # $/min
    "avg_relocation_distance": [1], # km
    "avg_swap_distance": [0.4], # km
    "avg_trip_duration": [10], # min

    "scooter_relocation_strategy": ["proactive"], # Piovono e-scooter: magic_relocation
                                                  # CNN+LSTM: predictive
                                                  # Baseline: proactive

    "scooter_relocation_scheduling": [True],
    "scooter_relocation_technique": [
        frozenset({
                       "start": "delta",
                       "end": "delta",
                       "window_width": 1,
                  }.items())],

    "scooter_scheduled_relocation_triggers": [
        frozenset({
                       "post_charge": False,
                       "post_trip": False,
                  }.items())],
}

Single_run = {

}
