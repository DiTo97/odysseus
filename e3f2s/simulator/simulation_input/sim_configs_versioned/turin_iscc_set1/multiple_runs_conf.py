import datetime
import numpy as np

sim_scenario_conf_grid = {

    "requests_rate_factor": np.arange(1, 6, 1),
    "n_vehicles_factor": np.arange(1, 6, 1),

    "time_estimation": [True],
    "queuing": [True],

    "alpha": [26],
    "beta": [100],
    "n_poles_n_vehicles_factor": np.arange(0.05, 0.3, 0.05),

    "hub": [False],
    "hub_zone_policy": [""],

    "distributed_cps": [True],
    "system_cps": [True],
    "cps_placement_policy": ["num_parkings"],
    "cps_zones_percentage": np.arange(0.05, 0.25, 0.05),

    "battery_swap": [False],
    "avg_reach_time": [20],
    "avg_service_time": [0],

    "n_workers": [1000],
    "relocation": [False],

    "user_contribution": [False],
    "willingness": [0],

}