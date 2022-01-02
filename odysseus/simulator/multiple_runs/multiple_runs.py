import os
import datetime
import multiprocessing as mp
import sys

import pandas as pd
import pickle5 as pickle

from tqdm import tqdm

from odysseus.simulator.simulation_input.sim_config_grid import EFFCS_SimConfGrid
from odysseus.simulator.simulation_output.sim_output_plotter import EFFCS_SimOutputPlotter

from odysseus.simulator.single_run.run_eventG_sim import get_eventG_sim_stats
from odysseus.simulator.single_run.run_traceB_sim import get_traceB_sim_stats


def multiple_runs(sim_general_conf, sim_scenario_conf_grid, sim_scenario_name,
                  exp_name, conf_id, n_cpus=mp.cpu_count()):
    """
    Parameters
    ----------
    sim_general_conf : dict
        A combination from sim_conf.General

    sim_scenario_conf_grid : dict[list]
        Lists of parameters to experiment, i.e., sim_conf.Multiple_runs

    sim_scenario_name : str
        Name of the scenario, i.e., sim_general_conf['sim_scenario_name']

    exp_name : str
        Name of the experiment

    conf_id : int
        General configuration Id

    n_cpus : int
        Number of cores for parallel execution. The default is mp.cpu_count()
    """

    sim_technique = sim_general_conf["sim_technique"]
    city = sim_general_conf["city"]

    results_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "experiments",
        exp_name,
        "results",
        city,
        "multiple_runs",
        sim_scenario_name,
        str(conf_id)
    )

    os.makedirs(results_path, exist_ok=True)

    with mp.Pool(n_cpus) as pool:

        sim_conf_grid = EFFCS_SimConfGrid(sim_scenario_conf_grid)

        # pool_stats_dict = {}
        conf_tuples = []

        for scenario_id, sim_scenario_conf in enumerate(sim_conf_grid.conf_list):  # List of conf dicts
            sim_scenario_conf["conf_id"]   = scenario_id
            # sim_scenario_conf["n_workers"] = sim_scenario_conf["n_vehicles"]

            if "const_load_factor" in sim_general_conf.keys():
                if sim_general_conf["const_load_factor"]:
                    round_lambda = round(sim_scenario_conf["requests_rate_factor"], 2)
                    round_vehicles_factor = round(sim_scenario_conf["n_vehicles_factor"], 2)

                    if round(round_lambda / round_vehicles_factor, 2) == sim_general_conf["const_load_factor"]:
                        conf_tuples += [(
                            sim_general_conf,
                            sim_scenario_conf,
                        )]
                else:
                    conf_tuples += [(
                        sim_general_conf,
                        sim_scenario_conf,
                    )]
            else:
                conf_tuples += [(
                    sim_general_conf,
                    sim_scenario_conf,
                )]

        with tqdm(total=len(conf_tuples), unit="sim", postfix=str(n_cpus) + " CPU(s)",
                  smoothing=0, dynamic_ncols=True) as pbar:

            def collect_result(res):
                stats, sim_output = res
                _scenario_id = stats["conf_id"]

                # Extract the scenario conf
                _sim_scenario_conf = [d for d in sim_conf_grid.conf_list
                                              if d['conf_id'] == _scenario_id]
                _sim_scenario_conf = _sim_scenario_conf[0]

                # Make a folder for every scenario conf
                scenario_path = os.path.join(results_path, str(_scenario_id))
                os.makedirs(scenario_path, exist_ok=True)

                stats.to_pickle(
                    os.path.join(
                        scenario_path,
                        "sim_stats.pickle"))

                stats.to_csv(
                    os.path.join(
                        scenario_path,
                        "sim_stats.csv"))

                pd.Series(_sim_scenario_conf).to_pickle(
                    os.path.join(
                        scenario_path,
                        "sim_scenario_conf.pickle"))

                pd.Series(_sim_scenario_conf).to_csv(
                    os.path.join(
                        scenario_path,
                        "sim_scenario_conf.csv"), header=True)

                pickle.dump(sim_output, open(
                    os.path.join(
                        scenario_path,
                        "sim_output.pickle"), "wb"))

                sim_output.grid.to_pickle(
                    os.path.join(
                        scenario_path,
                        "grid.pickle"
                    )
                )

                sim_output.grid.to_file(
                    os.path.join(
                        scenario_path,
                        "grid.dbf"
                    )
                )

                if sim_general_conf["save_history"]:
                    sim_output.sim_booking_requests.to_csv(
                        os.path.join(
                            scenario_path,
                            "sim_booking_requests.csv"
                        )
                    )

                    sim_output.sim_bookings.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_bookings.pickle"
                        )
                    )

                    sim_output.sim_charges.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_charges.pickle"
                        )
                    )

                    sim_output.sim_not_enough_energy_requests.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_unsatisfied_no-energy.pickle"
                        )
                    )

                    sim_output.sim_no_close_vehicle_requests.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_unsatisfied_no_close_vehicle.pickle"
                        )
                    )

                    sim_output.sim_unsatisfied_requests.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_unsatisfied_requests.pickle"
                        )
                    )

                    sim_output.sim_system_charges_bookings.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_system_charges_bookings.pickle"
                        )
                    )

                    sim_output.sim_users_charges_bookings.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_users_charges_bookings.pickle"
                        )
                    )

                    sim_output.sim_unfeasible_charge_bookings.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_unfeasible_charge_bookings.pickle"
                        )
                    )

                    sim_output.sim_charge_deaths.to_pickle(
                        os.path.join(
                            scenario_path,
                            "sim_unfeasible_charges.pickle"
                        )
                    )

                    sim_output.vehicles_history.to_csv(
                        os.path.join(
                            scenario_path,
                            "vehicles_history.csv"
                        )
                    )

                    sim_output.stations_history.to_csv(
                        os.path.join(
                            scenario_path,
                            "stations_history.csv"
                        )
                    )

                    sim_output.zones_history.to_csv(
                        os.path.join(
                            scenario_path,
                            "zones_history.csv"
                        )
                    )

                    if _sim_scenario_conf["scooter_relocation"]:
                        sim_output.relocation_history.to_csv(
                            os.path.join(
                                scenario_path,
                                "relocation_history.csv"
                            )
                        )

                    plotter = EFFCS_SimOutputPlotter(sim_output, city,
                                                     sim_scenario_name, scenario_path)

                    plotter.plot_events_profile_barh()
                    plotter.plot_events_t()
                    plotter.plot_fleet_status_t()
                    plotter.plot_events_hourly_count_boxplot("bookings_train", "start")
                    plotter.plot_events_hourly_count_boxplot("charges", "start")
                    plotter.plot_events_hourly_count_boxplot("unsatisfied", "start")
                    plotter.plot_n_vehicles_charging_hourly_mean_boxplot()
                    plotter.plot_city_zones()

                    for col in [
                        "origin_count",
                        "destination_count",
                        "charge_needed_system_zones_count",
                        "charge_needed_users_zones_count",
                        "unsatisfied_demand_origins_fraction",
                        "not_enough_energy_origins_count",
                        "charge_deaths_origins_count",
                    ]:
                        if col in sim_output.grid:
                            plotter.plot_choropleth(col)

                pbar.update()

            def print_error(e):
                tqdm.write(str(datetime.datetime.now())
                               + " Error: Simulation failed! Cause: "
                               + "-->{}<--".format(e.__cause__),
                           file=sys.stderr)

                pbar.update()

            futures = []

            run_func = get_eventG_sim_stats \
                if sim_technique == "eventG" \
                else get_traceB_sim_stats

            for conf_tuple in conf_tuples:
                future = pool.apply_async(
                    run_func, (conf_tuple,),
                    callback=collect_result,
                    error_callback=print_error)
                futures.append(future)

            [future.wait() for future in futures]

    print(datetime.datetime.now(), city, f'#{conf_id}', "multiple runs finished!")

    # # Convert the stats dict into a list ordered by key (configuration Id)
    # # and concatenates them row by row in a DataFrame
    # sim_stats_df = pd.concat([pool_stats_dict[res_id]
    # 						 	for res_id in sorted(pool_stats_dict)],
    # 						 axis=1, ignore_index=True).T
    # sim_stats_df.to_csv(os.path.join(results_path, "sim_stats.csv"))
    #
    # pd.Series(sim_general_conf).to_csv(os.path.join(results_path, "sim_general_conf.csv"), header=True)
    # pd.Series(sim_scenario_conf_grid).to_csv(os.path.join(results_path, "sim_scenario_conf_grid.csv"), header=True)
    #
    # sim_stats_df.to_pickle(os.path.join(results_path, "sim_stats.pickle"))
    # pd.Series(sim_general_conf).to_pickle(os.path.join(results_path, "sim_general_conf.pickle"))
    # pd.Series(sim_scenario_conf_grid).to_pickle(os.path.join(results_path, "sim_scenario_conf_grid.pickle"))

    # Store the general conf in the higher folder
    pd.Series(sim_general_conf).to_csv(os.path.join(results_path, "sim_general_conf.csv"), header=True)
    pd.Series(sim_general_conf).to_pickle(os.path.join(results_path, "sim_general_conf.pickle"))
