"""
Reproduce plots from Fig. 5 to Fig. 7 from the paper
A. Ciociola, M. Cocca, et al. - E-Scooter Sharing: Leveraging Open Data for System Design
"""
import pathlib

from plotter import Plotter

Constants = {
    'CITY': 'Louisville',
    'RUN_TYPE': 'multiple_runs',
    'SCENARIO_NAME': 'escooter_mobility'
}

Paths = {
    'EXPERIMENTS_DIR': 'experiments'
}


def get_experiment_dir(name, general_conf_id,
                       city=Constants['CITY']):
    simulator_dir = pathlib.Path(__file__) \
                           .absolute().parents[1]

    return str(simulator_dir
            / Paths['EXPERIMENTS_DIR']
            / name
            / 'results'
            / city
            / Constants['RUN_TYPE']
            / Constants['SCENARIO_NAME']
            / str(general_conf_id))


Fig5_dir = 'Workers-100_Clock-00-60-min_Prof_Charge-reactive_Not-ideal-times_Austin'


if __name__ == '__main__':
    # Figure 5 plot
    dir_fleet_size = get_experiment_dir(
        Fig5_dir,
        0,
        'Austin'
    )

    Plotter.plot_fleet_size(dir_fleet_size,
                            Fig5_dir)

    # # Figure 6 plot
    # dir_charging_thresh = get_experiment_dir(
    #     'Alpha_10_80',
    #     0
    # )
    #
    # Plotter.plot_charging_thresh(dir_charging_thresh,
    #                              'Alpha_10_80')

    # # Figure 7 plot
    # dir_charging_policy = get_experiment_dir(
    #     'Fig7_Charging-policy_No-willingness_Louisville_01_2020',
    #     0
    # )

    # Plotter.plot_charging_policy(dir_charging_policy,
    #                              'Fig7_Charging-policy_No-willingness_Louisville_01_2020')
