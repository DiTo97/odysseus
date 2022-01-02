import importlib
import argparse
import shutil
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument(
    "exp_name",
    help="Name of the experiment"
)

parser.add_argument(
    "--conf_filename",
    help="Name of the configuration file",
    default="sim_conf"
)

parser.add_argument(
    "--n_cpus",
    help="Max number of CPUs to be used"
)

args = parser.parse_args()

# Metadata configuration
parent_dir = pathlib.Path(__file__).parent.absolute()

sim_input_conf_dir = parent_dir / 'configs' / 'escooter_mobility'
output_dir         = parent_dir / 'experiments'

experiment_dir = output_dir / args.exp_name

pathlib.Path.mkdir(experiment_dir, parents=True, exist_ok=True)

# Copy configuration files
shutil.rmtree(experiment_dir)
shutil.copytree(sim_input_conf_dir, experiment_dir)

# Rename to the default name
conf_filepath = experiment_dir / (args.conf_filename + ".py")

default_conf_filename = parser.get_default("conf_filename")
default_conf_filepath = experiment_dir / (default_conf_filename + ".py")

conf_filepath.rename(default_conf_filepath)

# Delete all other potential conf files
for filename in experiment_dir.glob(
        default_conf_filename + "_*.py"):
    filename.unlink()

sim_conf = importlib.import_module('odysseus.simulator.experiments.{}.{}'
                                   .format(args.exp_name, default_conf_filename))

# Custom simulator imports
from odysseus.simulator.single_run.single_run import single_run
from odysseus.simulator.multiple_runs.multiple_runs import multiple_runs

from odysseus.simulator.simulation_input.sim_config_grid import EFFCS_SimConfGrid

confs_dict = {
    "single_run"   : sim_conf.Single_run,
    "multiple_runs": sim_conf.Multiple_runs }

sim_general_conf_list = EFFCS_SimConfGrid(sim_conf.General).conf_list

# Launch a simulation for each conf
for general_conf_id, sim_general_conf in enumerate(sim_general_conf_list):
    sim_run_mode = sim_general_conf["sim_run_mode"]

    if sim_run_mode == "single_run":
        single_run((
            sim_general_conf,
            confs_dict[sim_general_conf["sim_run_mode"]],
            sim_general_conf["sim_scenario_name"]
        ))
    elif sim_run_mode == "multiple_runs":
        if args.n_cpus is not None:
            multiple_runs(
                sim_general_conf,
                confs_dict[sim_general_conf["sim_run_mode"]],
                sim_general_conf["sim_scenario_name"],
                args.exp_name,
                general_conf_id,
                n_cpus=int(args.n_cpus)
            )
        else:
            multiple_runs(
                sim_general_conf,
                confs_dict[sim_general_conf["sim_run_mode"]],
                sim_general_conf["sim_scenario_name"],
                args.exp_name,
                general_conf_id
            )
