import os

sim_input_path = os.path.dirname(__file__)  # '/home/gianvito/Desktop/odysseus-master/odysseus/simulator/simulation_input'

simulation_input_paths = {
    'sim_configs_target': os.path.join(sim_input_path, 'sim_configs_target.json'), # <-- json con i path utilizzati
    'sim_configs_versioned': os.path.join(sim_input_path, 'sim_configs_versioned'),
    'sim_current_config': os.path.join(sim_input_path, 'sim_current_config')
}
