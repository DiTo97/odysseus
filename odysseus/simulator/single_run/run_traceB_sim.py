from odysseus.simulator.simulation.trace_driven_simulator import TraceDrivenSim
from odysseus.simulator.simulation_output.sim_output import SimOutput
from odysseus.simulator.simulation_input.sim_input import SimInput


def run_traceB_sim (simInput):

    sim_traceB = TraceDrivenSim(
    
                simInput=simInput,
    
            )

    sim_traceB.init_data_structures()
    sim_traceB.run()
    return sim_traceB


def get_traceB_sim_output(simInput):
    sim_traceB = run_traceB_sim(simInput)
    return SimOutput(sim_traceB)


def get_traceB_sim_stats(conf_tuple):
    """
    Parameters
    ----------
    conf_tuple : tuple
        Single combination of general conf and scenario conf
    """

    simInput = SimInput(conf_tuple)

    simInput.init_vehicles()
    simInput.init_relocation()
    simInput.init_workers()

    sim_traceB = run_traceB_sim(simInput)

    simOutput_traceB = SimOutput(sim_traceB)
    return simOutput_traceB.sim_stats, simOutput_traceB
