import argparse
import numpy as np
import sys
import os
import opensim as osim
from opensim import Millard2012EquilibriumMuscle

def run_simulation(dt, T, muscle_name, activation_array, continue_from=None, save_state=None):
    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    
    time_array = np.arange(0, T, dt)

    class ActivationController(osim.PrescribedController):
        def __init__(self, model, muscle_name, time_array, activation_array):
            super().__init__()
            self.setName("ActivationController")
            muscle = model.getMuscles().get(muscle_name)
            self.addActuator(muscle)

            func = osim.PiecewiseLinearFunction()
            for t, a in zip(time_array, activation_array):
                func.addPoint(t, float(a))

            self.prescribeControlForActuator(muscle_name, func)

    controller = ActivationController(model, muscle_name, time_array, activation_array)
    model.addController(controller)

    reporter = osim.TableReporter()
    reporter.setName("MuscleReporter")
    reporter.set_report_time_interval(dt)
    muscle = model.getMuscles().get(muscle_name)
    muscle = Millard2012EquilibriumMuscle.safeDownCast(muscle)
    reporter.addToReport(muscle.getOutput("fiber_length"), "fiber_length")
    model.addComponent(reporter)

    state = model.initSystem()
    
    # Initialize from a saved state if provided
    if continue_from and os.path.exists(continue_from):
        print(f"Loading simulation state from {continue_from}")
        sto = osim.Storage(continue_from)
        state = osim.StatesTrajectory.createFromStatesStorage(model, sto)
        T += state.getTime()  # Add the starting time to the total simulation time

    else:
        state.setTime(0)
    
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    
    manager.integrate(T)
    
    # Save the final state if requested
    if save_state:
        state = manager.getState()
        state.write(save_state)
        print(f"Simulation state saved to {save_state}")
    
    results_table = reporter.getTable()
    return results_table.getDependentColumn("fiber_length").to_numpy()

   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Muscle simulation')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscle', type=str, required=True, help='Muscle name')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input numpy array file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output numpy array')
    parser.add_argument('--continue_from', type=str, help='Path to state file to continue simulation from')
    parser.add_argument('--save_state', type=str, help='Path to save final simulation state')
    
    args = parser.parse_args()
    
    # Load the input array
    activation_array = np.load(args.input_file)
    
    # Run the simulation
    fiber_length = run_simulation(
        args.dt, 
        args.T, 
        args.muscle, 
        activation_array,
        continue_from=args.continue_from,
        save_state=args.save_state
    )
    
    # Save the result to the output file
    np.save(args.output_file, fiber_length)
