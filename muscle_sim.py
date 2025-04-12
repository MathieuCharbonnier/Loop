import argparse
import numpy as np
import sys
import opensim as osim

def run_simulation(dt, T, muscle_name, initial_fiber_length, activation_array):
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
    reporter.addToReport(muscle.getOutput("fiber_length"), "fiber_length")
    model.addComponent(reporter)

    state = model.initSystem()
    muscle.setFiberLength(state, initial_fiber_length)
    manager = osim.Manager(model)
    state.setTime(0)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(T)

    results_table = reporter.getTable()
    return results_table.getDependentColumn("fiber_length").to_numpy()

   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Muscle simulation')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscle', type=str, required=True, help='Muscle name')
    parser.add_argument('--L', type=float, required=True, help='initial fiber length')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input numpy array file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output numpy array')
    
    args = parser.parse_args()
    
    # Load the input array
    activation_array = np.load(args.input_file)
    # Run the simulation
    fiber_length=run_simulation(args.dt, args.T, args.muscle, args.L, activation_array)
    # Save the result to the output file
    np.save(args.output_file, fiber_length)
  
