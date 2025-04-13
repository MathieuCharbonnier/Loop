import argparse
import numpy as np
import sys
import os
import opensim as osim
from opensim import Millard2012AccelerationMuscle

def run_simulation(dt, T, muscle_name, activation_array,resume_from=None, L0=None, v0=None):
    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    
    time_array = np.arange(0, T, dt)

    # First initialize the system without controllers
    state = model.initSystem()
    
    # Now add controller after initial system creation
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
    
    # IMPORTANT: Reinitialize the system after adding controllers and reporters
    state = model.initSystem()
    if resume_from is not None:
        print("continue from previous state")
        try:
            # Load the .sto file as a TimeSeriesTable
            table = osim.TimeSeriesTable(resume_from)
            # Get the last row (time point)
            lastRowIndex = table.getNumRows() - 1
            lastRow = table.getRowAtIndex(lastRowIndex)
        
            # Get column labels
            columnLabels = table.getColumnLabels()
        
            # Set each state variable
            for i in range(len(columnLabels)):
                label = columnLabels[i]
                value = lastRow[i]

                try:
                    model.setStateVariableValue(state, label, value)
                except Exception as e:
                    print(f"Couldn't set state {label}: {e}")
            
            state.setTime(0)  # Reset time to 0 for the new simulation
            
        except Exception as e:
            print(f"Error loading state: {e}")
            # If loading fails, use the initialized state
            pass

    """
    if L0  is not None:
        muscle.setFiberLength(state, L0)
    if v0 is not None:
        muscle.setFiberVelocity(state, v0)
    """
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(T)


    # Get all states and save to .sto file
    statesTable = manager.getStatesTable()
    osim.STOFileAdapter.write(statesTable, "states.sto")

    results_table = reporter.getTable()
    return results_table.getDependentColumn("fiber_length").to_numpy()

   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Muscle simulation')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscle', type=str, required=True, help='Muscle name')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input numpy array file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output numpy array')
    parser.add_argument('--resume_from', type=str, help='initialize the system with the state.sto file')
    parser.add_argument('--L0', type=float, help='Initial fiber length')
    parser.add_argument('--v0', type=float, help='Initial fiber velocity')
    
    args = parser.parse_args()
    
    # Load the input array
    activation_array = np.load(args.input_file)

    optional_args = {}
    if args.L0 is not None:
        optional_args["L0"] = args.L0
    if args.v0 is not None:
        optional_args["v0"] = args.v0
    if args.resume_from is not None:
        optional_args["resume_from"]=args.resume_from

    # Run the simulation
    fiber_length = run_simulation(
        args.dt, 
        args.T, 
        args.muscle, 
        activation_array,
        **optional_args
    )
    
    # Save the result to the output file
    np.save(args.output_file, fiber_length)
