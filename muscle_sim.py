import argparse
import numpy as np
import sys
import os
import json
import opensim as osim

def run_simulation(dt, T, muscle_names, activation_array, output_all=None, initial_state=None, final_state=None, stretch_file=None):

    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    time_array = np.arange(0, T, dt)

    class ActivationController(osim.PrescribedController):
        def __init__(self, model, muscle_name, time_array, activation_array):
            super().__init__()
            self.setName("ActivationController")
            for i,muscle_name in enumerate(muscle_names):
                muscle = model.getMuscles().get(muscle_name)
                self.addActuator(muscle)

                func = osim.PiecewiseLinearFunction()
                for t, a in zip(time_array, activation_array[i]):
                    func.addPoint(t, float(a))

                self.prescribeControlForActuator(muscle_name, func)

    controller = ActivationController(model, muscle_names, time_array, activation_array)
    model.addController(controller)

    if stretch_file is not None:
        reporter = osim.TableReporter()
        reporter.setName("MuscleReporter")
        reporter.set_report_time_interval(dt)
        for i,muscle_name in enumerate(muscle_names):
            muscle = model.getMuscles().get(muscle_name)
            reporter.addToReport(muscle.getOutput("fiber_length"), f'{muscle_name}_fiber_length')
        model.addComponent(reporter)

    # Initialize state
    state = model.initSystem()
    if initial_state is not None:
        print("Continue from previous state")
        if os.path.isfile(initial_state):
            try:
                with open(initial_state, 'r') as f:
                    data = json.load(f)
                for label, value in data.items():
                    try:
                        model.setStateVariableValue(state, label, value)
                    except Exception as e:
                        print(f"Couldn't set state {label}: {e}")
                state.setTime(0)
            except Exception as e:
                print(f"Error loading initial state file: {e}")
        else:
            print(f"Initial state file not found: {initial_state}")

    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(T)

    if output_all is not None:
        statesTable = manager.getStatesTable()
        osim.STOFileAdapter.write(statesTable, output_all)
        print(f'{output_all} file is saved')

    if final_state is not None:
        json_ = {}
        statesTable = manager.getStatesTable()
        lastRowIndex = statesTable.getNumRows() - 1
        lastRow = statesTable.getRowAtIndex(lastRowIndex)
        columnLabels = statesTable.getColumnLabels()
        for i in range(len(columnLabels)):
            label = columnLabels[i]
            value = lastRow[i]
            json_[label] = value
        with open(final_state, "w") as f:
            json.dump(json_, f, indent=4)

    if stretch_file is not None:
        results_table = reporter.getTable()
        fiber_length=np.zeros((len(muscle_names),int(T/dt)+1))
        for i,muscle_name in enumerate(muscle_names):
            fiber_length[i] = results_table.getDependentColumn(f'{muscle_name}_fiber_length').to_numpy()
        np.save(stretch_file, fiber_length)

    if output_all is None and final_state is None and stretch_file is None:
        raise ValueError("At least one output target must be specified: 'output_all', 'final_state', or 'stretch_file'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Muscle simulation')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscles', type=str, required=True, help='Muscle name')
    parser.add_argument('--activations', type=str, required=True, help='Path to input numpy array file')
    parser.add_argument('--initial_state', type=str, help='Initial state JSON file')
    parser.add_argument('--output_all', type=str, help='Path to the saved states file (.sto)')
    parser.add_argument('--output_stretch', type=str, help='Path to save output numpy array')
    parser.add_argument('--output_final_state', type=str, help="Path to save final state JSON file")

    args = parser.parse_args()

    # Validate activation input file
    if not os.path.isfile(args.activations):
        raise FileNotFoundError(f"Activation file not found: {args.activations}")

    activation_array = np.load(args.activations)

    # Gather optional arguments
    optional_args = {
        'initial_state': args.initial_state,
        'final_state': args.output_final_state,
        'stretch_file': args.output_stretch,
    }

    run_simulation(
        args.dt,
        args.T,
        args.muscles.split(','),
        activation_array,
        output_all=args.output_all,
        **{k: v for k, v in optional_args.items() if v is not None}
    )

