import argparse
import numpy as np
import sys
import os
import json
import opensim as osim

def run_simulation(dt, T, muscles, activation_array=None, joint_name=None, torque_values=None, output_all=None, 
                  initial_state=None, final_state=None, stretch_file=None, joint_file=None):
    """
    Run an OpenSim simulation with muscle activations and/or direct joint torques.
    
    Parameters:
    -----------
    dt : float
        Time step for simulation
    T : float
        Total simulation time
    muscles : list
        List of muscle names to be recorded and/or activated
    activation_array : numpy.ndarray, optional
        Array of muscle activations over time [muscles × time_points]
    joint_name : string, optional
        Which joint coordinate to directly apply torque to and record
    torque_values: numpy.array, optional
         torque values [time_points]
    output_all : str, optional
        Path to save all states (.sto file)
    initial_state : str, optional
        Path to JSON file with initial state
    final_state : str, optional
        Path to save final state as JSON
    stretch_file : str, optional
        Path to save muscle fiber lengths
    joint_file : str, optional
        Path to save joint angles
    """

    model = osim.Model("Model/gait2392_millard2012_pelvislocked.osim")
    time_array = np.arange(0, T, dt)
    
    # Add muscle controller if activation provided
    if activation_array is not None:
        
        class ActivationController(osim.PrescribedController):
            def __init__(self, model, muscle_list, time_array, activation_array):
                super().__init__()
                self.setName("ActivationController")
                for i, muscle_name in enumerate(muscle_list):
                    muscle = model.getMuscles().get(muscle_name)
                    self.addActuator(muscle)

                    func = osim.PiecewiseLinearFunction()
                    for t, a in zip(time_array, activation_array[i]):
                        func.addPoint(t, float(a))

                    self.prescribeControlForActuator(muscle_name, func)

        controller = ActivationController(model, muscles, time_array, activation_array)
        model.addController(controller)
    
    # Add direct joint coordinate torque if provided
    if joint_name is not None and torque_values is not None:
        # Get the coordinate for the specified joint
        coordinate = model.getCoordinateSet().get(joint_name)
        if coordinate is None:
            raise ValueError(f"Coordinate '{joint_name}' not found in the model")
        
        # Create torque function
        torque_function = osim.PiecewiseLinearFunction()
        for t, torque in zip(time_array, torque_values):
            torque_function.addPoint(t, float(torque))
        
        # Create a CoordinateActuator to directly apply torque to the joint coordinate
        coord_actuator = osim.CoordinateActuator(joint_name)
        coord_actuator.setName(f"Torque_{joint_name}")
        
        # Set optimal force large enough to handle the maximum torque
        max_torque = max(abs(max(torque_values)), abs(min(torque_values)))
        coord_actuator.setOptimalForce(max_torque * 1.5)  # Set safely above maximum
        
        # Add actuator to model
        model.addForce(coord_actuator)
        
        # Create a controller for the coordinate actuator
        controller = osim.PrescribedController()
        controller.setName(f"Controller_{joint_name}")
        controller.addActuator(coord_actuator)
        controller.prescribeControlForActuator(coord_actuator.getName(), torque_function)
        model.addController(controller)
        
        print(f"Applying direct torque to coordinate '{joint_name}'")
  

    # Add muscle reporter to record fiber lengths
    if stretch_file is not None:
        reporter = osim.TableReporter()
        reporter.setName("MuscleReporter")
        reporter.set_report_time_interval(dt)
        for muscle_name in muscles:
            muscle = model.getMuscles().get(muscle_name)
            reporter.addToReport(muscle.getOutput("fiber_length"), f'{muscle_name}_fiber_length')
        model.addComponent(reporter)

    # Add joint reporter to record joint angles
    joint_reporter = None
    if joint_file is not None and joint_name is not None:
        joint_reporter = osim.TableReporter()
        joint_reporter.setName("JointReporter")
        joint_reporter.set_report_time_interval(dt)
        coordinate = model.getCoordinateSet().get(joint_name)
        if coordinate is not None:
            joint_reporter.addToReport(coordinate.getOutput("value"), f'{joint_name}_angle')
        else:
            print(f"Warning: Coordinate '{joint_name}' not found in model")
        
        model.addComponent(joint_reporter)

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
    # Set muscles to resting state if no initial state is provided
    else:
        print("Setting muscles to resting state")
        for muscle_name in muscles:
            muscle = model.getMuscles().get(muscle_name)
            # Set muscle activation to 0
            muscle.setActivation(state, 0.0)
                
        model.equilibrateMuscles(state)

    # Run the simulation
    manager = osim.Manager(model)
    manager.setIntegratorAccuracy(1e-4)
    manager.initialize(state)
    manager.integrate(T)

    # Save full simulation results if requested
    if output_all is not None:
        statesTable = manager.getStatesTable()
        osim.STOFileAdapter.write(statesTable, output_all)
        print(f'{output_all} file is saved')

    # Save final state if requested
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

    # Save muscle stretch data if requested
    if stretch_file is not None:
        results_table = reporter.getTable()
        fiber_length = np.zeros((len(muscles), results_table.getNumRows()))
        print("len fiber_length ", fiber_length.shape)
        for i, muscle_name in enumerate(muscles):
            print('results ',results_table.getDependentColumn(f'{muscle_name}_fiber_length').to_numpy().shape)
            fiber_length[i] = results_table.getDependentColumn(f'{muscle_name}_fiber_length').to_numpy()
            
        np.save(stretch_file, fiber_length)
    
    # Save joint angle data if requested
    if joint_file is not None and joint_reporter is not None:
        results_table = joint_reporter.getTable()
        joint_angles = results_table.getDependentColumn(f'{joint_name}_angle').to_numpy()
        joint_angles=joint_angles*180/np.pi
        np.save(joint_file, joint_angles)
        print(f'{joint_file} file is saved')

    if output_all is None and final_state is None and stretch_file is None and joint_file is None:
        raise ValueError("At least one output target must be specified: 'output_all', 'final_state', 'stretch_file', or 'joint_file'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Muscle and direct coordinate actuation simulation in OpenSim')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscles_names', type=str, required=True, help='Comma-separated list of muscles to activate/record')
    parser.add_argument('--initial_state', type=str, help='Initial state JSON file')
    parser.add_argument('--activations', type=str, help='Path to input numpy array file for muscle activations')
    parser.add_argument('--joint_name', type=str, help='Joint coordinate to directly actuate and record')
    parser.add_argument('--torque', type=str, help='Path to numpy array file with torque values')
    parser.add_argument('--output_all', type=str, help='Path to the saved states file (.sto)')
    parser.add_argument('--output_stretch', type=str, help='Path to save output numpy array of fiber lengths')
    parser.add_argument('--output_joint', type=str, help='Path to save output numpy array of joint angles')
    parser.add_argument('--output_final_state', type=str, help="Path to save final state JSON file")

    args = parser.parse_args()
    
    # Parse muscles list
    muscles = args.muscles_names.split(',')
    
    # Load activation data if provided
    activation_array = None
    if args.activations:
        if not os.path.isfile(args.activations):
            raise FileNotFoundError(f"Activation file not found: {args.activations}")
        activation_array = np.load(args.activations)
        
    # Load and prepare torque data if provided
    torque_values = None
    joint_name = None
    if args.joint_name and args.torque:
        if not os.path.isfile(args.torque):
            raise FileNotFoundError(f"Torque values file not found: {args.torque}")

        torque_values = np.load(args.torque)
        joint_name = args.joint_name
    
    # Ensure at least one of activation or torque is provided
    if activation_array is None and torque_values is None:
        raise ValueError("Must provide either muscle activations or external torques")

    # Run the simulation
    run_simulation(
        args.dt,
        args.T,
        muscles,
        activation_array=activation_array,
        joint_name=joint_name,
        torque_values=torque_values,
        output_all=args.output_all,
        initial_state=args.initial_state,
        final_state=args.output_final_state,
        stretch_file=args.output_stretch,
        joint_file=args.output_joint
    )
