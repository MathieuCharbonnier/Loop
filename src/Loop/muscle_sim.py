import argparse
from pickle import EMPTY_DICT
import numpy as np
import sys
import os
import json
import opensim as osim

def run_simulation(dt, T, muscles, joint_name, activation_array=None, torque_values=None, output_all=None, 
                  state_storage={}):
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
    if torque_values is not None:
        
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
        
        # Add actuator to model
        model.addForce(coord_actuator)
        
        # Create a controller for the coordinate actuator
        controller = osim.PrescribedController()
        controller.setName(f"Controller_{joint_name}")
        controller.addActuator(coord_actuator)
        controller.prescribeControlForActuator(coord_actuator.getName(), torque_function)
        model.addController(controller)
        
        print(f"Applying direct torque to coordinate '{joint_name}'")
  
    # Add muscle reporter to record fiber lengths, normalized forces and joint
    reporter = osim.TableReporter()
    reporter.setName("MuscleReporter")
    reporter.set_report_time_interval(dt)
    # Add this debugging code to see available outputs


    for muscle_name in muscles:
        muscle = model.getMuscles().get(muscle_name)
        reporter.addToReport(muscle.getOutput("fiber_length"), f'{muscle_name}_fiber_length')
        reporter.addToReport(muscle.getOutput("fiber_force"), f'{muscle_name}_fiber_force')
    coordinate = model.getCoordinateSet().get(joint_name)
    if coordinate is not None:
        reporter.addToReport(coordinate.getOutput("value"), f'{joint_name}_angle')
    else:
        print(f"Warning: Coordinate '{joint_name}' not found in model")
    model.addComponent(reporter)

    # Initialize state
    state = model.initSystem()
    if state_storage:
        for label, value in state_storage.items():
            try:
                model.setStateVariableValue(state, label, value)
            except Exception as e:
                print(f"Couldn't set state {label}: {e}")
        state.setTime(0)

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

    # Get muscle stretch, force and joint data 
    results_table = reporter.getTable()
    fiber_length = np.zeros((len(muscles), results_table.getNumRows()))
    normalized_force = np.zeros((len(muscles), results_table.getNumRows()))
    for i, muscle_name in enumerate(muscles):
        fiber_length[i] = results_table.getDependentColumn(f'{muscle_name}_fiber_length').to_numpy()
        muscle = model.getMuscles().get(muscle_name)
        normalized_force[i] = results_table.getDependentColumn(f'{muscle_name}_fiber_force').to_numpy()/muscle.getMaxIsometricForce()
    
    joint_angles = None
    if coordinate is not None:
        joint_angles = results_table.getDependentColumn(f'{joint_name}_angle').to_numpy()
        joint_angles = joint_angles * 180/np.pi
    
    json_ = {}
    statesTable= manager.getStatesTable()
    lastRowIndex = statesTable.getNumRows() - 1
    lastRow = statesTable.getRowAtIndex(lastRowIndex)
    columnLabels = statesTable.getColumnLabels()
    for i in range(len(columnLabels)):
        label = columnLabels[i]
        value = lastRow[i]
        json_[label] = value

    return fiber_length, normalized_force, joint_angles, json_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Muscle and direct coordinate actuation simulation in OpenSim')
    parser.add_argument('--dt', type=float, required=True, help='Time step')
    parser.add_argument('--T', type=float, required=True, help='Total simulation time')
    parser.add_argument('--muscles_names', type=str, required=True, help='Comma-separated list of muscles to activate/record')
    parser.add_argument('--joint_name', type=str, required=True, help='Joint coordinate to directly actuate and record')
    parser.add_argument('--state', type=str, help='State JSON file to initialise the simulation and save the final state')
    parser.add_argument('--activations', type=str, help='Path to input numpy array file for muscle activations')
    parser.add_argument('--torque', type=str, help='Path to numpy array file with torque values')
    parser.add_argument('--output_all', type=str, help='Path to the saved states file (.sto)')
    parser.add_argument('--output_stretch', type=str, help='Path to save output numpy array of fiber lengths')
    parser.add_argument('--output_force', type=str, help='Path to save output numpy array of normalized muscle forces')
    parser.add_argument('--output_joint', type=str, help='Path to save output numpy array of joint angles')


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
    joint_name = args.joint_name
    if args.torque:
        if not os.path.isfile(args.torque):
            raise FileNotFoundError(f"Torque values file not found: {args.torque}")
        torque_values = np.load(args.torque)
    
    # Load state file
    state = {}
    if args.state and os.path.isfile(args.state):
        try:
            with open(args.state, 'r') as f:
                state = json.load(f)
        except Exception as e:
            print(f"Error loading initial state file: {e}")

    # Run the simulation
    fiber_lengths, normalized_forces, joint_angles, json_ = run_simulation(
        args.dt,
        args.T,
        muscles,
        joint_name=joint_name,
        activation_array=activation_array,
        torque_values=torque_values,
        output_all=args.output_all,
        state_storage=state
    )
    
    # Save outputs if requested
    if args.output_stretch:
        np.save(args.output_stretch, fiber_lengths)
    
    if args.output_force:
        np.save(args.output_force, normalized_forces)
    
    if args.output_joint and joint_angles is not None:
        np.save(args.output_joint, joint_angles)
    
    
    with open(args.state, "w") as f:
        json.dump(json_, f, indent=4)
