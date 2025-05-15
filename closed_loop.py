from brian2 import *
import numpy as np
import pandas as pd
import os
import subprocess
import tempfile
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

from neural_simulations import run_one_muscle_neuron_simulation, run_flexor_extensor_neuron_simulation
from activation import decode_spikes_to_activation


def closed_loop(NUM_ITERATIONS, REACTION_TIME, TIME_STEP, NEURON_COUNTS, CONNECTIONS,
           SPINDLE_MODEL, BIOPHYSICAL_PARAMS, MUSCLE_NAMES, associated_joint, base_output_path, 
            EES_PARAMS=None, torque=None, fast=True, seed=42):
    """
    Neuromuscular Simulation Pipeline with Initial Dorsiflexion

    This script runs a neuromuscular simulation that integrates:
    1. EES stimulation
    2. Spike-to-activation decoding
    3. Muscle length/velocity simulation via OpenSim
    4. Proprioceptive feedback
    5. Initial dorsiflexion perturbation for clonus simulation

    Parameters:
    -----------
    NUM_ITERATIONS : int
        Number of simulation iterations to run
    REACTION_TIME : brian2.unit.second
        Duration of each simulation iteration
    TIME_STEP : brian2.unit.second
        Time step size for simulation
    EES_PARAMS : dict
        Parameters for electrical epidural stimulation
    NEURON_COUNTS : dict
        Number of neurons for each type
    CONNECTIONS : dict
        Neural connection configuration
    SPINDLE_MODEL : dict
        Model parameters for muscle spindles
    BIOPHYSICAL_PARAMS : dict
        Biophysical model parameters
    MUSCLE_NAMES : list
        List of muscle names strings
    associated_joint : str
        Name of the associated joint
    sto_path : str  
        Path to save STO results file
    torque : numpy.ndarray, optional
        External torque to apply at each time step
    seed : int, optional
        Random seed for simulation reproducibility (default: 42)
    csv_path : str, optional
        Path to save the combined data CSV file (default: None, will use sto_path with .csv extension)
    """
    
    # create CSV and sto paths 
    csv_path = base_output_path + '.csv'
    sto_path = base_output_path + '.sto'

    # Muscle configuration
    NUM_MUSCLES = len(MUSCLE_NAMES)
    # Validate muscle count
    if NUM_MUSCLES > 2:
        raise ValueError("This pipeline supports only 1 or 2 muscles!")
    

    # =============================================================================
    # Initialization
    # =============================================================================

    #Discritization configuration
    nb_points = int(REACTION_TIME/TIME_STEP)
                       
    # Initialize muscle activation
    activations = np.zeros(( NUM_MUSCLES, nb_points))
    time_points = np.arange(0, REACTION_TIME/second, TIME_STEP/second)

    initial_potentials = {
        "exc": BIOPHYSICAL_PARAMS['Eleaky'],
        "MN": BIOPHYSICAL_PARAMS['Eleaky']
    }
    if NUM_MUSCLES == 2:
        initial_potentials["inh"] = BIOPHYSICAL_PARAMS['Eleaky']

    # Initialize parameters for each motoneuron
    initial_params = [
        [{
            'u0': [0.0, 0.0],    # Initial fiber AP state
            'c0': [0.0, 0.0],    # Initial calcium concentration state
            'P0': 0.0,           # Initial calcium-troponin binding state
            'a0': 0.0            # Initial activation state
        } for _ in range(NEURON_COUNTS['MN'])]
        for _ in range(NUM_MUSCLES)]

    # Containers for simulation data
    muscle_data = [[] for _ in range(NUM_MUSCLES)]
    resting_lengths = [None] * NUM_MUSCLES
    joint_all=np.zeros((NUM_ITERATIONS*nb_points))
    joint_velocity_all=np.zeros((NUM_ITERATIONS*nb_points))
 

    spike_data = {
        muscle_name: {
            neuron_type: defaultdict(list)
            for neuron_type in NEURON_COUNTS.keys()
        }
        for muscle_name in MUSCLE_NAMES
    }

    # Use temporary file for state management across iterations
    state_file = None
    
    # =============================================================================
    # Main Simulation Loop
    # =============================================================================

    print("Start Simulation:")
    if EES_PARAMS is not None:
        print(f"EES frequency: {EES_PARAMS['ees_freq']}")
        print(f"Number Ia fibers recruited by EES: {EES_PARAMS['Ia_recruited']} / {NEURON_COUNTS['Ia']}")
        if "II" in NEURON_COUNTS and "II" in SPINDLE_MODEL:
            print(f"Number II fibers recruited by EES: {EES_PARAMS['II_recruited']} / {NEURON_COUNTS['II']}")
        print(f"Number Efferent fibers recruited by EES: {EES_PARAMS['eff_recruited']} / {NEURON_COUNTS['MN']}")
    
    # Create reusable temporary files for the whole simulation
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_activation, \
         tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_torque_file:
        
        input_activation_path = input_activation.name
        input_torque_path = input_torque_file.name

        for iteration in range(NUM_ITERATIONS):
            print(f"--- Iteration {iteration+1} of {NUM_ITERATIONS} ---")

            # Create temporary files for this iteration
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as output_stretch, \
                 tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as output_joint, \
                 tempfile.NamedTemporaryFile(suffix='.json', delete=False) as new_state_file:

                output_stretch_path = output_stretch.name
                output_joint_path = output_joint.name
                new_state_path = new_state_file.name

                # Save activations to temporary file
                np.save(input_activation_path, activations)

                # Build command for OpenSim muscle simulation
                cmd = [
                    'conda', 'run', '-n', 'opensim_env', 'python', 'muscle_sim.py',
                    '--dt', str(TIME_STEP/second),
                    '--T', str(REACTION_TIME/second),
                    '--muscles_names',','.join(MUSCLE_NAMES),
                    '--joint_name', associated_joint,
                    '--activation', input_activation_path,
                    '--output_stretch', output_stretch_path,
                    '--output_joint', output_joint_path,
                    '--output_final_state', new_state_path
                ]

                # Add initial state parameter if not the first iteration
                if state_file is not None:
                    cmd += ['--initial_state', state_file]
                
                # Add torque if provided
                if torque is not None:
                    start_idx = iteration * nb_points
                    end_idx = (iteration + 1) * nb_points
                    current_torque = torque[start_idx:end_idx]
                    np.save(input_torque_path, current_torque)
                    cmd += ['--torque', input_torque_path]


                # Run OpenSim simulation
                process = subprocess.run(cmd, capture_output=True, text=True)

                # Process OpenSim results
                if process.returncode == 0:
                    # Load muscle lengths and joint from simulation
                    fiber_lengths = np.load(output_stretch_path)
                    joint = np.load(output_joint_path)
                    # Remove the last value as it will be included in the next iteration
                    fiber_lengths = fiber_lengths[:, :-1]
                    joint = joint[:-1]
                    joint_all[iteration*nb_points: (iteration+1)*nb_points]=joint
                    joint_velocity=np.gradient(joint, time_points)
                    joint_velocity_all[iteration*nb_points: (iteration+1)*nb_points]=joint_velocity
                    if resting_lengths[0] is None:
                        resting_lengths = fiber_lengths[:, 0]

                    stretch = np.zeros((NUM_MUSCLES, nb_points))
                    stretch_velocity = np.zeros((NUM_MUSCLES, nb_points))
                    
                    # Process each muscle's data
                    for muscle_idx in range(NUM_MUSCLES):
                        # Set resting length on first iteration if not already set
                        if resting_lengths[muscle_idx] is None:
                            resting_lengths[muscle_idx] = fiber_lengths[muscle_idx, 0]

                        # Calculate stretch and velocity for next iteration
                        stretch[muscle_idx] = fiber_lengths[muscle_idx] / resting_lengths[muscle_idx] - 1
                        stretch_velocity[muscle_idx] = np.gradient(stretch[muscle_idx], time_points)

                else:
                    error_msg = f'Error in iteration {iteration+1}. STDERR: {process.stderr}'
                    raise RuntimeError(error_msg)

                # Clean up the old state file if it exists 
                if iteration > 0 and state_file is not None and state_file != new_state_path:
                    os.unlink(state_file)

                # Update state file for next iteration
                state_file = new_state_path

                # Clean up other temporary files
                os.unlink(output_stretch_path)
                os.unlink(output_joint_path)
                         
            # Run neural simulation based on muscle count
            if NUM_MUSCLES == 1:
                all_spikes, final_potentials, state_monitors = run_one_muscle_neuron_simulation(
                    stretch, stretch_velocity,joint, joint_velocity, NEURON_COUNTS, CONNECTIONS, TIME_STEP, REACTION_TIME, SPINDLE_MODEL, seed,
                    initial_potentials, **BIOPHYSICAL_PARAMS, ees_params=EES_PARAMS
                )
            else:  # NUM_MUSCLES == 2
                all_spikes, final_potentials, state_monitors = run_flexor_extensor_neuron_simulation(
                    stretch, stretch_velocity, NEURON_COUNTS, CONNECTIONS, TIME_STEP, REACTION_TIME, SPINDLE_MODEL, seed,
                    initial_potentials, **BIOPHYSICAL_PARAMS, ees_params=EES_PARAMS
                )
            initial_potentials.update(final_potentials)

            # Store spike times for visualization
            for muscle_idx, muscle_name in enumerate(MUSCLE_NAMES):
                muscle_spikes = all_spikes[muscle_idx]
                for fiber_type, fiber_spikes in muscle_spikes.items():
                    for neuron_id, spikes in fiber_spikes.items():
                        # Adjust spike times by iteration offset
                        adjusted_spikes = spikes/second + iteration * REACTION_TIME/second
                        spike_data[muscle_name][fiber_type][neuron_id].extend(adjusted_spikes)

            # Initialize arrays for mean values of all neurons per muscle
            mean_e, mean_u, mean_c, mean_P, mean_activation = [
                np.zeros((NUM_MUSCLES, int(REACTION_TIME/TIME_STEP))) for _ in range(5)
            ]

            # Process motor neuron spikes to get muscle activations
            for muscle_idx, muscle_spikes in enumerate(all_spikes):
                # Only process if we have motor neuron spikes
                if "MN" in muscle_spikes and len(muscle_spikes["MN"]) > 0:
                    # Convert spike times to seconds
                    mn_spikes_sec = [value/second for _, value in muscle_spikes["MN"].items()]
                    # Decode spikes to muscle activations
                    e, u, c, P, activations_result, final_values = decode_spikes_to_activation(
                        mn_spikes_sec,
                        TIME_STEP/second,
                        REACTION_TIME/second,
                        initial_params[muscle_idx],
                        fast=fast
                    )

                    # Store mean values across all neurons
                    mean_e[muscle_idx] = np.mean(e, axis=0)
                    mean_u[muscle_idx] = np.mean(u, axis=0)
                    mean_c[muscle_idx] = np.mean(c, axis=0)
                    mean_P[muscle_idx] = np.mean(P, axis=0)
                    mean_activation[muscle_idx] = np.mean(activations_result, axis=0)

                    # Update activation for next iteration
                    activations[muscle_idx] = mean_activation[muscle_idx]

                    # Save final state for next iteration
                    initial_params[muscle_idx] = final_values

                    # Create batch data for current iteration including joint data and torque if available
                    batch_data = {
                        'Time': time_points + iteration * REACTION_TIME/second,
                        f'Fiber_length_{MUSCLE_NAMES[muscle_idx]}': fiber_lengths[muscle_idx],
                        f'Stretch_{MUSCLE_NAMES[muscle_idx]}': stretch[muscle_idx],
                        f'Stretch_Velocity_{MUSCLE_NAMES[muscle_idx]}': stretch_velocity[muscle_idx],
                        f'mean_e_{MUSCLE_NAMES[muscle_idx]}': mean_e[muscle_idx],
                        f'mean_u_{MUSCLE_NAMES[muscle_idx]}': mean_u[muscle_idx],
                        f'mean_c_{MUSCLE_NAMES[muscle_idx]}': mean_c[muscle_idx],
                        f'mean_P_{MUSCLE_NAMES[muscle_idx]}': mean_P[muscle_idx],
                        f'Activation_{MUSCLE_NAMES[muscle_idx]}': mean_activation[muscle_idx],
                    }
                    
                    # Add state monitor data with muscle name as suffix
                    for key, value in state_monitors[muscle_idx].items():
                        batch_data[f'{key}_{MUSCLE_NAMES[muscle_idx]}'] = value

                    # Store batch data for this muscle
                    muscle_data[muscle_idx].append(pd.DataFrame(batch_data))

        # Clean up reusable temporary files
        os.unlink(input_activation_path)
        os.unlink(input_torque_path)

    # =============================================================================
    # Combine Results and Compute Firing rates
    # =============================================================================

    # Create a single dataframe with all muscle data
    all_data_dfs = []
    
    # Process data for each muscle
    for muscle_idx, muscle_name in enumerate(MUSCLE_NAMES):
          
        # Combine all iteration data for this muscle
        df=pd.concat(muscle_data[muscle_idx], ignore_index=True)

        # Compute firing rate for this muscle
        # Extract stretch and velocity values for this muscle
        stretch_values = df[f'Stretch_{muscle_name}'].values
        stretch_velocity_values = df[f'Stretch_Velocity_{muscle_name}'].values

        time = df['Time'].values
        
        # Compute Ia firing rate using spindle model
        Ia_rate = eval(SPINDLE_MODEL['Ia'], 
                       {"__builtins__": {'sign': np.sign, 'abs': np.abs, 'clip': np.clip}}, 
                       {"stretch": stretch_values, "stretch_velocity": stretch_velocity_values,
                        "joint": joint_all, "joint_velocity":joint_velocity_all})
        
        df[f'Ia_rate_baseline_{muscle_name}'] = Ia_rate

        # Compute II firing rate if applicable
        if "II" in NEURON_COUNTS and "II" in SPINDLE_MODEL:
            II_rate = eval(SPINDLE_MODEL['II'], 
                          {"__builtins__": {}}, 
                          {"stretch": stretch_values, "stretch_velocity": stretch_velocity_values,
                           "joint": joint_all, "joint_velocity":joint_velocity_all})

            df[f'II_rate_baseline_{muscle_name}'] = II_rate

        # Calculate all firing rate using KDE
        for fiber_name, fiber_spikes in spike_data[muscle_name].items():
        
            all_spike_times = np.concatenate(list(fiber_spikes.values())) 
            
            firing_rate = np.zeros_like(time)
            if len(all_spike_times) > 1:
                kde = gaussian_kde(all_spike_times, bw_method=0.3)
                firing_rate = kde(time) * len(all_spike_times) / max(len(fiber_spikes), 1)
            df[f'{fiber_name}_measured_rate_{muscle_name}'] = firing_rate
        all_data_dfs.append(df)
    
    # Combine all muscle data into a single dataframe
    if len(all_data_dfs) > 1:
        # If we have multiple muscles, merge on Time
        combined_df = pd.merge(all_data_dfs[0], all_data_dfs[1], on='Time', how='outer')
    else:
        # If only one muscle, use that dataframe
        combined_df = all_data_dfs[0]
    
    # Move Time column to be the first column
    combined_df = combined_df[['Time'] + [col for col in combined_df.columns if col != 'Time']]

    #add joint and torque if torque is applied
    combined_df[f'Joint_{associated_joint}'] = joint_all
    combined_df[f'Joint_Velocity_{associated_joint}'] = joint_velocity_all
    if current_torque is not None:
        combined_df['Torque'] = torque


    # Save the combined dataframe to CSV
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved combined data to {csv_path}")

    # ====================================================================================================
    # Run Complete Muscle Simulation for visualization of the dynamic on opensim
    # ======================================================================================================

    # Create full simulation STO file for visualization
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_activation_file, \
         tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_joint_file:
        
        input_activation_path = input_activation_file.name
        input_joint_path = input_joint_file.name

        # Save muscle activations for full simulation
        total_time_points = nb_points * NUM_ITERATIONS
        activations_array = np.zeros((NUM_MUSCLES, total_time_points))
        
        for muscle_idx, muscle_name in enumerate(MUSCLE_NAMES):
            # Extract activations from combined dataframe
            if f'Activation_{muscle_name}' in combined_df.columns:
                #Input activations are not exactly the output activations present in the combined_df dataframe       
                activations_array[muscle_idx, nb_points:] = combined_df[f'Activation_{muscle_name}'].values[nb_points:]
        np.save(input_activation_path, activations_array)

        # Build command for full muscle simulation
        cmd = [
            'conda', 'run', '-n', 'opensim_env', 'python', 'muscle_sim.py',
            '--dt', str(TIME_STEP/second),
            '--T', str(REACTION_TIME/second * NUM_ITERATIONS),
            '--muscles_names', ','.join(MUSCLE_NAMES),
            '--activation', input_activation_path,
            '--output_all', sto_path
        ]
        
        if torque is not None:
            np.save(input_joint_path, torque)
            cmd += [
                '--joint_name', associated_joint,
                '--torque', input_joint_path
            ]

        # Run OpenSim simulation for complete trajectory
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.stdout.strip():
            print("STDOUT:\n", process.stdout)
        if process.stderr.strip():
            print(f"STDERR: {process.stderr}")

        # Clean up temporary files
        os.unlink(input_activation_path)
        os.unlink(input_joint_path)

    return spike_data, combined_df
