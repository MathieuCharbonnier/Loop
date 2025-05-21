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

from neural_dynamics import run_monosynaptic_simulation, run_trisynaptic_simulation, run_flexor_extensor_neuron_simulation
from activation import decode_spikes_to_activation
from input_generator import transform_torque_params_in_array, transform_intensity_balance_in_recruitment


def closed_loop(n_iterations, reaction_time, time_step, neurons_population, connections,
              spindle_model, biophysical_params, muscles_names, num_muscles, associated_joint, 
              base_output_path, ees_recruitment_profile, ees_stimulation_params=None, 
              torque=None, fast=True, seed=42):
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
    n_iterations : int
        Number of simulation iterations to run
    reaction_time : brian2.unit.second
        Duration of each simulation iteration
    time_step : brian2.unit.second
        Time step size for simulation
    neurons_population : dict
        Number of neurons for each type
    connections : dict
        Neural connection configuration
    spindle_model : dict
        Model parameters for muscle spindles
    biophysical_params : dict
        Biophysical model parameters
    muscles_names : list
        List of muscle names strings
    num_muscles : int
        Number of muscles (length of muscles_names list)
    associated_joint : str
        Name of the associated joint
    base_output_path : str
        Base path for output files
    ees_recruitment_profile : dict
        Parameters to define recruitment curve
    ees_stimulation_params : dict, optional
        Parameters for electrical epidural stimulation
    torque : dict, optional
        Parameter to create the external torque profile applied at each time step
    fast : bool, optional
        Whether to use the fast spike-to-activation decoding algorithm (default: True)
    seed : int, optional
        Random seed for simulation reproducibility (default: 42)
    
    Returns:
    --------
    tuple
        (spikes, time_series) - Neuronal spikes and simulation time series data
    """

    # Create CSV and STO paths 
    csv_path = base_output_path + '.csv'
    sto_path = base_output_path + '.sto'

    # =============================================================================
    # Initialization
    # =============================================================================

    # Discretization configuration + vector initialization

    time_=np.arange(0, reaction_time, time_step)
    nb_points=len(time_)
    activations = np.zeros((num_muscles, nb_points))
    time_points = np.arange(0, reaction_time*n_iterations, time_step)
    joint_all = np.zeros((len(time_points)))
    activations_all = np.zeros((num_muscles, len(time_points)))

    print('reaction time ', reaction_time)
    print('n iteration ', n_iterations)
    print('nb_points ', nb_points)
    print('nb_total_time ', len(time_points))

    initial_potentials = {
        "exc": biophysical_params['Eleaky'],
        "MN": biophysical_params['Eleaky']
    }
    if num_muscles == 2:
        initial_potentials["inh"] = biophysical_params['Eleaky']

    # Initialize parameters for each motoneuron
    initial_params = [
        [{
            'u0': [0.0, 0.0],    # Initial fiber AP state
            'c0': [0.0, 0.0],    # Initial calcium concentration state
            'P0': 0.0,           # Initial calcium-troponin binding state
            'a0': 0.0            # Initial activation state
        } for _ in range(neurons_population['MN'])]
        for _ in range(num_muscles)]

    # Containers for simulation data
    muscle_data = [[] for _ in range(num_muscles)]
    resting_lengths = [None] * num_muscles

 
    spike_data = {
        muscle_name: {
            neuron_type: defaultdict(list)
            for neuron_type in neurons_population.keys()
        }
        for muscle_name in muscles_names
    }
    torque_array = None
    if torque is not None:
        torque_array = transform_torque_params_in_array(time_points, torque)

    ees_params = None
    if ees_stimulation_params is not None:
        ees_params = transform_intensity_balance_in_recruitment(
          ees_recruitment_profile, ees_stimulation_params, neurons_population, num_muscles)

    # =============================================================================
    # Main Simulation Loop
    # =============================================================================

    print("Start Simulation:")
    if ees_params is not None:
        freq = ees_params['freq']
        if isinstance(freq, tuple):
           print("Phase specific EES modulation")
           print(f"frequency swing phase: {freq[0]}")
           print(f"frequency stance phase: {freq[1]}")
        else:
            print(f"EES frequency: {freq}")

        for fiber_key, fiber_recruitment in ees_params['recruitment'].items():

            print(f"Number {' '.join(fiber_key.split('_'))} recruited by EES: {fiber_recruitment}/{neurons_population[fiber_key]}")
  

    # Create a simulator instance based on execution environment
    on_colab=is_running_on_colab()
    simulator = CoLabSimulator() if on_colab else LocalSimulator()

                       
    for iteration in range(n_iterations):
            print(f"--- Iteration {iteration+1} of {n_iterations} ---")

            # Prepare torque if provided
            current_torque = None
            if torque_array is not None:
                start_idx = iteration * nb_points
                end_idx = (iteration + 1) * nb_points
                current_torque = torque_array[start_idx:end_idx]

            # Run muscle simulation with file paths
            fiber_lengths, joint = simulator.run_muscle_simulation(
                time_step/second,
                reaction_time/second,
                muscles_names,
                associated_joint,
                activation=activations,
                torque=current_torque
                )
                
            # Process muscle simulation results
            fiber_lengths = fiber_lengths[:, :nb_points]
            joint = joint[:nb_points]
            joint_all[iteration*nb_points: (iteration+1)*nb_points] = joint
            
            # Set resting lengths on first iteration
            if resting_lengths[0] is None:
                resting_lengths = fiber_lengths[:, 0]

            # Calculate stretch and velocity for all muscles
            stretch = np.zeros((num_muscles, nb_points))
            stretch_velocity = np.zeros((num_muscles, nb_points))
                
            for muscle_idx in range(num_muscles):
                # Calculate stretch and velocity
                stretch[muscle_idx] = fiber_lengths[muscle_idx] / resting_lengths[muscle_idx] - 1
                stretch_velocity[muscle_idx] = np.gradient(stretch[muscle_idx], time_points[iteration*nb_points:(iteration+1)*nb_points])


            # Run neural simulation based on muscle count
            if num_muscles == 1:
            
                # Determine if we need a II/excitatory pathway simulation
                has_II_pathway = (
                    'II' in spindle_model and 
                    'II' in neurons_population and 
                    'exc' in neurons_population
                )
            
                if has_II_pathway:
                    all_spikes, final_potentials, state_monitors = run_trisynaptic_simulation(
                        stretch, stretch_velocity, neurons_population, connections, 
                        time_step, reaction_time, spindle_model, seed,
                        initial_potentials, **biophysical_params, ees_params=ees_params
                    )
                else:
                    all_spikes, final_potentials, state_monitors = run_monosynaptic_simulation(
                        stretch, stretch_velocity, neurons_population, connections, 
                        time_step, reaction_time, spindle_model, seed,
                        initial_potentials, **biophysical_params, ees_params=ees_params
                    )
            
            else:  # num_muscles == 2
                # Adjust EES frequency based on muscle activation if phase-dependent
                ees_params_copy = None
            
                if ees_params is not None:
                    ees_params_copy = ees_params.copy()
            
                    freq = ees_params.get("freq")
                    if isinstance(freq, tuple) and len(freq) == 2:
                        dominant = 0 if np.mean(activations[0]) >= np.mean(activations[1]) else 1
                        ees_params_copy["freq"] = freq[dominant]
            
                all_spikes, final_potentials, state_monitors = run_flexor_extensor_neuron_simulation(
                    stretch, stretch_velocity, neurons_population, connections, time_step, reaction_time, 
                    spindle_model, seed, initial_potentials, **biophysical_params, ees_params=ees_params_copy
                )

                
            # Update initial potentials for next iteration
            initial_potentials.update(final_potentials)

            # Store spike times for visualization
            for muscle_idx, muscle_name in enumerate(muscles_names):
                muscle_spikes = all_spikes[muscle_idx]
                for fiber_type, fiber_spikes in muscle_spikes.items():
                    for neuron_id, spikes in fiber_spikes.items():
                        # Adjust spike times by iteration offset
                        adjusted_spikes = spikes/second + iteration * reaction_time/second
                        spike_data[muscle_name][fiber_type][neuron_id].extend(adjusted_spikes)

            # Initialize arrays for mean values of all neurons per muscle
            mean_e, mean_u, mean_c, mean_P, mean_activation = [
                np.zeros((num_muscles, int(reaction_time/time_step))) for _ in range(5)
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
                        time_step/second,
                        reaction_time/second,
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
                    
                    # Store activations to relaunch the entire opensim simulation at the end
                    if (activations_all.shape[1]>=(iteration+2)*nb_points):
                        activations_all[muscle_idx,(iteration+1)*nb_points: (iteration+2)*nb_points] = mean_activation[muscle_idx]
                    
                    # Save final state for next iteration
                    initial_params[muscle_idx] = final_values

                    # Create batch data for current iteration
                    batch_data = {
                        'Time': time_points[iteration*nb_points:(iteration+1)*nb_points],
                        f'Fiber_length_{muscles_names[muscle_idx]}': fiber_lengths[muscle_idx],
                        f'Stretch_{muscles_names[muscle_idx]}': stretch[muscle_idx],
                        f'Stretch_Velocity_{muscles_names[muscle_idx]}': stretch_velocity[muscle_idx],
                        f'mean_e_{muscles_names[muscle_idx]}': mean_e[muscle_idx],
                        f'mean_u_{muscles_names[muscle_idx]}': mean_u[muscle_idx],
                        f'mean_c_{muscles_names[muscle_idx]}': mean_c[muscle_idx],
                        f'mean_P_{muscles_names[muscle_idx]}': mean_P[muscle_idx],
                        f'Activation_{muscles_names[muscle_idx]}': mean_activation[muscle_idx],
                    }
                    
                    # Add state monitor data with muscle name as suffix
                    for key, value in state_monitors[muscle_idx].items():
                        batch_data[f'{key}_{muscles_names[muscle_idx]}'] = value

                    # Store batch data for this muscle
                    muscle_data[muscle_idx].append(pd.DataFrame(batch_data))

    # =============================================================================
    # Combine Results and Compute Firing rates
    # =============================================================================

    # Create a single dataframe with all muscle data
    all_data_dfs = []
    
    # Process data for each muscle
    for muscle_idx, muscle_name in enumerate(muscles_names):
          
        # Combine all iteration data for this muscle
        df = pd.concat(muscle_data[muscle_idx], ignore_index=True)

        # Compute firing rate for this muscle
        # Extract stretch and velocity values for this muscle
        stretch_values = df[f'Stretch_{muscle_name}'].values
        stretch_velocity_values = df[f'Stretch_Velocity_{muscle_name}'].values

        time = df['Time'].values
        
        # Compute Ia firing rate using spindle model
        Ia_rate = eval(spindle_model['Ia'], 
                       {"__builtins__": {'sign': np.sign, 'abs': np.abs, 'clip': np.clip}}, 
                       {"stretch": stretch_values, "stretch_velocity": stretch_velocity_values}
                       )
        
        df[f'Ia_rate_baseline_{muscle_name}'] = Ia_rate

        # Compute II firing rate if applicable
        if "II" in neurons_population and "II" in spindle_model:
            II_rate = eval(spindle_model['II'], 
                          {"__builtins__": {}}, 
                          {"stretch": stretch_values, "stretch_velocity": stretch_velocity_values}
                           )

            df[f'II_rate_baseline_{muscle_name}'] = II_rate

        # Calculate all firing rate using 
        for fiber_name, fiber_spikes in spike_data[muscle_name].items():
            if fiber_spikes:
                all_spike_times = np.concatenate(list(fiber_spikes.values())) 
                
                firing_rate = np.zeros_like(time)
                if len(all_spike_times) > 1:
                    kde = gaussian_kde(all_spike_times, bw_method=0.3)
                    firing_rate = kde(time) * len(all_spike_times) / max(len(fiber_spikes), 1)
                df[f'{fiber_name}_rate_measured_{muscle_name}'] = firing_rate
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

    # Add joint and torque if torque is applied
    combined_df[f'Joint_{associated_joint}'] = joint_all
    combined_df[f'Joint_Velocity_{associated_joint}'] = np.gradient(joint_all, time_points)
    if torque_array is not None:
        combined_df['Torque'] = torque_array

    # Save the combined dataframe to CSV
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved combined data to {csv_path}")

    # ====================================================================================================
    # Run Complete Muscle Simulation for visualization of the dynamic on opensim
    # ======================================================================================================
    # Recreate a simulator since we want to start the simulation from the beginning
    simulator = CoLabSimulator() if on_colab else LocalSimulator()
  
    # Run final simulation for visualization
    fiber_lengths, joint=simulator.run_muscle_simulation(
            time_step/second,
            reaction_time/second * n_iterations,
            muscles_names,
            associated_joint,
            activations_all,
            torque_array,
            sto_path
        )
    """
    plt.plot(time_points, joint_all, label='loop simulation')
    plt.plot(time_points, joint[:len(joint_all)], label='final simulation')
    plt.legend()
    plt.show()
    """

    return spike_data, combined_df

class SimulatorBase:
    """Base class for simulation strategies"""
    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation_path, new_state_path, state_file=None, torque_path=None):
        """Run a single muscle simulation iteration"""
        raise NotImplementedError("Subclasses must implement this method")


class CoLabSimulator(SimulatorBase):
    """Simulator implementation for Google Colab using subprocess"""

    def __init__(self):
        # Create reusable temporary files for the whole simulation
        self.input_activation_path = tempfile.mktemp(suffix='.npy')
        self.input_torque_path = tempfile.mktemp(suffix='.npy')
        self.output_stretch_path = tempfile.mktemp(suffix='.npy')
        self.output_joint_path = tempfile.mktemp(suffix='.npy')
        self.state_path = tempfile.mktemp(suffix='.json')
        
    def __del__(self):
        """Clean up temporary files when the object is destroyed"""
        # List of all temporary files to remove
        temp_files = [
            self.input_activation_path,
            self.input_torque_path, 
            self.output_stretch_path,
            self.output_joint_path,
            self.state_path
        ]
        
        # Remove each temporary file if it exists
        for file_path in temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except (IOError, OSError):
                    # Silently ignore errors during cleanup
                    pass
    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation, torque=None, state_file=None):
        """Run a single muscle simulation iteration using conda subprocess"""
        
        # Save activation to temp file
        np.save(self.input_activation_path, activation)
        
        # Build command for OpenSim muscle simulation
        cmd = [
            'conda', 'run', '-n', 'opensim_env', 'python', 'muscle_sim.py',
            '--dt', str(dt),
            '--T', str(T),
            '--muscles_names', ','.join(muscle_names),
            '--joint_name', joint_name,
            '--activation', self.input_activation_path,
            '--output_stretch', self.output_stretch_path,
            '--output_joint', self.output_joint_path,
            '--state', self.state_path
        ]
        
        # Add torque if provided
        if torque is not None:
            np.save(self.input_torque_path, torque)
            cmd += ['--torque', self.input_torque_path]
        
        
        # Run OpenSim simulation
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for errors
        if process.returncode != 0:
            error_msg = f'Error in muscle simulation. STDERR: {process.stderr}'
            raise RuntimeError(error_msg)
        # Display messages:
        #if process.stdout.strip():
            #print("STDOUT:\n", process.stdout)

        # Load simulation results
        fiber_lengths = np.load(self.output_stretch_path)
        joint = np.load(self.output_joint_path)
        

        return fiber_lengths, joint
    
    

class LocalSimulator(SimulatorBase):
    """Simulator implementation for local execution with in-memory state"""
    
    def __init__(self):
        self.current_state = {}  # Store simulation state in memory
    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation, torque):
        """Run a single muscle simulation iteration using direct function call with in-memory state"""
        from muscle_sim import run_simulation
        # Run simulation directly with in-memory state
        fiber_lengths, joint, new_state = run_simulation(
            dt, T, muscle_names, joint_name, activation,
            self.current_state
        )
        # Update in-memory state
        self.current_state = new_state
        
        
        return fiber_lengths, joint
    
 
def is_running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
