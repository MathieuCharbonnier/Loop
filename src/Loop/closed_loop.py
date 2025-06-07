from brian2 import *
import numpy as np
import pandas as pd
import os
import time
import subprocess
import tempfile
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from .neural_dynamics import run_monosynaptic_simulation, run_disynaptic_simulation, run_disynaptic_simulation_with_ib, run_flexor_extensor_neuron_simulation, run_spinal_circuit_with_Ib
from .activation import decode_spikes_to_activation
from ..helpers.copy_brian_dict import copy_brian_dict


def closed_loop(n_iterations, reaction_time, time_step, neurons_population, connections,
              spindle_model, biophysical_params, muscles_names, num_muscles, resting_lengths, associated_joint, fast,
              initial_state_neurons, initial_condition_spike_activation, initial_state_opensim, 
                activation_function=None, stretch_history_function=None,
              ees_params=None, torque_array=None, seed=42, base_output_path=None):
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
    resting_lengths : list
        Resting lengths for each muscle
    associated_joint : str
        Name of the associated joint
    ees_params : dict, optional
        Parameters for electrical epidural stimulation
    torque_array: numpy.array, optional
         External torque profile applied at each time step
    fast : bool, optional
        Whether to use the fast spike-to-activation decoding algorithm (default: True)
    seed : int, optional
        Random seed for simulation reproducibility (default: 42)
    base_output_path : str
        Base path for output files
    
    Returns:
    --------
    tuple
        (spikes, time_series, final_state) - Neuronal spikes, simulation time series data, and final state
    """
    print(" seed : ", seed)
    # =============================================================================
    # Initialization
    # =============================================================================

    # Discretization configuration + vector initialization
    nb_points = int(reaction_time / time_step)
    time_ = np.linspace(0, reaction_time, nb_points)

    total_points = n_iterations * nb_points
    total_time=reaction_time * n_iterations
    time_points = np.linspace(0, total_time, total_points)
    joint_all = np.zeros(total_points)
    activations_all = np.zeros((num_muscles, total_points))

    activation_history = np.zeros((num_muscles, nb_points))
    if activation_function is not None:
        try:
            old_activation = activation_function(time_)
            if old_activation.shape[0] == num_muscles :
                activation_history = old_activation
            else:
                print(f"Warning: stretch_history shape mismatch. Expected ({num_muscles}, {nb_points}), got {old_activation.shape}")
        except Exception as e:
            print(f"Warning: Could not load activation history: {e}")
        

    delay = spindle_model.get("Ia_II_delta_delay", 0)  
    delay_points = int(delay / time_step)
    stretch_global_buffer = np.zeros((num_muscles, delay_points + total_points))

    # Initialize delay buffer with historical data if available
    if delay_points > 0:
      
        if stretch_history_function is not None:
            time_delay = np.linspace(-delay, 0, delay_points)
            try:
                old_stretch = stretch_history_function(time_delay)
                if old_stretch.shape[0] == num_muscles :
                    stretch_global_buffer[:, :delay_points] = old_stretch
                else:
                    print(f"Warning: stretch_history shape mismatch. Expected ({num_muscles}, {delay_points}), got {old_stretch.shape}")
            except Exception as e:
                print(f"Warning: Could not load stretch history: {e}")

    
    # Containers for simulation data
    muscle_data = [[] for _ in range(num_muscles)]

    neuron_types = list(dict.fromkeys(k.split('_')[0] for k in neurons_population.keys()))
    spike_data = {
        muscle_name: {
            neuron_type: defaultdict(list)
            for neuron_type in neuron_types
        }
        for muscle_name in muscles_names
    }
    recruitment=np.zeros(num_muscles)
    
    # =============================================================================
    # Main Simulation Loop
    # =============================================================================

    print("Start Simulation:")
    if ees_params is not None:
        freq = ees_params['frequency']
        if isinstance(freq, tuple):
           print("Phase specific EES modulation")
           print(f"frequency dorsiflexion phase: {freq[0]}")
           print(f"frequency plantarflexion phase: {freq[1]}")
        elif isinstance(freq, list):
            if len(freq) != n_iterations:
                raise ValueError(f"The length of the frequency array should match the number of iterations: n_iterations={n_iterations}, but len(freq)={len(freq)}")
            print('Temporal EES modulation')
        else:
            print(f"EES frequency: {freq}")

        for fiber_key, fiber_recruitment in ees_params['recruitment'].items():
            print(f"Number {' '.join(fiber_key.split('_'))} recruited by EES: {fiber_recruitment}/{neurons_population[fiber_key]}")

    # Create a simulator instance based on execution environment
    on_colab = is_running_on_colab()
    simulator = CoLabSimulator(initial_state_opensim) if on_colab else LocalSimulator(initial_state_opensim)

    for iteration in range(n_iterations):
        print(f"--- Iteration {iteration+1} of {n_iterations} ---")
        #start_opensim = time.time()
        
        # Prepare torque if provided
        current_torque = None
        if torque_array is not None:
            start_idx = iteration * nb_points
            end_idx = (iteration + 1) * nb_points
            current_torque = torque_array[start_idx:end_idx]

        # Run muscle simulation
        fiber_lengths, normalized_force, joint = simulator.run_muscle_simulation(
            time_step / second,
            reaction_time / second,
            muscles_names,
            associated_joint,
            activation=activation_history,
            torque=current_torque
        )
            
        # Process muscle simulation results
        fiber_lengths = fiber_lengths[:, :nb_points]
        normalized_force = normalized_force[:, :nb_points]
        joint = joint[:nb_points]
        joint_all[iteration * nb_points: (iteration + 1) * nb_points] = joint

        # Calculate stretch and velocity for all muscles
        stretch = np.zeros((num_muscles, nb_points))
        stretch_velocity = np.zeros((num_muscles, nb_points))
            
        for muscle_idx in range(num_muscles):
            # Calculate stretch
            stretch[muscle_idx] = fiber_lengths[muscle_idx] / resting_lengths[muscle_idx] - 1
            # Calculate velocity using time step
            stretch_velocity[muscle_idx] = np.gradient(stretch[muscle_idx], time_step / second)
        
        buffer_start = delay_points + iteration * nb_points
        buffer_end = delay_points + (iteration + 1) * nb_points
        stretch_global_buffer[:, buffer_start:buffer_end] = stretch
        
        # Extract delayed stretch for neural simulation 
        delayed_start = iteration * nb_points
        delayed_end = (iteration + 1) * nb_points
        stretch_II = stretch_global_buffer[:, delayed_start:delayed_end]
        
        #end_opensim = time.time()
        #start_neuron = time.time()
      
        # Adjust EES frequency based on muscle spiking if phase-dependent or if ees frequency is time dependent
        ees_params_copy = None
        if ees_params is not None:
            ees_params_copy = copy_brian_dict(ees_params)
            freq = ees_params.get("frequency")
            
            if isinstance(freq, tuple) and len(freq) == 2:
                if np.isclose(recruitment[0], recruitment[1], atol=1e-2):  
                    ees_params_copy["frequency"] = (freq[0] + freq[1]) / 2
                else:
                    dominant = 0 if recruitment[0] >= recruitment[1] else 1
                    ees_params_copy["frequency"] = freq[dominant]
        
            elif isinstance(freq, list): 
                ees_params_copy["frequency"] = freq[iteration]
        
            print('ees_freq', ees_params_copy['frequency'])
 
        # Run neural simulation based on muscle count
        if num_muscles == 1:
              
            # Determine if we need a II/excitatory pathway simulation
            has_II_pathway = (
                'II' in spindle_model and 
                'II' in neurons_population and 
                'exc' in neurons_population
            )
            has_Ib_pathway=(
                'Ib' in spindle_model and 
                'Ib' in neurons_population and 
                'inhb' in neurons_population
            )
            if has_II_pathway:
              
                if has_Ib_pathway:
                     all_spikes, final_state_neurons, state_monitors = run_disynaptic_simulation_with_ib(
                        stretch, stretch_velocity, stretch_II, normalized_force, neurons_population, connections, 
                        time_step, reaction_time, spindle_model, seed,
                        initial_state_neurons, **biophysical_params, ees_params=ees_params_copy)
                else:
                    all_spikes, final_state_neurons, state_monitors = run_disynaptic_simulation(
                        stretch, stretch_velocity, stretch_II, neurons_population, connections, 
                        time_step, reaction_time, spindle_model, seed,
                        initial_state_neurons, **biophysical_params, ees_params=ees_params_copy)
                
            else:
                all_spikes, final_state_neurons, state_monitors = run_monosynaptic_simulation(
                    stretch, stretch_velocity, neurons_population, connections, 
                    time_step, reaction_time, spindle_model, seed,
                    initial_state_neurons, **biophysical_params, ees_params=ees_params_copy
                )
        
        else:  # num_muscles == 2

            if "Ib_flexor" in neurons_population and "Ib_extensor" in neurons_population:
                all_spikes, final_state_neurons, state_monitors = run_spinal_circuit_with_Ib(
                    stretch, stretch_velocity, stretch_II, normalized_force, neurons_population, connections, time_step, reaction_time, 
                    spindle_model, seed, initial_state_neurons, **biophysical_params, ees_params=ees_params_copy
                )
            else:
                all_spikes, final_state_neurons, state_monitors = run_flexor_extensor_neuron_simulation(
                    stretch, stretch_velocity, stretch_II, neurons_population, connections, time_step, reaction_time, 
                    spindle_model, seed, initial_state_neurons, **biophysical_params, ees_params=ees_params_copy
                )

        # Update initial potentials for next iteration
        initial_state_neurons.update(final_state_neurons)

        for muscle_idx, muscle_name in enumerate(muscles_names):
            muscle_spikes = all_spikes[muscle_idx]
            recruited_MN = sum(1 for spikes in muscle_spikes['MN'].values() if len(spikes) > 0)
            print(f"Number of recruited {muscle_name} motoneuron: {recruited_MN}/{len(muscle_spikes['MN'])}") 
            recruitment[muscle_idx]=recruited_MN

        # Store spike times for visualization
        for muscle_idx, muscle_name in enumerate(muscles_names):
            muscle_spikes = all_spikes[muscle_idx]
            for fiber_type, fiber_spikes in muscle_spikes.items():
                for neuron_id, spikes in fiber_spikes.items():
                    # Adjust spike times by iteration offset
                    adjusted_spikes = spikes / second + iteration * reaction_time / second
                    spike_data[muscle_name][fiber_type][neuron_id].extend(adjusted_spikes)
        
        #end_neuron = time.time()
        #start_activation = time.time()
        
        # Initialize arrays for mean values of all neurons per muscle
        mean_e, mean_u, mean_c, mean_P, mean_activation = [
            np.zeros((num_muscles, nb_points)) for _ in range(5)
        ]

        # Process motor neuron spikes to get muscle activations
        for muscle_idx, muscle_spikes in enumerate(all_spikes):
            # Only process if we have motor neuron spikes
            if "MN" in muscle_spikes and len(muscle_spikes["MN"]) > 0:
                # Convert spike times to seconds
                mn_spikes_sec = [value / second for _, value in muscle_spikes["MN"].items()]
                # Decode spikes to muscle activations
                e, u, c, P, activations_result, final_values = decode_spikes_to_activation(
                    mn_spikes_sec,
                    time_step / second,
                    reaction_time / second,
                    initial_condition_spike_activation[muscle_idx],
                    fast=fast
                )

                # Store mean values across all neurons
                mean_e[muscle_idx] = np.mean(e, axis=0)
                mean_u[muscle_idx] = np.mean(u, axis=0)
                mean_c[muscle_idx] = np.mean(c, axis=0)
                mean_P[muscle_idx] = np.mean(P, axis=0)
                mean_activation[muscle_idx] = np.clip(np.mean(activations_result, axis=0), 0, 1)

                # Update activation for next iteration
                activation_history[muscle_idx] = mean_activation[muscle_idx]
                
                # Store activations for final simulation
                if activations_all.shape[1] >= (iteration + 2) * nb_points:
                    activations_all[muscle_idx, (iteration + 1) * nb_points: (iteration + 2) * nb_points] = mean_activation[muscle_idx]
                
                # Save final state for next iteration
                initial_condition_spike_activation[muscle_idx] = final_values
                
                #end_activation = time.time()
              
                
                # Create batch data for current iteration
                batch_data = {
                    'Time': time_points[iteration * nb_points:(iteration + 1) * nb_points],
                    f'Fiber_length_{muscles_names[muscle_idx]}': fiber_lengths[muscle_idx],
                    f'Stretch_{muscles_names[muscle_idx]}': stretch[muscle_idx],
                    f'Stretch_Velocity_{muscles_names[muscle_idx]}': stretch_velocity[muscle_idx],
                    f'Force_{muscles_names[muscle_idx]}': normalized_force[muscle_idx],
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
                

        #print(f"Opensim time: {end_opensim - start_opensim:.2f} s")
        #print(f"Neuron time: {end_neuron - start_neuron:.2f} s")
        #print(f"Activation time: {end_activation - start_activation:.2f} s")
  

    # =============================================================================
    # Prepare Final State 
    # =============================================================================
    
    # Create interpolation function for stretch history (last delay_points of buffer)
    stretch_history_interp = None
    if delay_points > 0:
        # Extract the last delay_points from the buffer (most recent history)
        recent_stretch_history = stretch_global_buffer[:, total_points:total_points + delay_points]
        time_history = np.linspace(-delay, 0, delay_points)
        
        stretch_history_interp = interp1d(
            time_history, recent_stretch_history, 
            axis=1, kind='linear', 
            bounds_error=False, fill_value='extrapolate'
        )

    final_state = {
        "opensim": simulator.recover_final_state(),
        "neurons": final_state_neurons,
        "spikes_activations": initial_condition_spike_activation,
        "last_activations": interp1d(time_, activation_history, axis=1, kind='linear', bounds_error=False, fill_value='extrapolate')
    }
    
    if stretch_history_interp is not None:
        final_state["stretch_history"] = stretch_history_interp

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
        time_values = df['Time'].values
        # Extract stretch and velocity values for this muscle
        stretch_values = df[f'Stretch_{muscle_name}'].values
        stretch_velocity_values = df[f'Stretch_Velocity_{muscle_name}'].values
        force_normalized_values=df[f'Force_{muscle_name}'].values
        stretch_delay_values = stretch_global_buffer[muscle_idx, :len(time_values)]
           
        # Compute Ia firing rate using spindle model
        Ia_rate = eval(spindle_model['Ia'], 
                       {"__builtins__": {'sign': np.sign, 'abs': np.abs, 'clip': np.clip}}, 
                       {"stretch": stretch_values, "stretch_velocity": stretch_velocity_values}
                       )
        
        df[f'Ia_rate_baseline_{muscle_name}'] = Ia_rate

        # Compute II firing rate if applicable
        if "II" in spindle_model:
            II_rate = eval(spindle_model['II'], 
                          {"__builtins__": {}}, 
                          {"stretch": stretch_values,
                          "stretch_delay": stretch_delay_values }
                           )
         
            df[f'II_rate_baseline_{muscle_name}'] = II_rate

        # Compute Ib firing rate if applicable
        if "Ib" in neurons_population and "Ib" in spindle_model:
            Ib_rate = eval(spindle_model['Ib'], 
                          {"__builtins__": {}}, 
                          {"force_normalized": force_normalized_values}
                           )

            df[f'Ib_rate_baseline_{muscle_name}'] = Ib_rate

        # Calculate all firing rate using KDE
        try:
            for fiber_name, fiber_spikes in spike_data[muscle_name].items():
                if fiber_spikes:
                    all_spike_times = np.concatenate(list(fiber_spikes.values()))
                    
                    firing_rate = np.zeros_like(time)
                    if len(all_spike_times) > 1:
                        kde = gaussian_kde(all_spike_times, bw_method=0.3)
                        firing_rate = kde(time_values) * len(all_spike_times) / max(len(fiber_spikes), 1)
                        
                    df[f'{fiber_name}_rate_{muscle_name}'] = firing_rate
        except Exception as e:
            print(f"Error while calculating firing rate, {muscle_name}: {e}")

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
    
    if base_output_path is not None:
 
        csv_path = base_output_path + '.csv'
  
        # Save the combined dataframe to CSV
        combined_df.to_csv(csv_path, index=False)
        print(f"Saved combined data to {csv_path}")

        get_sto_file(time_step/second,reaction_time/second*n_iterations,
          muscles_names, associated_joint, activations_all, torque_array, base_output_path )

    return spike_data, combined_df, final_state

  
def get_sto_file(time_step, total_time, muscles_names, associated_joint,activations_all, torque_array, base_output_path):
    # ====================================================================================================
    # Run Complete Muscle Simulation for visualization of the dynamic on opensim
    # ======================================================================================================
  
    if base_output_path is not None:

        sto_path = base_output_path + '.sto'
        # Recreate a simulator since we want to start the simulation from the beginning
        on_colab = is_running_on_colab()
        simulator = CoLabSimulator() if on_colab else LocalSimulator()
      
       # Run final simulation for visualization
        fiber_lengths, force_normalized, joint=simulator.run_muscle_simulation(
                time_step,
                total_time,
                muscles_names,
                associated_joint,
                activations_all,
                torque_array,
                sto_path=sto_path
            )

    

class SimulatorBase:
    """Base class for simulation strategies"""
    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation_path, new_state_path, state_file=None, torque_path=None):
        """Run a single muscle simulation iteration"""
        raise NotImplementedError("Subclasses must implement this method")


class CoLabSimulator(SimulatorBase):
    """Simulator implementation for Google Colab using subprocess"""

    def __init__(self, initial_state=None):
        # Create reusable temporary files for the whole simulation
        self.input_activation_path = tempfile.mktemp(suffix='.npy')
        self.input_torque_path = tempfile.mktemp(suffix='.npy')
        self.output_stretch_path = tempfile.mktemp(suffix='.npy')
        self.output_force_path = tempfile.mktemp(suffix='.npy')
        self.output_joint_path = tempfile.mktemp(suffix='.npy')
        self.state_path = tempfile.mktemp(suffix='.json')

        if initial_state is not None:
              with open(self.state_path, "w") as f:
                  json.dump(initial_state, f, indent=4)

    def recover_final_state(self):
        with open(self.state_path, 'r') as f:
                state = json.load(f)
        return state  

    def __del__(self):
        """Clean up temporary files when the object is destroyed"""
        # List of all temporary files to remove
        temp_files = [
            self.input_activation_path,
            self.input_torque_path, 
            self.output_stretch_path,
            self.output_force_path,
            self.output_joint_path,
            self.state_path
        ]
        
        # Remove each temporary file if it exists
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation, torque=None, initial_state=None, sto_path=None):
        """Run a single muscle simulation iteration using conda subprocess"""
        
        # Save activation to temp file
        np.save(self.input_activation_path, activation)

        
        # Build command for OpenSim muscle simulation
        cmd = [
            'conda', 'run', '-n', 'opensim_env', 'python', 'src/Loop/muscle_sim.py',
            '--dt', str(dt),
            '--T', str(T),
            '--muscles_names', ','.join(muscle_names),
            '--joint_name', joint_name,
            '--activation', self.input_activation_path,
            '--output_stretch', self.output_stretch_path,
            '--output_force', self.output_force_path,
            '--output_joint', self.output_joint_path,
            '--state', self.state_path
        ]
        
        # Add torque if provided
        if torque is not None:
            np.save(self.input_torque_path, torque)
            cmd += ['--torque', self.input_torque_path]
        if sto_path is not None:
            cmd += ['--output_all', sto_path]
        
        # Run OpenSim simulation
        process = subprocess.run(cmd, capture_output=True, text=True)
        # Display messages:
        #if process.stdout.strip():
            #print("STDOUT:\n", process.stdout)
        # Check for errors
        if process.returncode != 0:
            error_msg = f'Error in muscle simulation. STDERR: {process.stderr}'
            raise RuntimeError(error_msg)


        # Load simulation results
        fiber_lengths = np.load(self.output_stretch_path)
        normalized_force=np.load(self.output_force_path)
        joint = np.load(self.output_joint_path)
        

        return fiber_lengths,normalized_force, joint
    
    

class LocalSimulator(SimulatorBase):
    """Simulator implementation for local execution with in-memory state"""
    
    def __init__(self, current_state={}):
        self.current_state = current_state 

    def recover_final_state(self):
        return self.current_state   
    
    def run_muscle_simulation(self, dt, T, muscle_names, joint_name, 
                              activation, torque, sto_path=None):
        """Run a single muscle simulation iteration using direct function call with in-memory state"""
        from .muscle_sim import run_simulation
        # Run simulation directly with in-memory state
        fiber_lengths,normalized_force, joint, new_state = run_simulation(
            dt, T, muscle_names, joint_name, activation,
            state_storage=self.current_state, output_all=sto_path
        )
        # Update in-memory state
        self.current_state = new_state
        
        
        return fiber_lengths, normalized_force, joint
    
 
def is_running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

