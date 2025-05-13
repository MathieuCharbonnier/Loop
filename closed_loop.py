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


def closed_loop(NUM_ITERATIONS, REACTION_TIME, TIME_STEP, EES_PARAMS, NEURON_COUNTS, CONNECTIONS,
           SPINDLE_MODEL, BIOPHYSICAL_PARAMS, MUSCLE_NAMES_STR, associated_joint, sto_path, 
           torque, seed=42):
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
  Initial_perturbation : dict, optional
      Parameters for initial dorsiflexion:
      - 'initial_stretch': float, initial stretch value to apply to muscles
      - 'amplitude': float, amplitude of perturbation
      - 'target_muscles': list, indices of muscles to apply perturbation to
  """
  # Muscle configuration
  MUSCLE_NAMES = MUSCLE_NAMES_STR.split(",")
  NUM_MUSCLES = len(MUSCLE_NAMES)

  # Validate muscle count
  if NUM_MUSCLES > 2:
      raise ValueError("This pipeline supports only 1 or 2 muscles!")
    
  # For symmetric afferent recruitment 
  if 'aff_recruited' in EES_PARAMS:
    value = EES_PARAMS.pop('aff_recruited')
    EES_PARAMS['Ia_recruited'] = value
    EES_PARAMS['II_recruited'] = value


  # =============================================================================
  # Initialization
  # =============================================================================

  # Initialize muscle activation
  activations = np.zeros((NUM_MUSCLES, int(REACTION_TIME/TIME_STEP)))
  time_points = np.arange(0, REACTION_TIME/second, TIME_STEP/second)

  initial_potentials = {
      "exc": BIOPHYSICAL_PARAMS['Eleaky'],
      "moto": BIOPHYSICAL_PARAMS['Eleaky']
  }
  if NUM_MUSCLES == 2:
      initial_potentials["inh"] = BIOPHYSICAL_PARAMS['Eleaky']

  neuron_types = NEURON_COUNTS.keys()

  # Initialize parameters for each motoneuron
  initial_params = [
      [{
              'u0': [0.0, 0.0],    # Initial fiber AP state
              'c0': [0.0, 0.0],    # Initial calcium concentration state
              'P0': 0.0,           # Initial calcium-troponin binding state
              'a0': 0.0            # Initial activation state
          }for _ in range(NEURON_COUNTS['motor'])]
      for _ in range(NUM_MUSCLES)]

  # Containers for simulation data
  muscle_data = [[] for _ in range(NUM_MUSCLES)]
  resting_lengths = [None] * NUM_MUSCLES

  spike_data = {
      muscle_name: {
          neuron_type: defaultdict(list)
          for neuron_type in neuron_types
      }
      for muscle_name in MUSCLE_NAMES
  }

  # Use temporary file for state management across iterations
  state_file = None
  
       
  # =============================================================================
  # Main Simulation Loop
  # =============================================================================

  print("Start Simulation :")
  print("EES frequency : " + str(EES_PARAMS['ees_freq']))
  print("Number Ia fibers recruited by EES: " + str(EES_PARAMS['Ia_recruited'])+ " / "+str(NEURON_COUNTS['Ia']))
  if "II" in NEURON_COUNTS and "II" in SPINDLE_MODEL:
      print("Number II fibers recruited by EES : " + str(EES_PARAMS['II_recruited'])+ " / "+str(NEURON_COUNTS['II']))
  print("Number Efferent fibers recruited by EES : " + str(EES_PARAMS['eff_recruited'])+" / "+str(NEURON_COUNTS['motor']))
  

  for iteration in range(NUM_ITERATIONS):
      print(f"--- Iteration {iteration+1} of {NUM_ITERATIONS} ---")


      # Run OpenSim muscle simulation using the computed activations
      with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_activation, \
          tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_torque, \
          tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as output_stretch, \
          tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as output_joint, \
          tempfile.NamedTemporaryFile(suffix='.json', delete=False) as new_state_file:

          input_activation_path = input_activation.name
          input_torque_path = input_torque.name
          output_stretch_path = output_stretch.name
          output_joint_path=output_joint.name
          new_state_path = new_state_file.name

          # Save activations and torque to temporary file
          np.save(input_activation, activations)
          np.save(input_torque, torque)

          # Build command for OpenSim muscle simulation
          cmd = [
              'conda', 'run', '-n', 'opensim_env', 'python', 'muscle_sim.py',
              '--dt', str(TIME_STEP/second),
              '--T', str(REACTION_TIME/second),
              '--muscles_names', MUSCLE_NAMES_STR,
              '--joint_name', associated_joint,
              '--activation', input_activation_path,
              '--torque', input_torque_path
              '--output_stretch', output_stretch_path,
              '--output_joint', output_joint_path
              '--output_final_state', new_state_path
          ]

          # Add initial state parameter if not the first iteration
          if state_file is not None:
              cmd += ['--initial_state', state_file]

          # Run OpenSim simulation
          process = subprocess.run(cmd, capture_output=True, text=True)

          # Process OpenSim results
          if process.returncode == 0 and os.path.getsize(output_path) > 0:
              # Load muscle lengths from simulation
              fiber_lengths = np.load(output_path)
              # Remove the last value as it will be included in the next iteration
              fiber_lengths = fiber_lengths[:, :-1]

              # Process each muscle's data
              for muscle_idx in range(NUM_MUSCLES):
                  # Set resting length on first iteration if not already set
                  if resting_lengths[muscle_idx] is None:
                      resting_lengths[muscle_idx] = fiber_lengths[muscle_idx, 0]

                  # Calculate stretch and velocity for next iteration
                  stretch[muscle_idx] = fiber_lengths[muscle_idx] / resting_lengths[muscle_idx] - 1
                  velocity[muscle_idx] = np.gradient(stretch[muscle_idx], time_points)

      # Run neural simulation based on muscle count
      if NUM_MUSCLES == 1:
          all_spikes, final_potentials, state_monitors = run_one_muscle_neuron_simulation(
              stretch, velocity, NEURON_COUNTS,CONNECTIONS, TIME_STEP, REACTION_TIME,SPINDLE_MODEL,seed,
              initial_potentials, **EES_PARAMS, **BIOPHYSICAL_PARAMS
          )
      else:  # NUM_MUSCLES == 2
          all_spikes,  final_potentials, state_monitors = run_flexor_extensor_neuron_simulation(
              stretch, velocity, NEURON_COUNTS, CONNECTIONS, TIME_STEP, REACTION_TIME, SPINDLE_MODEL,seed,
              initial_potentials,**EES_PARAMS, **BIOPHYSICAL_PARAMS
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
          if len(muscle_spikes["MN"]) > 0:
              # Convert spike times to seconds
              mn_spikes_sec = [value/second for _, value in muscle_spikes["MN"].items()]
              # Decode spikes to muscle activations
              e, u, c, P, activations, final_values = decode_spikes_to_activation(
                  mn_spikes_sec,
                  TIME_STEP/second,
                  REACTION_TIME/second,
                  initial_params[muscle_idx]
              )

              # Store mean values across all neurons
              mean_e[muscle_idx] = np.mean(e, axis=0)
              mean_u[muscle_idx] = np.mean(u, axis=0)
              mean_c[muscle_idx] = np.mean(c, axis=0)
              mean_P[muscle_idx] = np.mean(P, axis=0)
              mean_activation[muscle_idx] = np.mean(activations, axis=0)

              # Save final state for next iteration
              initial_params[muscle_idx] = final_values


                  # Create batch data for current iteration
                  batch_data = {
                      **state_monitors[muscle_idx],
                      'mean_e': mean_e[muscle_idx],
                      'mean_u': mean_u[muscle_idx],
                      'mean_c': mean_c[muscle_idx],
                      'mean_P': mean_P[muscle_idx],
                      'Activation': mean_activation[muscle_idx],
                      'Fiber_length': fiber_lengths[muscle_idx],
                      'Stretch': stretch[muscle_idx],
                      'Velocity': velocity[muscle_idx]
                  }

                  # Store batch data for this muscle
                  muscle_data[muscle_idx].append(pd.DataFrame(batch_data))
          else:
              error_msg = f'Error in iteration {iteration+1}. STDERR: {process.stderr}'
              raise RuntimeError(error_msg)

          # Clean up the old state file if it exists (don't delete the initial file in the first iteration, need it to do the entire simulation opensim at the end)
          if iteration>0 and state_file is not None and state_file != new_state_path:
              os.unlink(state_file)

          # Update state file for next iteration
          state_file = new_state_path

          # Clean up other temporary files
          os.unlink(input_path)
          os.unlink(output_path)

  # =============================================================================
  # Combine Results and Compute Firing rates
  # =============================================================================

  muscle_dataframes = []

  # Process and save data for each muscle
  for muscle_idx, muscle_name in enumerate(MUSCLE_NAMES):

      # Combine all iteration data for this muscle
      combined_df = pd.concat(muscle_data[muscle_idx], ignore_index=True)

      # Add time column as first column
      time = np.arange(len(combined_df)) * (TIME_STEP/second)
      combined_df['Time'] = time
      combined_df = combined_df[['Time'] + [col for col in combined_df.columns if col != 'Time']]

      # Compute all firing rate:
      # First calculate initial stretch
      stretch_init = np.append(stretch0[muscle_idx], combined_df['Stretch'].values)
      stretch_init = stretch_init[:len(time)]
      velocity_init = np.gradient(stretch_init, time)
      Ia_rate = eval(SPINDLE_MODEL['Ia'], {"__builtins__": {'sign': np.sign, 'abs': np.abs}}, {
        "stretch": stretch_init,
        "velocity": velocity_init
      })
      Ia_rate += EES_PARAMS['ees_freq']/hertz * EES_PARAMS['Ia_recruited']/NEURON_COUNTS['Ia']
      combined_df['Ia_rate'] = 1/((1/Ia_rate)+BIOPHYSICAL_PARAMS['T_refr']/second)

      if "II" in NEURON_COUNTS and "II" in SPINDLE_MODEL:
          II_rate = eval(SPINDLE_MODEL['II'], {"__builtins__": {}}, {
            "stretch": stretch_init,
            "velocity": velocity_init
          })
          II_rate += EES_PARAMS['ees_freq']/hertz * EES_PARAMS['II_recruited']/NEURON_COUNTS['II']
          combined_df['II_rate'] = 1/((1/II_rate)+BIOPHYSICAL_PARAMS['T_refr']/second)

      all_spike_times = np.concatenate(list(spike_data[muscle_name]['MN'].values())) if spike_data[muscle_name]['MN'] else []
      firing_rate = np.zeros_like(time)
      if len(all_spike_times) > 1:
          kde = gaussian_kde(all_spike_times, bw_method=0.3)
          firing_rate = kde(time) * len(all_spike_times) / len(spike_data[muscle_name]['MN'])
      combined_df['MN_rate'] = firing_rate
  
      # Store dataframe for plotting
      muscle_dataframes.append(combined_df)

  # ====================================================================================================
  # Run Complete Muscle Simulation for:
  # - visualization of the the dynamic on opensim
  # - joint analysis
  # ======================================================================================================

  # Create full simulation STO file for visualization
  # Create temporary file for activation data
  with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as input_file:
      input_path = input_file.name

      # Save muscle activations without time column
      activations_array = np.zeros((NUM_MUSCLES, len((muscle_dataframes[0]))))
      for i in range(NUM_MUSCLES):
          activations_array[i] = muscle_dataframes[i]['Activation'].T.to_numpy()

      np.save(input_path, activations_array)

      # Build command for full muscle simulation
      cmd = [
          'conda', 'run', '-n', 'opensim_env', 'python', 'muscle_sim.py',
          '--dt', str(TIME_STEP/second),
          '--T', str(REACTION_TIME/second * NUM_ITERATIONS),
          '--muscles', MUSCLE_NAMES_STR,
          '--activation', input_path,
          '--output_all', sto_path
      ]
      if torque is not None:
        cmd += ['--initial_state', initial_state_path]


      # Run OpenSim simulation for complete trajectory
      process = subprocess.run(cmd, capture_output=True, text=True)

      if process.stdout.strip():
          print("STDOUT:\n", process.stdout)
      if process.stderr.strip():
          print(f"STDERR: {process.stderr}")

      # Clean up temporary files
      os.unlink(input_path)
      os.unlink(initial_state_path)

  return spike_data, muscle_dataframes
