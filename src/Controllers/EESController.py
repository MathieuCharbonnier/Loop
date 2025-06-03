from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import os
from scipy.interpolate import interp1d
from ..BiologicalSystems.BiologicalSystem import BiologicalSystem

class EESController:
    """
    Model Predictive Controller for EES parameters to achieve desired joint trajectories.
    
    This controller uses a model-based approach.
    It optimizes EES parameters by increasing/decreasing EES frequency if the absolute amplitude of the movement is too weak/strong, 
    """
    
    def __init__(self, biological_system, ees_intensity=0.5, ees_frequency_guess=50*hertz):
        """
        Initialize the EES controller.
        
        Parameters:
        -----------
        biological_system : BiologicalSystem
            The biological system to control
        """
        self.biological_system = biological_system.clone_with()
        self.ees_intensity = ees_intensity
        self.ees_frequency_guess = ees_frequency_guess
        
        # Load trajectory data
        mot_file = 'data/BothLegsWalk.mot'

        # Use tab separator instead of space separator
        df = pd.read_csv(mot_file, sep='\t', skiprows=6)
        # Clean column names (remove extra whitespace)
        df.columns = df.columns.str.strip()

        def move_side_to_suffix(col):
            if col.startswith('r_'):
                return col[2:] + '_r'
            elif col.startswith('l_'):
                return col[2:] + '_l'
            else:
                return col

        df.columns = [move_side_to_suffix(col) for col in df.columns]

        # Extract time and joint data
        self.time = df['time'].values
        self.desired_trajectory = df[self.biological_system.associated_joint].values
        self.total_time = self.time[-1]
        self.desired_trajectory_function = interp1d(self.time, self.desired_trajectory, kind='cubic', fill_value="extrapolate")
        
        # Define gait events: max dorsiflexion and max plantarflexion
        max_dorsi_idx = np.argmax(self.desired_trajectory)
        min_plantar_idx = np.argmin(self.desired_trajectory)
        
        # Create event times for the three phases
        self.event_times = np.array([0, self.time[max_dorsi_idx], self.time[min_plantar_idx], self.total_time])
        self.site_stimulation = ['L4', 'S1', 'L4']  # dorsiflexion, plantarflexion, recovery
        
        
        # Frequency search parameters
        self.freq_step = 5 * hertz  # Step size for frequency adjustment
        self.min_frequency = 10 * hertz
        self.max_frequency = 100 * hertz
        self.amplitude_tolerance = 0.1  # Tolerance for amplitude matching
        self.max_iterations = 10
        
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        self.optimization_trajectories = []  # Now stores list of phase optimization data

    
    def _get_trajectory_amplitude(self, trajectory):
        """
        Calculate the amplitude (peak-to-peak) of a trajectory.
        
        Parameters:
        -----------
        trajectory : np.ndarray
            Joint trajectory
            
        Returns:
        --------
        float
            Amplitude of the trajectory
        """
        return np.max(trajectory) - np.min(trajectory)
  
    
    def run(self, time_step=0.1*ms):
        """
        Run the complete control simulation.
        
        Parameters:
        -----------
        time_step : float
            Time step for simulation (in seconds)
            
        Returns:
        --------
        tuple
            (trajectory_history, desired_trajectory_history, ees_params_history, time_history)
        """
        current_time = 0.0
        phase_idx = 0
        
        print(f"Starting EES control simulation...")
        print(f"Event times: {self.event_times}")
        print(f"Stimulation sites: {self.site_stimulation}")
        
        while current_time < self.total_time and phase_idx < len(self.site_stimulation):
            # Determine next phase transition time
            next_event_time = self.event_times[phase_idx + 1] if phase_idx + 1 < len(self.event_times) else self.total_time
            phase_duration = next_event_time - current_time
            
            # Get stimulation site for current phase
            site = self.site_stimulation[phase_idx]
            
            # Calculate number of iterations for this phase
            update_iterations = max(1, int(phase_duration / (self.biological_system.reaction_time / second)))
            
            print(f"\nPhase {phase_idx + 1}: {current_time:.3f}s to {next_event_time:.3f}s")
            print(f"Duration: {phase_duration:.3f}s, Iterations: {update_iterations}, Site: {site}")
            
            simulation_time = np.arange(update_iterations) * time_step
            prediction_time = simulation_time + current_time
            desired_trajectory_segment = self.desired_trajectory_function(prediction_time)
            desired_amplitude = self._get_trajectory_amplitude(desired_trajectory_segment)
            self.time_history.append(prediction_time)
            self.desired_trajectory_history.append(desired_trajectory_segment)
            
            iteration = 1
            freq = self.ees_frequency_guess
            amplitude_error = float('inf')  # Initialize amplitude_error
                
            # Store optimization data for this phase
            phase_optimization_data = []
            
            while amplitude_error > self.amplitude_tolerance and iteration < self.max_iterations:
                # Optimize EES parameters for this phase
                ees_params = {
                    'frequency': freq,
                    'intensity': self.ees_intensity,
                    'site': site
                }
                
                # Run simulation with test parameters
                spikes, time_series = self.biological_system.run_simulation(
                    n_iterations=update_iterations,
                    time_step=time_step,
                    ees_stimulation_params=ees_params
                )
                
                # Extract joint trajectory
                joint_col = f"Joint_{self.biological_system.associated_joint}"
                actual_trajectory = time_series[joint_col].values
                actual_amplitude = self._get_trajectory_amplitude(actual_trajectory)
                    
                # Calculate amplitude error
                amplitude_error = abs(actual_amplitude - desired_amplitude)
                
                # Store optimization trajectory data correctly
                optimization_data = {
                    'trajectory': actual_trajectory.copy(),
                    'time': prediction_time.copy(),
                    'params': ees_params.copy(),
                    'cost': amplitude_error,
                    'amplitude': actual_amplitude
                }
                phase_optimization_data.append(optimization_data)
                
                self.cost_history.append(amplitude_error)
                
                if actual_amplitude > desired_amplitude:
                    freq = freq - self.freq_step
                else:
                    freq = freq + self.freq_step
                
                # Clamp frequency to valid range
                freq = max(self.min_frequency, min(self.max_frequency, freq))
            
                iteration += 1
            
            # Store optimization data for this phase
            self.optimization_trajectories.append(phase_optimization_data)
            
            # Update biological system state
            self.biological_system.update_state()
            
            # Store results for this phase
            self.trajectory_history.append(phase_optimization_data[-1]['trajectory'])
            self.ees_params_history.append(phase_optimization_data[-1]['params'])

            # Update time to next phase
            current_time = next_event_time   
            phase_idx += 1
        
        print(f"\nControl simulation completed!")
        
        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)
    
    def plot(self, base_output_path=None):
        """
        Plot the control results including trajectory tracking and EES parameter evolution.
        
        Parameters:
        -----------
        base_output_path : str, optional
            Base path for saving plots
        """
        if not self.trajectory_history:
            raise ValueError("No simulation results to plot. Run control simulation first.")
        
        # Convert lists to numpy arrays for easier handling
        time_array = np.concatenate(self.time_history)
        actual_traj = np.concatenate(self.trajectory_history)
        desired_traj = np.concatenate(self.desired_trajectory_history)
        
        # Extract EES parameters over time
        frequencies = [params['frequency'] / hertz for params in self.ees_params_history]
        sites = [params['site'] for params in self.ees_params_history]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Color-blind friendly palette
        cb_colors = [
            '#000000',  # Black
            '#E69F00',  # Orange
            '#56B4E9',  # Sky Blue
            '#009E73',  # Bluish Green
            '#F0E442',  # Yellow
            '#0072B2',  # Blue
            '#D55E00',  # Vermillion
            '#CC79A7',  # Reddish Purple
        ]
        
        # Plot 1: Enhanced trajectory comparison with optimization trajectories
        axes[0].plot(time_array, desired_traj, color='#000000', linewidth=2, 
                    label='Desired trajectory', zorder=10)
        axes[0].plot(time_array, actual_traj, color='#E69F00', linestyle='--', linewidth=2, 
                    label='Actual trajectory (selected)', zorder=9)
        
        # Add phase boundaries
        for i, event_time in enumerate(self.event_times[1:-1], 1):
            axes[0].axvline(x=event_time, color='red', linestyle=':', alpha=0.7, 
                          label=f'Phase {i} boundary' if i == 1 else "")
        
        # Plot optimization trajectories with transparency
        param_to_color = {}
        plotted_params = set()
        
        if self.optimization_trajectories:
            for phase_idx, phase_optimization_data in enumerate(self.optimization_trajectories):
                for opt_data in phase_optimization_data:
                    # Create parameter key for consistent coloring
                    param_key = (float(opt_data['params']['frequency'] / hertz), opt_data['params']['site'])
                    param_label = f"f={opt_data['params']['frequency']/hertz:.0f}Hz, {opt_data['params']['site']}"
                    
                    # Assign color if not already assigned
                    if param_key not in param_to_color:
                        color_idx = (len(param_to_color) + 2) % len(cb_colors)
                        param_to_color[param_key] = cb_colors[color_idx]
                    
                    # Determine if this should be in legend
                    show_in_legend = param_key not in plotted_params
                    if show_in_legend:
                        plotted_params.add(param_key)
                    
                    # Plot with transparency (skip the best trajectory as it's already shown in orange)
                    if not np.array_equal(opt_data['trajectory'], self.trajectory_history[phase_idx]):
                        axes[0].plot(opt_data['time'], opt_data['trajectory'], 
                                   color=param_to_color[param_key], 
                                   alpha=0.4, linewidth=1.0, 
                                   label=param_label if show_in_legend else "", 
                                   zorder=1)
        
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)', fontsize=12)
        axes[0].set_title('Trajectory Tracking Performance with EES Parameter Optimization', 
                         fontsize=14, fontweight='bold')
        
        legend = axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                              fontsize=10, frameon=True, fancybox=True, 
                              shadow=True, framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: EES Frequency evolution
        # Create time array for frequency plot (one point per phase)
        freq_time_array = []
        for i, time_segment in enumerate(self.time_history):
            freq_time_array.append(time_segment[0])  # Use start time of each phase
        
        axes[1].plot(freq_time_array, frequencies, color='#009E73', linewidth=2, marker='o', markersize=4)
        for event_time in self.event_times[1:-1]:
            axes[1].axvline(x=event_time, color='red', linestyle=':', alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('EES Frequency (Hz)')
        axes[1].set_title('EES Frequency Evolution Across Gait Phases')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([self.min_frequency/hertz - 5, self.max_frequency/hertz + 5])
        
        # Plot 3: Stimulation sites
        unique_sites = list(set(sites))
        site_to_num = {site: i for i, site in enumerate(unique_sites)}
        site_nums = [site_to_num[site] for site in sites]
        
        axes[2].plot(freq_time_array, site_nums, color='#0072B2', linewidth=2, marker='s', markersize=4)
        for event_time in self.event_times[1:-1]:
            axes[2].axvline(x=event_time, color='red', linestyle=':', alpha=0.7)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Stimulation Site')
        axes[2].set_title('Stimulation Site Across Gait Phases')
        axes[2].set_yticks(range(len(unique_sites)))
        axes[2].set_yticklabels(unique_sites)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, 'ees_control_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Control results plot saved to {base_output_path}")
        
        plt.show()
        
        # Plot cost evolution
        if len(self.cost_history) > 0:
            plt.figure(figsize=(10, 6))
            phase_points = np.arange(len(self.cost_history))
            
            plt.plot(phase_points, self.cost_history, 'o-', color='#0072B2', linewidth=2, markersize=6)
            plt.xlabel('Optimization Phase')
            plt.ylabel('Optimization Cost (MSE)')
            plt.title('EES Parameter Optimization Cost Evolution by Phase')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            if base_output_path:
                plt.savefig(os.path.join(base_output_path, 'optimization_cost.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
