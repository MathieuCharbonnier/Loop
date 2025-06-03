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
        self.ees_intensity=ees_intensity
        self.ees_frequency_guess=ees_frequency_guess
        
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
        self.total_time=time[-1]
        self.desired_trajectory_function = interp1d(time, data, kind='cubic', fill_value="extrapolate")
        self.event=np.array([self.time[np.argmax(self.desired_trajectory)], self.time[np.argmin(self.desired_trajectory)])*second# define the 3 phases: first dorsiflexion, then plantarflexion, and recovery 
        self.site_stimulation=['L4', 'S1', 'L4']
    
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        self.optimization_trajectories = []

    def _compute_trajectory_cost(self, actual_trajectory, desired_trajectory):
        """
        Compute the cost between actual and desired trajectories.
        
        Parameters:
        -----------
        actual_trajectory : np.ndarray
            Actual joint trajectory from simulation
        desired_trajectory : np.ndarray
            Desired joint trajectory
            
        Returns:
        --------
        float
            Cost value (lower is better)
        """
        return np.mean((actual_trajectory - desired_trajectory) ** 2)
        
    
    def _optimize_ees_parameters(self,time_step, update_iteration,site):
        """
        Optimize EES parameters by increasing/decreasing EES frequency if the absolute amplitude of the movement is too weak/strong, 
        and optionally optimize the site of stimulation if the joint is too much in dorsi or plantar flexion..
        Returns the system state, results, and parameters for the best combination.
        
        Parameters:
        -----------
        current_time : float
            Current simulation time
            
        Returns:
        --------
        tuple
            (best_system_state, best_spikes, best_time_series, best_params, best_cost, optimization_trajectories)
        """
        best_cost = float('inf')
        best_params = BiologicalSystem.copy_brian_dict(self.current_ees_params)
        best_spikes = None
        best_time_series = None
        best_system_state = None
        
        print(f"Optimizing EES parameters at time {current_time:.3f}s...")
        
        # Store trajectories for this optimization cycle
        optimization_cycle_trajectories = []
        
        #here need to define the optimization: if amplitude of movement too  weak/strong increase/decrease frequency
            test_params={'frequency': freq,
                        'intensity':self.ees_intensity,
                        'site':site
                       }
    
            # Run simulation with test parameters
            spikes, time_series = self.biological_system.run_simulation(
                n_iterations=update_iterations,
                time_step=time_step,
                ees_stimulation_params=test_params
            )
            
            # Extract joint trajectory
            joint_col = f"Joint_{self.biological_system.associated_joint}"
            if joint_col in time_series.columns:
                actual_trajectory = time_series[joint_col].values
                prediction_time = time_series['Time'] + current_time
                desired_trajectory = self.desired_trajectory_function(prediction_time)

                # Compute cost
                cost = self._compute_trajectory_cost(actual_trajectory, desired_trajectory)
                
                # Store this trajectory for plotting
                optimization_cycle_trajectories.append({
                    'time': prediction_time[:len(actual_trajectory)],
                    'trajectory': actual_trajectory,
                    'params': test_params,
                    'cost': cost
                })
                    
                # Update best parameters if this is better
                if cost < best_cost:
                    best_cost = cost
                    best_params = test_params
                    best_spikes = spikes
                    best_time_series = time_series
                    # Save the system state after the best simulation
                    best_system_state = self.biological_system.get_system_state()
        
        print(f"Best cost: {best_cost:.6f}")
        print(f"Best parameters: frequency={best_params['frequency']:.1f} Hz")
        
        return best_system_state, best_spikes, best_time_series, best_params, best_cost, optimization_cycle_trajectories
    
    def run(self, time_step=0.1*ms):
        """
        Run the complete control simulation.
        
        Parameters:
        -----------
        total_iterations : int
            Total number of iterations to simulate
        time_step : float
            Time step for simulation (in seconds)
        base_output_path : str, optional
            Base path for saving output files
            
        Returns:
        --------
        tuple
            (trajectory_history, desired_trajectory_history, ees_params_history, time_history)
        """
        current_time = 0.0 * ms
           
        print(f"Starting EES control simulation...")
        
        while current_time < self.total_time:
            #Find the next gait event and the number of iteration of the closed loop to achieve it
            next_indices = np.where(self.time> current_time)[0]
            if next_indices.size > 0:
                next_index = next_indices[0]
                next_time = self.time[next_index]
                site=self.stimulation_site[next_index]
            else:
                next_time=self.total_time
                site=self.stimulation_site[-1]
            
            update_iteration=int((next_time-current_time)/self.biological_system.reaction_time) 
            
            # Optimize EES parameters
            (best_system_state, spikes, time_series, 
             optimal_params, cost, optimization_trajectories) = self._optimize_ees_parameters(time_step, update_iteration, site)
            
            # Update current parameters and cost history
            self.current_ees_params = optimal_params
            self.cost_history.append(cost)
            
            # Store optimization trajectories
            self.optimization_trajectories.append(optimization_trajectories)
            
            # Apply the best system state to our main biological system
            # This avoids re-running the simulation with the best parameters
            self.biological_system.set_system_state(best_system_state)
           
            # Extract joint trajectory from the already-computed best simulation
            joint_col = f"Joint_{self.biological_system.associated_joint}"
            
            actual_trajectory = time_series[joint_col].values
            
            # Create time array for this segment
            segment_time = np.arange(
                current_time,
                current_time + len(actual_trajectory) * time_step,
                time_step
            )[:len(actual_trajectory)] * second
      
            # Get desired trajectory for this segment
            desired_trajectory = self.desired_trajectory_function(segment_time)
            
            # Store results
            self.trajectory_history.extend(actual_trajectory)
            self.desired_trajectory_history.extend(desired_trajectory)
            self.time_history.extend(segment_time)
            
            # Store EES parameters for this segment
            for _ in range(len(actual_trajectory)):
                self.ees_params_history.append(deepcopy(self.current_ees_params))

            # Update counters
            current_iteration += iterations_this_cycle
            current_time = segment_time[-1] + time_step if len(segment_time) > 0 else current_time + time_step
        
        print(f"\nControl simulation completed!")
        print(f"Final trajectory length: {len(self.trajectory_history)} points")
        
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
        time_array = np.array(self.time_history)
        actual_traj = np.array(self.trajectory_history)
        desired_traj = np.array(self.desired_trajectory_history)
        
        # Extract EES parameters over time
        frequencies = [params['frequency'] for params in self.ees_params_history]
        
        # Determine number of subplots based on system type
        if self.has_multiple_muscles:
            sites = [params['site'] for params in self.ees_params_history]
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Color-blind friendly palette (Wong 2011)
        cb_colors = [
            '#000000',  # Black
            '#E69F00',  # Orange
            '#56B4E9',  # Sky Blue
            '#009E73',  # Bluish Green
            '#F0E442',  # Yellow
            '#0072B2',  # Blue
            '#D55E00',  # Vermillion
            '#CC79A7',  # Reddish Purple
            '#999999',  # Gray
            '#8B4513',  # Brown
            '#FF1493',  # Deep Pink
            '#32CD32',  # Lime Green
            '#4169E1',  # Royal Blue
            '#FF8C00',  # Dark Orange
            '#9370DB',  # Medium Purple
            '#20B2AA'   # Light Sea Green
        ]
        
        # Plot 1: Enhanced trajectory comparison with optimization trajectories
        axes[0].plot(time_array, desired_traj, color='#000000', linewidth=2, 
                    label='Desired trajectory', zorder=10)
        axes[0].plot(time_array, actual_traj, color='#E69F00', linestyle='--', linewidth=2, 
                    label='Actual trajectory (selected)', zorder=9)
        
        # Create a mapping from parameter combinations to colors
        param_to_color = {}
        plotted_params = set()
        
        for optimization_cycle in self.optimization_trajectories:
            # Sort trajectories by cost to show best ones more prominently
            sorted_trajectories = sorted(optimization_cycle, key=lambda x: x['cost'])
            
            for traj_idx, traj_data in enumerate(sorted_trajectories):
                # Skip the best trajectory as it's already shown as "Actual trajectory"
                if traj_idx == 0:
                    continue
                
                # Create parameter key for consistent coloring
                if self.has_multiple_muscles:
                    param_key = (float(traj_data['params']['frequency']), 
                               float(traj_data['params']['site']))
                    param_label = f"f={traj_data['params']['frequency']:.0f}Hz, b={traj_data['params']['site']:.2f}"
                else:
                    param_key = (float(traj_data['params']['frequency']),)
                    param_label = f"f={traj_data['params']['frequency']:.0f}Hz"
                
                # Assign color if not already assigned
                if param_key not in param_to_color:
                    color_idx = len(param_to_color) % len(cb_colors)
                    if color_idx == 0:  # Skip black
                        color_idx = 2
                    elif color_idx == 1:  # Skip orange  
                        color_idx = 3
                    param_to_color[param_key] = cb_colors[color_idx]
                
                # Determine if this should be in legend
                show_in_legend = param_key not in plotted_params
                if show_in_legend:
                    plotted_params.add(param_key)
                
                # Plot with transparency and thinner lines for non-optimal trajectories
                alpha = 0.4 if traj_idx > 2 else 0.7
                linewidth = 0.8 if traj_idx > 2 else 1.2
                
                axes[0].plot(traj_data['time'], traj_data['trajectory'], 
                           color=param_to_color[param_key], 
                           alpha=alpha, linewidth=linewidth, 
                           label=param_label if show_in_legend else "", 
                           zorder=1)
        
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)', fontsize=12)
        axes[0].set_title('Trajectory Tracking Performance with EES Parameter Optimization', 
                         fontsize=14, fontweight='bold')
        
        legend = axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                              fontsize=11, frameon=True, fancybox=True, 
                              shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: EES Frequency evolution
        axes[1].plot(time_array, frequencies, color='#009E73', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('EES Frequency (Hz)')
        axes[1].set_title('EES Frequency Evolution')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([self.frequency_grid[0] - 5*hertz, self.frequency_grid[-1] + 5*hertz])
        

        
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
            optimization_points = np.arange(len(self.cost_history)) * self.update_iterations
            
            plt.plot(optimization_points, self.cost_history, 'o-', color='#0072B2', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Optimization Cost (MSE)')
            plt.title('EES Parameter Optimization Cost Evolution')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            if base_output_path:
                plt.savefig(os.path.join(base_output_path, 'optimization_cost.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def get_performance_metrics(self):
        """
        Calculate and return performance metrics for the control.
        
        Returns:
        --------
        dict
            Dictionary containing various performance metrics
        """
        if not self.trajectory_history:
            raise ValueError("No simulation results available. Run control simulation first.")
        
        actual_traj = np.array(self.trajectory_history)
        desired_traj = np.array(self.desired_trajectory_history)
        
        # Calculate metrics
        mse = np.mean((actual_traj - desired_traj) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual_traj - desired_traj))
        max_error = np.max(np.abs(actual_traj - desired_traj))
        
        # R-squared coefficient
        ss_res = np.sum((actual_traj - desired_traj) ** 2)
        ss_tot = np.sum((desired_traj - np.mean(desired_traj)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r_squared': r_squared,
            'final_frequency': self.ees_params_history[-1]['frequency'] if self.ees_params_history else None,
        }
        
        
        return metrics
