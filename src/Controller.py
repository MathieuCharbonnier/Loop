from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import os

class EESController:
    """
    Model Predictive Controller for EES parameters to achieve desired joint trajectories.
    
    This controller uses a model-based approach where it simulates different EES parameter
    combinations and selects the one that best matches the desired trajectory.
    """
    
    def __init__(self, biological_system, desired_trajectory_func, update_iterations, 
                 initial_ees_params=None, frequency_range=(20, 100)*hertz, balance_range=(-1.0, 1.0),
                 frequency_step=20*hertz, balance_step=0.5, time_step=0.1*ms):
        """
        Initialize the EES controller.
        
        Parameters:
        -----------
        biological_system : BiologicalSystem
            The biological system to control
        desired_trajectory_func : callable
            Function that takes time array and returns desired joint angles
        update_iterations : int
            Number of iterations after which to update EES parameters
        initial_ees_params : dict, optional
            Initial EES parameters. If None, uses default values
        frequency_range : tuple
            (min_freq, max_freq) range for frequency optimization
        balance_range : tuple
            (min_balance, max_balance) range for balance optimization (ignored for single muscle)
        frequency_step : float
            Step size for frequency optimization
        balance_step : float
            Step size for balance optimization (ignored for single muscle)
        """
        self.biological_system = biological_system.clone_with()
        self.desired_trajectory_func = desired_trajectory_func
        self.update_iterations = update_iterations
        self.time_step=time_step
        # Check if system has multiple muscles
        self.has_multiple_muscles = self.biological_system.number_muscles > 1
        
        # Default EES parameters
        if initial_ees_params is None:
            self.current_ees_params = {
                'frequency': 50 * hertz,  # Hz
                'intensity': 0.6,  # Fixed intensity
            }
            # Only add balance parameter for multiple muscle systems
            if self.has_multiple_muscles:
                self.current_ees_params['balance'] = 0.0  # Neutral balance
        else:
            self.current_ees_params = deepcopy(initial_ees_params)
            # Remove balance parameter if system has only one muscle
            if not self.has_multiple_muscles and 'balance' in self.current_ees_params:
                del self.current_ees_params['balance']
            
        # Optimization ranges
        self.frequency_range = frequency_range
        self.balance_range = balance_range
        self.frequency_step = frequency_step
        self.balance_step = balance_step
        
        # Generate parameter grids for optimization
        self.frequency_grid = np.arange(frequency_range[0], frequency_range[1], frequency_step)*hertz
        
        # Only create balance grid for multiple muscle systems
        if self.has_multiple_muscles:
            self.balance_grid = np.arange(balance_range[0], balance_range[1], balance_step)
        else:
            self.balance_grid = []  # Empty grid for single muscle
        
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        
        # Storage for optimization trajectories
        self.optimization_trajectories = []  # List of dicts with 'time', 'trajectory', 'params', 'cost'

    def copy_brian_dict(self,d):
        if isinstance(d, dict):
            # If d is a dictionary, apply the function to each value
            return {k: self.copy_brian_dict(v) for k, v in d.items()}
        elif hasattr(d, 'copy'):
            # If the object has a `.copy()` method (e.g., Brian2 Quantity), use it
            return d.copy()
        else:
            # Otherwise, return the value as-is (int, float, string, etc.)
            return d 

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
        # Use mean squared error as cost function
        return np.mean((actual_trajectory - desired_trajectory) ** 2)
    
    def _optimize_ees_parameters(self, current_time):
        """
        Optimize EES parameters by testing different combinations.
        
        Parameters:
        -----------
        current_time : float
            Current simulation time
            
        Returns:
        --------
        dict
            Optimal EES parameters
        """
        best_cost = float('inf')
        best_params = self.copy_brian_dict(self.current_ees_params)

        
        print(f"Optimizing EES parameters at time {current_time:.3f}s...")
        
        # Store trajectories for this optimization cycle
        optimization_cycle_trajectories = []
        
        if self.has_multiple_muscles:
            print(f"Testing {len(self.frequency_grid)} frequency × {len(self.balance_grid)} balance combinations")
            # Test all combinations of frequency and balance for multiple muscles
            for freq, balance in product(self.frequency_grid, self.balance_grid):
                # Create test parameters
                test_params = self.copy_brian_dict(self.current_ees_params)
                test_params['frequency'] = freq
                test_params['balance'] = balance

                # Clone the biological system to avoid modifying the original
                test_system = self.biological_system
                    
                # Run simulation with test parameters
                spikes, time_series = test_system.run_simulation(
                        n_iterations=self.update_iterations,
                        time_step=self.time_step,
                        ees_stimulation_params=test_params
                )
                
                # Extract joint trajectory
                joint_col = f"Joint_{self.biological_system.associated_joint}"
                if joint_col in time_series.columns:
                    actual_trajectory = time_series[joint_col].values
                    prediction_time=time_series['Time']+current_time
                    desired_trajectory= self.desired_trajectory_func(prediction_time)

                    # Compute cost
                    cost = self._compute_trajectory_cost(actual_trajectory, desired_trajectory)
                    
                    # Store this trajectory for plotting
                    optimization_cycle_trajectories.append({
                        'time': prediction_time[:len(actual_trajectory)],
                        'trajectory': actual_trajectory,
                        'params': deepcopy(test_params),
                        'cost': cost
                    })
                        
                    # Update best parameters if this is better
                    if cost < best_cost:
                        best_cost = cost
                        best_params = test_params
                            
        else:
            print(f"Testing {len(self.frequency_grid)} frequency values ")
            # Test only frequency for single muscle systems
            for freq in self.frequency_grid:
                # Create test parameters
                test_params = self.copy_brian_dict(self.current_ees_params)
                test_params['frequency'] = freq

                # Clone the biological system to avoid modifying the original
                test_system = self.biological_system
                    
                # Run simulation with test parameters
                spikes, time_series = test_system.run_simulation(
                    n_iterations=self.update_iterations,
                    time_step=self.time_step,
                    ees_stimulation_params=test_params
                )
                test_system.plot() 
                # Extract joint trajectory
                joint_col = f"Joint_{self.biological_system.associated_joint}"
                if joint_col in time_series.columns:
                    actual_trajectory = time_series[joint_col].values
                    prediction_time=time_series['Time']+current_time
                    desired_trajectory= self.desired_trajectory_func(prediction_time)    

                    # Compute cost
                    cost = self._compute_trajectory_cost(actual_trajectory, desired_trajectory)
                    
                    # Store this trajectory for plotting
                    optimization_cycle_trajectories.append({
                        'time': prediction_time[:len(actual_trajectory)],
                        'trajectory': actual_trajectory,
                        'params': deepcopy(test_params),
                        'cost': cost
                    })
                        
                    # Update best parameters if this is better
                    if cost < best_cost:
                        best_cost = cost
                        best_params = test_params
        
        # Store all trajectories from this optimization cycle
        self.optimization_trajectories.append(optimization_cycle_trajectories)
        
        print(f"Best cost: {best_cost:.6f}")
        if self.has_multiple_muscles:
            print(f"Best parameters: frequency={best_params['frequency']:.1f}Hz, balance={best_params['balance']:.3f}")
        else:
            print(f"Best parameters: frequency={best_params['frequency']:.1f}Hz")
        
        return best_params, best_cost
    
    def run_control(self, total_iterations, time_step=0.1*ms, base_output_path=None):
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
        current_iteration = 0
        current_time = 0.0*ms
        
        print(f"Starting EES control simulation...")
        print(f"Total iterations: {total_iterations}")
        print(f"Update every: {self.update_iterations} iterations")
        print(f"Time step: {time_step:.1f}")
        print(f"System type: {'Multiple muscles' if self.has_multiple_muscles else 'Single muscle'}")
        
        while current_iteration < total_iterations:
            # Determine how many iterations to run this cycle
            remaining_iterations = total_iterations - current_iteration
            iterations_this_cycle = min(self.update_iterations, remaining_iterations)
            
            print(f"\nControl cycle: iterations {current_iteration} to {current_iteration + iterations_this_cycle}")
            
            # Optimize EES parameters if not the first iteration
            if current_iteration > 0:
                optimal_params, cost = self._optimize_ees_parameters(current_time)
                self.current_ees_params = optimal_params
                self.cost_history.append(cost)
            else:
                self.cost_history.append(0.0)  # No optimization cost for first iteration
         
            # Run simulation with current EES parameters
            spikes, time_series = self.biological_system.run_simulation(
                n_iterations=iterations_this_cycle,
                time_step=time_step,
                ees_stimulation_params=self.current_ees_params,
                base_output_path=base_output_path
            )
           
            # Extract joint trajectory
            joint_col = f"Joint_{self.biological_system.associated_joint}"
            if joint_col not in time_series.columns:
                raise ValueError(f"Joint column '{joint_col}' not found in time series")
            
            actual_trajectory = time_series[joint_col].values
            
            # Create time array for this segment
            segment_time = np.arange(
                current_time,
                current_time + len(actual_trajectory) * time_step,
                time_step
            )[:len(actual_trajectory)]*second
      
            # Get desired trajectory for this segment
            desired_trajectory = self.desired_trajectory_func(segment_time)
            
            # Store results
            self.trajectory_history.extend(actual_trajectory)
            self.desired_trajectory_history.extend(desired_trajectory)
            self.time_history.extend(segment_time)
            
            # Store EES parameters for this segment
            for _ in range(len(actual_trajectory)):
                self.ees_params_history.append(deepcopy(self.current_ees_params))

            # Update system state for next iteration
            self.biological_system.update_system_state()
            
            # Update counters
            current_iteration += iterations_this_cycle
            current_time = segment_time[-1] + time_step if len(segment_time) > 0 else current_time + time_step
        
        print(f"\nControl simulation completed!")
        print(f"Final trajectory length: {len(self.trajectory_history)} points")
        
        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)
    
    
        
        # Plot cost evolution
        if len(self.cost_history) > 1:  # Only plot if we have optimization costs
            plt.figure(figsize=(10, 6))
            optimization_points = np.arange(1, len(self.cost_history)) * self.update_iterations
            costs = self.cost_history[1:]  # Skip first point (no optimization)
            
            plt.plot(optimization_points, costs, 'o-', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Optimization Cost (MSE)')
            plt.title('EES Parameter Optimization Cost Evolution')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale often better for cost visualization
            
            if base_output_path:
                plt.savefig(os.path.join(base_output_path, 'optimization_cost.png'), 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
    
    def plot_results(self, base_output_path=None):
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
            balances = [params['balance'] for params in self.ees_params_history]
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))  # Made figure taller
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 12))  # Made figure taller
        
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
        axes[0].plot(time_array, desired_traj, color='#000000', linewidth=3, 
                    label='Desired trajectory', zorder=10)
        axes[0].plot(time_array, actual_traj, color='#E69F00', linestyle='--', linewidth=2, 
                    label='Actual trajectory (selected)', zorder=9)
        
        # Create a mapping from parameter combinations to colors
        param_to_color = {}
        plotted_params = set()  # Track which parameter combinations have been plotted
        
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
                               float(traj_data['params']['balance']))
                    param_label = f"f={traj_data['params']['frequency']:.0f}Hz, b={traj_data['params']['balance']:.2f}"
                else:
                    param_key = (float(traj_data['params']['frequency']),)
                    param_label = f"f={traj_data['params']['frequency']:.0f}Hz"
                
                # Assign color if not already assigned
                if param_key not in param_to_color:
                    color_idx = len(param_to_color) % len(cb_colors)
                    # Skip black and orange as they're used for desired and actual trajectories
                    if color_idx == 0:  # Skip black
                        color_idx = 2
                    elif color_idx == 1:  # Skip orange  
                        color_idx = 3
                    param_to_color[param_key] = cb_colors[color_idx]
                
                # Determine if this should be in legend (only first occurrence of each param combination)
                show_in_legend = param_key not in plotted_params
                if show_in_legend:
                    plotted_params.add(param_key)
                
                # Plot with transparency and thinner lines for non-optimal trajectories
                alpha = 0.4 if traj_idx > 2 else 0.7  # Make worst trajectories more transparent
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
        
        # Create a smaller legend
        legend = axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                              fontsize=8, frameon=True, fancybox=True, 
                              shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: EES Frequency evolution
        axes[1].plot(time_array, frequencies, color='#009E73', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('EES Frequency (Hz)')
        axes[1].set_title('EES Frequency Evolution')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([self.frequency_range[0] - 5*hertz, self.frequency_range[1] + 5*hertz])
        
        # Plot 3: EES Balance evolution (only for multiple muscle systems)
        if self.has_multiple_muscles:
            axes[2].plot(time_array, balances, color='#D55E00', linewidth=2)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('EES Balance')
            axes[2].set_title('EES Balance Evolution (Flexor-Extensor)')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([self.balance_range[0] - 0.1, self.balance_range[1] + 0.1])
        
        plt.tight_layout()
        
        # Save plot if path provided
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, 'ees_control_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Control results plot saved to {base_output_path}")
        
        plt.show()
        
        # Plot cost evolution
        if len(self.cost_history) > 1:  # Only plot if we have optimization costs
            plt.figure(figsize=(10, 6))
            optimization_points = np.arange(1, len(self.cost_history)) * self.update_iterations
            costs = self.cost_history[1:]  # Skip first point (no optimization)
            
            plt.plot(optimization_points, costs, 'o-', color='#0072B2', linewidth=2, markersize=6)
            plt.xlabel('Iteration')
            plt.ylabel('Optimization Cost (MSE)')
            plt.title('EES Parameter Optimization Cost Evolution')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale often better for cost visualization
            
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
        
        # Only include balance metrics for multiple muscle systems
        if self.has_multiple_muscles:
            metrics['final_balance'] = self.ees_params_history[-1]['balance'] if self.ees_params_history else None
        
        return metrics


# Example usage and utility functions
def create_sinusoidal_trajectory(amplitude=30, frequency=0.5, offset=0):
    """
    Create a sinusoidal desired trajectory function.
    
    Parameters:
    -----------
    amplitude : float
        Amplitude of the sinusoid (degrees)
    frequency : float
        Frequency of the sinusoid (Hz)
    offset : float
        Vertical offset (degrees)
        
    Returns:
    --------
    callable
        Function that takes time array and returns trajectory
    """
    def trajectory_func(time_array):
        return amplitude * np.sin(2 * np.pi * frequency * time_array) + offset
    
    return trajectory_func


def create_step_trajectory(step_times, step_values, initial_value=0):
    """
    Create a step function trajectory.
    
    Parameters:
    -----------
    step_times : list
        Times at which steps occur
    step_values : list
        Values after each step
    initial_value : float
        Initial value before first step
        
    Returns:
    --------
    callable
        Function that takes time array and returns trajectory
    """
    def trajectory_func(time_array):
        trajectory = np.full_like(time_array, initial_value)
        
        for step_time, step_value in zip(step_times, step_values):
            trajectory[time_array >= step_time] = step_value
            
        return trajectory
    
    return trajectory_func
