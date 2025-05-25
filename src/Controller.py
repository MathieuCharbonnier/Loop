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
                 initial_ees_params=None, frequency_range=(20, 100), balance_range=(-1.0, 1.0),
                 frequency_steps=5, balance_steps=5):
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
            (min_balance, max_balance) range for balance optimization
        frequency_steps : int
            Number of frequency values to test
        balance_steps : int
            Number of balance values to test
        """
        self.biological_system = biological_system
        self.desired_trajectory_func = desired_trajectory_func
        self.update_iterations = update_iterations
        
        # Default EES parameters
        if initial_ees_params is None:
            self.current_ees_params = {
                'frequency': 50,  # Hz
                'intensity': 1.0,  # Fixed intensity
                'balance': 0.0    # Neutral balance
            }
        else:
            self.current_ees_params = deepcopy(initial_ees_params)
            
        # Optimization ranges
        self.frequency_range = frequency_range
        self.balance_range = balance_range
        self.frequency_steps = frequency_steps
        self.balance_steps = balance_steps
        
        # Generate parameter grids for optimization
        self.frequency_grid = np.linspace(frequency_range[0], frequency_range[1], frequency_steps)
        self.balance_grid = np.linspace(balance_range[0], balance_range[1], balance_steps)
        
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        
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
        best_params = deepcopy(self.current_ees_params)
        
        # Create time array for the prediction horizon
        time_step = 0.1e-3  # Assuming 0.1ms time step (adjust as needed)
        prediction_time = np.arange(
            current_time, 
            current_time + self.biological_system.reaction_time * self.update_iterations,
            time_step
        )
        
        # Get desired trajectory for prediction horizon
        desired_trajectory = self.desired_trajectory_func(prediction_time)
        
        print(f"Optimizing EES parameters at time {current_time:.3f}s...")
        print(f"Testing {len(self.frequency_grid)} frequency × {len(self.balance_grid)} balance combinations")
        
        # Test all combinations of frequency and balance
        for freq, balance in product(self.frequency_grid, self.balance_grid):
            # Create test parameters
            test_params = deepcopy(self.current_ees_params)
            test_params['frequency'] = freq
            test_params['balance'] = balance
            
            try:
                # Clone the biological system to avoid modifying the original
                test_system = self.biological_system.clone_with()
                
                # Run simulation with test parameters
                spikes, time_series = test_system.run_simulation(
                    n_iterations=self.update_iterations,
                    ees_stimulation_params=test_params
                )
                
                # Extract joint trajectory
                joint_col = f"joint_{self.biological_system.associated_joint}"
                if joint_col in time_series.columns:
                    actual_trajectory = time_series[joint_col].values
                    
                    # Ensure trajectories have the same length
                    min_len = min(len(actual_trajectory), len(desired_trajectory))
                    actual_trajectory = actual_trajectory[:min_len]
                    desired_trajectory_truncated = desired_trajectory[:min_len]
                    
                    # Compute cost
                    cost = self._compute_trajectory_cost(actual_trajectory, desired_trajectory_truncated)
                    
                    # Update best parameters if this is better
                    if cost < best_cost:
                        best_cost = cost
                        best_params = test_params
                        
            except Exception as e:
                print(f"Warning: Simulation failed for freq={freq}, balance={balance}: {e}")
                continue
        
        print(f"Best cost: {best_cost:.6f}")
        print(f"Best parameters: frequency={best_params['frequency']:.1f}Hz, balance={best_params['balance']:.3f}")
        
        return best_params, best_cost
    
    def run_control(self, total_iterations, time_step=0.1e-3, base_output_path=None):
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
        current_time = 0.0
        
        print(f"Starting EES control simulation...")
        print(f"Total iterations: {total_iterations}")
        print(f"Update every: {self.update_iterations} iterations")
        print(f"Time step: {time_step*1000:.1f}ms")
        
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
            joint_col = f"joint_{self.biological_system.associated_joint}"
            if joint_col not in time_series.columns:
                raise ValueError(f"Joint column '{joint_col}' not found in time series")
            
            actual_trajectory = time_series[joint_col].values
            
            # Create time array for this segment
            segment_time = np.arange(
                current_time,
                current_time + len(actual_trajectory) * time_step,
                time_step
            )[:len(actual_trajectory)]
            
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
        balances = [params['balance'] for params in self.ees_params_history]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Trajectory comparison
        axes[0].plot(time_array, desired_traj, 'b-', linewidth=2, label='Desired trajectory')
        axes[0].plot(time_array, actual_traj, 'r--', linewidth=1.5, label='Actual trajectory')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
        axes[0].set_title('Trajectory Tracking Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: EES Frequency evolution
        axes[1].plot(time_array, frequencies, 'g-', linewidth=2)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('EES Frequency (Hz)')
        axes[1].set_title('EES Frequency Evolution')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([self.frequency_range[0] - 5, self.frequency_range[1] + 5])
        
        # Plot 3: EES Balance evolution
        axes[2].plot(time_array, balances, 'm-', linewidth=2)
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
            'final_balance': self.ees_params_history[-1]['balance'] if self.ees_params_history else None
        }
        
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
