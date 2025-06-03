from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, golden
from ..BiologicalSystems.BiologicalSystem import BiologicalSystem

class EESControllerAdaptative:
    """
    Model Predictive Controller for EES parameters to achieve desired joint trajectories.
    
    This controller uses adaptive optimization algorithms for better convergence.
    """
    
    def __init__(self, biological_system, ees_intensity=0.5, ees_frequency_guess=50*hertz, 
                 optimization_method='adaptive_pid'):
        """
        Initialize the EES controller.
        
        Parameters:
        -----------
        biological_system : BiologicalSystem
            The biological system to control
        optimization_method : str
            Optimization method: 'adaptive_pid', 'proportional', 'golden_section', 'gradient_descent'
        """
        self.biological_system = biological_system.clone_with()
        self.ees_intensity = ees_intensity
        self.ees_frequency_guess = ees_frequency_guess
        self.optimization_method = optimization_method
        
        # Load trajectory data (keeping original code)
        mot_file = 'data/BothLegsWalk.mot'
        df = pd.read_csv(mot_file, sep='\t', skiprows=6)
        df.columns = df.columns.str.strip()

        def move_side_to_suffix(col):
            if col.startswith('r_'):
                return col[2:] + '_r'
            elif col.startswith('l_'):
                return col[2:] + '_l'
            else:
                return col

        df.columns = [move_side_to_suffix(col) for col in df.columns]

        self.time = df['time'].values
        self.desired_trajectory = df[self.biological_system.associated_joint].values
        self.total_time = self.time[-1]
        self.desired_trajectory_function = interp1d(self.time, self.desired_trajectory, kind='cubic', fill_value="extrapolate")
        
        max_dorsi_idx = np.argmax(self.desired_trajectory)
        min_plantar_idx = np.argmin(self.desired_trajectory)
        
        self.event_times = np.array([0, self.time[max_dorsi_idx], self.time[min_plantar_idx], self.total_time])
        self.site_stimulation = ['L4', 'S1', 'L4']
        
        # Enhanced optimization parameters
        self.min_frequency = 10 * hertz
        self.max_frequency = 100 * hertz
        self.amplitude_tolerance = 0.1
        self.max_iterations = 15
        
        # Adaptive PID controller parameters
        self.kp = 2.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.5  # Derivative gain
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # Adaptive step size parameters
        self.initial_step = 10 * hertz
        self.min_step = 0.5 * hertz
        self.step_decay = 0.8
        self.step_increase = 1.2
        
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        self.optimization_trajectories = []
        self.frequency_history = []  # Track frequency evolution within each phase

    def _get_trajectory_amplitude(self, trajectory):
        """Calculate the amplitude (peak-to-peak) of a trajectory."""
        return np.max(trajectory) - np.min(trajectory)
    
    def _simulate_with_frequency(self, freq, site, update_iterations, time_step):
        """
        Simulate the biological system with given frequency and return amplitude error.
        
        Returns:
        --------
        tuple: (amplitude_error, actual_trajectory, actual_amplitude)
        """
        ees_params = {
            'frequency': freq,
            'intensity': self.ees_intensity,
            'site': site
        }
        
        spikes, time_series = self.biological_system.run_simulation(
            n_iterations=update_iterations,
            time_step=time_step,
            ees_stimulation_params=ees_params
        )
        
        joint_col = f"Joint_{self.biological_system.associated_joint}"
        actual_trajectory = time_series[joint_col].values
        actual_amplitude = self._get_trajectory_amplitude(actual_trajectory)
        
        return actual_amplitude, actual_trajectory, ees_params
    
    def _optimize_frequency_adaptive_pid(self, desired_amplitude, site, update_iterations, 
                                       time_step, prediction_time):
        """
        Optimize frequency using adaptive PID controller.
        """
        freq = self.ees_frequency_guess
        phase_optimization_data = []
        
        # Reset PID variables for new phase
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        for iteration in range(self.max_iterations):
            actual_amplitude, actual_trajectory, ees_params = self._simulate_with_frequency(
                freq, site, update_iterations, time_step)
            
            # Calculate amplitude error (positive if too small, negative if too large)
            amplitude_error = desired_amplitude - actual_amplitude
            abs_error = abs(amplitude_error)
            
            # Store optimization data
            optimization_data = {
                'trajectory': actual_trajectory.copy(),
                'time': prediction_time.copy(),
                'params': ees_params.copy(),
                'cost': abs_error,
                'amplitude': actual_amplitude
            }
            phase_optimization_data.append(optimization_data)
            self.cost_history.append(abs_error)
            
            # Check convergence
            if abs_error <= self.amplitude_tolerance:
                print(f"    Converged in {iteration + 1} iterations (PID)")
                break
            
            # PID control calculation
            self.integral_error += amplitude_error
            derivative_error = amplitude_error - self.previous_error
            
            # PID output (frequency adjustment)
            pid_output = (self.kp * amplitude_error + 
                         self.ki * self.integral_error + 
                         self.kd * derivative_error)
            
            # Update frequency
            freq += pid_output
            freq = max(self.min_frequency, min(self.max_frequency, freq))
            
            self.previous_error = amplitude_error
            
            print(f"    Iteration {iteration + 1}: freq={freq/hertz:.1f}Hz, "
                  f"error={abs_error:.3f}, PID_out={pid_output/hertz:.1f}")
        
        return phase_optimization_data
    
    def _optimize_frequency_proportional(self, desired_amplitude, site, update_iterations, 
                                       time_step, prediction_time):
        """
        Optimize frequency using proportional controller with adaptive step size.
        """
        freq = self.ees_frequency_guess
        phase_optimization_data = []
        current_step = self.initial_step
        previous_error = float('inf')
        
        for iteration in range(self.max_iterations):
            actual_amplitude, actual_trajectory, ees_params = self._simulate_with_frequency(
                freq, site, update_iterations, time_step)
            
            amplitude_error = abs(actual_amplitude - desired_amplitude)
            
            optimization_data = {
                'trajectory': actual_trajectory.copy(),
                'time': prediction_time.copy(),
                'params': ees_params.copy(),
                'cost': amplitude_error,
                'amplitude': actual_amplitude
            }
            phase_optimization_data.append(optimization_data)
            self.cost_history.append(amplitude_error)
            
            if amplitude_error <= self.amplitude_tolerance:
                print(f"    Converged in {iteration + 1} iterations (Proportional)")
                break
            
            # Adaptive step size
            if amplitude_error < previous_error:
                current_step *= self.step_increase
            else:
                current_step *= self.step_decay
            
            current_step = max(self.min_step, current_step)
            
            # Proportional control
            if actual_amplitude > desired_amplitude:
                freq -= current_step
            else:
                freq += current_step
            
            freq = max(self.min_frequency, min(self.max_frequency, freq))
            previous_error = amplitude_error
            
            print(f"    Iteration {iteration + 1}: freq={freq/hertz:.1f}Hz, "
                  f"error={amplitude_error:.3f}, step={current_step/hertz:.1f}")
        
        return phase_optimization_data
    
    def _optimize_frequency_golden_section(self, desired_amplitude, site, update_iterations, 
                                         time_step, prediction_time):
        """
        Optimize frequency using golden section search.
        """
        phase_optimization_data = []
        
        def objective_function(freq_hz):
            freq = freq_hz * hertz
            actual_amplitude, actual_trajectory, ees_params = self._simulate_with_frequency(
                freq, site, update_iterations, time_step)
            
            amplitude_error = abs(actual_amplitude - desired_amplitude)
            
            # Store optimization data
            optimization_data = {
                'trajectory': actual_trajectory.copy(),
                'time': prediction_time.copy(),
                'params': ees_params.copy(),
                'cost': amplitude_error,
                'amplitude': actual_amplitude
            }
            phase_optimization_data.append(optimization_data)
            self.cost_history.append(amplitude_error)
            
            return amplitude_error
        
        # Golden section search
        result = minimize_scalar(objective_function, 
                               bounds=(self.min_frequency/hertz, self.max_frequency/hertz),
                               method='bounded')
        
        print(f"    Golden section converged: freq={result.x:.1f}Hz, error={result.fun:.3f}")
        
        return phase_optimization_data
    
    def _optimize_frequency_gradient_descent(self, desired_amplitude, site, update_iterations, 
                                          time_step, prediction_time):
        """
        Optimize frequency using gradient descent with finite differences.
        """
        freq = self.ees_frequency_guess
        phase_optimization_data = []
        learning_rate = 5.0 * hertz
        epsilon = 1.0 * hertz  # For finite differences
        
        for iteration in range(self.max_iterations):
            # Current point
            actual_amplitude, actual_trajectory, ees_params = self._simulate_with_frequency(
                freq, site, update_iterations, time_step)
            
            current_error = abs(actual_amplitude - desired_amplitude)
            
            optimization_data = {
                'trajectory': actual_trajectory.copy(),
                'time': prediction_time.copy(),
                'params': ees_params.copy(),
                'cost': current_error,
                'amplitude': actual_amplitude
            }
            phase_optimization_data.append(optimization_data)
            self.cost_history.append(current_error)
            
            if current_error <= self.amplitude_tolerance:
                print(f"    Converged in {iteration + 1} iterations (Gradient Descent)")
                break
            
            # Finite difference gradient
            freq_plus = min(freq + epsilon, self.max_frequency)
            freq_minus = max(freq - epsilon, self.min_frequency)
            
            amp_plus, _, _ = self._simulate_with_frequency(freq_plus, site, update_iterations, time_step)
            amp_minus, _, _ = self._simulate_with_frequency(freq_minus, site, update_iterations, time_step)
            
            error_plus = abs(amp_plus - desired_amplitude)
            error_minus = abs(amp_minus - desired_amplitude)
            
            # Gradient approximation
            gradient = (error_plus - error_minus) / (2 * epsilon / hertz)
            
            # Update frequency
            freq -= learning_rate * gradient
            freq = max(self.min_frequency, min(self.max_frequency, freq))
            
            # Adaptive learning rate
            learning_rate *= 0.95
            
            print(f"    Iteration {iteration + 1}: freq={freq/hertz:.1f}Hz, "
                  f"error={current_error:.3f}, grad={gradient:.3f}")
        
        return phase_optimization_data

    def run(self, time_step=0.1*ms):
        """
        Run the complete control simulation with improved optimization.
        """
        current_time = 0.0
        phase_idx = 0
        
        print(f"Starting EES control simulation with {self.optimization_method} optimization...")
        print(f"Event times: {self.event_times}")
        print(f"Stimulation sites: {self.site_stimulation}")
        
        while current_time < self.total_time and phase_idx < len(self.site_stimulation):
            next_event_time = self.event_times[phase_idx + 1] if phase_idx + 1 < len(self.event_times) else self.total_time
            phase_duration = next_event_time - current_time
            
            site = self.site_stimulation[phase_idx]
            update_iterations = max(1, int(phase_duration / (self.biological_system.reaction_time / second)))
            
            print(f"\nPhase {phase_idx + 1}: {current_time:.3f}s to {next_event_time:.3f}s")
            print(f"Duration: {phase_duration:.3f}s, Iterations: {update_iterations}, Site: {site}")
            
            simulation_time = np.arange(update_iterations) * time_step
            prediction_time = simulation_time + current_time
            desired_trajectory_segment = self.desired_trajectory_function(prediction_time)
            desired_amplitude = self._get_trajectory_amplitude(desired_trajectory_segment)
            
            self.time_history.append(prediction_time)
            self.desired_trajectory_history.append(desired_trajectory_segment)
            
            # Choose optimization method
            if self.optimization_method == 'adaptive_pid':
                phase_optimization_data = self._optimize_frequency_adaptive_pid(
                    desired_amplitude, site, update_iterations, time_step, prediction_time)
            elif self.optimization_method == 'proportional':
                phase_optimization_data = self._optimize_frequency_proportional(
                    desired_amplitude, site, update_iterations, time_step, prediction_time)
            elif self.optimization_method == 'golden_section':
                phase_optimization_data = self._optimize_frequency_golden_section(
                    desired_amplitude, site, update_iterations, time_step, prediction_time)
            elif self.optimization_method == 'gradient_descent':
                phase_optimization_data = self._optimize_frequency_gradient_descent(
                    desired_amplitude, site, update_iterations, time_step, prediction_time)
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            # Store optimization data for this phase
            self.optimization_trajectories.append(phase_optimization_data)
            
            # Update biological system state
            self.biological_system.update_state()
            
            # Store results for this phase (best result)
            best_result = min(phase_optimization_data, key=lambda x: x['cost'])
            self.trajectory_history.append(best_result['trajectory'])
            self.ees_params_history.append(best_result['params'])
            
            # Track frequency evolution
            freq_evolution = [data['params']['frequency']/hertz for data in phase_optimization_data]
            self.frequency_history.append(freq_evolution)
            
            current_time = next_event_time   
            phase_idx += 1
        
        print(f"\nControl simulation completed!")
        
        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)

    def plot(self, base_output_path=None):
        """
        Enhanced plotting with optimization method comparison.
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
        fig, axes = plt.subplots(4, 1, figsize=(12, 18))
        
        # Color-blind friendly palette
        cb_colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        
        # Plot 1: Trajectory comparison (same as original)
        axes[0].plot(time_array, desired_traj, color='#000000', linewidth=2, 
                    label='Desired trajectory', zorder=10)
        axes[0].plot(time_array, actual_traj, color='#E69F00', linestyle='--', linewidth=2, 
                    label='Actual trajectory (selected)', zorder=9)
        
        # Add phase boundaries
        for i, event_time in enumerate(self.event_times[1:-1], 1):
            axes[0].axvline(x=event_time, color='red', linestyle=':', alpha=0.7, 
                          label=f'Phase {i} boundary' if i == 1 else "")
        
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)', fontsize=12)
        axes[0].set_title(f'Trajectory Tracking Performance ({self.optimization_method})', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Frequency evolution within each phase
        for phase_idx, freq_evolution in enumerate(self.frequency_history):
            iterations = range(len(freq_evolution))
            axes[1].plot(iterations, freq_evolution, 'o-', 
                        label=f'Phase {phase_idx + 1}', linewidth=2, markersize=4)
        
        axes[1].set_xlabel('Optimization Iteration')
        axes[1].set_ylabel('EES Frequency (Hz)')
        axes[1].set_title('Frequency Evolution During Optimization')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Cost evolution
        if len(self.cost_history) > 0:
            axes[2].plot(range(len(self.cost_history)), self.cost_history, 'o-', 
                        color='#0072B2', linewidth=2, markersize=4)
            axes[2].set_xlabel('Total Optimization Steps')
            axes[2].set_ylabel('Amplitude Error')
            axes[2].set_title('Optimization Cost Evolution')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
        
        # Plot 4: Final frequency per phase
        freq_time_array = [time_segment[0] for time_segment in self.time_history]
        axes[3].plot(freq_time_array, frequencies, color='#009E73', linewidth=2, marker='o', markersize=6)
        for event_time in self.event_times[1:-1]:
            axes[3].axvline(x=event_time, color='red', linestyle=':', alpha=0.7)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Final EES Frequency (Hz)')
        axes[3].set_title('Final Optimized Frequency Across Gait Phases')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'ees_control_results_{self.optimization_method}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Control results plot saved to {base_output_path}")
        
        plt.show()

# Example usage comparing different optimization methods
def compare_optimization_methods(biological_system, methods=['adaptive_pid', 'proportional', 'golden_section']):
    """
    Compare different optimization methods.
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing {method.upper()} optimization")
        print(f"{'='*50}")
        
        controller = EESController(biological_system, optimization_method=method)
        trajectory_history, desired_trajectory_history, ees_params_history, time_history = controller.run()
        
        results[method] = {
            'controller': controller,
            'total_cost': sum(controller.cost_history),
            'total_iterations': len(controller.cost_history),
            'final_errors': [min([data['cost'] for data in phase_data]) 
                           for phase_data in controller.optimization_trajectories]
        }
        
        print(f"Total cost: {results[method]['total_cost']:.3f}")
        print(f"Total iterations: {results[method]['total_iterations']}")
        print(f"Final errors per phase: {results[method]['final_errors']}")
    
    return results
