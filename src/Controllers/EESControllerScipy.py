from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize, differential_evolution, basinhopping
from scipy.optimize import OptimizeResult
from ..BiologicalSystems.BiologicalSystem import BiologicalSystem

class EESControllerSciPy:
    """
    Model Predictive Controller for EES parameters using SciPy optimization functions.
    
    This controller leverages established numerical optimization methods for robust convergence.
    """
    
    def __init__(self, biological_system, ees_intensity=0.5, ees_frequency_guess=50*hertz, 
                 optimization_method='brent'):
        """
        Initialize the EES controller.
        
        Parameters:
        -----------
        biological_system : BiologicalSystem
            The biological system to control
        optimization_method : str
            SciPy optimization method: 'brent', 'bounded', 'nelder_mead', 'differential_evolution', 'basinhopping'
        """
        self.biological_system = biological_system.clone_with()
        self.ees_intensity = ees_intensity
        self.ees_frequency_guess = ees_frequency_guess
        self.optimization_method = optimization_method
        
        # Load trajectory data (same as original)
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
        
        # Optimization parameters
        self.min_frequency = 10 * hertz
        self.max_frequency = 100 * hertz
        self.amplitude_tolerance = 0.1
        
        # Storage for results
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        self.optimization_trajectories = []
        self.optimization_results = []  # Store scipy optimization results
        
        # For tracking function evaluations
        self.current_phase_data = []
        self.current_prediction_time = None
        self.current_site = None
        self.current_update_iterations = None
        self.current_time_step = None
        self.desired_amplitude = None

    def _get_trajectory_amplitude(self, trajectory):
        """Calculate the amplitude (peak-to-peak) of a trajectory."""
        return np.max(trajectory) - np.min(trajectory)
    
    def _objective_function(self, freq_hz):
        """
        Objective function for scipy optimization.
        
        Parameters:
        -----------
        freq_hz : float or array-like
            Frequency in Hz (scalar for 1D optimization, array for multi-dimensional)
            
        Returns:
        --------
        float
            Amplitude error to minimize
        """
        # Handle both scalar and array inputs
        if isinstance(freq_hz, (list, np.ndarray)):
            freq = freq_hz[0] * hertz  # For multi-dimensional optimizers
        else:
            freq = freq_hz * hertz
        
        # Clamp frequency to valid range
        freq = max(self.min_frequency, min(self.max_frequency, freq))
        
        # Run simulation
        ees_params = {
            'frequency': freq,
            'intensity': self.ees_intensity,
            'site': self.current_site
        }
        
        try:
            spikes, time_series = self.biological_system.run_simulation(
                n_iterations=self.current_update_iterations,
                time_step=self.current_time_step,
                ees_stimulation_params=ees_params
            )
            
            joint_col = f"Joint_{self.biological_system.associated_joint}"
            actual_trajectory = time_series[joint_col].values
            actual_amplitude = self._get_trajectory_amplitude(actual_trajectory)
            
            amplitude_error = abs(actual_amplitude - self.desired_amplitude)
            
            # Store optimization data
            optimization_data = {
                'trajectory': actual_trajectory.copy(),
                'time': self.current_prediction_time.copy(),
                'params': ees_params.copy(),
                'cost': amplitude_error,
                'amplitude': actual_amplitude
            }
            self.current_phase_data.append(optimization_data)
            self.cost_history.append(amplitude_error)
            
            return amplitude_error
            
        except Exception as e:
            print(f"Simulation failed for freq={freq/hertz:.1f}Hz: {e}")
            return 1e6  # Return large penalty for failed simulations
    
    def _optimize_frequency_scipy(self, desired_amplitude, site, update_iterations, 
                                time_step, prediction_time):
        """
        Optimize frequency using specified scipy method.
        """
        # Set up current optimization context
        self.current_phase_data = []
        self.current_prediction_time = prediction_time
        self.current_site = site
        self.current_update_iterations = update_iterations
        self.current_time_step = time_step
        self.desired_amplitude = desired_amplitude
        
        freq_min = self.min_frequency / hertz
        freq_max = self.max_frequency / hertz
        freq_guess = self.ees_frequency_guess / hertz
        
        print(f"    Optimizing with {self.optimization_method}...")
        
        if self.optimization_method == 'brent':
            # Brent's method - good for unimodal functions
            result = minimize_scalar(
                self._objective_function,
                bounds=(freq_min, freq_max),
                method='bounded',
                options={'xatol': 0.5}  # Tolerance in Hz
            )
            
        elif self.optimization_method == 'golden':
            # Golden section search
            result = minimize_scalar(
                self._objective_function,
                bounds=(freq_min, freq_max),
                method='golden',
                options={'xtol': 0.5}
            )
            
        elif self.optimization_method == 'nelder_mead':
            # Nelder-Mead simplex - robust but can be slow
            result = minimize(
                self._objective_function,
                x0=[freq_guess],
                method='Nelder-Mead',
                options={'xatol': 0.5, 'fatol': self.amplitude_tolerance}
            )
            
        elif self.optimization_method == 'powell':
            # Powell's method - good for optimization without derivatives
            result = minimize(
                self._objective_function,
                x0=[freq_guess],
                method='Powell',
                bounds=[(freq_min, freq_max)],
                options={'xtol': 0.5, 'ftol': self.amplitude_tolerance}
            )
            
        elif self.optimization_method == 'differential_evolution':
            # Differential Evolution - global optimization, handles multimodal functions
            result = differential_evolution(
                self._objective_function,
                bounds=[(freq_min, freq_max)],
                seed=42,  # For reproducibility
                atol=self.amplitude_tolerance,
                tol=0.01,
                maxiter=20,
                popsize=5  # Small population for speed
            )
            
        elif self.optimization_method == 'basinhopping':
            # Basin hopping - global optimization with local refinement
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': [(freq_min, freq_max)]
            }
            result = basinhopping(
                self._objective_function,
                x0=[freq_guess],
                minimizer_kwargs=minimizer_kwargs,
                niter=10,
                T=5.0,  # Temperature for acceptance
                stepsize=5.0  # Step size in Hz
            )
            
        elif self.optimization_method == 'ternary_search':
            # Custom ternary search implementation
            result = self._ternary_search(freq_min, freq_max)
            
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")
        
        # Store optimization result
        self.optimization_results.append(result)
        
        # Print results
        if hasattr(result, 'x') and isinstance(result.x, (list, np.ndarray)):
            optimal_freq = result.x[0]
        else:
            optimal_freq = result.x if hasattr(result, 'x') else result
            
        optimal_error = result.fun if hasattr(result, 'fun') else self._objective_function(optimal_freq)
        n_evaluations = result.nfev if hasattr(result, 'nfev') else len(self.current_phase_data)
        
        print(f"    Converged: freq={optimal_freq:.1f}Hz, error={optimal_error:.3f}, "
              f"evaluations={n_evaluations}")
        
        return self.current_phase_data
    
    def _ternary_search(self, left, right, epsilon=0.5):
        """
        Custom ternary search implementation for comparison.
        """
        class TernaryResult:
            def __init__(self, x, fun, nfev):
                self.x = x
                self.fun = fun
                self.nfev = nfev
        
        nfev = 0
        while abs(right - left) > epsilon:
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            
            f1 = self._objective_function(m1)
            f2 = self._objective_function(m2)
            nfev += 2
            
            if f1 > f2:
                left = m1
            else:
                right = m2
        
        optimal_x = (left + right) / 2
        optimal_fun = self._objective_function(optimal_x)
        nfev += 1
        
        return TernaryResult(optimal_x, optimal_fun, nfev)

    def run(self, time_step=0.1*ms):
        """
        Run the complete control simulation with SciPy optimization.
        """
        current_time = 0.0
        phase_idx = 0
        
        print(f"Starting EES control simulation with SciPy {self.optimization_method} optimization...")
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
            
            print(f"    Target amplitude: {desired_amplitude:.3f}")
            
            # Optimize using SciPy
            phase_optimization_data = self._optimize_frequency_scipy(
                desired_amplitude, site, update_iterations, time_step, prediction_time)
            
            # Store optimization data for this phase
            self.optimization_trajectories.append(phase_optimization_data)
            
            # Update biological system state
            self.biological_system.update_state()
            
            # Store results for this phase (best result)
            best_result = min(phase_optimization_data, key=lambda x: x['cost'])
            self.trajectory_history.append(best_result['trajectory'])
            self.ees_params_history.append(best_result['params'])
            
            current_time = next_event_time   
            phase_idx += 1
        
        print(f"\nControl simulation completed!")
        
        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)

    def plot(self, base_output_path=None):
        """
        Plot the control results with SciPy optimization details.
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
        
        # Plot 1: Trajectory comparison
        axes[0].plot(time_array, desired_traj, color='#000000', linewidth=2, 
                    label='Desired trajectory', zorder=10)
        axes[0].plot(time_array, actual_traj, color='#E69F00', linestyle='--', linewidth=2, 
                    label='Actual trajectory', zorder=9)
        
        # Add phase boundaries
        for i, event_time in enumerate(self.event_times[1:-1], 1):
            axes[0].axvline(x=event_time, color='red', linestyle=':', alpha=0.7, 
                          label=f'Phase {i} boundary' if i == 1 else "")
        
        axes[0].set_xlabel('Time (s)', fontsize=12)
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)', fontsize=12)
        axes[0].set_title(f'Trajectory Tracking Performance (SciPy {self.optimization_method})', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Optimization convergence per phase
        for phase_idx, phase_data in enumerate(self.optimization_trajectories):
            costs = [data['cost'] for data in phase_data]
            iterations = range(len(costs))
            axes[1].semilogy(iterations, costs, 'o-', 
                           label=f'Phase {phase_idx + 1}', linewidth=2, markersize=4)
        
        axes[1].set_xlabel('Function Evaluations')
        axes[1].set_ylabel('Amplitude Error (log scale)')
        axes[1].set_title('Optimization Convergence per Phase')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=self.amplitude_tolerance, color='red', linestyle='--', 
                       alpha=0.7, label='Tolerance')
        
        # Plot 3: Function evaluations and final errors
        n_evaluations = [len(phase_data) for phase_data in self.optimization_trajectories]
        final_errors = [min(data['cost'] for data in phase_data) for phase_data in self.optimization_trajectories]
        
        ax3_twin = axes[2].twinx()
        bars1 = axes[2].bar([f'Phase {i+1}' for i in range(len(n_evaluations))], 
                           n_evaluations, alpha=0.7, color='#56B4E9', label='Function Evaluations')
        bars2 = ax3_twin.bar([f'Phase {i+1}' for i in range(len(final_errors))], 
                            final_errors, alpha=0.7, color='#E69F00', width=0.6, label='Final Error')
        
        axes[2].set_ylabel('Function Evaluations', color='#56B4E9')
        ax3_twin.set_ylabel('Final Amplitude Error', color='#E69F00')
        axes[2].set_title('Optimization Efficiency per Phase')
        
        # Combined legend
        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Plot 4: Final frequency per phase
        freq_time_array = [time_segment[0] for time_segment in self.time_history]
        axes[3].plot(freq_time_array, frequencies, color='#009E73', linewidth=2, marker='o', markersize=6)
        for event_time in self.event_times[1:-1]:
            axes[3].axvline(x=event_time, color='red', linestyle=':', alpha=0.7)
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Optimized EES Frequency (Hz)')
        axes[3].set_title('Final Optimized Frequency Across Gait Phases')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if path provided
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'ees_scipy_results_{self.optimization_method}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Control results plot saved to {base_output_path}")
        
        plt.show()
        
        # Print optimization summary
        self.print_optimization_summary()
    
    def print_optimization_summary(self):
        """Print a summary of the optimization results."""
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION SUMMARY ({self.optimization_method.upper()})")
        print(f"{'='*60}")
        
        total_cost = sum(self.cost_history)
        total_evaluations = len(self.cost_history)
        final_errors = [min(data['cost'] for data in phase_data) for phase_data in self.optimization_trajectories]
        avg_final_error = np.mean(final_errors)
        
        print(f"Total function evaluations: {total_evaluations}")
        print(f"Total cumulative cost: {total_cost:.3f}")
        print(f"Average final error per phase: {avg_final_error:.3f}")
        print(f"Tolerance: {self.amplitude_tolerance}")
        print(f"Phases achieving tolerance: {sum(1 for err in final_errors if err <= self.amplitude_tolerance)}/{len(final_errors)}")
        
        print(f"\nPer-phase results:")
        for i, (phase_data, result) in enumerate(zip(self.optimization_trajectories, self.optimization_results)):
            n_evals = len(phase_data)
            final_error = min(data['cost'] for data in phase_data)
            
            if hasattr(result, 'x') and isinstance(result.x, (list, np.ndarray)):
                optimal_freq = result.x[0]
            else:
                optimal_freq = result.x if hasattr(result, 'x') else 'N/A'
            
            success = "✓" if final_error <= self.amplitude_tolerance else "✗"
            print(f"  Phase {i+1}: {n_evals:2d} evals, error={final_error:.3f}, freq={optimal_freq:.1f}Hz {success}")

# Example usage and comparison
def compare_scipy_methods(biological_system, methods=['brent', 'differential_evolution', 'nelder_mead', 'powell']):
    """
    Compare different SciPy optimization methods.
    """
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing SciPy {method.upper()} optimization")
        print(f"{'='*50}")
        
        try:
            controller = EESControllerSciPy(biological_system, optimization_method=method)
            trajectory_history, desired_trajectory_history, ees_params_history, time_history = controller.run()
            
            results[method] = {
                'controller': controller,
                'total_cost': sum(controller.cost_history),
                'total_evaluations': len(controller.cost_history),
                'final_errors': [min([data['cost'] for data in phase_data]) 
                               for phase_data in controller.optimization_trajectories],
                'success': True
            }
            
            controller.print_optimization_summary()
            
        except Exception as e:
            print(f"Method {method} failed: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Evaluations':<12} {'Total Cost':<12} {'Avg Error':<12} {'Success':<8}")
    print(f"{'-'*70}")
    
    for method, result in results.items():
        if result.get('success', False):
            avg_error = np.mean(result['final_errors'])
            print(f"{method:<20} {result['total_evaluations']:<12} {result['total_cost']:<12.3f} {avg_error:<12.3f} {'Yes':<8}")
        else:
            print(f"{method:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'No':<8}")
    
    return results
