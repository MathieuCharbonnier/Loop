from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.optimize import OptimizeResult

class EESControllerSciPy:
    """
    Simplified EES Controller using only RMS error with robust convergence handling.
    """
    
    def __init__(self, biological_system, ees_intensity=0.5, ees_frequency_guess=50*hertz, 
                 optimization_method='brent'):
        """
        Initialize the simplified EES controller.
        
        Parameters:
        -----------
        biological_system : BiologicalSystem
            The biological system to control
        optimization_method : str
            SciPy optimization method ('brent', 'nelder_mead', 'differential_evolution')
        """
        self.biological_system = biological_system.clone_with()
        self.ees_intensity = ees_intensity
        self.ees_frequency_guess = ees_frequency_guess
        self.optimization_method = optimization_method
        
        # Convergence parameters
        self.max_iterations = 3
        self.rms_tolerance = 10.0  # RMS error tolerance
        
        # Load trajectory data
        self._load_trajectory_data()
        
        # Storage for results
        self._initialize_storage()
        

    def _load_trajectory_data(self):
        """Load and process trajectory data."""
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
        self.desired_trajectory_function = interp1d(
            self.time, self.desired_trajectory, kind='cubic', fill_value="extrapolate"
        )
        
        max_dorsi_idx = np.argmax(self.desired_trajectory)
        min_plantar_idx = np.argmin(self.desired_trajectory)
        
        self.event_times = np.array([0, self.time[max_dorsi_idx], self.time[min_plantar_idx], self.total_time])
        self.site_stimulation = ['L3', 'S2', 'L3']
        
        # Frequency bounds
        self.min_frequency = 10 * hertz
        self.max_frequency = 100 * hertz

    def _initialize_storage(self):
        """Initialize storage for results."""
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        self.cost_history = []
        self.optimization_trajectories = []
        self.optimization_results = []
        
        # Enhanced storage for convergence information
        self.convergence_info = []
        self.intermediate_trajectories = []  # Store all intermediate trajectories
        
        # Current optimization context
        self.current_phase_data = []
        self.current_prediction_time = None
        self.current_site = None
        self.current_update_iterations = None
        self.current_time_step = None
        self.current_desired_trajectory = None
        self.current_phase_idx = None

    def _compute_rms_error(self, actual_trajectory, desired_trajectory):
        """
        Compute RMS error between actual and desired trajectories.

        """
        squared_errors = (actual_trajectory - desired_trajectory) ** 2
        rms_error = np.sqrt(np.mean(squared_errors))

        return rms_error

    def _objective_function(self, freq_hz):
        
        # Clamp frequency to valid range
        freq = max(self.min_frequency/hertz, min(self.max_frequency/hertz, freq_hz))
        
        # Run simulation
        ees_params = {
            'frequency': freq * hertz,
            'intensity': self.ees_intensity,
            'site': self.current_site
        }
        spikes, time_series = self.biological_system.run_simulation(
                n_iterations=self.current_update_iterations,
                time_step=self.time_step,
                ees_stimulation_params=ees_params
        )
            
        joint_col = f"Joint_{self.biological_system.associated_joint}"
        actual_trajectory = time_series[joint_col].values
            
        # Compute RMS error
        rms_error = self._compute_rms_error(actual_trajectory, self.current_desired_trajectory)
            
        # Store optimization data
        optimization_data = {
            'trajectory': actual_trajectory.copy(),
            'time': self.current_prediction_time.copy(),
            'params': ees_params.copy(),
            'rms_error': rms_error,
            'frequency_hz': freq_hz,
            'phase_idx': self.current_phase_idx,
            'evaluation_count': len(self.current_phase_data) + 1
        }
        self.current_phase_data.append(optimization_data)
        self.cost_history.append(rms_error)
        
        # Store intermediate trajectory for plotting
        self.intermediate_trajectories.append({
            'phase_idx': self.current_phase_idx,
            'evaluation': len(self.current_phase_data),
            'trajectory': actual_trajectory.copy(),
            'time': self.current_prediction_time.copy(),
            'frequency': freq_hz,
            'rms_error': rms_error
        })
            
        return rms_error
  

    def _optimize_frequency(self):
        """
        Optimize frequency with fallback handling and enhanced convergence tracking.
        """
        # Set up optimization context
        self.current_phase_data = []
        
        freq_min = self.min_frequency / hertz
        freq_max = self.max_frequency / hertz
        freq_guess = self.ees_frequency_guess / hertz
        
        print(f"    Optimizing with {self.optimization_method} (RMS error)...")
        
        # Track optimization start time and initial error
        start_function_evals = len(self.cost_history)
        
        if self.optimization_method == 'brent':
            result = minimize_scalar(
                self._objective_function,
                bounds=(freq_min, freq_max),
                method='bounded',
                options={'xatol': 0.5, 'maxfun': self.max_iterations}
            )
                
        elif self.optimization_method == 'differential_evolution':
            result = differential_evolution(
                self._objective_function,
                bounds=[(freq_min, freq_max)],
                seed=42,
                maxiter=min(20, self.max_iterations//5),
                popsize=5,
                atol=self.rms_tolerance
            )
                
        elif self.optimization_method == 'nelder_mead':
            result = minimize(
                self._objective_function,
                x0=[freq_guess],
                method='Nelder-Mead',
                options={
                    'maxiter': self.max_iterations,
                    'xatol': 0.5,
                    'fatol': self.rms_tolerance
                }
            )
        
        # Calculate convergence information
        end_function_evals = len(self.cost_history)
        total_function_evals = end_function_evals - start_function_evals
        
        # Get RMS errors for this phase
        phase_rms_errors = [data['rms_error'] for data in self.current_phase_data]
        
        convergence_data = {
            'phase_idx': self.current_phase_idx,
            'optimization_method': self.optimization_method,
            'function_evaluations': total_function_evals,
            'initial_rms_error': phase_rms_errors[0] if phase_rms_errors else None,
            'final_rms_error': phase_rms_errors[-1] if phase_rms_errors else None,
            'best_rms_error': min(phase_rms_errors) if phase_rms_errors else None,
            'rms_improvement': (phase_rms_errors[0] - min(phase_rms_errors)) if phase_rms_errors else 0,
            'converged': result.success if hasattr(result, 'success') else True,
            'convergence_reason': result.message if hasattr(result, 'message') else 'Completed',
            'optimal_frequency': result.x if hasattr(result, 'x') else result.x[0] if hasattr(result, 'x') else None,
            'rms_tolerance_met': min(phase_rms_errors) <= self.rms_tolerance if phase_rms_errors else False
        }
        

        convergence_data['optimal_frequency'] = result.x
        
        self.convergence_info.append(convergence_data)
        
        # Store result
        self.optimization_results.append(result)
        
        # Print convergence information
        print(f"    Convergence: {convergence_data['function_evaluations']} evals, "
              f"RMS: {convergence_data['initial_rms_error']:.3f} → {convergence_data['final_rms_error']:.3f}, "
              f"Best: {convergence_data['best_rms_error']:.3f}")
        
        return self.current_phase_data

    def run(self, time_step=0.1*ms):
        """
        Run the complete control simulation.
        """
        current_time = 0.0
        phase_idx = 0
        self.time_step = time_step

        print(f"Starting EES control simulation with RMS error optimization...")
        print(f"Optimization method: {self.optimization_method}")
        print(f"RMS tolerance: {self.rms_tolerance}")
        
        while current_time < self.total_time and phase_idx < len(self.site_stimulation):
            next_event_time = self.event_times[phase_idx + 1] if phase_idx + 1 < len(self.event_times) else self.total_time
            phase_duration = next_event_time - current_time
            
            site = self.site_stimulation[phase_idx]
            update_iterations = max(1, int(phase_duration / (self.biological_system.reaction_time / second)))
            
            print(f"\nPhase {phase_idx + 1}: {current_time:.3f}s to {next_event_time:.3f}s")
            print(f"Duration: {phase_duration:.3f}s, Iterations: {update_iterations}, Site: {site}")
            
            prediction_time = np.linspace(self.event_times[phase_idx], self.event_times[phase_idx+1], update_iterations)
            desired_trajectory_segment = self.desired_trajectory_function(prediction_time)

            self.current_prediction_time = prediction_time
            self.current_desired_trajectory = desired_trajectory_segment
            self.current_site = site
            self.current_update_iterations = update_iterations
            self.current_phase_idx = phase_idx
            self.time_history.append(prediction_time)
            self.desired_trajectory_history.append(desired_trajectory_segment)
 
        
            # Optimize frequency
            phase_optimization_data = self._optimize_frequency()
            
            self.optimization_trajectories.append(phase_optimization_data)
            
            # Update biological system state
            self.biological_system.update_system_state()
            
            # Store best result for this phase
            if phase_optimization_data:
                best_result = min(phase_optimization_data, key=lambda x: x['rms_error'])
                self.trajectory_history.append(best_result['trajectory'])
                self.ees_params_history.append(best_result['params'])
            
            current_time = next_event_time   
            phase_idx += 1

        
        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)

    def print_convergence_summary(self):
        """Print a detailed convergence summary."""
        print("\n" + "="*80)
        print("CONVERGENCE SUMMARY")
        print("="*80)
        
        for conv_info in self.convergence_info:
            print(f"\nPhase {conv_info['phase_idx'] + 1}:")
            print(f"  Method: {conv_info['optimization_method']}")
            print(f"  Function Evaluations: {conv_info['function_evaluations']}")
            print(f"  RMS Error: {conv_info['initial_rms_error']:.4f} → {conv_info['final_rms_error']:.4f}")
            print(f"  Best RMS Error: {conv_info['best_rms_error']:.4f}")
            print(f"  Improvement: {conv_info['rms_improvement']:.4f}")
            print(f"  Optimal Frequency: {conv_info['optimal_frequency']:.2f} Hz")
            print(f"  Tolerance Met: {'Yes' if conv_info['rms_tolerance_met'] else 'No'}")
            print(f"  Converged: {'Yes' if conv_info['converged'] else 'No'}")
            print(f"  Status: {conv_info['convergence_reason']}")
        
        # Overall statistics
        total_evals = sum(conv['function_evaluations'] for conv in self.convergence_info)
        avg_improvement = np.mean([conv['rms_improvement'] for conv in self.convergence_info])
        tolerance_met_count = sum(1 for conv in self.convergence_info if conv['rms_tolerance_met'])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Function Evaluations: {total_evals}")
        print(f"  Average RMS Improvement: {avg_improvement:.4f}")
        print(f"  Phases Meeting Tolerance: {tolerance_met_count}/{len(self.convergence_info)}")
        print("="*80)

    def plot(self, base_output_path=None):
        """
        Plot the control results focusing on RMS error performance with intermediate trajectories.
        """
        if not self.trajectory_history:
            raise ValueError("No simulation results to plot. Run simulation first.")
        
        # Prepare data
        time_array = np.concatenate(self.time_history)
        actual_traj = np.concatenate(self.trajectory_history)
        desired_traj = np.concatenate(self.desired_trajectory_history)
        
        frequencies = [params['frequency'] / hertz for params in self.ees_params_history]
        
        # Create plots - now with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Plot 1: Trajectory comparison
        axes[0].plot(time_array, desired_traj, 'k-', linewidth=2, label='Desired trajectory')
        axes[0].plot(time_array, actual_traj, 'r--', linewidth=2, label='Actual trajectory')
        
        # Add phase boundaries
        for i, event_time in enumerate(self.event_times[1:-1], 1):
            axes[0].axvline(x=event_time, color='gray', linestyle=':', alpha=0.7)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
        axes[0].set_title(f'RMS-Optimized Trajectory Tracking ({self.optimization_method})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RMS error convergence per phase
        for phase_idx, phase_data in enumerate(self.optimization_trajectories):
            rms_errors = [data['rms_error'] for data in phase_data]
            iterations = range(len(rms_errors))
            axes[1].semilogy(iterations, rms_errors, 'o-', 
                           label=f'Phase {phase_idx + 1}', linewidth=2, markersize=4)
        
        axes[1].axhline(y=self.rms_tolerance, color='red', linestyle='--', 
                       alpha=0.7, label='RMS Tolerance')
        axes[1].set_xlabel('Function Evaluations')
        axes[1].set_ylabel('RMS Error (log scale)')
        axes[1].set_title('RMS Error Convergence per Phase')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Optimized frequencies
        freq_time_array = [time_segment[0] for time_segment in self.time_history]
        axes[2].plot(freq_time_array, frequencies, 'go-', linewidth=2, markersize=8)
        for event_time in self.event_times[1:-1]:
            axes[2].axvline(x=event_time, color='gray', linestyle=':', alpha=0.7)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Optimized EES Frequency (Hz)')
        axes[2].set_title('RMS-Optimized Frequency Profile')
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Intermediate trajectories during optimization
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.convergence_info)))
        
        for phase_idx in range(len(self.convergence_info)):
            # Get intermediate trajectories for this phase
            phase_intermediates = [traj for traj in self.intermediate_trajectories 
                                 if traj['phase_idx'] == phase_idx]
            
            if phase_intermediates:
                # Plot desired trajectory for this phase
                phase_time = phase_intermediates[0]['time']
                phase_desired = self.desired_trajectory_function(phase_time)
                axes[3].plot(phase_time, phase_desired, 'k-', linewidth=2, alpha=0.7)
                
                # Plot intermediate trajectories with increasing opacity
                n_intermediates = len(phase_intermediates)
                for i, intermediate in enumerate(phase_intermediates):
                    alpha = 0.3 + 0.7 * (i / max(1, n_intermediates - 1))  # Increasing opacity
                    linestyle = '-' if i == n_intermediates - 1 else '--'  # Solid line for final
                    linewidth = 2 if i == n_intermediates - 1 else 1
                    
                    label = None
                    if i == 0:
                        label = f'Phase {phase_idx + 1} iterations'
                    elif i == n_intermediates - 1:
                        label = f'Phase {phase_idx + 1} final'
                    
                    axes[3].plot(intermediate['time'], intermediate['trajectory'], 
                               color=colors[phase_idx], alpha=alpha, 
                               linestyle=linestyle, linewidth=linewidth, label=label)
        
        # Add phase boundaries to intermediate plot
        for event_time in self.event_times[1:-1]:
            axes[3].axvline(x=event_time, color='gray', linestyle=':', alpha=0.7)
        
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
        axes[3].set_title('Intermediate Optimization Trajectories')
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'rms_ees_results_{self.optimization_method}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {base_output_path}")
        
        plt.show()
        
        # Print convergence summary
        self.print_convergence_summary()

    def plot_intermediate_trajectories_detailed(self, base_output_path=None):
        """
        Create a detailed plot showing intermediate trajectories for each phase separately.
        """
        if not self.intermediate_trajectories:
            print("No intermediate trajectories to plot.")
            return
        
        n_phases = len(self.convergence_info)
        fig, axes = plt.subplots(n_phases, 1, figsize=(12, 4*n_phases))
        
        if n_phases == 1:
            axes = [axes]
        
        for phase_idx in range(n_phases):
            ax = axes[phase_idx]
            
            # Get intermediate trajectories for this phase
            phase_intermediates = [traj for traj in self.intermediate_trajectories 
                                 if traj['phase_idx'] == phase_idx]
            
            if not phase_intermediates:
                continue
            
            # Plot desired trajectory
            phase_time = phase_intermediates[0]['time']
            phase_desired = self.desired_trajectory_function(phase_time)
            ax.plot(phase_time, phase_desired, 'k-', linewidth=3, label='Desired', alpha=0.8)
            
            # Color map for iterations
            n_intermediates = len(phase_intermediates)
            colors = plt.cm.plasma(np.linspace(0, 1, n_intermediates))
            
            # Plot each intermediate trajectory
            for i, intermediate in enumerate(phase_intermediates):
                alpha = 0.4 + 0.6 * (i / max(1, n_intermediates - 1))
                linewidth = 1.5 if i == n_intermediates - 1 else 1
                
                label = f"Eval {i+1} (f={intermediate['frequency']:.1f}Hz, RMS={intermediate['rms_error']:.3f})"
                if i == n_intermediates - 1:
                    label = f"FINAL: {label}"
                
                ax.plot(intermediate['time'], intermediate['trajectory'], 
                       color=colors[i], alpha=alpha, linewidth=linewidth, 
                       label=label if i < 5 or i == n_intermediates - 1 else None)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
            ax.set_title(f'Phase {phase_idx + 1} - Optimization Trajectory Evolution\n'
                        f'({self.convergence_info[phase_idx]["function_evaluations"]} evaluations, '
                        f'RMS: {self.convergence_info[phase_idx]["initial_rms_error"]:.3f} → '
                        f'{self.convergence_info[phase_idx]["best_rms_error"]:.3f})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'intermediate_trajectories_{self.optimization_method}.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Intermediate trajectories plot saved to {base_output_path}")
        
        plt.show()
