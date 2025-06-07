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
    Enhanced EES Controller optimizing frequency, gait events, and stimulation sites.
    """
    
    def __init__(self, biological_system, ees_intensity=0.5,
        sites=["L3", "S2", "L3"], gait_times=[0, 0.39, 0.56, 1]*second,
        max_iterations=4, rms_tolerance=10):
        """
        Initialize the enhanced EES controller.
        """
        self.biological_system = biological_system.clone_with()
        self.ees_intensity = ees_intensity    
        self.initial_event_times = gait_times
        self.initial_sites = sites
        
        # Available stimulation sites
        self.available_sites = ['L3', 'L4', 'L5', 'S1', 'S2']
        
        # Convergence parameters
        self.max_iterations = max_iterations
        self.rms_tolerance = rms_tolerance
        
        # Load trajectory data
        self._load_trajectory_data()
        
        # Storage for results
        self._initialize_storage()
        
        # Optimization parameters
        self.optimize_events = True
        self.optimize_sites = True

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
        self.total_time = self.time[-1]*second
        self.desired_trajectory_function = interp1d(
            self.time, self.desired_trajectory, kind='cubic', fill_value="extrapolate"
        )
        
        # Frequency bounds
        self.min_frequency = 10 * hertz
        self.max_frequency = 100 * hertz

    def _initialize_storage(self):
        """Initialize storage for results."""
        # Main results for plotting
        self.trajectory_history = []
        self.desired_trajectory_history = []
        self.ees_params_history = []
        self.time_history = []
        
        # Convergence tracking for plotting
        self.convergence_info = []
        self.intermediate_trajectories = []
        
        # Current optimization context
        self.current_phase_data = []
        self.current_prediction_time = None
        self.current_desired_trajectory = None
        self.current_update_iterations = None
        self.current_time_step = None
        self.current_phase_idx = None
        
        # Store initial system state
        self.initial_system_state = self.biological_system.get_system_state()
        
        # Optimized parameters
        self.optimized_event_times = None
        self.optimized_sites = None

    def _compute_rms_error(self, actual_trajectory, desired_trajectory):
        """Compute RMS error between actual and desired trajectories."""
        squared_errors = (actual_trajectory - desired_trajectory) ** 2
        rms_error = np.sqrt(np.mean(squared_errors))
        return rms_error

    def _global_objective_function(self, params):
        """
        Global objective function for optimizing event times and sites.
        params: [event_time_1, event_time_2, ..., site_idx_0, site_idx_1, ...]
        """
        n_events = len(self.initial_event_times) - 2  # Exclude first and last (fixed)
        n_sites = len(self.initial_sites)
        
        # Extract event times (keep first=0 and last=total_time fixed)
        event_times = [0]
        for i in range(n_events):
            event_times.append(max(0.1, min(self.total_time/second - 0.1, params[i])))
        event_times.append(self.total_time/second)
        event_times = sorted(event_times)  # Ensure monotonic
        event_times = [t * second for t in event_times]
        
        # Extract sites
        sites = []
        for i in range(n_sites):
            site_idx = int(np.clip(params[n_events + i], 0, len(self.available_sites) - 1))
            sites.append(self.available_sites[site_idx])
        
        # Reset system state
        self.biological_system.set_system_state(self.initial_system_state)
        
        # Run full simulation with these parameters
        total_rms_error = 0
        n_phases = 0
        
        for phase_idx in range(len(sites)):
            if phase_idx + 1 >= len(event_times):
                break
                
            phase_duration = event_times[phase_idx + 1] - event_times[phase_idx]
            update_iterations = max(1, int(phase_duration / (self.biological_system.reaction_time / second)))
            
            prediction_time = np.linspace(event_times[phase_idx], event_times[phase_idx+1], 
                                        update_iterations*int(self.biological_system.reaction_time/self.current_time_step))
            desired_trajectory_segment = self.desired_trajectory_function(prediction_time)
            
            # Optimize frequency for this phase
            self.current_prediction_time = prediction_time
            self.current_desired_trajectory = desired_trajectory_segment
            self.current_site = sites[phase_idx]
            self.current_update_iterations = update_iterations
            self.current_phase_idx = phase_idx
            
            freq_min = self.min_frequency / hertz
            freq_max = self.max_frequency / hertz
            
            result = minimize_scalar(
                self._phase_objective_function,
                bounds=(freq_min, freq_max),
                method='bounded',
                options={'xatol': 1.0, 'maxiter': 3}  # Reduced for global optimization
            )
            
            total_rms_error += result.fun
            n_phases += 1
        
        avg_rms_error = total_rms_error / max(1, n_phases)
        return avg_rms_error

    def _phase_objective_function(self, freq_hz):
        """Objective function for frequency optimization within a phase."""
        freq = max(self.min_frequency/hertz, min(self.max_frequency/hertz, freq_hz))
        
        ees_params = {
            'frequency': freq * hertz,
            'intensity': self.ees_intensity,
            'site': self.current_site
        }
        
        spikes, time_series = self.biological_system.run_simulation(
            n_iterations=self.current_update_iterations,
            time_step=self.current_time_step,
            ees_stimulation_params=ees_params
        )
        
        joint_col = f"Joint_{self.biological_system.associated_joint}"
        actual_trajectory = time_series[joint_col].values
        
        rms_error = self._compute_rms_error(actual_trajectory, self.current_desired_trajectory)
        return rms_error

    def _objective_function(self, freq_hz):
        """Original objective function for detailed optimization."""
        freq = max(self.min_frequency/hertz, min(self.max_frequency/hertz, freq_hz))
        
        ees_params = {
            'frequency': freq * hertz,
            'intensity': self.ees_intensity,
            'site': self.current_site
        }
        spikes, time_series = self.biological_system.run_simulation(
                n_iterations=self.current_update_iterations,
                time_step=self.current_time_step,
                ees_params=ees_params
        )
            
        joint_col = f"Joint_{self.biological_system.associated_joint}"
        actual_trajectory = time_series[joint_col].values
            
        rms_error = self._compute_rms_error(actual_trajectory, self.current_desired_trajectory)
        
        optimization_data = {
            'trajectory': actual_trajectory.copy(),
            'time': self.current_prediction_time.copy(),
            'system_state': self.biological_system.get_system_state(),
            'params': ees_params.copy(),
            'rms_error': rms_error,
            'frequency_hz': freq_hz,
            'phase_idx': self.current_phase_idx,
            'evaluation_count': len(self.current_phase_data) + 1
        }
        self.current_phase_data.append(optimization_data)
        
        self.intermediate_trajectories.append({
            'phase_idx': self.current_phase_idx,
            'evaluation': len(self.current_phase_data),
            'trajectory': actual_trajectory.copy(),
            'time': self.current_prediction_time.copy(),
            'frequency': freq_hz,
            'rms_error': rms_error
        })
            
        return rms_error

    def _optimize_global_parameters(self):
        """Optimize event times and stimulation sites globally."""
        print("Optimizing gait events and stimulation sites...")
        
        # Setup bounds for optimization
        n_events = len(self.initial_event_times) - 2  # Exclude first and last
        n_sites = len(self.initial_sites)
        
        bounds = []
        
        # Bounds for event times (between 0.1 and total_time-0.1)
        for i in range(n_events):
            bounds.append((0.1, self.total_time/second - 0.1))
        
        # Bounds for site indices
        for i in range(n_sites):
            bounds.append((0, len(self.available_sites) - 1))
        
        # Initial guess
        initial_params = []
        for i in range(1, len(self.initial_event_times) - 1):
            initial_params.append(self.initial_event_times[i]/second)
        for site in self.initial_sites:
            initial_params.append(self.available_sites.index(site))
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            self._global_objective_function,
            bounds,
            seed=42,
            maxiter=10,  # Limited iterations for practical runtime
            popsize=5,
            disp=True
        )
        
        # Extract optimized parameters
        params = result.x
        
        # Extract optimized event times
        self.optimized_event_times = [0 * second]
        for i in range(n_events):
            self.optimized_event_times.append(params[i] * second)
        self.optimized_event_times.append(self.total_time)
        self.optimized_event_times = sorted(self.optimized_event_times)
        
        # Extract optimized sites
        self.optimized_sites = []
        for i in range(n_sites):
            site_idx = int(np.clip(params[n_events + i], 0, len(self.available_sites) - 1))
            self.optimized_sites.append(self.available_sites[site_idx])
        
        print(f"Original event times: {[t/second for t in self.initial_event_times]}")
        print(f"Optimized event times: {[t/second for t in self.optimized_event_times]}")
        print(f"Original sites: {self.initial_sites}")
        print(f"Optimized sites: {self.optimized_sites}")
        print(f"Global optimization RMS: {result.fun:.3f}")

    def _optimize_frequency(self):
        """Optimize frequency with fallback handling and enhanced convergence tracking."""
        self.current_phase_data = []
        
        freq_min = self.min_frequency / hertz
        freq_max = self.max_frequency / hertz
        
        result = minimize_scalar(
                self._objective_function,
                bounds=(freq_min, freq_max),
                method='bounded',
                options={'xatol': 0.5, 'disp': 3, 'maxiter': self.max_iterations}
            )
                
        phase_rms_errors = [data['rms_error'] for data in self.current_phase_data]
        
        convergence_data = {
            'phase_idx': self.current_phase_idx,
            'function_evaluations': len(self.current_phase_data),
            'initial_rms_error': phase_rms_errors[0] if phase_rms_errors else None,
            'final_rms_error': phase_rms_errors[-1] if phase_rms_errors else None,
            'best_rms_error': min(phase_rms_errors) if phase_rms_errors else None,
            'rms_improvement': (phase_rms_errors[0] - min(phase_rms_errors)) if phase_rms_errors else 0,
            'converged': result.success if hasattr(result, 'success') else True,
            'convergence_reason': result.message if hasattr(result, 'message') else 'Completed',
            'optimal_frequency': result.x,
            'rms_tolerance_met': min(phase_rms_errors) <= self.rms_tolerance if phase_rms_errors else False
        }
        
        self.convergence_info.append(convergence_data)
        
        print(f"    Convergence: {convergence_data['function_evaluations']} evals, "
              f"RMS: {convergence_data['initial_rms_error']:.3f} → {convergence_data['final_rms_error']:.3f}, "
              f"Best: {convergence_data['best_rms_error']:.3f}")
        
        return self.current_phase_data

    def run(self, time_step=0.1*ms):
        """Run the complete control simulation with global optimization."""
        self.current_time_step = time_step
        
        # Step 1: Global optimization of events and sites
        if self.optimize_events or self.optimize_sites:
            self._optimize_global_parameters()
            event_times = self.optimized_event_times
            sites = self.optimized_sites
        else:
            event_times = self.initial_event_times
            sites = self.initial_sites
        
        # Step 2: Detailed simulation with optimized parameters
        self.biological_system.set_system_state(self.initial_system_state)
        current_time = 0.0
        phase_idx = 0
        self.time_step = time_step

        print(f"\nStarting detailed EES control simulation...")
        print(f"RMS tolerance: {self.rms_tolerance}")
        
        while current_time < self.total_time and phase_idx < len(sites):
            next_event_time = event_times[phase_idx + 1] if phase_idx + 1 < len(event_times) else self.total_time
            phase_duration = next_event_time - current_time
            
            site = sites[phase_idx]
            update_iterations = max(1, int(phase_duration / (self.biological_system.reaction_time / second)))
            
            print(f"\nPhase {phase_idx + 1}: {current_time:.3f}s to {next_event_time:.3f}s")
            print(f"Duration: {phase_duration:.3f}s, Iterations: {update_iterations}, Site: {site}")
            
            prediction_time = np.linspace(event_times[phase_idx], event_times[phase_idx+1], 
                                        update_iterations*int(self.biological_system.reaction_time/self.time_step))
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
            
            # Find the best result and set system state
            if phase_optimization_data:
                best_result = min(phase_optimization_data, key=lambda x: x['rms_error'])
                self.biological_system.set_system_state(best_result['system_state'])
                self.trajectory_history.append(best_result['trajectory'])
                self.ees_params_history.append(best_result['params'])
                print(f"    Set system to best state with RMS error: {best_result['rms_error']:.3f}")
            else:
                print("    Warning: No optimization data available for this phase")
            
            current_time = next_event_time   
            phase_idx += 1

        return (self.trajectory_history, self.desired_trajectory_history, 
                self.ees_params_history, self.time_history)

    def plot(self, base_output_path=None):
        """Plot the control results with optimized parameters."""
        if not self.trajectory_history:
            raise ValueError("No simulation results to plot. Run simulation first.")
        
        # Use optimized parameters if available
        event_times = self.optimized_event_times if self.optimized_event_times else self.initial_event_times
        sites = self.optimized_sites if self.optimized_sites else self.initial_sites
        
        # Prepare data
        time_array = np.concatenate(self.time_history)
        actual_traj = np.concatenate(self.trajectory_history)
        desired_traj = np.concatenate(self.desired_trajectory_history)
        
        frequencies = [params['frequency'] / hertz for params in self.ees_params_history]
        
        # Create plots
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # Plot 1: Trajectory comparison
        axes[0].plot(time_array, desired_traj, 'k-', linewidth=2, label='Desired trajectory')
        axes[0].plot(time_array, actual_traj, 'r--', linewidth=2, label='Actual trajectory')
        
        # Add phase boundaries with site labels
        for i, (event_time, site) in enumerate(zip(event_times[1:-1], sites)):
            axes[0].axvline(x=event_time/second, color='gray', linestyle=':', alpha=0.7)
            if i < len(sites):
                axes[0].text(event_time/second, axes[0].get_ylim()[1], f'{site}', 
                           rotation=90, verticalalignment='top', fontsize=8)
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
        axes[0].set_title('Optimized EES Control: Events, Sites & Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RMS error convergence per phase
        for phase_idx in range(len(self.convergence_info)):
            phase_rms_errors = [traj['rms_error'] for traj in self.intermediate_trajectories 
                               if traj['phase_idx'] == phase_idx]
            iterations = range(len(phase_rms_errors))
            axes[1].semilogy(iterations, phase_rms_errors, 'o-', 
                           label=f'Phase {phase_idx + 1} ({sites[phase_idx]})', 
                           linewidth=2, markersize=4)
        
        axes[1].axhline(y=self.rms_tolerance, color='red', linestyle='--', 
                       alpha=0.7, label='RMS Tolerance')
        axes[1].set_xlabel('Function Evaluations')
        axes[1].set_ylabel('RMS Error (log scale)')
        axes[1].set_title('RMS Error Convergence per Phase')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Optimized frequencies with sites
        freq_time_array = [time_segment[0] for time_segment in self.time_history]
        bars = axes[2].bar(freq_time_array, frequencies, width=0.05, alpha=0.7)
        
        # Color bars by site
        site_colors = {'L3': 'red', 'L4': 'orange', 'L5': 'yellow', 'S1': 'green', 'S2': 'blue'}
        for bar, site in zip(bars, sites):
            bar.set_color(site_colors.get(site, 'gray'))
        
        for event_time in event_times[1:-1]:
            axes[2].axvline(x=event_time/second, color='gray', linestyle=':', alpha=0.7)
        
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Optimized EES Frequency (Hz)')
        axes[2].set_title('Frequency Profile by Stimulation Site')
        axes[2].grid(True, alpha=0.3)
        
        # Add legend for sites
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=site_colors.get(site, 'gray'), 
                                       label=site) for site in set(sites)]
        axes[2].legend(handles=legend_elements, title='Sites')
        
        # Plot 4: Event timing comparison
        axes[3].plot([t/second for t in self.initial_event_times], 
                    range(len(self.initial_event_times)), 'ro-', label='Initial Events', markersize=8)
        if self.optimized_event_times:
            axes[3].plot([t/second for t in self.optimized_event_times], 
                        range(len(self.optimized_event_times)), 'go-', label='Optimized Events', markersize=8)
        
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Event Index')
        axes[3].set_title('Gait Event Timing Optimization')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        # Save if requested
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'optimized_ees_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {base_output_path}")
        plt.tight_layout()
        plt.show()
        
        # Print optimization summary
        self.print_optimization_summary()

    def print_optimization_summary(self):
        """Print summary of all optimizations."""
        print("\n" + "="*60)
        print("OPTIMIZATION SUMMARY")
        print("="*60)
        
        if self.optimized_event_times and self.optimized_sites:
            print("Global Parameter Optimization:")
            print(f"  Original event times: {[f'{t/second:.3f}' for t in self.initial_event_times]}")
            print(f"  Optimized event times: {[f'{t/second:.3f}' for t in self.optimized_event_times]}")
            print(f"  Original sites: {self.initial_sites}")
            print(f"  Optimized sites: {self.optimized_sites}")
        
        self.print_convergence_summary()

    def print_convergence_summary(self):
        """Print a summary of convergence performance."""
        if not self.convergence_info:
            print("No convergence information available.")
            return
        
        print("\nFrequency Optimization Convergence:")
        total_evaluations = sum(info['function_evaluations'] for info in self.convergence_info)
        converged_phases = sum(1 for info in self.convergence_info if info['converged'])
        tolerance_met_phases = sum(1 for info in self.convergence_info if info['rms_tolerance_met'])
        
        print(f"Total phases: {len(self.convergence_info)}")
        print(f"Total function evaluations: {total_evaluations}")
        print(f"Converged phases: {converged_phases}/{len(self.convergence_info)}")
        print(f"Phases meeting RMS tolerance: {tolerance_met_phases}/{len(self.convergence_info)}")
        
        sites = self.optimized_sites if self.optimized_sites else self.initial_sites
        print("\nPer-phase details:")
        for i, info in enumerate(self.convergence_info):
            site = sites[i] if i < len(sites) else 'Unknown'
            print(f"  Phase {info['phase_idx'] + 1} ({site}): "
                  f"{info['function_evaluations']} evals, "
                  f"RMS {info['initial_rms_error']:.3f} → {info['best_rms_error']:.3f} "
                  f"({'✓ converged' if info['converged'] else '✗ not converged'})")

    def plot_intermediate_trajectories_detailed(self, base_output_path=None):
        """Create a detailed plot showing intermediate trajectories for each phase separately."""
        if not self.intermediate_trajectories:
            print("No intermediate trajectories to plot.")
            return
        
        sites = self.optimized_sites if self.optimized_sites else self.initial_sites
        n_phases = len(self.convergence_info)
        fig, axes = plt.subplots(n_phases, 1, figsize=(12, 4*n_phases))
        
        if n_phases == 1:
            axes = [axes]
        
        for phase_idx in range(n_phases):
            ax = axes[phase_idx]
            
            phase_intermediates = [traj for traj in self.intermediate_trajectories 
                                 if traj['phase_idx'] == phase_idx]
            
            if not phase_intermediates:
                continue
            
            phase_time = phase_intermediates[0]['time']
            phase_desired = self.desired_trajectory_function(phase_time)
            ax.plot(phase_time, phase_desired, 'k-', linewidth=3, label='Desired', alpha=0.8)
            
            n_intermediates = len(phase_intermediates)
            colors = plt.cm.viridis(np.linspace(0, 1, n_intermediates))
            
            for i, intermediate in enumerate(phase_intermediates):
                alpha = 0.4 + 0.6 * (i / max(1, n_intermediates - 1))
                linewidth = 1.5 if i == n_intermediates - 1 else 1
                
                label = f"Eval {i+1} (f={intermediate['frequency']:.1f}Hz, RMS={intermediate['rms_error']:.3f})"
                if i == n_intermediates - 1:
                    label = f"FINAL: {label}"
                
                ax.plot(intermediate['time'], intermediate['trajectory'], 
                       color=colors[i], alpha=alpha, linewidth=linewidth, 
                       label=label if i < 5 or i == n_intermediates - 1 else None)
            
            site = sites[phase_idx] if phase_idx < len(sites) else 'Unknown'
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'Joint {self.biological_system.associated_joint} (deg)')
            ax.set_title(f'Phase {phase_idx + 1} - Site {site} - Optimization Evolution\n'
                        f'({self.convergence_info[phase_idx]["function_evaluations"]} evaluations, '
                        f'RMS: {self.convergence_info[phase_idx]["initial_rms_error"]:.3f} → '
                        f'{self.convergence_info[phase_idx]["best_rms_error"]:.3f})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, f'intermediate_trajectories_optimized.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Intermediate trajectories plot saved to {base_output_path}")
        
        plt.show()
