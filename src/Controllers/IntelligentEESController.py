from brian2 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from copy import deepcopy
import os
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from ..BiologicalSystems.BiologicalSystem import BiologicalSystem
from .BaseEESController import BaseEESController

class IntelligentEESController(BaseEESController):
    """
    Enhanced Model Predictive Controller for EES parameters with physiological feedback rules
    and gait event-based parameter updates.
    
    This controller incorporates:
    1. Physiological feedback rules (dorsiflexion/plantarflexion corrections)
    2. Amplitude-based frequency adaptation
    3. Gait event detection and targeted parameter updates
    4. Adaptive learning from tracking errors
    """
    
    def __init__(self, biological_system, update_iterations, 
                 initial_ees_params=None, frequency_grid=[30, 70]*hertz, site_grid=['L4', 'L5', 'S1'],
                 time_step=0.1*ms, enable_physiological_feedback=True, 
                 enable_gait_events=True, enable_amplitude_adaptation=True):
        """
        Initialize the Intelligent EES controller.
        
        Additional Parameters:
        ---------------------
        enable_physiological_feedback : bool
            Enable physiological feedback rules for balance adjustment
        enable_gait_events : bool
            Enable gait event detection and targeted updates
        enable_amplitude_adaptation : bool
            Enable frequency adaptation based on movement amplitude
        """
        super().__init__(biological_system, update_iterations, initial_ees_params, 
                        frequency_grid, site_grid, time_step)
        
        # Intelligent controller features
        self.enable_physiological_feedback = enable_physiological_feedback
        self.enable_gait_events = enable_gait_events
        self.enable_amplitude_adaptation = enable_amplitude_adaptation
        
        # Physiological feedback parameters
        self.error_threshold = 5.0  # degrees
        self.frequency_adaptation_rate = 0.15
        
        # Gait event detection parameters
        self.gait_events_detected = []
        self.last_gait_event_time = -1.0
        self.min_event_interval = 0.1  # minimum time between events (seconds)
        
        # Amplitude tracking
        self.amplitude_history = []
        self.target_amplitude = None
        
        # Enhanced tracking
        self.physiological_adjustments_history = []
        self.gait_events_history = []
        self.amplitude_adaptations_history = []
        
        print(f"Intelligent EES Controller initialized with:")
        print(f"  - Physiological feedback: {enable_physiological_feedback}")
        print(f"  - Gait event detection: {enable_gait_events}")
        print(f"  - Amplitude adaptation: {enable_amplitude_adaptation}")
    
    def _detect_gait_events(self, trajectory, time_array, desired_trajectory):
        """
        Detect key gait events in the ankle trajectory.
        
        Returns:
        --------
        list : List of detected events with timing and type
        """
        if len(trajectory) < 10:
            return []
        
        # Smooth the trajectories for better peak detection
        smooth_actual = savgol_filter(trajectory, 
                                    window_length=min(11, len(trajectory)//2*2+1), 
                                    polyorder=3)
        smooth_desired = savgol_filter(desired_trajectory, 
                                     window_length=min(11, len(trajectory)//2*2+1), 
                                     polyorder=3)
        
        events = []
        
        # Find peaks (dorsiflexion) and valleys (plantarflexion) in desired trajectory
        peaks, peak_props = find_peaks(smooth_desired, height=None, distance=len(trajectory)//10)
        valleys, valley_props = find_peaks(-smooth_desired, height=None, distance=len(trajectory)//10)
        
        # Classify events based on gait phase
        for peak_idx in peaks:
            if peak_idx < len(time_array):
                events.append({
                    'time': time_array[peak_idx],
                    'type': 'dorsiflexion_peak',
                    'desired_angle': smooth_desired[peak_idx],
                    'actual_angle': smooth_actual[peak_idx],
                    'error': smooth_actual[peak_idx] - smooth_desired[peak_idx],
                    'index': peak_idx
                })
        
        for valley_idx in valleys:
            if valley_idx < len(time_array):
                events.append({
                    'time': time_array[valley_idx],
                    'type': 'plantarflexion_peak',
                    'desired_angle': smooth_desired[valley_idx],
                    'actual_angle': smooth_actual[valley_idx],
                    'error': smooth_actual[valley_idx] - smooth_desired[valley_idx],
                    'index': valley_idx
                })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        return events
    
    def _apply_physiological_feedback(self, current_params, trajectory, desired_trajectory):
        """
        Apply physiological feedback rules to adjust EES parameters.
        
        Parameters:
        -----------
        current_params : dict
            Current EES parameters
        trajectory : np.ndarray
            Recent actual trajectory
        desired_trajectory : np.ndarray
            Recent desired trajectory
            
        Returns:
        --------
        dict : Adjusted parameters
        """
        if not self.has_multiple_muscles:
            return current_params
    
        adjusted_params = BiologicalSystem.copy_brian_dict(current_params)
        
        # Calculate recent tracking error
        recent_error = np.mean(trajectory[-10:] - desired_trajectory[-10:])
        
        adjustment_made = False
        adjustment_reason = ""
        
        # Define stimulation site order from more flexor-dominant to more extensor-dominant
        site_order = ["L1","L2","L3","L4", "L5", "S1", "S2"]  # Ordered from flexor-dominant to extensor-dominant
    
        current_site = adjusted_params.get("site")  # now a string like "L5"
        if current_site not in site_order:
            return adjusted_params  # Invalid site, skip adjustment
    
        current_index = site_order.index(current_site)
    
        # Rule 1: If ankle is too dorsiflexed (positive error), move to more extensor-dominant site
        if recent_error > self.error_threshold and current_index < len(site_order) - 1:
            new_site = site_order[current_index + 1]
            adjustment_reason = f"Excessive dorsiflexion (error: {recent_error:.1f}°) -> shift to more extensor site: {new_site}"
            adjusted_params['site'] = new_site
            adjustment_made = True
    
        # Rule 2: If ankle is too plantarflexed (negative error), move to more flexor-dominant site
        elif recent_error < -self.error_threshold and current_index > 0:
            new_site = site_order[current_index - 1]
            adjustment_reason = f"Excessive plantarflexion (error: {recent_error:.1f}°) -> shift to more flexor site: {new_site}"
            adjusted_params['site'] = new_site
            adjustment_made = True
    
        if adjustment_made:
            self.physiological_adjustments_history.append({
                'time': len(self.time_history) * self.time_step if self.time_history else 0,
                'reason': adjustment_reason,
                'error': recent_error,
                'balance_change': f"{current_site} → {adjusted_params['site']}"
            })
        
        return adjusted_params

    
    def _apply_amplitude_adaptation(self, current_params, trajectory, desired_trajectory):
        """
        Adapt frequency based on movement amplitude requirements.
        
        Parameters:
        -----------
        current_params : dict
            Current EES parameters
        trajectory : np.ndarray
            Recent actual trajectory
        desired_trajectory : np.ndarray
            Recent desired trajectory
            
        Returns:
        --------
        dict : Adjusted parameters
        """
        adjusted_params = BiologicalSystem.copy_brian_dict(current_params)
        
        if len(trajectory) < 20:
            return adjusted_params
        
        # Calculate current and desired amplitude
        actual_amplitude = np.max(trajectory[-20:]) - np.min(trajectory[-20:])
        desired_amplitude = np.max(desired_trajectory[-20:]) - np.min(desired_trajectory[-20:])
        
        # Store amplitude for tracking
        self.amplitude_history.append({
            'actual': actual_amplitude,
            'desired': desired_amplitude,
            'ratio': actual_amplitude / desired_amplitude if desired_amplitude > 0 else 1.0
        })
        
        amplitude_ratio = actual_amplitude / desired_amplitude if desired_amplitude > 0 else 1.0
        
        adjustment_made = False
        adjustment_reason = ""
        
        # Rule 3: If amplitude is too small, increase frequency
        if amplitude_ratio < 0.7:  # Less than 70% of desired amplitude
            freq_adjustment = self.frequency_adaptation_rate * (self.frequency_grid[-1] - self.frequency_grid[0])
            new_freq = min(adjusted_params['frequency'] + freq_adjustment, self.frequency_grid[-1])
            adjusted_params['frequency'] = new_freq
            adjustment_made = True
            adjustment_reason = f"Low amplitude ({amplitude_ratio:.2f}) -> increase frequency"
        
        # Rule 4: If amplitude is too large, decrease frequency
        elif amplitude_ratio > 1.3:  # More than 130% of desired amplitude
            freq_adjustment = self.frequency_adaptation_rate * (self.frequency_grid[-1] - self.frequency_grid[0])
            new_freq = max(adjusted_params['frequency'] - freq_adjustment, self.frequency_grid[0])
            adjusted_params['frequency'] = new_freq
            adjustment_made = True
            adjustment_reason = f"High amplitude ({amplitude_ratio:.2f}) -> decrease frequency"
        
        if adjustment_made:
            self.amplitude_adaptations_history.append({
                'time': len(self.time_history) * self.time_step if self.time_history else 0,
                'reason': adjustment_reason,
                'amplitude_ratio': amplitude_ratio,
                'frequency_change': adjusted_params['frequency'] - current_params['frequency']
            })
        
        return adjusted_params
    
    def _apply_gait_event_adaptations(self, current_params, gait_events):
        """
        Apply targeted parameter adjustments based on detected gait events.
        
        Parameters:
        -----------
        current_params : dict
            Current EES parameters
        gait_events : list
            List of detected gait events
            
        Returns:
        --------
        dict : Adjusted parameters
        """
        if not gait_events or not self.has_multiple_muscles:
            return current_params
        
        adjusted_params = BiologicalSystem.copy_brian_dict(current_params)
        
        # Focus on the most recent significant event
        recent_events = [e for e in gait_events if abs(e['error']) > self.error_threshold/2]
        
        if not recent_events:
            return adjusted_params
        
        # Get the event with largest error
        critical_event = max(recent_events, key=lambda x: abs(x['error']))
        
        adjustment_made = False
        adjustment_reason = ""
        
        # Event-specific adaptations
        if critical_event['type'] == 'dorsiflexion_peak':
            if critical_event['error'] > 0:  # Too much dorsiflexion
                balance_adjustment = -self.balance_adaptation_rate * 1.5  # Stronger correction at peaks
                adjusted_params['balance'] = max(adjusted_params['balance'] + balance_adjustment, 
                                               self.balance_grid[0])
                adjustment_made = True
                adjustment_reason = f"Dorsiflexion peak error ({critical_event['error']:.1f}°) -> boost extensor"
            elif critical_event['error'] < -2:  # Insufficient dorsiflexion
                balance_adjustment = self.balance_adaptation_rate * 1.5
                adjusted_params['balance'] = min(adjusted_params['balance'] + balance_adjustment, 
                                               self.balance_grid[-1])
                adjustment_made = True
                adjustment_reason = f"Insufficient dorsiflexion ({critical_event['error']:.1f}°) -> boost flexor"
        
        elif critical_event['type'] == 'plantarflexion_peak':
            if critical_event['error'] < 0:  # Too much plantarflexion
                balance_adjustment = self.balance_adaptation_rate * 1.5
                adjusted_params['balance'] = min(adjusted_params['balance'] + balance_adjustment, 
                                               self.balance_grid[-1])
                adjustment_made = True
                adjustment_reason = f"Plantarflexion peak error ({critical_event['error']:.1f}°) -> boost flexor"
            elif critical_event['error'] > 2:  # Insufficient plantarflexion
                balance_adjustment = -self.balance_adaptation_rate * 1.5
                adjusted_params['balance'] = max(adjusted_params['balance'] + balance_adjustment, 
                                               self.balance_grid[0])
                adjustment_made = True
                adjustment_reason = f"Insufficient plantarflexion ({critical_event['error']:.1f}°) -> boost extensor"
        
        if adjustment_made:
            self.gait_events_history.append({
                'time': critical_event['time'],
                'event_type': critical_event['type'],
                'reason': adjustment_reason,
                'error': critical_event['error'],
                'balance_change': adjusted_params['balance'] - current_params['balance']
            })
        
        return adjusted_params
    
    def _optimize_ees_parameters(self, current_time):
        """
        Enhanced optimization with intelligent parameter adjustments.
        """
        # First, apply intelligent adjustments if we have trajectory history
        if len(self.trajectory_history) > 20:
            # Get recent trajectory data
            recent_actual = np.array(self.trajectory_history[-50:])
            recent_desired = np.array(self.desired_trajectory_history[-50:])
            recent_time = np.array(self.time_history[-50:])
            
            # Apply physiological feedback
            if self.enable_physiological_feedback:
                self.current_ees_params = self._apply_physiological_feedback(
                    self.current_ees_params, recent_actual, recent_desired
                )
            
            # Apply amplitude adaptation
            if self.enable_amplitude_adaptation:
                self.current_ees_params = self._apply_amplitude_adaptation(
                    self.current_ees_params, recent_actual, recent_desired
                )
            
            # Detect and respond to gait events
            if self.enable_gait_events:
                gait_events = self._detect_gait_events(recent_actual, recent_time, recent_desired)
                if gait_events:
                    self.gait_events_detected.extend(gait_events)
                    self.current_ees_params = self._apply_gait_event_adaptations(
                        self.current_ees_params, gait_events
                    )
        
        # Create refined search grid around current parameters
        refined_frequency_grid = self._create_refined_grid(
            self.current_ees_params['frequency'], 
            self.frequency_grid, 
            n_points=5
        )
        
        if self.has_multiple_muscles:
            refined_balance_grid = self._create_refined_grid(
                self.current_ees_params['balance'], 
                self.balance_grid, 
                n_points=5
            )
        else:
            refined_balance_grid = []
        
        # Run optimization with refined grid
        return self._run_grid_optimization(current_time, refined_frequency_grid, refined_balance_grid)
    
    def _create_refined_grid(self, current_value, full_grid, n_points=5):
        """
        Create a refined search grid around the current parameter value.
        """
        if isinstance(full_grid, list):
            grid_min, grid_max = min(full_grid), max(full_grid)
        else:
            grid_min, grid_max = full_grid[0], full_grid[-1]
        
        # Create search range around current value
        search_range = (grid_max - grid_min) * 0.3  # 30% of full range
        search_min = max(grid_min, current_value - search_range/2)
        search_max = min(grid_max, current_value + search_range/2)
        
        # Always include current value
        refined_grid = [current_value]
        
        # Add points around current value
        for i in range(1, n_points//2 + 1):
            step = search_range / n_points
            if search_min + i * step <= search_max:
                refined_grid.append(search_min + i * step)
            if search_max - i * step >= search_min:
                refined_grid.append(search_max - i * step)
        
        return sorted(list(set(refined_grid)))
    
    def _run_grid_optimization(self, current_time, frequency_grid, balance_grid):
        """
        Run the actual grid search optimization.
        """
        best_cost = float('inf')
        best_params = BiologicalSystem.copy_brian_dict(self.current_ees_params)
        best_spikes = None
        best_time_series = None
        best_system_state = None
        
        print(f"Optimizing EES parameters at time {current_time:.3f}s...")
        
        # Store trajectories for this optimization cycle
        optimization_cycle_trajectories = []
        
        # Create list of parameter combinations to test
        if self.has_multiple_muscles:
            param_combinations = list(product(frequency_grid, balance_grid))
            print(f"Testing {len(frequency_grid)} frequency × {len(balance_grid)} balance combinations (refined grid)")
        else:
            param_combinations = [(freq, None) for freq in frequency_grid]
            print(f"Testing {len(frequency_grid)} frequency values (refined grid)")
        
        for i, (freq, balance) in enumerate(param_combinations):
            # Create test parameters
            test_params = BiologicalSystem.copy_brian_dict(self.current_ees_params)
            test_params['frequency'] = freq
            if self.has_multiple_muscles:
                test_params['balance'] = balance
            
            # Run simulation with test parameters
            spikes, time_series = self.biological_system.run_simulation(
                n_iterations=self.update_iterations,
                time_step=self.time_step,
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
        if self.has_multiple_muscles:
            print(f"Best parameters: frequency={best_params['frequency']:.1f}Hz, balance={best_params['balance']:.3f}")
        else:
            print(f"Best parameters: frequency={best_params['frequency']:.1f}Hz")
        
        return best_system_state, best_spikes, best_time_series, best_params, best_cost, optimization_cycle_trajectories
    
    def plot(self, base_output_path=None):
        """
        Plot additional analysis showing the intelligent adaptations.
        """
        if not self.trajectory_history:
            raise ValueError("No simulation results to plot. Run control simulation first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Physiological adjustments over time
        if self.physiological_adjustments_history:
            times = [adj['time'] for adj in self.physiological_adjustments_history]
            balance_changes = [adj['balance_change'] for adj in self.physiological_adjustments_history]
            errors = [adj['error'] for adj in self.physiological_adjustments_history]
            
            axes[0,0].scatter(times, balance_changes, c=errors, cmap='RdBu_r', s=50, alpha=0.7)
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Balance Change')
            axes[0,0].set_title('Physiological Feedback Adjustments')
            axes[0,0].grid(True, alpha=0.3)
            cbar = plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])
            cbar.set_label('Tracking Error (deg)')
        else:
            axes[0,0].text(0.5, 0.5, 'No physiological\nadjustments made', 
                          ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Physiological Feedback Adjustments')
        
        # Plot 2: Gait events detection
        if self.gait_events_detected:
            event_times = [event['time'] for event in self.gait_events_detected]
            event_errors = [event['error'] for event in self.gait_events_detected]
            event_types = [event['type'] for event in self.gait_events_detected]
            
            # Color code by event type
            colors = ['red' if 'dorsi' in et else 'blue' for et in event_types]
            axes[0,1].scatter(event_times, event_errors, c=colors, s=60, alpha=0.7)
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Tracking Error at Event (deg)')
            axes[0,1].set_title('Detected Gait Events')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend(['Dorsiflexion peaks', 'Plantarflexion peaks'], 
                           handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8),
                                   plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8)])
        else:
            axes[0,1].text(0.5, 0.5, 'No gait events\ndetected', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Detected Gait Events')
        
        # Plot 3: Amplitude adaptation
        if self.amplitude_history:
            actual_amps = [amp['actual'] for amp in self.amplitude_history]
            desired_amps = [amp['desired'] for amp in self.amplitude_history]
            ratios = [amp['ratio'] for amp in self.amplitude_history]
            
            time_points = np.linspace(0, len(self.time_history) * self.time_step, len(actual_amps))
            
            axes[1,0].plot(time_points, actual_amps, label='Actual amplitude', color='orange')
            axes[1,0].plot(time_points, desired_amps, label='Desired amplitude', color='green')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('Movement Amplitude (deg)')
            axes[1,0].set_title('Amplitude Tracking')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No amplitude\ndata available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Amplitude Tracking')
        
        # Plot 4: Parameter evolution summary
        if self.ees_params_history:
            time_array = np.array(self.time_history)
            frequencies = [params['frequency'] for params in self.ees_params_history]
            
            axes[1,1].plot(time_array, frequencies, label='Frequency', color='purple', linewidth=2)
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('Frequency (Hz)', color='purple')
            axes[1,1].tick_params(axis='y', labelcolor='purple')
            
            if self.has_multiple_muscles:
                balances = [params['balance'] for params in self.ees_params_history]
                ax2 = axes[1,1].twinx()
                ax2.plot(time_array, balances, label='Balance', color='brown', linewidth=2)
                ax2.set_ylabel('Balance', color='brown')
                ax2.tick_params(axis='y', labelcolor='brown')
            
            axes[1,1].set_title('Intelligent Parameter Evolution')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            plt.savefig(os.path.join(base_output_path, 'intelligence_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Intelligence analysis plot saved to {base_output_path}")
        
        plt.show()
    
    def get_metrics(self):
        """
        Get metrics about the intelligent adaptations made during control.
        """
        metrics = super().get_performance_metrics()
        
        # Add intelligence-specific metrics
        metrics['physiological_adjustments'] = len(self.physiological_adjustments_history)
        metrics['gait_events_detected'] = len(self.gait_events_detected)
        metrics['amplitude_adaptations'] = len(self.amplitude_adaptations_history)
        
        if self.amplitude_history:
            amplitude_ratios = [amp['ratio'] for amp in self.amplitude_history]
            metrics['mean_amplitude_ratio'] = np.mean(amplitude_ratios)
            metrics['amplitude_stability'] = np.std(amplitude_ratios)
        
        return metrics
