import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd
from copy import deepcopy
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
import matplotlib.pyplot as plt

from ..BiologicalSystems.BiologicalSystem import BiologicalSystem


class Sensitivity:
  
    def __init__(self, system: BiologicalSystem):
        self.biological_system = system
        self.simulation_data = {} 
          
    def run(self, 
            biophysical_variations: Optional[Dict[str, List]] = None,
            connection_variations: Optional[Dict[str, Dict[str, List]]] = None,
            neuron_count_variations: Optional[Dict[str, List[int]]] = None,
            n_iterations: int = 10,
            time_step=0.1*ms,
            ees_stimulation_params: Optional[Dict] = None,
            torque_profile: Optional[Dict] = None,
            seed: int = 42,
            base_output_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive sensitivity analysis on the biological system.
        
        Parameters:
        -----------
        biophysical_variations : dict, optional
            Dictionary specifying variations in biophysical parameters
            Format: {param_name: [values_list]}
            Example: {'gL': [5*nS, 10*nS, 15*nS]}
            
        connection_variations : dict, optional
            Dictionary specifying variations in connection parameters
            Format: {connection_name: {param: [values_list]}}
            Example: {'Ia_flexor_to_MN_flexor': {'w': [1.0*nS, 2.1*nS, 3.0*nS], 'p': 0.5}}
            
        neuron_count_variations : dict, optional
            Dictionary specifying variations in neuron population sizes
            Format: {population_name: [count_list]}
            Example: {'MN_flexor': [200, 300, 400]}
            
        n_iterations : int
            Number of simulation iterations for each parameter combination
            
        time_step : float
            Time step for simulation (in seconds)
            
        ees_stimulation_params : dict, optional
            EES stimulation parameters
            
        torque_profile : dict, optional
            External torque profile
            
        seed : int
            Random seed for reproducibility
            
        base_output_path : str, optional
            Base path for saving results
            
        Returns:
        --------
        dict
            Dictionary containing sensitivity analysis results for each parameter type
        """
        
        # Define the specific metrics we want
        metrics = ['max_joint_angle', 'min_joint_angle', 'joint_velocity_l2', 'joint_acceleration_l2']
        
        results = {
            'biophysical': pd.DataFrame(),
            'connections': pd.DataFrame(),
            'neuron_counts': pd.DataFrame(),
            'summary': {}
        }
        
        # Store original system state
        original_system = self.biological_system.clone_with()
        
        # 1. Biophysical parameter sensitivity
        if biophysical_variations:
            print("Analyzing biophysical parameter sensitivity...")
            bio_results = self._analyze_biophysical_sensitivity(
                biophysical_variations, n_iterations, time_step, 
                ees_stimulation_params, torque_profile, seed, metrics)
            results['biophysical'] = bio_results
            
        # 2. Connection parameter sensitivity
        if connection_variations:
            print("Analyzing connection parameter sensitivity...")
            conn_results = self._analyze_connection_sensitivity(
                connection_variations, n_iterations, time_step,
                ees_stimulation_params, torque_profile, seed, metrics)
            results['connections'] = conn_results
            
        # 3. Neuron count sensitivity
        if neuron_count_variations:
            print("Analyzing neuron count sensitivity...")
            neuron_results = self._analyze_neuron_count_sensitivity(
                neuron_count_variations, n_iterations, time_step,
                ees_stimulation_params, torque_profile, seed, metrics)
            results['neuron_counts'] = neuron_results
            
        # Save results if path provided
        if base_output_path:
            self._save_sensitivity_results(results, base_output_path)
            
        # Store results in instance
        self.sensitivity_results = results
            
        
        return results
    
    def _analyze_biophysical_sensitivity(self, variations: Dict[str, List], n_iterations: int, 
                                       time_step: float, ees_params: Optional[Dict],
                                       torque_profile: Optional[Dict], seed: int,
                                       metrics: List[str]) -> pd.DataFrame:
        """Analyze sensitivity to biophysical parameter variations."""
        
        results_list = []
        
        for param_name, values_list in variations.items():
            for value in values_list:
                #try:
                    # Create modified biophysical parameters
                    modified_params = BiologicalSystem.copy_brian_dict(self.biological_system.biophysical_params)
                    modified_params[param_name] = value
                    
                    # Run simulation with modified parameters
                    modified_system = self.biological_system.clone_with(
                        biophysical_params=modified_params)
                        
                    spikes, time_series = modified_system.run_simulation(
                            n_iterations=n_iterations,
                            time_step=time_step,
                            ees_stimulation_params=ees_params,
                            torque_profile=torque_profile,
                            seed=seed
                    )
                        
                    # Calculate metrics
                    metric_values = self._calculate_joint_metrics(time_series, metrics)
                    print('metric_values ', metric_values)    
                    # Store results
                    result_row = {
                            'parameter_type': 'biophysical',
                            'parameter_name': param_name,
                            'parameter_value': float(value),
                            **metric_values
                        }
                    results_list.append(result_row)
                  
                    # Store simulation data
                    if 'biophysical' not in self.simulation_data:
                        self.simulation_data['biophysical'] = {}
                    if param_name not in self.simulation_data['biophysical']:
                        self.simulation_data['biophysical'][param_name] = {}
                    key = float(value)
                    self.simulation_data['biophysical'][param_name][key] = {
                      'Spikes': {muscle_name: spikes[muscle_name]['MN'] for muscle_name in self.biological_system.muscles_names},
                      'Joint': time_series[f'Joint_{self.biological_system.associated_joint}'],
                      'Time': time_series['Time']
                    }   
                #except Exception as e:
                    #warnings.warn(f"Simulation failed for {param_name} = {value}: {e}")
                    #continue
        
        return pd.DataFrame(results_list)
    def analyze_connection_sensitivity(self, variations: Dict, n_iterations: int,
                                      time_step: float, ees_params: Optional[Dict],
                                      torque_profile: Optional[Dict], seed: int,
                                      metrics: List[str]) -> pd.DataFrame:
        """
        Analyze sensitivity to connection parameter variations.
        
        Args:
            variations: {connection_tuple: {param_name: [values_list]}}
                       e.g., {("Ia", "MN"): {"w": [1.0*nS, 2.1*nS], "p": [0.3, 0.45]}}
        """
        results_list = []
        
        for connection_tuple, param_variations in variations.items():
            if connection_tuple not in self.biological_system.connections:
                warnings.warn(f"Connection {connection_tuple} not found in system. Skipping.")
                continue
                
            for param_name, values_list in param_variations.items():
                for value in values_list:
                    try:
                        # Create modified connections
                        modified_connections = BiologicalSystem.copy_brian_dict(self.biological_system.connections)
                        modified_connections[connection_tuple][param_name] = value
                        
                        # Run simulation with modified system
                        modified_system = self.biological_system.clone_with(connections=modified_connections)
                        spikes, time_series = modified_system.run_simulation(
                            n_iterations=n_iterations, time_step=time_step,
                            ees_stimulation_params=ees_params, torque_profile=torque_profile, seed=seed
                        )
                        
                        # Calculate metrics and store results
                        metric_values = self._calculate_joint_metrics(time_series, metrics)
                        results_list.append({
                            'parameter_type': 'connection',
                            'connection': f"{connection_tuple[0]}_to_{connection_tuple[1]}",
                            'parameter_name': param_name,
                            'parameter_value': float(value) if hasattr(value, 'magnitude') else value,
                            **metric_values
                        })
                        
                        # Store simulation data
                        if 'connection' not in self.simulation_data:
                            self.simulation_data['connection'] = {}
                        connection_key = f"{connection_tuple[0]}_to_{connection_tuple[1]}"
                        if connection_key not in self.simulation_data['connection']:
                            self.simulation_data['connection'][connection_key] = {}
                        if param_name not in self.simulation_data['connection'][connection_key]:
                            self.simulation_data['connection'][connection_key][param_name] = {}
                        
                        self.simulation_data['connection'][connection_key][param_name][float(value) if hasattr(value, 'magnitude') else value] = {
                            'Spikes': {muscle_name: spikes[muscle_name]['MN'] for muscle_name in self.biological_system.muscles_names},
                            'Joint': time_series[f'Joint_{self.biological_system.associated_joint}'],
                            'Time': time_series['Time']
                        }
                        
                    except Exception as e:
                        warnings.warn(f"Simulation failed for {connection_tuple}, {param_name}={value}: {e}")
                        continue
        
        return pd.DataFrame(results_list)
    
    def _analyze_neuron_count_sensitivity(self, variations: Dict, n_iterations: int,
                                        time_step: float, ees_params: Optional[Dict],
                                        torque_profile: Optional[Dict], seed: int,
                                        metrics: List[str]) -> pd.DataFrame:
        """Analyze sensitivity to neuron population count variations."""
        
        results_list = []
        
        for population_name, count_list in variations.items():
            for count in count_list:
                try:
                    # Create modified neuron populations
                    modified_populations = BiologicalSystem.copy_brian_dict(self.biological_system.neurons_population)
                    modified_populations[population_name] = count
                    
                    modified_system = self.biological_system.clone_with(
                        neurons_population=modified_populations)
                    
                    spikes, time_series = modified_system.run_simulation(
                        n_iterations=n_iterations,
                        time_step=time_step,
                        ees_stimulation_params=ees_params,
                        torque_profile=torque_profile,
                        seed=seed
                    )
                    
                    metric_values = self._calculate_joint_metrics(time_series, metrics)
                    
                    result_row = {
                        'parameter_type': 'neuron_count',
                        'parameter_name': population_name,
                        'neuron_count': count,
                        'parameter_value': count,
                        **metric_values
                    }
                    results_list.append(result_row)
                  
                    # Store simulation data
                    if 'neuron_count' not in self.simulation_data:
                        self.simulation_data['neuron_count'] = {}
                    if param_name not in self.simulation_data['neuron_count']:
                        self.simulation_data['neuron_count'][param_name] = {}
                    self.simulation_data['neuron_count'][param_name][float(value)] = {
                      'Spikes': {muscle_name: spikes[muscle_name]['MN'] for muscle_name in self.biological_system.muscles_names},
                      'Time': time_series['Time'],
                      'Joint': time_series[f'Joint_{self.biological_system.associated_joint}'],
                    }   
                except Exception as e:
                    warnings.warn(f"Simulation failed for {population_name} count = {count}: {e}")
                    continue
        
        return pd.DataFrame(results_list)
    
    def _calculate_joint_metrics(self, time_series: Dict, metrics: List[str]) -> Dict[str, float]:
        """Calculate joint-specific metrics from simulation results."""
        
        metric_values = {}
        # Get joint angle data
        joint_angle_key = f'Joint_{self.biological_system.associated_joint}'
        joint_angles = time_series.get(joint_angle_key, [])
        joint_velocity_key=f'Joint_Velocity_{self.biological_system.associated_joint}'  
        joint_velocity=time_series.get(joint_velocity_key, [])
      
        # Get time data for derivatives
        time_data = time_series.get('Time', [])
            
        if len(joint_angles) == 0 or len(time_data) == 0:
            # Return NaN values if no data
            for metric in metrics:
                metric_values[metric] = np.nan
            return metric_values
            
        joint_angles = np.array(joint_angles)
        time_data = np.array(time_data)
            
        for metric in metrics:
            if metric == 'max_joint_angle':
                metric_values[metric] = np.max(joint_angles)
                
            elif metric == 'min_joint_angle':
                metric_values[metric] = np.min(joint_angles)
                
            elif metric == 'joint_velocity_l2':
                # Calculate joint velocity from angle and time
                if len(joint_velocity) > 1:
                    metric_values[metric] = np.linalg.norm(joint_velocity)
                else:
                    metric_values[metric] = 0.0
                
            elif metric == 'joint_acceleration_l2':
                # Calculate joint acceleration from velocity
                if len(joint_velocity) > 2:
                    acceleration = np.gradient(joint_velocity, time_data)
                    metric_values[metric] = np.linalg.norm(acceleration)
                else:
                    metric_values[metric] = 0.0
                
            else:
                metric_values[metric] = np.nan
                    
        
        return metric_values


    def _plot_joint_angle(self, param_name: str, param_type: str, save_path: Optional[str] = None):
        """Plot joint angle for all parameter variations."""
        
        data = self.simulation_data[param_type][param_name]
        n_plots = len(data)
        
        # Calculate subplot layout
        n_cols = min(3, n_plots)  # Max 3 columns for better readability
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        
        # Handle case where there's only one subplot
        if n_plots == 1:
            axs = [axs]
        elif n_rows == 1:
            axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        else:
            axs = axs.flatten()
        
        for i, (param_value, simulation_results) in enumerate(data.items()):
            if i < len(axs):
                ax = axs[i]
                ax.plot(simulation_results['Time'], simulation_results['Joint'], 
                       label=f"{param_name}: {param_value}", linewidth=2)
                ax.set_title(f"{param_name} = {param_value}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Joint Angle")
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Hide unused subplots
        for i in range(len(data), len(axs)):
            axs[i].set_visible(False)
        
        fig.suptitle(f'Joint Angle Analysis - {param_name}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{param_name}_Joint_Angle_{timestamp}.png'
        fig_path = self._get_save_path(save_path, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig_path
    
    
    def _plot_raster(self, param_name: str, param_type: str, save_path: Optional[str] = None):
        """Plot raster plot for all parameter variations."""
        
        data = self.simulation_data[param_type][param_name]
        n_plots = len(data)
        
        # Calculate subplot layout
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
        
        # Handle case where there's only one subplot
        if n_plots == 1:
            axs = [axs]
        elif n_rows == 1:
            axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        else:
            axs = axs.flatten()
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']  # Colors for different muscles
        
        for i, (param_value, simulation_results) in enumerate(data.items()):
            if i < len(axs):
                ax = axs[i]
                neuron_offset = 0
                
                # Plot spikes for each muscle
                for muscle_idx, (muscle_name, spikes_muscle) in enumerate(simulation_results['Spikes'].items()):
                    color = colors[muscle_idx % len(colors)]
                    
                    for neuron_id, neuron_spikes in spikes_muscle.items():
                        if neuron_spikes:  # Only plot if there are spikes
                            spike_times = np.array(neuron_spikes)
                            neuron_positions = np.ones_like(spike_times) * (int(neuron_id) + neuron_offset)
                            ax.scatter(spike_times, neuron_positions, s=15, c=color, 
                                     marker='|', alpha=0.8, label=muscle_name if int(neuron_id) == 0 else "")
                    
                    # Update offset for next muscle
                    if spikes_muscle:
                        neuron_offset += len(spikes_muscle)
                
                ax.set_title(f"{param_name} = {param_value}")
                ax.set_ylabel("Neuron Index")
                ax.grid(True, alpha=0.3)
                
                # Add legend only for the first subplot to avoid clutter
                if i == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))  # Remove duplicate labels
                    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Hide unused subplots
        for i in range(len(data), len(axs)):
            axs[i].set_visible(False)
        
        # Set common x-label for bottom row
        for i in range(max(0, len(axs) - n_cols), len(axs)):
            if axs[i].get_visible():
                axs[i].set_xlabel("Time (s)")
        
        fig.suptitle(f'Spikes Raster Plot - {param_name}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{param_name}_Raster_Plot_{timestamp}.png'
        fig_path = self._get_save_path(save_path, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig_path
    
    
    def _plot_frequency_spectrum(self, param_name: str, param_type: str, save_path: Optional[str] = None):
        """Plot frequency spectrum for all parameter variations."""
        
        data = self.simulation_data[param_type][param_name]
        n_plots = len(data)
        
        # Calculate subplot layout
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), 
                               sharex=True, sharey=True)
        
        # Handle case where there's only one subplot
        if n_plots == 1:
            axs = [axs]
        elif n_rows == 1:
            axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
        else:
            axs = axs.flatten()
        
        for i, (param_value, simulation_results) in enumerate(data.items()):
            if i < len(axs):
                ax = axs[i]
                joint = simulation_results['Joint']
                time = simulation_results['Time']
                
                # Calculate sampling rate and perform FFT
                if len(time) > 1:
                    dt = float(time.iloc[1] - time.iloc[0])  # Ensure float conversion
                    n = len(joint)
                    freqs = np.fft.fftfreq(n, d=dt)
                    fft_values = np.fft.fft(joint)
                    
                    # Keep only positive frequencies
                    pos_mask = freqs > 0
                    freqs = freqs[pos_mask]
                    magnitudes = np.abs(fft_values[pos_mask])
                    
                    # Plot frequency spectrum
                    ax.loglog(freqs, magnitudes, linewidth=2)  # Log-log plot for better visualization
                    ax.set_title(f"{param_name} = {param_value}")
                    ax.set_ylabel("Amplitude")
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, 
                           ha='center', va='center')
        
        # Hide unused subplots
        for i in range(len(data), len(axs)):
            axs[i].set_visible(False)
        
        # Set common x-label for bottom row
        for i in range(max(0, len(axs) - n_cols), len(axs)):
            if axs[i].get_visible():
                axs[i].set_xlabel("Frequency (Hz)")
        
        fig.suptitle(f'Frequency Spectrum - {param_name}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{param_name}_Frequency_Spectrum_{timestamp}.png'
        fig_path = self._get_save_path(save_path, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig_path
    
    
    def _get_save_path(self,save_path: Optional[str], filename: str) -> str:
        """Helper function to determine the save path for figures."""
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            return os.path.join(save_path, filename)
        else:
            os.makedirs("Sensitivity", exist_ok=True)
            return os.path.join("Results", filename)
    
    
    def detailed_plot(self, save_path: Optional[str] = None) -> Dict[str, list]:
        """
        Plot detailed analysis including joint angles, raster plots, and frequency spectra
        for each parameter variation.
        
        Returns:
            Dict[str, list]: Dictionary containing paths to saved figures organized by plot type
        """
        
        if not hasattr(self, 'simulation_data'):
            raise ValueError("No simulation data available. Run run() method first.")
        
        
        for param_type in ['biophysical', 'connections', 'neuron_counts']:
            if param_type not in self.simulation_data:
                continue
                
            for param_name in self.simulation_data[param_type].keys():
                print(f"Plotting {param_type} parameter: {param_name}")
              
                # Plot joint angles
                self._plot_joint_angle(param_name, param_type, save_path)
                  
                # Plot raster plots
                self._plot_raster(param_name, param_type, save_path)
         
                # Plot frequency spectra
                self._plot_frequency_spectrum(param_name, param_type, save_path)
                    
    
    def _get_parameter_variations(self, param_name: str, param_value: Any, param_type: str) -> List[Any]:
        """
        Get appropriate parameter variations based on parameter type and name.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter
        param_value : Any
            Original parameter value
        param_type : str
            Type of parameter ('biophysical', 'connection', 'neuron_population')
            
        Returns:
        --------
        List[Any]
            List of parameter variations
        """
        
        # Define the multiplicative factors for regular parameters
        multiplicative_factors = [1/2, 1/np.sqrt(2), 10/11, 1, 11/10, np.sqrt(2), 2]
        
        # Define explicit values for voltage potentials (in mV)
        voltage_variations = {
            'Eleaky': [-80, -75, -70, -65, -60],  # Typical resting potentials
            'E_ex': [-10, -5, 0, 5, 10],          # Excitatory reversal potentials
            'E_inh': [-85, -80, -75, -70, -65],   # Inhibitory reversal potentials  
            'threshold_v': [-60, -55, -50, -45, -40]  # Action potential thresholds
        }
        
        # Define explicit values for probability parameters
        probability_values = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
        
        variations = []
        
        if param_type == 'biophysical':
            # Handle voltage parameters (potentials and thresholds)
            if param_name in voltage_variations:
                # Use predefined biologically realistic voltage values
                for val in voltage_variations[param_name]:
                    if hasattr(param_value, 'dim'):  # Brian2 quantity
                        variations.append(val * mV)
                    else:
                        variations.append(val)
            else:
                # For other biophysical parameters (conductances, capacitances, time constants), use multiplicative factors
                for factor in multiplicative_factors:
                    if hasattr(param_value, 'magnitude'):  # Brian2 quantity
                        new_val = param_value.magnitude * factor
                        variations.append(new_val * param_value.dim)
                    else:
                        variations.append(param_value * factor)
        
        elif param_type == 'connection':
            if param_name == 'p':
                # This is a probability parameter, use explicit probability values
                # Filter to only include values that make biological sense
                original_p = float(param_value) if hasattr(param_value, 'magnitude') else param_value
                
                # Create variations around the original probability
                if original_p >= 0.8:  # High probability connections
                    variations = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                elif original_p >= 0.5:  # Medium probability connections  
                    variations = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
                else:  # Low probability connections
                    variations = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                    
                # Ensure we don't exceed probability bounds
                variations = [p for p in variations if 0 < p <= 1]
                
            else:
                # For weight parameters, use multiplicative factors
                for factor in multiplicative_factors:
                    if hasattr(param_value, 'magnitude'):  # Brian2 quantity
                        new_val = param_value.magnitude * factor
                        variations.append(new_val * param_value.dim)
                    else:
                        variations.append(param_value * factor)
        
        elif param_type == 'neuron_population':
            # For neuron counts, use multiplicative factors but ensure integer values >= 1
            for factor in multiplicative_factors:
                new_count = int(param_value * factor)
                new_count = max(1, new_count)  # Ensure at least 1 neuron
                variations.append(new_count)
            
            # Remove duplicates while preserving order
            seen = set()
            variations = [x for x in variations if not (x in seen or seen.add(x))]
        
        return variations

    
    def global_variance_analysis(self, 
                              n_iterations: int = 10,
                              time_step=0.1*ms,
                              ees_stimulation_params: Optional[Dict] = None,
                              torque_profile: Optional[Dict] = None,
                              seed: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Perform global variance analysis by varying all parameters using specific variation factors.
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations per parameter variation
        time_step : float
            Time step for simulation
        ees_stimulation_params : dict, optional
            EES stimulation parameters
        torque_profile : dict, optional
            External torque profile
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing variance analysis results and top impactful parameters
        """
        
        metrics = ['max_joint_angle', 'min_joint_angle', 'joint_velocity_l2', 'joint_acceleration_l2']
        
        # Store original system
        original_system = self.biological_system.clone_with()
        
        # Get baseline metrics
        print("Calculating baseline metrics...")
        spikes_base, time_series_base = original_system.run_simulation(
            n_iterations=n_iterations,
            time_step=time_step,
            ees_stimulation_params=ees_stimulation_params,
            torque_profile=torque_profile,
            seed=seed
        )
        baseline_metrics = self._calculate_joint_metrics(time_series_base, metrics)
            
        # 1. Prepare biophysical parameter variations
        print("Analyzing biophysical parameter variances...")
        biophysical_variations = {}
        for param_name, param_value in self.biological_system.biophysical_params.items():
            biophysical_variations[param_name] = self._get_parameter_variations(
                param_name, param_value, 'biophysical')
            
        bio_results = self._analyze_biophysical_sensitivity(
            biophysical_variations, n_iterations, time_step,
            ees_stimulation_params, torque_profile, seed, metrics
        )
            
        # 2. Prepare connection parameter variations
        print("Analyzing connection parameter variances...")
        connection_variations = {}
        for connection_key, connection_params in self.biological_system.connections.items():
            connection_variations[connection_key] = {}
            for param_name, param_value in connection_params.items():
                # Pass the original value, let _analyze_connection_sensitivity generate variations
                connection_variations[connection_key][param_name] = param_value
            
        conn_results = self._analyze_connection_sensitivity(
            connection_variations, n_iterations, time_step,
            ees_stimulation_params, torque_profile, seed, metrics
        )
            
        # 3. Prepare neuron population variations
        print("Analyzing neuron population variances...")
        neuron_variations = {}
        for pop_name, pop_count in self.biological_system.neurons_population.items():
            neuron_variations[pop_name] = self._get_parameter_variations(
                pop_name, pop_count, 'neuron_population')
            
        neuron_results = self._analyze_neuron_count_sensitivity(
            neuron_variations, n_iterations, time_step,
            ees_stimulation_params, torque_profile, seed, metrics
        )
            
        # Combine all results
        all_results = pd.concat([bio_results, conn_results, neuron_results], ignore_index=True)
            
        # Calculate variances for each parameter-metric combination
        parameter_variances = []
            
        for param_name in all_results['parameter_name'].unique():
            param_data = all_results[all_results['parameter_name'] == param_name]
            if len(param_data) > 1:  # Need at least 2 points to calculate variance
                    
                param_type = param_data['parameter_type'].iloc[0]
                    
                for metric in metrics:
                    if metric in param_data.columns:
                        values = param_data[metric].dropna()
                        if len(values) > 1:
                            variance = np.var(values)
                            parameter_variances.append({
                                'parameter_type': param_type,
                                'parameter_name': param_name,
                                'metric': metric,
                                'variance': variance,
                                'baseline_value': baseline_metrics[metric],
                                'coefficient_of_variation': np.sqrt(variance) / abs(baseline_metrics[metric]) if baseline_metrics[metric] != 0 else np.inf
                            })
            
        # Convert to DataFrame and find top impactful parameters
        variance_df = pd.DataFrame(parameter_variances)
            
        # Get top 15 most impactful parameters for each metric
        top_parameters = {}
        for metric in metrics:
            metric_data = variance_df[variance_df['metric'] == metric].copy()
            metric_data = metric_data.sort_values('variance', ascending=False)
            top_parameters[metric] = metric_data.head(15)
            
        results = {
            'all_variances': variance_df,
            'top_parameters': top_parameters,
            'baseline_metrics': baseline_metrics,
            'detailed_results': {
                'biophysical': bio_results,
                'connections': conn_results,
                'neuron_counts': neuron_results
            }
        }
            
        # Store results
        self.global_variance_results = results
            
        return results
      
    def plot(self, analysis_type: str = 'all', save_path: Optional[str] = None):
        """
        Plot sensitivity analysis results with joint-specific metrics.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis to plot ('biophysical', 'connections', 'neuron_counts', or 'all')
        save_path : str, optional
            Path to save plots
        """
        
        if not hasattr(self, 'sensitivity_results') or not self.sensitivity_results:
            raise ValueError("No sensitivity analysis results available. Run run() method first.")
        
        metrics = ['max_joint_angle', 'min_joint_angle', 'joint_velocity_l2', 'joint_acceleration_l2']
        metric_labels = {
            'max_joint_angle': 'Max Joint Angle (degree)',
            'min_joint_angle': 'Min Joint Angle (degree)', 
            'joint_velocity_l2': 'Joint Velocity L2 Norm (degree/s)',
            'joint_acceleration_l2': 'Joint Acceleration L2 Norm (degree/s²)'
        }
        
        types_to_plot = (['biophysical', 'connections', 'neuron_counts'] 
                        if analysis_type == 'all' else [analysis_type])
        
        for analysis in types_to_plot:
            df = self.sensitivity_results.get(analysis, pd.DataFrame())
            if df.empty:
                continue
            
            # Get unique parameters
            unique_params = df['parameter_name'].unique()
            
            for param in unique_params:
                param_data = df[df['parameter_name'] == param].copy()
                if param_data.empty:
                    continue
                
                # Create subplots for each metric
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, metric in enumerate(metrics):
                    ax = axes[i]
                    
                    if metric in param_data.columns:
                        # Sort by parameter value for smooth plotting
                        param_data_sorted = param_data.sort_values('parameter_value')
                        
                        ax.plot(param_data_sorted['parameter_value'], 
                               param_data_sorted[metric], 
                               'o-', linewidth=2, markersize=6, alpha=0.8)
                        
                        ax.set_xlabel('Parameter Value')
                        ax.set_ylabel(metric_labels[metric])
                        ax.set_title(f'{metric_labels[metric]} vs {param}')
                        ax.grid(True, alpha=0.3)
                        
                        # Add some styling
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                
                plt.suptitle(f'Sensitivity Analysis: {param} ({analysis.title()})', fontsize=16)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(f"{save_path}/sensitivity_{analysis}_{param.replace(' ', '_')}.png", 
                               dpi=300, bbox_inches='tight')
                
                plt.show()
    
    
    def plot_global_variance(self, save_path: Optional[str] = None):
        """
        Plot global variance analysis results as bar plots showing top 15 most impactful parameters.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save plots
        """
        
        if not hasattr(self, 'global_variance_results'):
            raise ValueError("No global variance analysis results available. Run global_variance_analysis() first.")
        
        top_parameters = self.global_variance_results['top_parameters']
        metrics = ['max_joint_angle', 'min_joint_angle', 'joint_velocity_l2', 'joint_acceleration_l2']
        metric_labels = {
            'max_joint_angle': 'Max Joint Angle Variance',
            'min_joint_angle': 'Min Joint Angle Variance', 
            'joint_velocity_l2': 'Joint Velocity L2 Variance',
            'joint_acceleration_l2': 'Joint Acceleration L2 Variance'
        }
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for parameter types
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric in top_parameters:
                data = top_parameters[metric]
                
                if not data.empty:
                    # Create colors based on parameter type
                    color_map = {'biophysical': '#1f77b4', 'connection': '#ff7f0e', 'neuron_population': '#2ca02c'}
                    bar_colors = [color_map.get(ptype, '#gray') for ptype in data['parameter_type']]
                    
                    # Create bar plot
                    bars = ax.bar(range(len(data)), data['variance'], color=bar_colors, alpha=0.7)
                    
                    # Customize plot
                    ax.set_xlabel('Parameters (Ranked by Impact)')
                    ax.set_ylabel('Variance')
                    ax.set_title(f'Top 15 Most Impactful Parameters\n{metric_labels[metric]}')
                    
                    # Set x-tick labels (rotated for readability)
                    ax.set_xticks(range(len(data)))
                    ax.set_xticklabels(data['parameter_name'], rotation=45, ha='right')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add value labels on bars
                    for j, (bar, value) in enumerate(zip(bars, data['variance'])):
                        if not np.isnan(value):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data['variance'])*0.01,
                                   f'{value:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.7, label='Biophysical'),
                          plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.7, label='Connection'),
                          plt.Rectangle((0,0),1,1, facecolor='#2ca02c', alpha=0.7, label='Neuron Population')]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.suptitle('Global Parameter Impact Analysis - Top 15 Most Influential Parameters', 
                     fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/global_variance_analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _save_sensitivity_results(self, results: Dict, base_path: str):
        """Save sensitivity analysis results to files."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for analysis_type, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(f"{base_path}/sensitivity_{analysis_type}.csv", index=False)
