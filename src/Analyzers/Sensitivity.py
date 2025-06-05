import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd
import os
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
            Example: {('Ia', 'MN'): {'w': [1.0*nS, 2.1*nS, 3.0*nS], 'p': 0.5}}
            
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
                ees_stimulation_params, torque_profile, metrics)
            results['biophysical'] = bio_results
            
        # 2. Connection parameter sensitivity
        if connection_variations:
            print("Analyzing connection parameter sensitivity...")
            conn_results = self._analyze_connection_sensitivity(
                connection_variations, n_iterations, time_step,
                ees_stimulation_params, torque_profile,  metrics)
            results['connections'] = conn_results
            
        # 3. Neuron count sensitivity
        if neuron_count_variations:
            print("Analyzing neuron count sensitivity...")
            neuron_results = self._analyze_neuron_count_sensitivity(
                neuron_count_variations, n_iterations, time_step,
                ees_stimulation_params, torque_profile, metrics)
            results['neuron_counts'] = neuron_results
            
        # Save results if path provided
        if base_output_path:
            self._save_sensitivity_results(results, base_output_path)
            
        # Store results in instance
        self.sensitivity_results = results
            
        
        return results
    
    def _analyze_biophysical_sensitivity(self, variations: Dict[str, List], n_iterations: int, 
                                       time_step: float, ees_params: Optional[Dict],
                                       torque_profile: Optional[Dict],
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
                            torque_profile=torque_profile
                    )
                        
                    # Calculate metrics
                    metric_values = self._calculate_joint_metrics(time_series, metrics)
                     
                    # Store results
                    result_row = {
                            'parameter_type': 'biophysical',
                            'parameter_name': param_name,
                            'parameter_value': float(value),
                            'original_value': value,
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
                      'Time': time_series['Time'],
                      'original_value': value
                    }   
                #except Exception as e:
                    #warnings.warn(f"Simulation failed for {param_name} = {value}: {e}")
                    #continue
        
        return pd.DataFrame(results_list)
    def _analyze_connection_sensitivity(self, variations: Dict, n_iterations: int,
                                      time_step: float, ees_params: Optional[Dict],
                                      torque_profile: Optional[Dict],
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
                    #try:
                        # Create modified connections
                        modified_connections = BiologicalSystem.copy_brian_dict(self.biological_system.connections)
                        modified_connections[connection_tuple][param_name] = value
                        
                        # Run simulation with modified system
                        modified_system = self.biological_system.clone_with(connections=modified_connections)
                        spikes, time_series = modified_system.run_simulation(
                            n_iterations=n_iterations, time_step=time_step,
                            ees_stimulation_params=ees_params, torque_profile=torque_profile
                        )
                        
                        # Calculate metrics and store results
                        metric_values = self._calculate_joint_metrics(time_series, metrics)
                        results_list.append({
                            'parameter_type': 'connection',
                            'connection': f"{connection_tuple[0]}_to_{connection_tuple[1]}",
                            'parameter_name': param_name,
                            'parameter_value': float(value) if hasattr(value, 'magnitude') else value,
                            'original_value': value,
                            **metric_values
                        })
                        
                        # Store simulation data
                        if 'connection' not in self.simulation_data:
                            self.simulation_data['connection'] = {}
                        connection_key = f"{param_name}_{connection_tuple[0]}_to_{connection_tuple[1]}"
                        if connection_key not in self.simulation_data['connection']:
                            self.simulation_data['connection'][connection_key] = {}
                        
                        self.simulation_data['connection'][connection_key][float(value)] = {
                            'Spikes': {muscle_name: spikes[muscle_name]['MN'] for muscle_name in self.biological_system.muscles_names},
                            'Joint': time_series[f'Joint_{self.biological_system.associated_joint}'],
                            'Time': time_series['Time'],
                            'original_value': value
                        }
                        
                    #except Exception as e:
                    #    warnings.warn(f"Simulation failed for {connection_tuple}, {param_name}={value}: {e}")
                    #    continue
        
        return pd.DataFrame(results_list)
    
    def _analyze_neuron_count_sensitivity(self, variations: Dict, n_iterations: int,
                                        time_step: float, ees_params: Optional[Dict],
                                        torque_profile: Optional[Dict], 
                                        metrics: List[str]) -> pd.DataFrame:
        """Analyze sensitivity to neuron population count variations."""
        
        results_list = []
        
        for population_name, count_list in variations.items():
            for count in count_list:
                #try:
                    # Create modified neuron populations
                    modified_populations = BiologicalSystem.copy_brian_dict(self.biological_system.neurons_population)
                    modified_populations[population_name] = count
                    
                    modified_system = self.biological_system.clone_with(
                        neurons_population=modified_populations,
                         initial_condition_spike_activation=None)
                    
                    spikes, time_series = modified_system.run_simulation(
                        n_iterations=n_iterations,
                        time_step=time_step,
                        ees_stimulation_params=ees_params,
                        torque_profile=torque_profile
                    )
                    
                    metric_values = self._calculate_joint_metrics(time_series, metrics)
                    
                    result_row = {
                        'parameter_type': 'neuron_count',
                        'parameter_name': population_name,
                        'parameter_value': count,
                        'original_value': count,
                        **metric_values
                    }
                    results_list.append(result_row)
                  
                    # Store simulation data
                    if 'neuron_count' not in self.simulation_data:
                        self.simulation_data['neuron_count'] = {}
                    if population_name not in self.simulation_data['neuron_count']:
                        self.simulation_data['neuron_count'][population_name] = {}
                    self.simulation_data['neuron_count'][population_name][float(count)] = {
                      'Spikes': {muscle_name: spikes[muscle_name]['MN'] for muscle_name in self.biological_system.muscles_names},
                      'Time': time_series['Time'],
                      'Joint': time_series[f'Joint_{self.biological_system.associated_joint}'],
                      'original_value': count
                    }   
                #except Exception as e:
                #    warnings.warn(f"Simulation failed for {population_name} count = {count}: {e}")
                #    continue
        
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
                    metric_values[metric] =  np.sqrt(np.mean(joint_velocity**2))
                else:
                    metric_values[metric] = 0.0
                
            elif metric == 'joint_acceleration_l2':
                # Calculate joint acceleration from velocity
                if len(joint_velocity) > 2:
                    acceleration = np.gradient(joint_velocity, time_data)
                    metric_values[metric] = np.sqrt(np.mean(acceleration)**2)
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
                #formatted_value = f"${param_value:.2e}$"
                formatted_value=simulation_results['original_value']
                #if isinstance(original_value, Quantity):
                #    unit = original_value.get_best_unit()
                #    formatted_value = round(float(original_value / unit), 1) * unit
                #else:
                #    formatted_value = round(original_value, 1)
     
                ax.plot(simulation_results['Time'], simulation_results['Joint'])
                ax.set_title(f"{param_name} = {formatted_value}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Joint Angle")
              
        
        # Hide unused subplots
        for i in range(len(data), len(axs)):
            axs[i].set_visible(False)
        
        fig.suptitle(f'Joint Angle Analysis - {param_name}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        filename = f'{param_name}_Joint_Angle.png'
        fig_path = self._get_save_path(save_path, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    
    
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
                            ax.scatter(spike_times, neuron_positions, label=muscle_name, marker='.', c=color, s=9)
                    
                    # Update offset for next muscle
                    if spikes_muscle:
                        neuron_offset += len(spikes_muscle)

                formatted_value=simulation_results['original_value']
                #if isinstance(original_value, Quantity):
                #    unit = original_value.get_best_unit()
                #    formatted_value = round(float(original_value / unit), 1) * unit
                #else:
                #    formatted_value = round(original_value, 1)

                #formatted_value = f"${param_value:.2e}$"
                ax.set_title(f"{param_name} = {formatted_value}")
                ax.set_ylabel("Neuron Index")
                ax.set_xlim([simulation_results['Time'].iloc[0], simulation_results['Time'].iloc[-1]])
                ax.set_ylim([-0.5, neuron_offset + 0.5])
                
                # Add legend only for the first subplot to avoid clutter
                #if i == 0:
                #    handles, labels = ax.get_legend_handles_labels()
                #    by_label = dict(zip(labels, handles))  # Remove duplicate labels
                #    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        # Hide unused subplots
        for i in range(len(data), len(axs)):
            axs[i].set_visible(False)
        
        # Set common x-label for bottom row
        for i in range(max(0, len(axs) - n_cols), len(axs)):
            if axs[i].get_visible():
                axs[i].set_xlabel("Time (s)")
        
        fig.suptitle(f'Spikes Raster Plot - {param_name}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        filename = f'{param_name}_Raster_Plot.png'
        fig_path = self._get_save_path(save_path, filename)
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        
        
    def _get_save_path(self,save_path: Optional[str], filename: str) -> str:
        """Helper function to determine the save path for figures."""
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            return os.path.join(save_path, filename)
        else:
            os.makedirs("Sensitivity", exist_ok=True)
            return os.path.join("Sensitivity", filename)
    
    
    def detailed_plot(self, save_path: Optional[str] = None) -> Dict[str, list]:
        """
        Plot detailed analysis including joint angles, raster plots, and frequency spectra
        for each parameter variation.
        
        Returns:
            Dict[str, list]: Dictionary containing paths to saved figures organized by plot type
        """
        
        if not hasattr(self, 'simulation_data'):
            raise ValueError("No simulation data available. Run run() method first.")
        
        for param_type in ['biophysical', 'connection', 'neuron_count']:
           
            if param_type not in self.simulation_data:
                continue 
            for param_name in self.simulation_data[param_type].keys():
                print(f"Plotting {param_type} parameter: {param_name}")
              
                # Plot joint angles
                self._plot_joint_angle(param_name, param_type, save_path)
                  
                # Plot raster plots
                self._plot_raster(param_name, param_type, save_path)
                    
    
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
        multiplicative_factors = [1/2, 1/np.sqrt(2), 10/11, 11/10, np.sqrt(2), 2]
        
        # Define explicit values for voltage potentials (in mV)
        voltage_variations = {
            'Eleaky': [-80, -76, -72, -68, -64, -60],  # Typical resting potentials
            'E_ex': [-10, -6, -2, 2, 6, 10],          # Excitatory reversal potentials
            'E_inh': [-90, -85, -80, -75, -70, -65],   # Inhibitory reversal potentials  
            'threshold_v': [ -55,-53,-51,-49,-47, -45]  # Action potential thresholds
        }
        
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
                        unit=param_value.get_best_unit()
                        new_val =round( param_value/unit * factor, 1)*unit
                        variations.append(new_val * param_valu)
                    else:
                        variations.append(param_value * factor)
        
        elif param_type == 'connection':
            if param_name == 'p':
                # This is a probability parameter, use explicit probability values
                # Filter to only include values that make biological sense
                original_p = float(param_value) if hasattr(param_value, 'magnitude') else param_value
                
                # Create variations around the original probability
                if original_p >= 0.8:  # High probability connections
                    variations = [0.7,0.75, 0.8, 0.85, 0.9, 0.95]
                elif original_p >= 0.5:  # Medium probability connections  
                    variations = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
                else:  # Low probability connections
                    variations = [ 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                    
                # Ensure we don't exceed probability bounds
                variations = [p for p in variations if 0 < p <= 1]
                
            else:
                # For weight parameters, use multiplicative factors
                for factor in multiplicative_factors:
                    if hasattr(param_value, 'magnitude'):  # Brian2 quantity
                        unit=param_value.get_best_unit()
                        new_val =round( param_value/unit * factor, 1)*unit
                        variations.append(new_val * param_valu)
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

    def global_sensitivity_analysis(self, 
                          n_iterations: int = 10,
                          time_step=0.1*ms,
                          ees_stimulation_params: Optional[Dict] = None,
                          torque_profile: Optional[Dict] = None, 
                                   what=['biophysic','neuron', 'connection']) -> Dict[str, pd.DataFrame]:
        """
        Perform global normalized sensitivity analysis by varying all parameters.
        
        Returns normalized sensitivity coefficients: (relative output change) / (relative parameter change)
        """
        
        metrics = ['max_joint_angle', 'min_joint_angle', 'joint_velocity_l2', 'joint_acceleration_l2']
        
        # Store original system and get baseline
        original_system = self.biological_system.clone_with()
        
        print("Calculating baseline metrics...")
        spikes_base, time_series_base = original_system.run_simulation(
            n_iterations=n_iterations, time_step=time_step,
            ees_stimulation_params=ees_stimulation_params,
            torque_profile=torque_profile
        )
        baseline_metrics = self._calculate_joint_metrics(time_series_base, metrics)
        
        # Collect original parameter values
        original_params = {}
        original_params.update(self.biological_system.biophysical_params)
        for conn_key, conn_params in self.biological_system.connections.items():
            for param_name, param_value in conn_params.items():
                original_params[f"{conn_key}_{param_name}"] = param_value
        original_params.update(self.biological_system.neurons_population)
        
        # Prepare variations for the run function
        print("Preparing parameter variations...")
        
        # 1. Biophysical parameter variations
        if 'biophysic' in what:
            biophysical_variations = {name: self._get_parameter_variations(name, value, 'biophysical')
                                for name, value in self.biological_system.biophysical_params.items()}
        else:
            biophysical_variations=None
          
        # 2. Connection parameter variations                      
        connection_variations = {}
        if 'connection' in what:
            for connection_key, connection_params in self.biological_system.connections.items():
                connection_variations[connection_key] = {
                    param_name: self._get_parameter_variations(param_name, param_value, 'connection')
                    for param_name, param_value in connection_params.items()
                }
        
        # 3. Neuron population variations
        if 'neuron' in what:
            neuron_variations = {name: self._get_parameter_variations(name, count, 'neuron_population')
                          for name, count in self.biological_system.neurons_population.items()}
        else:
            neuron_variations=None
        
        # Use the run function to get all sensitivity results
        print("Running comprehensive sensitivity analysis...")
        results = self.run(
            biophysical_variations=biophysical_variations,
            connection_variations=connection_variations,
            neuron_count_variations=neuron_variations,
            n_iterations=n_iterations,
            time_step=time_step,
            ees_stimulation_params=ees_stimulation_params,
            torque_profile=torque_profile
        )
        
        # Combine all results from the run function
        all_results = pd.concat([
            results['biophysical'], 
            results['connections'], 
            results['neuron_counts']
        ], ignore_index=True)
            
        # Calculate sensitivity coefficients
        sensitivities = []
        
        for param_name in all_results['parameter_name'].unique():
            param_data = all_results[all_results['parameter_name'] == param_name]
            if len(param_data) <= 1:
                continue
                
            param_type = param_data['parameter_type'].iloc[0]
            original_value = original_params.get(param_name)
            
            # Handle connection parameter naming
            if original_value is None:
                for key, value in original_params.items():
                    if param_name in key:
                        original_value = value
                        break
            
            if original_value is None:
                continue
                
            for metric in metrics:
                if metric not in param_data.columns:
                    continue
                    
                metric_values = param_data[metric]
                param_values = param_data['parameter_value']

                if len(metric_values) > 1 and len(param_values) > 1:
                    sensitivity = self._calculate_sensitivity_coefficient(
                        baseline_metrics[metric], metric_values,
                        original_value, param_values
                    )
                    
                    sensitivities.append({
                        'parameter_type': param_type,
                        'parameter_name': param_name,
                        'metric': metric,
                        'sensitivity_coefficient': sensitivity,
                        'baseline_value': baseline_metrics[metric]
                    })
        
        # Create results DataFrame
        sensitivity_df = pd.DataFrame(sensitivities)
        
        # Get top 15 most sensitive parameters for each metric
        top_parameters = {}
        for metric in metrics:
            metric_data = sensitivity_df[sensitivity_df['metric'] == metric].copy()
            metric_data = metric_data.sort_values('sensitivity_coefficient', ascending=False)
            top_parameters[metric] = metric_data.head(15)
        
        final_results = {
            'all_sensitivities': sensitivity_df,
            'top_parameters': top_parameters,
            'baseline_metrics': baseline_metrics,
            'detailed_results': results  # Include the detailed results from run function
        }
        
        self.global_variance_results = final_results
        return final_results

    def _calculate_sensitivity_coefficient(self, baseline_metric, varied_results, 
                                         original_param, varied_params):
        """
        Calculate sensitivity coefficient: (relative output change) / (relative parameter change)
        """
        sensitivities = []
        
        for result, param_value in zip(varied_results, varied_params):
            if baseline_metric != 0 and original_param != 0:
                output_change_pct = abs(result - baseline_metric) / abs(baseline_metric)
                param_change_pct = abs(float(param_value) - float(original_param)) / abs(float(original_param))
                
                if param_change_pct > 0:
                    sensitivity = output_change_pct / param_change_pct
                    sensitivities.append(sensitivity)
        
        return np.mean(sensitivities) if sensitivities else 0.0
      
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
            'joint_velocity_l2': 'RMS Joint Velocity  (degree/s)',
            'joint_acceleration_l2': 'RMS Joint Acceleration (degree/s²)'
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
                        unit=param_data_sorted['original_value'].iloc[0].get_best_unit()
                        ax.plot(param_data_sorted['original_value']/unit, 
                               param_data_sorted[metric], 
                               'o-', linewidth=2, markersize=6, alpha=0.8)
                        
                        ax.set_xlabel(f'{param} ({unit})')
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
    
    
    def plot_global_sensitivity(self, save_path: Optional[str] = None):
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
            'max_joint_angle': 'Sensitivity Max Joint Angle',
            'min_joint_angle': 'Sensitivity Min Joint Angle', 
            'joint_velocity_l2': 'Sensitivity Joint Velocity ',
            'joint_acceleration_l2': 'Sensitivity Joint Acceleration '
        }
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            if metric in top_parameters:
                data = top_parameters[metric]
                
                if not data.empty:
                   
                    color_map = {'biophysical': '#1f77b4', 'connection': '#ff7f0e', 'neuron_population': '#2ca02c'}
                    bar_colors = [color_map.get(ptype, '#808080') for ptype in data['parameter_type']]  # Changed '#gray' to '#808080'
                    
                    # Create bar plot
                    bars = ax.bar(range(len(data)), data['sensitivity_coefficient'], color=bar_colors, alpha=0.7)
                    
                    # Customize plot
                    ax.set_xlabel('Parameters (Ranked by Impact)')
                    ax.set_ylabel('Sensibility')
                    ax.set_title(f'{metric_labels[metric]}')
                    
                    # Set x-tick labels (rotated for readability)
                    ax.set_xticks(range(len(data)))
                    ax.set_xticklabels(data['parameter_name'], rotation=45, ha='right')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Remove top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add value labels on bars
                    for j, (bar, value) in enumerate(zip(bars, data['sensitivity_coefficient'])):
                        if not np.isnan(value):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data['sensitivity_coefficient'])*0.01,
                                  f'{value:.2e}', ha='center', va='bottom',  rotation=90)
        
        # Add legend
        #legend_elements = [plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.7, label='Biophysical'),
        #                  plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.7, label='Connection'),
        #                  plt.Rectangle((0,0),1,1, facecolor='#2ca02c', alpha=0.7, label='Neuron Population')]
        #fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.65, 0.5))
        
        plt.suptitle('Global Parameter Impact Analysis - Top 15 Most Influential Parameters')
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at the top for suptitle

        
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
