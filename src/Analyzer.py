import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd
from copy import deepcopy
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from .Visualization.plot_parameters_variations import plot_delay_results, plot_excitability_results, plot_twitch_results, plot_ees_analysis_results


class Analyzer:
  
    def __init__(self, system):
        self.original_system = system
    
          
    def analyze_frequency_effects(self, freq_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying EES frequency with fixed afferent and efferent recruitment.
        
        Parameters:
        -----------
        freq_range : array-like
            Range of frequencies to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : float
            Time step for simulations (in ms, will be converted to brian2 units)
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        vary_param = {
            'param_name': 'ees_freq',
            'values': freq_range,
            'label': 'EES Frequency (Hz)'
        }

        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="frequency_analysis", seed=seed)

        

    def analyze_intensity_effects(self, intensity_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying stimulation intensity.
        
        Parameters:
        -----------
        intensity_range : array-like
            Range of normalized intensity values to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : float
            Time step for simulations (in ms, will be converted to brian2 units)
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        vary_param = {
            'param_name': 'intensity',
            'values': intensity_range,
            'label': 'Stimulation Intensity'
        }
        
        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param, 
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="intensity_analysis", seed=seed)
        
      
    def analyse_unbalanced_recruitment_effects(self, b_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of unbalanced afferent recruitment between antagonistic muscles.
        
        Parameters:
        -----------
        b_range : array-like
            Range of balance values to analyze (0-1 where 0.5 is balanced)
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results
        """
        vary_param = {
            'param_name': 'balance',
            'values': b_range,
            'label': 'Afferent Fiber Unbalanced Recruitment'
        }
        
        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="balance_analysis", seed=seed)    

    def _compute_ees_parameter_sweep(self, param_dict, vary_param, n_iterations, time_step=0.1*ms, seed=42):
        """
        Compute EES stimulation analysis by varying a parameter of interest.
        
        Parameters:
        -----------
        param_dict : dict
            Dictionary containing all EES parameters with their default values
        vary_param : dict
            Dictionary specifying which parameter to vary with its range of values
            Format: {'param_name': [values_to_test], 'label': 'Display Label'}
        n_iterations : int
            Number of iterations for each simulation
        time_step : float
            Time step for simulation (in ms)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'param_values': list of parameter values tested
            - 'param_name': name of the varied parameter
            - 'param_label': display label for the parameter
            - 'time_data': time array from simulations
            - 'simulation_data': list of main_data dictionaries for each parameter value
            - 'spikes_data': list of spikes dictionaries for each parameter value
            - 'muscle_names': list of muscle names
            - 'associated_joint': joint name
        """
        
        # Get parameter info
        param_name = vary_param['param_name']
        param_values = vary_param['values']
        param_label = vary_param['label']
        
        # Initialize storage for results
        simulation_data = []
        spikes_data = []
        time_data = None
        activities = None
        
        print(f"Running parameter sweep for {param_label}...")
        print('time_step ', time_step)
        # Run simulations for each parameter value
        for i, value in enumerate(param_values):
            print(f"  Computing {param_label} = {value} ({i+1}/{len(param_values)})")
            
            # Create a copy of the base parameters
            current_params = param_dict.copy()
            
            # Update the parameter we're varying
            current_params[param_name] = value
            
            # Run simulation
            spikes, main_data = self.original_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=current_params,
                torque_profile=None,
                seed=seed, 
                base_output_path=None)
            
            # Store results
            simulation_data.append(main_data)
            spikes_data.append(spikes)
            
            # Extract time data (same for all simulations)
            if time_data is None:
                time_data = main_data['Time']
                
        
        # Return comprehensive results dictionary
        results = {
            'param_values': param_values,
            'param_name': param_name,
            'param_label': param_label,
            'time_data': time_data,
            'simulation_data': simulation_data,
            'spikes_data': spikes_data,
            'muscle_names': self.original_system.muscles_names,
            'associated_joint': self.original_system.associated_joint,
            'num_muscles': self.original_system.number_muscles
        }
        
        return results
      
     
    def clonus_analysis(self, torque_profile, delay_values=None, 
                    threshold_values=None, duration=1, 
                    time_step=0.1, fast_type_mu=True, seed=41):
        """
        Analyze clonus behavior by varying one parameter at a time.
        
        Parameters:
        -----------
        torque_profile : dict
            Dictionary with torque profile parameters
        delay_values : list, optional
            List of delay values to test (in ms)
        threshold_values : list, optional
            List of threshold values to test (in mV)
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        fast_type_mu : bool
            If True, use fast twitch motor units
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (delay_results, fast_twitch_results, threshold_results)
            Each containing list of (parameter_value, spikes, time_series) tuples
        """
        
        # Set default values if not provided
        if delay_values is None:
            delay_values = [10, 25, 50, 75, 100]  # in ms
        if threshold_values is None:
            threshold_values = [-45, -50, -55]  # in mV
                        
        delay_results = []
        fast_twitch_results = []
        threshold_results = []
        
        # 1. Vary delay (reaction time)
        print("Running delay variation analysis...")
        for delay in tqdm(delay_values, desc="Varying delay"):
            delay_seconds = delay / 1000.0  # Convert ms to seconds
            n_iterations = int(duration / delay_seconds)
           
            new_system = self.original_system.deepcopy()
            new_system.reaction_time = delay_seconds
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            delay_results.append((delay, spikes, time_series))

        plot_delay_results(delay_results, delay_values, self.original_system.muscles_names, self.original_system.associated_joint)
                        
        n_iterations = int(duration / self.original_system.reaction_time)
        time_points = np.arange(0, self.original_system.reaction_time * n_iterations, time_step / 1000.0)
        torque_array = transform_torque_params_in_array(time_points, torque_profile)
                        
        # 2. Vary fast twitch parameter
        print("Running fast twitch variation analysis...")
        fast_twitch_values = [False, True]
        
        for fast in tqdm(fast_twitch_values, desc="Varying fast twitch parameter"):
            new_system = self.original_system.deepcopy()
            new_system.fast = fast
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            fast_twitch_results.append((fast, spikes, time_series))
            
        plot_twitch_result(fast_twitch_results, self.original_system.muscles_names, self.original_system.associated_joint)
                        
        # 3. Vary threshold voltage
        print("Running threshold variation analysis...")
        for threshold in tqdm(threshold_values, desc="Varying threshold voltage"):
            new_system = self.original_system.deepcopy()
            new_system.biophysical_params['v_threshold'] = threshold / 1000.0  # Convert mV to V
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            threshold_results.append((threshold, spikes, time_series))

        plot_excitability_results(threshold_results, threshold_values, self.original_system.muscles_names, self.original_system.associated_joint)
        
        return delay_results, fast_twitch_results, threshold_results

      


    def perform_sensitivity_analysis(self, 
                                   biophysical_variations: Optional[Dict[str, Dict[str, List]]] = None,
                                   connection_variations: Optional[Dict[str, Dict[str, List]]] = None,
                                   neuron_count_variations: Optional[Dict[str, List[int]]] = None,
                                   n_iterations: int = 10,
                                   time_step=0.1e-3,  # 0.1 ms in seconds
                                   ees_stimulation_params: Optional[Dict] = None,
                                   torque_profile: Optional[Dict] = None,
                                   seed: int = 42,
                                   metrics: Optional[List[str]] = None,
                                   base_output_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive sensitivity analysis on the biological system.
        
        Parameters:
        -----------
        biophysical_variations : dict, optional
            Dictionary specifying variations in biophysical parameters
            Format: {param_name: {sub_param: [values_list]}}
            Example: {'leak_conductance': {'value': [0.1, 0.2, 0.3]}}
            
        connection_variations : dict, optional
            Dictionary specifying variations in connection parameters
            Format: {connection_name: {param: [values_list]}}
            Example: {'excitatory_weight': {'weight': [0.5, 1.0, 1.5]}}
            
        neuron_count_variations : dict, optional
            Dictionary specifying variations in neuron population sizes
            Format: {population_name: [count_list]}
            Example: {'motor_neurons': [50, 100, 150]}
            
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
            
        metrics : list, optional
            List of metrics to calculate. If None, uses default metrics
            
        base_output_path : str, optional
            Base path for saving results
            
        Returns:
        --------
        dict
            Dictionary containing sensitivity analysis results for each parameter type
        """
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['mean_spike_rate', 'max_activation', 'joint_angle_range', 
                      'muscle_force_variance', 'neural_synchrony']
        
        results = {
            'biophysical': pd.DataFrame(),
            'connections': pd.DataFrame(),
            'neuron_counts': pd.DataFrame(),
            'summary': {}
        }
        
        # Store original system state
        original_system = deepcopy(self.biological_system)
        
        try:
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
            
            # Generate summary statistics
            results['summary'] = self._generate_sensitivity_summary(results)
            
            # Save results if path provided
            if base_output_path:
                self._save_sensitivity_results(results, base_output_path)
            
            # Store results in instance
            self.sensitivity_results = results
            
        finally:
            # Restore original system
            self.biological_system = original_system
        
        return results
    
    def _analyze_biophysical_sensitivity(self, variations: Dict, n_iterations: int, 
                                       time_step: float, ees_params: Optional[Dict],
                                       torque_profile: Optional[Dict], seed: int,
                                       metrics: List[str]) -> pd.DataFrame:
        """Analyze sensitivity to biophysical parameter variations."""
        
        results_list = []
        
        for param_name, values_list in variations.items():
            # Handle the flat dictionary structure of biophysical_params
            for value in values_list:
                # Create modified biophysical parameters
                modified_params = deepcopy(self.biological_system.biophysical_params)
                
                # Apply parameter modification directly
                modified_params[param_name] = value
                
                # Run simulation with modified parameters
                try:
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
                    metric_values = self._calculate_metrics(spikes, time_series, metrics)
                    
                    # Store results
                    result_row = {
                        'parameter_type': 'biophysical',
                        'parameter_name': param_name,
                        'parameter_value': float(value) if hasattr(value, 'magnitude') else value,
                        **metric_values
                    }
                    results_list.append(result_row)
                    
                except Exception as e:
                    warnings.warn(f"Simulation failed for {param_name} = {value}: {e}")
                    continue
        
        return pd.DataFrame(results_list)
    
    def _analyze_connection_sensitivity(self, variations: Dict, n_iterations: int,
                                      time_step: float, ees_params: Optional[Dict],
                                      torque_profile: Optional[Dict], seed: int,
                                      metrics: List[str]) -> pd.DataFrame:
        """Analyze sensitivity to connection parameter variations."""
        
        results_list = []
        
        for connection_key, param_variations in variations.items():
            # connection_key should be a tuple like ("Ia", "MN") or string representation
            if isinstance(connection_key, str):
                # Try to parse string representation like "Ia_to_MN"
                parts = connection_key.split('_to_')
                if len(parts) == 2:
                    connection_tuple = (parts[0], parts[1])
                else:
                    # If not parseable, use as is
                    connection_tuple = connection_key
            else:
                connection_tuple = connection_key
            
            for param_name, values_list in param_variations.items():
                for value in values_list:
                    # Create modified connections
                    modified_connections = deepcopy(self.biological_system.connections)
                    
                    # Apply connection modification
                    if connection_tuple not in modified_connections:
                        modified_connections[connection_tuple] = {}
                    
                    modified_connections[connection_tuple][param_name] = value
                    
                    try:
                        modified_system = self.biological_system.clone_with(
                            connections=modified_connections)
                        
                        spikes, time_series = modified_system.run_simulation(
                            n_iterations=n_iterations,
                            time_step=time_step,
                            ees_stimulation_params=ees_params,
                            torque_profile=torque_profile,
                            seed=seed
                        )
                        
                        metric_values = self._calculate_metrics(spikes, time_series, metrics)
                        
                        result_row = {
                            'parameter_type': 'connection',
                            'connection': str(connection_tuple),
                            'parameter_name': param_name,
                            'parameter_value': float(value) if hasattr(value, 'magnitude') else value,
                            **metric_values
                        }
                        results_list.append(result_row)
                        
                    except Exception as e:
                        warnings.warn(f"Simulation failed for connection {connection_tuple}, {param_name} = {value}: {e}")
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
                # Create modified neuron populations
                modified_populations = deepcopy(self.biological_system.neurons_population)
                
                # Modify neuron count directly (integers in the dict)
                modified_populations[population_name] = count
                
                try:
                    modified_system = self.biological_system.clone_with(
                        neurons_population=modified_populations)
                    
                    spikes, time_series = modified_system.run_simulation(
                        n_iterations=n_iterations,
                        time_step=time_step,
                        ees_stimulation_params=ees_params,
                        torque_profile=torque_profile,
                        seed=seed
                    )
                    
                    metric_values = self._calculate_metrics(spikes, time_series, metrics)
                    
                    result_row = {
                        'parameter_type': 'neuron_count',
                        'population_name': population_name,
                        'neuron_count': count,
                        **metric_values
                    }
                    results_list.append(result_row)
                    
                except Exception as e:
                    warnings.warn(f"Simulation failed for {population_name} count = {count}: {e}")
                    continue
        
        return pd.DataFrame(results_list)
    
    def _calculate_metrics(self, spikes: Dict, time_series: Dict, 
                          metrics: List[str]) -> Dict[str, float]:
        """Calculate specified metrics from simulation results."""
        
        metric_values = {}
        
        try:
            for metric in metrics:
                if metric == 'mean_spike_rate':
                    # Calculate average spike rate across all populations
                    total_spikes = sum(len(spike_times) for spike_times in spikes.values())
                    total_neurons = sum(len(np.unique(spike_indices)) 
                                      for spike_indices in spikes.values())
                    simulation_time = max(max(times) for times in spikes.values() if len(times) > 0)
                    metric_values[metric] = total_spikes / (total_neurons * simulation_time) if simulation_time > 0 else 0
                
                elif metric == 'max_activation':
                    # Maximum muscle activation achieved
                    activations = [time_series.get(f'{muscle}_activation', [0]) 
                                 for muscle in self.biological_system.muscles_names]
                    metric_values[metric] = max(max(act) for act in activations if len(act) > 0)
                
                elif metric == 'joint_angle_range':
                    # Range of joint angles
                    joint_angles = time_series.get(f'{self.biological_system.associated_joint}_angle', [0])
                    metric_values[metric] = max(joint_angles) - min(joint_angles) if len(joint_angles) > 0 else 0
                
                elif metric == 'muscle_force_variance':
                    # Variance in muscle forces
                    forces = []
                    for muscle in self.biological_system.muscles_names:
                        muscle_force = time_series.get(f'{muscle}_force', [0])
                        if len(muscle_force) > 0:
                            forces.extend(muscle_force)
                    metric_values[metric] = np.var(forces) if forces else 0
                
                elif metric == 'neural_synchrony':
                    # Measure of neural synchrony (simplified)
                    # This is a placeholder - would need more sophisticated analysis
                    spike_times_list = [times for times in spikes.values() if len(times) > 0]
                    if spike_times_list:
                        all_spike_times = np.concatenate(spike_times_list)
                        metric_values[metric] = np.std(np.diff(np.sort(all_spike_times)))
                    else:
                        metric_values[metric] = 0
                
                else:
                    # Unknown metric
                    metric_values[metric] = np.nan
                    
        except Exception as e:
            warnings.warn(f"Error calculating metrics: {e}")
            for metric in metrics:
                metric_values[metric] = np.nan
        
        return metric_values
    
    def _generate_sensitivity_summary(self, results: Dict) -> Dict:
        """Generate summary statistics for sensitivity analysis."""
        
        summary = {}
        
        for analysis_type, df in results.items():
            if analysis_type == 'summary' or df.empty:
                continue
                
            summary[analysis_type] = {}
            
            # Get numeric columns (metrics)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols if col not in 
                          ['parameter_type', 'parameter_name']]
            
            for metric in metric_cols:
                if metric in df.columns:
                    summary[analysis_type][metric] = {
                        'mean': df[metric].mean(),
                        'std': df[metric].std(),
                        'min': df[metric].min(),
                        'max': df[metric].max(),
                        'coefficient_of_variation': df[metric].std() / df[metric].mean() 
                                                  if df[metric].mean() != 0 else np.inf
                    }
        
        return summary
    
    def _save_sensitivity_results(self, results: Dict, base_path: str):
        """Save sensitivity analysis results to files."""
        
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Save DataFrames
        for analysis_type, df in results.items():
            if analysis_type != 'summary' and not df.empty:
                df.to_csv(os.path.join(base_path, f'sensitivity_{analysis_type}.csv'), 
                         index=False)
        
        # Save summary as JSON
        import json
        summary_path = os.path.join(base_path, 'sensitivity_summary.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary_serializable = json.loads(json.dumps(results['summary'], default=convert_numpy))
        
        with open(summary_path, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
    
    def plot_sensitivity_results(self, analysis_type: str = 'all', 
                               metrics: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """
        Plot sensitivity analysis results.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis to plot ('biophysical', 'connections', 'neuron_counts', or 'all')
        metrics : list, optional
            Specific metrics to plot
        save_path : str, optional
            Path to save plots
        """
        
        if not self.sensitivity_results:
            raise ValueError("No sensitivity analysis results available. Run perform_sensitivity_analysis first.")
        
        types_to_plot = (['biophysical', 'connections', 'neuron_counts'] 
                        if analysis_type == 'all' else [analysis_type])
        
        for analysis in types_to_plot:
            df = self.sensitivity_results.get(analysis, pd.DataFrame())
            if df.empty:
                continue
                
            # Get metric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols 
                          if col not in ['parameter_type', 'parameter_name']]
            
            if metrics:
                metric_cols = [col for col in metric_cols if col in metrics]
            
            # Create subplots
            n_metrics = len(metric_cols)
            if n_metrics == 0:
                continue
                
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(metric_cols):
                ax = axes[i]
                
                # Group by parameter name and plot
                for param_name in df['parameter_name'].unique():
                    param_data = df[df['parameter_name'] == param_name]
                    
                    # Find the parameter column for x-axis
                    param_cols = [col for col in param_data.columns 
                                 if col.startswith(param_name) and col != 'parameter_name']
                    
                    if param_cols:
                        x_col = param_cols[0]
                        ax.plot(param_data[x_col], param_data[metric], 
                               'o-', label=param_name, alpha=0.7)
                
                ax.set_xlabel('Parameter Value')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Sensitivity - {analysis.title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/sensitivity_{analysis}.png", dpi=300, bbox_inches='tight')
            
            plt.show()
    
  
