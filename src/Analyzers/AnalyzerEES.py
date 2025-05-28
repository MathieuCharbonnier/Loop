import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd

from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from .Visualization_helpers.plot_parameters_variations import  plot_ees_analysis_results
from .BiologicalSystems.BiologicalSystem import BiologicalSystem

class AnalyzerEES:
  
    def __init__(self, system: BiologicalSystem):
        self.original_system = system
        self.results=None
          
    def analyze_frequency_effects(self, freq_range=[0, 40, 80]*hertz, base_ees_params=None, n_iterations=20, time_step=0.1*ms, seed=42):
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
        if base_ees_params is None:
            
            base_ees_params={'frequency': 50*hertz,
                             'intensity': 0.5}
        vary_param = {
            'param_name': 'frequency',
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
        self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            n_iterations,
            time_step, 
            seed
        )
      
    def plot(self):
      
        if self.results is None:
            raise ValueError("No results to plots!")
        else:
            plot_ees_analysis_results(self.results, save_dir="balance_analysis", seed=seed) 

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
        # Run simulations for each parameter value
        for i, value in enumerate(param_values):
            print(f"  Computing {param_label} = {value} ({i+1}/{len(param_values)})")
            
            # Create a copy of the base parameters
            current_params = BiologicalSystem.copy_brian_dict(param_dict)
            
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
                
        
        self.results = {
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
        
      
     
   
   
