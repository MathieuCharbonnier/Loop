import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd
from copy import deepcopy
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
from .Visualization_helpers.plot_parameters_variations import plot_delay_results, plot_excitability_results, plot_twitch_results, plot_ees_analysis_results
from .BiologicalSystems.BiologicalSystem import BiologicalSystem

class AnalyzerReflex:
  
    def __init__(self, system : BiologicalSystem):
        self.original_system = system
    
      
    def run_delay (self, torque_profile=None, delay_values=None, 
                    threshold_values=None, duration=1*second, 
                    time_step=0.1, fast_type_mu=True, seed=41):
        """
        Analyze reflex behavior by varying one parameter at a time.
        
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
        if torque_profile is None:
            torque_profile={
                "type": "bump",
                "t_peak": 100 * ms,
                "sigma": 50 * ms,
                "max_amplitude": 25,
                "sustained_amplitude": 10
            }
        if delay_values is None:
            delay_values = [10, 25, 50, 75, 100]*ms  
        if threshold_values is None:
            threshold_values = [-45, -50, -55]*mV  
                        
        delay_results = []
        fast_twitch_results = []
        threshold_results = []
        
        # 1. Vary delay (reaction time)
        print("Running delay variation analysis...")
        for delay in tqdm(delay_values, desc="Varying delay"):
       
            n_iterations = int(duration / delay)
           
            new_system = self.original_system.clone_with(reaction_time=delay)
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
                        
        # 2. Vary fast twitch parameter
        print("Running fast twitch variation analysis...")
        fast_twitch_values = [False, True]
        
        for fast in tqdm(fast_twitch_values, desc="Varying fast twitch parameter"):
            new_system = self.original_system.clone_with(fast_type_mu=fast)
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            fast_twitch_results.append((fast, spikes, time_series))
            
        plot_twitch_results(fast_twitch_results, self.original_system.muscles_names, self.original_system.associated_joint)
                        
        # 3. Vary threshold voltage
        print("Running threshold variation analysis...")
        for threshold in tqdm(threshold_values, desc="Varying threshold voltage"):
            bio_phys=BiologicalSystem.copy_brian_dict(self.original_system.biohysical_params)
            bio_phys['v_threshold'] = threshold
            new_system = self.original_system.clone_with(biophysical_params=bio_phys)
           
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

    
