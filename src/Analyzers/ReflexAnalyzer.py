import os
import numpy as np

from tqdm import tqdm
from brian2 import *
import pandas as pd

from typing import Dict, List, Tuple, Any, Optional, Callable
import matplotlib.pyplot as plt

from ..BiologicalSystems.BiologicalSystem import BiologicalSystem


class ReflexAnalyzer:
    """
    Analyzer for reflex behavior under different parameter variations.
    
    This class provides methods to analyze how different parameters (delay, muscle fiber type,
    neuron excitability) affect reflex responses in a biological system.
    """
  
    def __init__(self, system: BiologicalSystem):
        """
        Initialize the analyzer with a biological system.
        
        Parameters:
        -----------
        system : BiologicalSystem
            The biological system to analyze
        """
        self.original_system = system.clone_with()
        self.delay_results = []
        self.fast_twitch_results = []
        self.threshold_results = []
      
    def plot (self, analysis_type: str = "all", output_dir: str = "reflex_analysis"):
        """
        Unified plotting function for all analysis results.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis to plot ("delay", "twitch", "threshold", or "all")
        output_dir : str
            Directory to save the plots
        """
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory '{output_dir}' for saving plots")
        
        plots_to_generate = []
        
        # Determine which plots to generate
        if analysis_type == "all":
            if self.delay_results:
                plots_to_generate.append(("delay", self.delay_results, "Joint Angle and Muscle Activation for different reaction times"))
            if self.fast_twitch_results:
                plots_to_generate.append(("twitch", self.fast_twitch_results, "Joint Angle and Muscle Activation of Slow and Fast type motor units"))
            if self.threshold_results:
                plots_to_generate.append(("threshold", self.threshold_results, "Effect of neuron excitability on Joint Angle and Muscle Activation"))
        else:
            if analysis_type == "delay" and self.delay_results:
                plots_to_generate.append(("delay", self.delay_results, "Joint Angle and Muscle Activation for different reaction times"))
            elif analysis_type == "twitch" and self.fast_twitch_results:
                plots_to_generate.append(("twitch", self.fast_twitch_results, "Joint Angle and Muscle Activation of Slow and Fast type motor units"))
            elif analysis_type == "threshold" and self.threshold_results:
                plots_to_generate.append(("threshold", self.threshold_results, "Effect of neuron excitability on Joint Angle and Muscle Activation"))
            else:
                print(f"No {analysis_type} results to plot. Run the corresponding analysis first.")
                return
        
        if not plots_to_generate:
            print("No results to plot. Run analyses first.")
            return
        
        # Generate plots
        for plot_type, results, title in plots_to_generate:
            fig, axs = plt.subplots(len(results), 2, figsize=(12, 3*len(results)), sharex=True)
            if len(results) == 1:
                axs = axs.reshape(1, -1)
            
            for i, result_data in enumerate(results):
                param_value, spikes, time_series = result_data
                
                # Plot joint angle
                joint_col = f'Joint_{self.original_system.associated_joint}'
                if joint_col in time_series:
                    axs[i, 0].plot(time_series['Time'], time_series[joint_col], 'b-')
                
                # Set appropriate y-label based on analysis type
                if plot_type == "delay":
                    ylabel = f"Delay = {int(param_value/ms)} ms\nJoint angle (deg)"
                elif plot_type == "twitch":
                    label = "Fast MU" if param_value else "Slow MU"
                    ylabel = f"{label}\nJoint angle (deg)"
                elif plot_type == "threshold":
                    ylabel = f"Vth = {int(param_value/mV)} mV\nJoint angle (deg)"
                
                axs[i, 0].set_ylabel(ylabel)
                
                # Plot muscle activations
                for muscle in self.original_system.muscles_names:
                    activation_col = f"Activation_{muscle}"
                    if activation_col in time_series:
                        axs[i, 1].plot(time_series['Time'], time_series[activation_col], label=muscle)
                
                axs[i, 1].set_ylabel("Muscle activation")
                axs[i, 1].legend()
            
            axs[-1, 0].set_xlabel("Time (s)")
            axs[-1, 1].set_xlabel("Time (s)")
            fig.suptitle(title)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save with appropriate filename
            filename = f'{plot_type}_variation.png'
            fig.savefig(os.path.join(output_dir, filename))
            plt.show()
            print(f"Saved {filename}")
          
    def run_delay(self, torque_profile=None, delay_values=None, 
                  duration=1*second, time_step=0.1*ms, seed=41):
        """
        Analyze reflex behavior by varying delay parameter.
        
        Parameters:
        -----------
        torque_profile : dict, optional
            Dictionary with torque profile parameters
        delay_values : list, optional
            List of delay values to test (in ms)
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        list
            List of tuples containing (delay, spikes, time_series) for each simulation
        """
        
        # Set default values if not provided
        if torque_profile is None:
            torque_profile = {
                "type": "bump",
                "t_peak": 100 * ms,
                "sigma": 50 * ms,
                "max_amplitude": 25,
                "sustained_amplitude": 10
            }
        if delay_values is None:
            delay_values = [10, 50, 100] * ms  
        
        # Clear previous results
        self.delay_results = []
        
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
                base_output_path=None
            )
            
            self.delay_results.append((delay, spikes, time_series))
        
        return self.delay_results
                      
    def run_mu_type(self, torque_profile=None, duration=1*second, 
                    time_step=0.1*ms, seed=41):
        """
        Analyze reflex behavior by varying motor unit type (fast vs slow twitch).
        
        Parameters:
        -----------
        torque_profile : dict, optional
            Dictionary with torque profile parameters
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        list
            List of tuples containing (fast_twitch_bool, spikes, time_series) for each simulation
        """
        
        # Set default values if not provided
        if torque_profile is None:
            torque_profile = {
                "type": "bump",
                "t_peak": 100 * ms,
                "sigma": 50 * ms,
                "max_amplitude": 25,
                "sustained_amplitude": 10
            }
                        
        n_iterations = int(duration / self.original_system.reaction_time)
        
        # Clear previous results
        self.fast_twitch_results = []
                        
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
                base_output_path=None
            )
            
            self.fast_twitch_results.append((fast, spikes, time_series))
        
        return self.fast_twitch_results

    def run_excitability(self, torque_profile=None, threshold_values=None, 
                        duration=1*second, time_step=0.1*ms, seed=41):
        """
        Analyze reflex behavior by varying neuron excitability threshold.
        
        Parameters:
        -----------
        torque_profile : dict, optional
            Dictionary with torque profile parameters
        threshold_values : list, optional
            List of threshold values to test (in mV)
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        list
            List of tuples containing (threshold, spikes, time_series) for each simulation
        """
        
        # Set default values if not provided
        if torque_profile is None:
            torque_profile = {
                "type": "bump",
                "t_peak": 100 * ms,
                "sigma": 50 * ms,
                "max_amplitude": 25,
                "sustained_amplitude": 10
            }
        if threshold_values is None:
            threshold_values = [-45, -50, -55] * mV
        
        n_iterations = int(duration / self.original_system.reaction_time)
        
        # Clear previous results
        self.threshold_results = []
                        
        print("Running threshold variation analysis...")
        for threshold in tqdm(threshold_values, desc="Varying threshold voltage"):
            bio_phys = BiologicalSystem.copy_brian_dict(self.original_system.biophysical_params)
            bio_phys['threshold_v'] = threshold
            new_system = self.original_system.clone_with(biophysical_params=bio_phys)
           
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None
            )
            
            self.threshold_results.append((threshold, spikes, time_series))
        
        return self.threshold_results

    def run(self, torque_profile=None, delay_values=None, 
                                  threshold_values=None, duration=1*second, 
                                  time_step=0.1*ms, seed=41, output_dir="reflex_analysis"):
        """
        Run all analysis methods and generate plots.
        
        Parameters:
        -----------
        torque_profile : dict, optional
            Dictionary with torque profile parameters
        delay_values : list, optional
            List of delay values to test (in ms)
        threshold_values : list, optional
            List of threshold values to test (in mV)
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        seed : int
            Random seed for reproducibility
        output_dir : str
            Directory to save the plots
            
        Returns:
        --------
        dict
            Dictionary containing all analysis results with keys 'delay', 'twitch', 'threshold'
        """
        
        print("Starting reflex analysis...")
        
        # Run all analyses and collect results
        results = {}
        results['delay'] = self.run_delay(torque_profile, delay_values, duration, time_step, seed)
        results['twitch'] = self.run_mu_type(torque_profile, duration, time_step, seed)
        results['threshold'] = self.run_excitability(torque_profile, threshold_values, duration, time_step, seed)
        
        # Generate all plots using the unified plotting function
        print("Generating plots...")
        self.plot("all", output_dir)
        
        print(f"Analysis complete! Results saved to {output_dir}")
        return results
