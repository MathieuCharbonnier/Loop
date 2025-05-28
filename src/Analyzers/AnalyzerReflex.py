import copy
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
from brian2 import *
import pandas as pd
from copy import deepcopy
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings
import matplotlib.pyplot as plt
from .Visualization_helpers.plot_parameters_variations import plot_delay_results, plot_excitability_results, plot_twitch_results, plot_ees_analysis_results
from .BiologicalSystems.BiologicalSystem import BiologicalSystem


class AnalyzerReflex:
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
      
    def plot_delay(self, output_dir="reflex_analysis"):
        """
        Plot delay variation results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the plots
        """
        if not self.delay_results:
            print("No delay results to plot. Run run_delay() first.")
            return

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory '{output_dir}' for saving plots")
        
        fig1, axs1 = plt.subplots(len(self.delay_results), 2, figsize=(15, 4*len(self.delay_results)), sharex=True)
        if len(self.delay_results) == 1:
            axs1 = axs1.reshape(1, -1)
        
        for i, (delay, spikes, time_series) in enumerate(self.delay_results):
            # Plot joint angle
            joint_col = f'Joint_{self.original_system.associated_joint}'
            if joint_col in time_series:
                axs1[i, 0].plot(time_series['Time'], time_series[joint_col], 'b-')
            axs1[i, 0].set_ylabel(f"Delay = {int(delay/ms)} ms\nJoint angle (deg)")
            
            # Plot muscle activations
            for muscle in self.original_system.muscles_names:
                activation_col = f"Activation_{muscle}"
                if activation_col in time_series:
                    axs1[i, 1].plot(time_series['Time'], time_series[activation_col], label=muscle)
            
            axs1[i, 1].set_ylabel("Muscle activation")
            axs1[i, 1].legend()
        
        axs1[-1, 0].set_xlabel("Time (s)")
        axs1[-1, 1].set_xlabel("Time (s)")
        fig1.suptitle("Joint Angle and Muscle Activation for different reaction times")
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        fig1.savefig(os.path.join(output_dir, f'delay_variation_{timestamp}.png'))
        plt.close(fig1)

    def plot_twitch_results(self, output_dir="reflex_analysis"):
        """
        Plot muscle fiber type variation results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the plots
        """
        if not self.fast_twitch_results:
            print("No fast twitch results to plot. Run run_mu_type() first.")
            return

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory '{output_dir}' for saving plots")
        
        fig2, axs2 = plt.subplots(len(self.fast_twitch_results), 2, figsize=(15, 4*len(self.fast_twitch_results)), sharex=True)
        if len(self.fast_twitch_results) == 1:
            axs2 = axs2.reshape(1, -1)
        
        for i, (fast, spikes, time_series) in enumerate(self.fast_twitch_results):
            # Plot joint angle
            joint_col = f'Joint_{self.original_system.associated_joint}'
            if joint_col in time_series:
                axs2[i, 0].plot(time_series['Time'], time_series[joint_col], 'b-')
            label = "Fast type Motor Unit" if fast else "Slow type Motor Unit"
            axs2[i, 0].set_ylabel(f"{label}\nJoint angle (deg)")
            
            # Plot muscle activations
            for muscle in self.original_system.muscles_names:
                activation_col = f"Activation_{muscle}"
                if activation_col in time_series:
                    axs2[i, 1].plot(time_series['Time'], time_series[activation_col], label=muscle)
            
            axs2[i, 1].set_ylabel("Muscle activation")
            axs2[i, 1].legend()
        
        axs2[-1, 0].set_xlabel("Time (s)")
        axs2[-1, 1].set_xlabel("Time (s)")
        fig2.suptitle("Joint Angle and Muscle Activation of Slow and Fast type motor units")
        fig2.tight_layout(rect=[0, 0, 1, 0.95])
        fig2.savefig(os.path.join(output_dir, f'fast_twitch_variation_{timestamp}.png'))
        plt.close(fig2)

    def plot_excitability_results(self, output_dir="reflex_analysis"):
        """
        Plot neuron excitability variation results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the plots
        """
        if not self.threshold_results:
            print("No threshold results to plot. Run run_excitability() first.")
            return

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory '{output_dir}' for saving plots")
        
        fig3, axs3 = plt.subplots(len(self.threshold_results), 2, figsize=(15, 4*len(self.threshold_results)), sharex=True)
        if len(self.threshold_results) == 1:
            axs3 = axs3.reshape(1, -1)
        
        for i, (threshold, spikes, time_series) in enumerate(self.threshold_results):
            # Plot joint angle
            joint_col = f'Joint_{self.original_system.associated_joint}'
            if joint_col in time_series:
                axs3[i, 0].plot(time_series['Time'], time_series[joint_col], 'b-')
            axs3[i, 0].set_ylabel(f"Threshold = {int(threshold/mV)} mV\nJoint angle (deg)")
            
            # Plot muscle activations
            for muscle in self.original_system.muscles_names:
                activation_col = f"Activation_{muscle}"
                if activation_col in time_series:
                    axs3[i, 1].plot(time_series['Time'], time_series[activation_col], label=muscle)
            
            axs3[i, 1].set_ylabel("Muscle activation")
            axs3[i, 1].legend()
        
        axs3[-1, 0].set_xlabel("Time (s)")
        axs3[-1, 1].set_xlabel("Time (s)")
        fig3.suptitle("Effect of neuron excitability on Joint Angle and Muscle Activation")
        fig3.tight_layout(rect=[0, 0, 1, 0.95])
        fig3.savefig(os.path.join(output_dir, f'threshold_variation_{timestamp}.png'))
        plt.close(fig3)
          
    def run_delay(self, torque_profile=None, delay_values=None, 
                  duration=1*second, time_step=0.1, seed=41):
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
            delay_values = [10, 25, 50, 75, 100] * ms  
        
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
                      
    def run_mu_type(self, torque_profile=None, duration=1*second, 
                    time_step=0.1, seed=41):
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
                base_output_path=None,
                plot=False
            )
            
            self.fast_twitch_results.append((fast, spikes, time_series))

    def run_excitability(self, torque_profile=None, threshold_values=None, 
                        duration=1*second, time_step=0.1, seed=41):
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
            bio_phys['v_threshold'] = threshold
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

    def run_comprehensive_analysis(self, torque_profile=None, delay_values=None, 
                                  threshold_values=None, duration=1*second, 
                                  time_step=0.1, seed=41, output_dir="reflex_analysis"):
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
        """
        
        print("Starting comprehensive reflex analysis...")
        
        # Run all analyses
        self.run_delay(torque_profile, delay_values, duration, time_step, seed)
        self.run_mu_type(torque_profile, duration, time_step, seed)
        self.run_excitability(torque_profile, threshold_values, duration, time_step, seed)
        
        # Generate all plots
        print("Generating plots...")
        self.plot_delay(output_dir)
        self.plot_twitch_results(output_dir)
        self.plot_excitability_results(output_dir)
        
        print(f"Analysis complete! Results saved to {output_dir}")

