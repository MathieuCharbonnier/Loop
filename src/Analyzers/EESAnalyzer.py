import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any, Optional, Callable
import warnings


from ..BiologicalSystems.BiologicalSystem import BiologicalSystem
from ..helpers.copy_brian_dict import copy_brian_dict

class EESAnalyzer:
    """
    A class for analyzing Epidural Electrical Stimulation (EES) effects on biological systems.
    
    This class provides methods to analyze various EES parameters and their effects on
    muscle activation, motor neuron recruitment, and joint dynamics.
    """
    
    def __init__(self, system: BiologicalSystem):
        """
        Initialize the EES analyzer.
        
        Parameters:
        -----------
        system : BiologicalSystem
            The biological system to analyze
        """
        self.original_system = system
        self.results = None
        self._default_ees_params = {
            'frequency': 50*hertz,
            'intensity': 0.5,
            'site': 'L5'
        }
    
    def analyze_frequency_effects(self, freq_range=None, base_ees_params=None, torque_profile=None,
                                n_iterations=20, time_step=0.1*ms):
        """
        Analyze the effects of varying EES frequency with fixed afferent and efferent recruitment.
        
        Parameters:
        -----------
        freq_range : array-like, optional
            Range of frequencies to analyze (default: [0, 40, 80] Hz)
        base_ees_params : dict, optional
            Base parameters for EES (uses defaults if None)
        n_iterations : int
            Number of iterations for each simulation
        time_step : Quantity
            Time step for simulations (in ms, will be converted to brian2 units)
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        if freq_range is None:
            freq_range = [0,40,80] * hertz
        
        if base_ees_params is None:
            base_ees_params = self._default_ees_params.copy()
        
        vary_param = {
            'param_name': 'frequency',
            'values': freq_range,
            'label': 'EES Frequency (Hz)'
        }

        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            torque_profile,
            n_iterations,
            time_step
        )
        
        return results

    def analyze_intensity_effects(self, intensity_range, base_ees_params=None, torque_profile=None, 
                                n_iterations=20, time_step=0.1*ms):
        """
        Analyze the effects of varying stimulation intensity.
        
        Parameters:
        -----------
        intensity_range : array-like
            Range of normalized intensity values to analyze
        base_ees_params : dict, optional
            Base parameters for EES (uses defaults if None)
        n_iterations : int
            Number of iterations for each simulation
        time_step : Quantity
            Time step for simulations (in ms, will be converted to brian2 units)
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        if base_ees_params is None:
            base_ees_params = self._default_ees_params.copy()
            
        vary_param = {
            'param_name': 'intensity',
            'values': intensity_range,
            'label': 'Stimulation Intensity'
        }
        
        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param, 
            torque_profile,
            n_iterations,
            time_step
        )
        
        return results
      
    def analyze_stimulation_sites(self, different_sites=None, base_ees_params=None, torque_profile=None,
                                             n_iterations=20, time_step=0.1*ms):
        """
        Analyze the effects of unbalanced afferent recruitment between antagonistic muscles.
        
        Parameters:
        -----------
        b_range : array-like
            Range of balance values to analyze (0-1 where 0.5 is balanced)
        base_ees_params : dict, optional
            Base parameters for EES (uses defaults if None)
        n_iterations : int
            Number of iterations for each simulation
        time_step : Quantity
            Time step for simulations
        
        Returns:
        --------
        dict
            Analysis results
        """
        if base_ees_params is None:
            base_ees_params = self._default_ees_params.copy()
        if different_sites is None:
            different_sites=['L3', 'L4','L5','S1', 'S2']
        vary_param = {
            'param_name': 'site',
            'values': different_sites,
            'label': 'Stimulation site variations'
        }
        
        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            torque_profile,
            n_iterations,
            time_step
        )
        
        return results
    
       
    def plot (self, results=None, save_dir="stimulation_analysis", show_plots=True):
        """
        Plot the analysis results.
        
        Parameters:
        -----------
        results : dict, optional
            Results dictionary from analysis methods (uses self.results if None)
        save_dir : str
            Directory to save plots
        seed : int
            Random seed (for filename generation)
        show_plots : bool
            Whether to display plots
        """
        if results is None:
            if self.results is None:
                raise ValueError("No results to plot! Run an analysis method first or provide results.")
            results = self.results
        
        self._plot_ees_analysis_results(results, save_dir, show_plots)
        
        # Store results for future plotting
        self.results = results
    
    def _compute_ees_parameter_sweep(self, param_dict, vary_param, torque_profile, n_iterations, 
                                   time_step=0.1*ms):
        """
        Compute EES stimulation analysis by varying a parameter of interest.
        
        Parameters:
        -----------
        param_dict : dict
            Dictionary containing all EES parameters with their default values
        vary_param : dict
            Dictionary specifying which parameter to vary with its range of values
            Format: {'param_name': str, 'values': list, 'label': str}
        n_iterations : int
            Number of iterations for each simulation
        time_step : Quantity
            Time step for simulation (in ms)
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        
        # Get parameter info
        param_name = vary_param['param_name']
        param_values = vary_param['values']
        param_label = vary_param['label']
        
        # Initialize storage for results
        simulation_data = []
        spikes_data = []
        time_data = None
        
        print(f"Running parameter sweep for {param_label}...")
        
        # Run simulations for each parameter value
        for i, value in enumerate(tqdm(param_values, desc=f"Analyzing {param_label}")):
            print(f"  Computing {param_label} = {value} ({i+1}/{len(param_values)})")
            
            current_params = copy_brian_dict(param_dict)
   
            
            # Update the parameter we're varying
            current_params[param_name] = value
         
            # Run simulation
            spikes, main_data = self.original_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=current_params,
                torque_profile=torque_profile,
                base_output_path=None
            )
            
            # Store results
            simulation_data.append(main_data)
            spikes_data.append(spikes)
            
            # Extract time data (same for all simulations)
            if time_data is None:
                if isinstance(main_data, dict) and 'Time' in main_data:
                    time_data = main_data['Time']
                else:
                    print("Warning: Could not extract time data from simulation results")
        
        # Prepare results dictionary
        results = {
            'param_values': param_values,
            'param_name': param_name,
            'param_label': param_label,
            'time_data': time_data,
            'simulation_data': simulation_data,
            'spikes_data': spikes_data,
            'muscle_names': getattr(self.original_system, 'muscles_names', []),
            'associated_joint': getattr(self.original_system, 'associated_joint', 'Unknown'),
            'num_muscles': getattr(self.original_system, 'number_muscles', 0)
        }
        
        # Store results in the instance
        self.results = results
        
        return results
    
    def _plot_ees_analysis_results(self, results, save_dir="stimulation_analysis", show_plots=True):
        """
        Plot the results from EES parameter sweep analysis.

        Parameters:
        -----------
        results : dict
            Results dictionary from compute_ees_parameter_sweep
        save_dir : str
            Directory to save plots
        seed : int
            Random seed (for filename generation)
        show_plots : bool
            Whether to display plots
        """

        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory '{save_dir}' for saving plots")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract data from results
        param_values = results['param_values']
        param_name = results['param_name']
        param_label = results['param_label']
        simulation_data = results['simulation_data']
        spikes_data = results['spikes_data']
        muscles_names = results['muscle_names']
        associated_joint = results['associated_joint']
        num_muscles = results['num_muscles']

        time_series_to_plot = ['Ia_rate_baseline', 'II_rate_baseline', 'MN_rate', 'Activation', 'Stretch']

        # Define muscle colors
        muscle_colors = {
            muscles_names[i]: plt.cm.tab10(i % 10) for i in range(len(muscles_names))
        }

        n_rows = len(param_values)

        # --- Plot time series data ---
        figs = {}
        axs_dict = {}
        for var in time_series_to_plot:
            fig, axs = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
            if n_rows == 1:
                axs = [axs]
            figs[var] = fig
            axs_dict[var] = axs

            for i, (value, main_data) in enumerate(zip(param_values, simulation_data)):
                ax = axs[i]
                ax.set_title(f"{param_label}: {value}")
               
                ylabel = var.replace('_', ' ').title()
                ax.set_ylabel(f"{ylabel} (Hz)" if "rate" in var else f"{ylabel} (dimless)")
    
                if 'Time' in main_data:
                    time_data = main_data['Time']
                    for idx, muscle_name in enumerate(muscles_names):
                        col_name = f"{var}_{muscle_name}"
                
                        if col_name in main_data:
                            ax.plot(time_data, main_data[col_name], label=muscle_name,
                                    color=muscle_colors[muscle_name])
             axs[-1].set_xlabel("Time (s)")            
            if muscles_names:
                fig.legend(muscles_names, loc='upper right')
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{var}_{timestamp}.png"))

        # --- Plot joint angle ---
        if associated_joint != 'Unknown':
            fig_joint, axs_joint = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
            if n_rows == 1:
                axs_joint = [axs_joint]
            for i, (value, main_data) in enumerate(zip(param_values, simulation_data)):
                ax = axs_joint[i]
                ax.set_title(f"{param_label}: {value}")
                ax.set_ylabel(f"{associated_joint} (deg)")
                
                joint_col = f'Joint_{associated_joint}'
                if joint_col in main_data and 'Time' in main_data:
                    ax.plot(main_data['Time'], main_data[joint_col],
                            color='darkred', label='Joint Angle')
                    ax.legend()
            axs_joint[-1].set_xlabel("Time (s)")
            fig_joint.tight_layout()
            fig_joint.savefig(os.path.join(save_dir, f"joint_angle_{timestamp}.png"))

        # --- Raster plot for MN spikes ---
        fig_raster, axs_raster = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
        if n_rows == 1:
            axs_raster = [axs_raster]
        for i, (value, spikes) in enumerate(zip(param_values, spikes_data)):
            ax = axs_raster[i]
            ax.set_title(f"MN Raster Plot — {param_label}: {value}")
            ax.set_ylabel("Neuron ID")

            for idx, muscle_name in enumerate(muscles_names):
                if muscle_name in spikes and 'MN' in spikes[muscle_name]:
                    for neuron_id, neuron_spikes in spikes[muscle_name]['MN'].items():
                        if neuron_spikes:
                            ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id),
                                    '.', markersize=4, color=muscle_colors[muscle_name])
        axs_raster[-1].set_xlabel("Time (s)")
        fig_raster.tight_layout()
        fig_raster.savefig(os.path.join(save_dir, f"mn_raster_{timestamp}.png"))

        # --- Plot MN Recruitment Fraction ---
        recruitment_fractions = []
        for spikes in spikes_data:
            total_recruited = 0
            total_neurons = 0
            for muscle_name in muscles_names:
                if muscle_name in spikes and 'MN' in spikes[muscle_name]:
                    MN_dict = spikes[muscle_name]['MN']
                    total_neurons += len(MN_dict)
                    total_recruited += sum(1 for s in MN_dict.values() if len(s) > 0)
            recruitment_fractions.append(total_recruited / total_neurons if total_neurons else 0)

        fig_recruit, ax_recruit = plt.subplots(figsize=(10, 6))
        ax_recruit.plot(param_values, recruitment_fractions, marker='o', linestyle='-', color='blue')
        ax_recruit.set_title("Fraction of Recruited MNs vs " + param_label)
        ax_recruit.set_xlabel(param_label)
        ax_recruit.set_ylabel("Fraction of Recruited Motoneurons")
        ax_recruit.grid(True, linestyle='--', alpha=0.3)
        fig_recruit.tight_layout()
        fig_recruit.savefig(os.path.join(save_dir, f"mn_recruitment_fraction_{timestamp}.png"))

        print(f"All plots saved to '{save_dir}' directory.")

        # Coactivation analysis for 2-muscle systems
        if num_muscles == 2: 
            self._plot_coactivation_analysis(results, save_dir, timestamp)
        
        if show_plots:
            plt.show()

    def _plot_coactivation_analysis(self, results, save_dir, timestamp):
        """
        Perform and plot coactivation analysis for 2-muscle systems.
        
        Parameters:
        -----------
        results : dict
            Results dictionary from compute_ees_parameter_sweep
        save_dir : str
            Directory to save plots
        timestamp : str
            Timestamp for filename
        seed : int
            Random seed for filename
        """
        
        print("\nPerforming flexor-extensor activation analysis...")
        
        param_values = results['param_values']
        param_name = results['param_name']
        param_label = results['param_label']
        simulation_data = results['simulation_data']
        muscles_names = results['muscle_names']
        time_data=simulation_data[0]['Time']
        # Extract activities - fix the syntax error from original code
        activities = []
        for muscle_name in muscles_names:
            muscle_activities = []
            for i, main_data in enumerate(simulation_data):
                activation_col = f'Activation_{muscle_name}'
                if activation_col in main_data:
                    muscle_activities.append(main_data[activation_col].values if hasattr(main_data[activation_col], 'values') else main_data[activation_col])
                else:
                    muscle_activities.append(np.zeros_like(time_data))
            activities.append(muscle_activities)
        
        activities = np.array(activities)
        
        # Define activation threshold
        activation_threshold = 0.1
        
        flexor_idx = 0  # First muscle (e.g., tibialis anterior - flexor)
        extensor_idx = 1  # Second muscle (e.g., medial gastrocnemius - extensor)
        
        # Calculate grid layout
        n_cols = 2
        n_rows = math.ceil(len(param_values) / n_cols)
        
        # Create figures
        fig_scatter, axs_scatter = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        fig_scatter.suptitle("Flexor vs Extensor Activity")
        
        fig_coact, axs_coact = plt.subplots(1, 2, figsize=(15, 6))
        fig_coact.suptitle("Coactivation Analysis")
        
        fig_time, axs_time = plt.subplots(1, 2, figsize=(15, 6))
        fig_time.suptitle("Muscle Activation Time Analysis")
        
        # Ensure axs_scatter is 2D array
        if n_rows == 1 and n_cols == 1:
            axs_scatter = np.array([[axs_scatter]])
        elif n_rows == 1:
            axs_scatter = axs_scatter.reshape(1, -1)
        elif n_cols == 1:
            axs_scatter = axs_scatter.reshape(-1, 1)
        
        # Initialize metric arrays
        min_coactivation = np.zeros(len(param_values))
        product_coactivation = np.zeros(len(param_values))
        flexor_active_time = np.zeros(len(param_values))
        extensor_active_time = np.zeros(len(param_values))
        
        # Get simulation parameters
        if hasattr(time_data, 'iloc'):
            total_time = time_data.iloc[-1]
            dt = time_data.iloc[1] - time_data.iloc[0] if len(time_data) > 1 else 0.001
        else:
            total_time = time_data[-1] if len(time_data) > 0 else 1.0
            dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0.001
        
        # Analyze each parameter value
        for i, value in enumerate(param_values):
            flexor_activation = activities[flexor_idx][i]
            extensor_activation = activities[extensor_idx][i]
            
            # Scatter plot
            row = i // n_cols
            col = i % n_cols
            ax = axs_scatter[row, col]
            
            ax.scatter(flexor_activation, extensor_activation, alpha=0.6, s=10)
            ax.set_xlabel("Flexor Activation")
            ax.set_ylabel("Extensor Activation")
            ax.set_title(f"{param_label}: {value}")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Diagonal reference line
            max_val = max(np.max(flexor_activation), np.max(extensor_activation))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
            
            # Calculate coactivation metrics
            min_coact = np.sum(np.minimum(flexor_activation, extensor_activation)) * dt / total_time
            min_coactivation[i] = min_coact
            
            prod_coact = np.sum(flexor_activation * extensor_activation) * dt / total_time
            product_coactivation[i] = prod_coact
            
            # Calculate activation time
            flexor_active = np.sum(flexor_activation > activation_threshold) * dt / total_time
            extensor_active = np.sum(extensor_activation > activation_threshold) * dt / total_time
            
            flexor_active_time[i] = flexor_active
            extensor_active_time[i] = extensor_active
        
        # Plot coactivation metrics
        axs_coact[0].plot(param_values, min_coactivation, 'o-', linewidth=2)
        axs_coact[0].set_xlabel(param_label)
        axs_coact[0].set_ylabel("Min-based Coactivation")
        axs_coact[0].set_title("Coactivation: min(flexor, extensor)")
        axs_coact[0].grid(True)
        
        axs_coact[1].plot(param_values, product_coactivation, 'o-', linewidth=2, color='orange')
        axs_coact[1].set_xlabel(param_label)
        axs_coact[1].set_ylabel("Product-based Coactivation")
        axs_coact[1].set_title("Coactivation: flexor * extensor")
        axs_coact[1].grid(True)
        
        # Plot activation time metrics
        axs_time[0].plot(param_values, flexor_active_time, 'o-', linewidth=2, color='blue', label='Flexor')
        axs_time[0].plot(param_values, extensor_active_time, 'o-', linewidth=2, color='green', label='Extensor')
        axs_time[0].set_xlabel(param_label)
        axs_time[0].set_ylabel("Fraction of Time Active")
        axs_time[0].set_title(f"Time Active (threshold = {activation_threshold})")
        axs_time[0].legend()
        axs_time[0].grid(True)
        
        # Plot activation time ratio
        ratio = np.divide(flexor_active_time, extensor_active_time, 
                         out=np.ones_like(flexor_active_time), 
                         where=extensor_active_time!=0)
        axs_time[1].plot(param_values, ratio, 'o-', linewidth=2, color='purple')
        axs_time[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  
        axs_time[1].set_xlabel(param_label)
        axs_time[1].set_ylabel("Flexor/Extensor Ratio")
        axs_time[1].set_title("Balance of Activation")
        axs_time[1].grid(True)
        
        # Hide unused subplots
        for j in range(len(param_values), n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            if row < axs_scatter.shape[0] and col < axs_scatter.shape[1]:
                axs_scatter[row, col].axis('off')
        
        # Adjust layout and save
        for fig in [fig_scatter, fig_coact, fig_time]:
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
        
        # Save figures
        for fig, name in zip([fig_scatter, fig_coact, fig_time], 
                            ["flexor_vs_extensor", "coactivation_metrics", "activation_time"]):
            filename = f"{name}_{param_name}_{min(param_values)}to{max(param_values)}_{timestamp}_{seed}.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, bbox_inches='tight')
            print(f"Saved analysis plot: {filename}")
    
   
