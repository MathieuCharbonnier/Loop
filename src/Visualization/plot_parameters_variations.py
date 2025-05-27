from brian2 import *
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_delay_results(delay_results, delay_values, muscle_names, associated_joint, output_dir="clonus_analysis"):

    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for saving plots")
    
    fig1, axs1 = plt.subplots(len(delay_values), 2, figsize=(15, 4*len(delay_values)), sharex=True)
    if len(delay_values) == 1:
        axs1 = axs1.reshape(1, -1)
    
    for i, (delay, spikes, time_series) in enumerate(delay_results):
        # Plot joint angle
        axs1[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
        axs1[i, 0].set_ylabel(f"Delay = {int(delay/ms)} ms\nJoint angle (deg)")
        
        # Plot muscle activations
        for muscle in muscles_names:
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

def plot_twitch_results(fast_twitch_results,  muscle_names, associated_joint, output_dir="clonus_analysis"):
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for saving plots")
    
    fig2, axs2 = plt.subplots(len(fast_twitch_results), 2, figsize=(15, 4*len(fast_twitch_results)), sharex=True)
    if len(fast_twitch_results) == 1:
        axs2 = axs2.reshape(1, -1)
    
    for i, (fast, spikes, time_series) in enumerate(fast_twitch_results):
        # Plot joint angle
        axs2[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
        label = "Fast type Motor Unit" if fast else "Slow type Motor Unit"
        axs2[i, 0].set_ylabel(f"{label}\nJoint angle (deg)")
        
        # Plot muscle activations
        for muscle in muscles_names:
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

def plot_excitability_results(threshold_results, threshold_values, muscles_names, associated_joint, output_dir="clonus_analysis"):

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for saving plots")
    
    fig3, axs3 = plt.subplots(len(threshold_values), 2, figsize=(15, 4*len(threshold_values)), sharex=True)
    if len(threshold_values) == 1:
        axs3 = axs3.reshape(1, -1)
    
    for i, (threshold, spikes, time_series) in enumerate(threshold_results):
        # Plot joint angle
        axs3[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
        axs3[i, 0].set_ylabel(f"Threshold = {int(threshold/mV)} mV\nJoint angle (deg)")
        
        # Plot muscle activations
        for muscle in muscles_names:
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
    
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def plot_ees_analysis_results(results, save_dir="stimulation_analysis", seed=42):
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
        muscles_names[i]: plt.cm.tab10(i % 10) for i in range(num_muscles)
    }

    n_rows = len(param_values)

    # --- Plot time series data ---
    figs = {}
    axs_dict = {}
    for var in time_series_to_plot:
        fig, axs = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True)
        if n_rows == 1:
            axs = [axs]
        figs[var] = fig
        axs_dict[var] = axs

        for i, (value, main_data) in enumerate(zip(param_values, simulation_data)):
            ax = axs[i]
            ax.set_title(f"{param_label}: {value}")
            ax.set_xlabel("Time (s)")
            ylabel = var.replace('_', ' ').title()
            ax.set_ylabel(f"{ylabel} (Hz)" if "rate" in var else f"{ylabel} (dimless)")
            ax.grid(True, linestyle='--', alpha=0.3)

            time_data = main_data['Time']
            for idx, muscle_name in enumerate(muscles_names):
                col_name = f"{var}_{muscle_name}"
                if col_name in main_data:
                    ax.plot(time_data, main_data[col_name], label=muscle_name,
                            color=muscle_colors[muscle_name], linewidth=2.0, alpha=0.8)
        fig.legend(muscles_names, loc='upper right')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{var}_{timestamp}.png"))

    # --- Plot joint angle ---
    fig_joint, axs_joint = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axs_joint = [axs_joint]
    for i, (value, main_data) in enumerate(zip(param_values, simulation_data)):
        ax = axs_joint[i]
        ax.set_title(f"{param_label}: {value}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{associated_joint} Angle (deg)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.plot(main_data['Time'], main_data[f'Joint_{associated_joint}'],
                color='darkred', label='Joint Angle', linewidth=2.5)
        ax.legend()
    fig_joint.tight_layout()
    fig_joint.savefig(os.path.join(save_dir, f"joint_angle_{timestamp}.png"))

    # --- Raster plot for MN spikes ---
    fig_raster, axs_raster = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axs_raster = [axs_raster]
    for i, (value, spikes) in enumerate(zip(param_values, spikes_data)):
        ax = axs_raster[i]
        ax.set_title(f"MN Raster Plot — {param_label}: {value}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Neuron ID")
        ax.grid(True, linestyle='--', alpha=0.3)

        for idx, muscle_name in enumerate(muscles_names):
            if muscle_name in spikes:
                for neuron_id, neuron_spikes in spikes[muscle_name]['MN'].items():
                    if neuron_spikes:
                        ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id),
                                '.', markersize=4, color=muscle_colors[muscle_name])
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

    print("All plots saved to", save_dir)

    # Coactivation analysis for 2-muscle systems
    if num_muscles == 2 and activities is not None:
        plot_coactivation_analysis(results, save_dir, timestamp, seed)
    
    plt.show()
    print(f"All plots saved to '{save_dir}' directory.")


def plot_coactivation_analysis(results, save_dir, timestamp, seed):
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
    activities = results['activities']
    time_data = results['time_data']
    
    # Define activation threshold
    activation_threshold = 0.1
    
    flexor_idx = 0  # tib_ant_r (tibialis anterior - flexor)
    extensor_idx = 1  # med_gas_r (medial gastrocnemius - extensor)
    
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
    axs_scatter = np.atleast_2d(axs_scatter)
    
    # Initialize metric arrays
    min_coactivation = np.zeros(len(param_values))
    product_coactivation = np.zeros(len(param_values))
    flexor_active_time = np.zeros(len(param_values))
    extensor_active_time = np.zeros(len(param_values))
    
    # Get simulation parameters
    total_time = time_data.iloc[-1] if hasattr(time_data, 'iloc') else time_data[-1]
    dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0.001
    
    # Analyze each parameter value
    for i, value in enumerate(param_values):
        flexor_activation = activities[flexor_idx, i, :]
        extensor_activation = activities[extensor_idx, i, :]
        
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
    
    plt.show()
    print("Flexor-extensor activation analysis complete!")
