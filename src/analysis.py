from brian2 import *
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from closed_loop import closed_loop

def delay_excitability_MU_type_analysis(delay_values, threshold_values, duration, time_step, reaction_time,
                                      neurons_population, connections, 
                                      spindle_model, biophysical_params, muscles_names, num_muscles,
                                      associated_joint, torque_profile, ees_stimulation_params, 
                                      ees_recruitment_profile, fast_type_mu, seed=41):
        """
        Analyze clonus behavior by varying one parameter at a time and create visualization plots.
        
        Parameters:
        -----------
        delay_values:list
            list of reaction time to test
        threshold_values
            list of spikes threshold potential to test
        duration: brian2.unit
            Duration of the simulation 
        time_step: brian2.unit
            Time step for simulation
        reaction_time: brian2.unit
            Feedback time
        neurons_population: dict
            Neurons count for each fiber type
        connections: dict
            Connections between neurons
        spindle_model: dict
            Equations to transform stretch or joint angle into afferent firing rate
        biophysical_params: dict
            Parameters for neural dynamics simulations
        muscles_names: list
            List of muscle names
        num_muscles: int
            Number of muscles
        associated_joint: str
            Name of the joint to analyze
        torque_profile: dict
            Dictionary with torque profile parameters
        ees_stimulation_params: dict
            Parameters for EES stimulation
        ees_recruitment_profile: dict
            Parameters for recruitment profiles
        fast_type_mu: bool
            Whether to use fast twitch motor units
        seed: int
            Random seed for reproducibility
        """
        # Create base_output_path for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "clonus_analysis"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Created directory '{save_dir}' for saving plots")
    
        # Create a base output path
        fig_dir = os.path.join(save_dir, timestamp)
    
        fast_twitch_values = [False, True]  # False for slow, True for fast
    
        # 1. Vary delay
        fig1, axs1 = plt.subplots(len(delay_values), 2, figsize=(15, 4*len(delay_values)), sharex=True)
        for i, delay in enumerate(tqdm(delay_values, desc="Varying delay")):
            # Set reaction time to the current delay value
            current_reaction_time = delay
            # Run simulation with current parameters
            n_iterations = int(duration/current_reaction_time) 
            spikes, time_series = closed_loop(
                    n_iterations, 
                    current_reaction_time, 
                    time_step, 
                    neurons_population, 
                    connections,
                    spindle_model, 
                    biophysical_params, 
                    muscles_names,
                    num_muscles, 
                    associated_joint,
                    f"{fig_dir}_delay",
                    ees_recruitment_profile,
                    ees_stimulation_params,
                    torque_profile, 
                    fast_type_mu, 
                    seed
                )
            
            # Plot joint angle
            axs1[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
            axs1[i, 0].set_ylabel(f"Delay = {int(delay/ms)} ms\nJoint angle (deg)")  # Fixed: delay*1000 -> delay/ms
            
            # Plot muscle activations
            for muscle in muscles_names:  
                activation_col = f"Activation_{muscle}"
                axs1[i, 1].plot(time_series['Time'], time_series[activation_col], 
                              label=muscle)
            
            axs1[i, 1].set_ylabel("Muscle activation")
            axs1[i, 1].legend()
        
        axs1[-1, 0].set_xlabel("Time (s)")
        axs1[-1, 1].set_xlabel("Time (s)")
        fig1.suptitle("Effect of Delay on Joint Angle and Muscle Activation", fontsize=16)
        fig1.tight_layout()
        fig1.savefig(os.path.join(fig_dir, 'delay_variation.png'), dpi=300)
        
        n_iterations = int(duration/reaction_time) 
        # 2. Vary fast twitch parameter
        fig2, axs2 = plt.subplots(len(fast_twitch_values), 2, figsize=(15, 4*len(fast_twitch_values)), sharex=True)
        
        for i, fast in enumerate(tqdm(fast_twitch_values, desc="Varying fast twitch parameter")):
            
            # Run simulation with current parameters
            n_iterations = int(duration/reaction_time) 
            spikes, time_series = closed_loop(
                n_iterations, 
                reaction_time, 
                time_step, 
                neurons_population, 
                connections,
                spindle_model, 
                biophysical_params, 
                muscles_names,
                num_muscles, 
                associated_joint,
                f"{fig_dir}_delay",
                ees_recruitment_profile,
                ees_stimulation_params,
                torque_profile, 
                fast, 
                seed
            )
            
            # Plot joint angle
            axs2[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
            axs2[i, 0].set_ylabel(f"Fast = {fast}\nJoint angle (deg)")
            
            # Plot muscle activations
            for muscle in muscles_names :  
                activation_col = f"Activation_{muscle}"
                axs2[i, 1].plot(time_series['Time'], time_series[activation_col], 
                              label=muscle)
            
            axs2[i, 1].set_ylabel("Muscle activation")
            axs2[i, 1].legend()
        
        axs2[-1, 0].set_xlabel("Time (s)")
        axs2[-1, 1].set_xlabel("Time (s)")
        fig2.suptitle("Effect of Fast Twitch Parameter on Joint Angle and Muscle Activation", fontsize=16)
        fig2.tight_layout()
        fig2.savefig(os.path.join(fig_dir, 'fast_twitch_variation.png'), dpi=300)
        
        # 3. Vary threshold voltage
        fig3, axs3 = plt.subplots(len(threshold_values), 2, figsize=(15, 4*len(threshold_values)), sharex=True)
        
        for i, threshold in enumerate(tqdm(threshold_values, desc="Varying threshold voltage")):
            # Create a copy of biophysical params and update the threshold
            current_biophysical_params = biophysical_params.copy()  
            current_biophysical_params['threshold_v'] = threshold
            
            # Run simulation with current parameters
            n_iterations = int(duration/reaction_time) 
            spikes, time_series = closed_loop(
                n_iterations, 
                reaction_time, 
                time_step, 
                neurons_population, 
                connections,
                spindle_model, 
                current_biophysical_params, 
                muscles_names,
                num_muscles, 
                associated_joint,
                f"{fig_dir}_threshold",
                ees_recruitment_profile,
                ees_stimulation_params,
                torque_profile, 
                fast_type_mu, 
                seed
            )
            
            # Plot joint angle
            axs3[i, 0].plot(time_series['Time'], time_series[f'Joint_{associated_joint}'], 'b-')
            axs3[i, 0].set_ylabel(f"Threshold = {int(threshold/mV)} mV\nJoint angle (deg)")  
            
            # Plot muscle activations
            for muscle in muscles_names:  
                activation_col = f"Activation_{muscle}"
                axs3[i, 1].plot(time_series['Time'], time_series[activation_col],  
                              label=muscle)
            
            axs3[i, 1].set_ylabel("Muscle activation")
            axs3[i, 1].legend()
        
        axs3[-1, 0].set_xlabel("Time (s)")
        axs3[-1, 1].set_xlabel("Time (s)")
        fig3.suptitle("Effect of Threshold Voltage on Joint Angle and Muscle Activation", fontsize=16)
        fig3.tight_layout()
        fig3.savefig(os.path.join(fig_dir, 'threshold_variation.png'), dpi=300)
        
        # Close all figures to free memory
        plt.close('all')

def EES_stim_analysis(param_dict, vary_param, n_iterations, reaction_time, 
                     neurons_population, connections, spindle_model, biophysical_params, 
                     muscles_names, num_muscles, associated_joint, ees_recruitment_profile, 
                     time_step=0.1*ms, seed=42):
    """
    Generalized EES stimulation analysis that can vary any parameter of interest.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary containing all EES parameters with their default values
        Expected keys: 'ees_freq', 'afferent_recruited', 'MN_recruited', 'B'
    vary_param : dict
        Dictionary specifying which parameter to vary with its range of values
        Format: {'param_name': [values_to_test], 'label': 'Display Label'}
        Example: {'param_name': 'ees_freq', 'values': [10, 20, 30], 'label': 'EES Frequency (Hz)'}
    n_iterations : int
        Number of iterations for each simulation
    reaction_time : brian2.unit
        Feedback time
    neurons_population : dict
        Neurons count for each fiber type
    connections : dict
        Connections between neurons
    spindle_model : dict
        Equations to transform stretch or joint angle into afferent firing rate
    biophysical_params : dict
        Parameters for neural dynamics simulations
    muscles_names : list
        List of muscle names
    num_muscles : int
        Number of muscles
    associated_joint : str
        Name of the joint to analyze
    ees_recruitment_profile: dict
        threshold and saturations to define the recruitment curves
    time_step : brian2.unit
        Time step for simulation
    seed : int
        Random seed for reproducibility
    """
    # Create a directory for saving plots if it doesn't exist
    save_dir = "stimulation_analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory '{save_dir}' for saving plots")
    
    # Create a base output path
    base_path = os.path.join(save_dir, "output")
  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define a custom color palette for muscles
    # Using a colorblind-friendly palette
    muscle_colors = {
        muscles_names[i]: plt.cm.tab10(i % 10) for i in range(num_muscles) 
    }
    
    # Define a custom style for the plots
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (15, 4),
        'figure.dpi': 100
    })
    
    time_series_to_plot = ['Ia_rate', 'II_rate', 'MN_rate', 'Raster_MN', 'Activation', 'Stretch', 'Joints']
    
    # Get parameter info
    param_name = vary_param['param_name']
    param_values = vary_param['values']
    param_label = vary_param['label']
    
    n_rows = len(param_values)
    
    # Initialize figures and axes
    figs = {}
    axs_dict = {}
    for var in time_series_to_plot:
        fig, axs = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True, sharey=True)
        if n_rows == 1:
            axs = [axs]  # Ensure axs is always a list
        figs[var] = fig
        axs_dict[var] = axs
      
    if (num_muscles == 2):  # coactivation analysis
        # Preallocate activities array
        activities = None  # Will initialize inside loop
    
    # Define fast parameter for closed_loop call
    fast = True  # Added default value for fast parameter
    
    # Run simulations for each parameter value
    for i, value in enumerate(param_values):
        # Create a copy of the base parameters
        current_params = param_dict.copy()
        
        # Update the parameter we're varying
        current_params[param_name] = value
        
        # Create a descriptive name for the output file
        param_str_parts = []
        for key in current_params.keys():
            param_str_parts.append(f"{key}_{current_params[key]}")
        
        param_str = '_'.join(param_str_parts)
        sto_name = f'All_opensim_{param_str}_{timestamp}_{seed}.sto'
        sto_path = os.path.join(save_dir, sto_name)
    
        # --- Run simulation ---
           
        spikes, main_data = closed_loop(
            n_iterations, 
            reaction_time, 
            time_step, 
            neurons_population, 
            connections,
            spindle_model, 
            biophysical_params, 
            muscles_names,
            num_muscles, 
            associated_joint,
            base_path,
            ees_recruitment_profile,
            current_params, # EES_STIMULATION_PARAMS
            None,  # TORQUE 
            fast, 
            seed
        )
 
        
        # Extract time from the dataframe
        time_data = main_data['Time']
        
        # Get time length for preallocation on first iteration
        if num_muscles == 2 and activities is None:
            T = len(time_data)
            activities = np.zeros((num_muscles, n_rows, T))  
    
        # --- Plot each variable ---
        for var in time_series_to_plot:
            # Get the axis for this variable and parameter value
            ax = axs_dict[var][i]
            
            # Set title with parameter information
            ax.set_title(f"{param_label}: {value} ", fontweight='bold')
            
            ax.set_xlabel("Time (s)", fontweight='bold')
            if "rate" in var:
                ax.set_ylabel(var.replace('_', ' ').title() + " (hertz)", fontweight='bold')
            else:
                ax.set_ylabel(var.replace('_', ' ').title() + " (dimless)", fontweight='bold')
            
            # Add a light background grid for better readability
            ax.grid(True, linestyle='--', alpha=0.3)
    
            if var == 'Joints':
                # Check if 'Joints' column exists, otherwise use 'Joint_ankle' or similar
                joint_col = 'Joints' if 'Joints' in main_data.columns else f"Joint_{param_dict.get('associated_joint', 'ankle')}"
                if joint_col in main_data.columns:
                    ax.plot(time_data, main_data[joint_col], color='darkred', 
                           label='Ankle Angle', linewidth=2.5)
                    ax.set_ylabel(var.replace('_', ' ').title() + " (degree)", fontweight='bold')
    
            elif var == 'Raster_MN':
                # Add different colors for each muscle in the raster plot
                for idx, muscle_name in enumerate(muscles_names):  # Fixed: muscles_names -> MUSCLES_NAMES
                    if muscle_name in spikes:
                        color = muscle_colors[muscle_name]
                        
                        # Plot spikes for this muscle with a distinct color
                        for neuron_id, neuron_spikes in spikes[muscle_name]['MN'].items():
                            if neuron_spikes:
                                ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), 
                                       '.', markersize=4, color=color)
                        
                        # Add a label for this muscle at its position
                        ax.text(0.01 + idx*0.09, 1.05, muscle_name, 
                               transform=ax.get_xaxis_transform(), color=color,
                               fontweight='bold', verticalalignment='center')
                
                # Add a more descriptive y-axis label for raster plot
                ax.set_ylabel("Neuron ID ", fontweight='bold')
                    
            else:
                # Plot data for each muscle with consistent colors
                for idx, muscle_name in enumerate(muscles_names):  
                    # Construct column name with muscle suffix
                    col_name = f"{var}_{muscle_name}"
                    
                    if col_name in main_data.columns:
                        ax.plot(time_data, main_data[col_name], label=muscle_name, 
                               color=muscle_colors[muscle_name], linewidth=2.0, alpha=0.8)
                        
                        # Store mean activation for coactivation analysis
                        if var == 'Activation' and num_muscles == 2:
                            activities[idx, i, :] = main_data[col_name].values
                
            # Add legend with improved styling
            if var != 'Raster_MN':  # Raster plot has text labels instead
                legend = ax.legend(frameon=True, fancybox=True, framealpha=0.9, 
                                  loc='upper right', ncol=1, fontsize='x-large')
                legend.get_frame().set_edgecolor('lightgray')
    
    # --- Final layout adjustments, saving, and display ---
    
    # Process all figures
    for var in time_series_to_plot:
        # Add a main title to each figure with improved styling
        figs[var].suptitle(f"{var.replace('_', ' ').title()} Response Across {param_label} Values", 
                          fontsize=16, fontweight='bold', y=0.98)
        
        # Make sure all plots have data and proper formatting
        for ax in axs_dict[var]:
            # If the axis is empty (no lines plotted), add a dummy invisible line
            if not ax.lines:
                ax.plot([0], [0], alpha=0)
                ax.text(0.5, 0.5, 'No data available', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold')
        
        # Add a global x-label at the bottom of the figure
        figs[var].text(0.5, 0.04, 'Time (seconds)', ha='center', 
                      fontsize=14, fontweight='bold')
        
        # Add a global y-label for the entire figure
        if var != 'Raster_MN':
            figs[var].text(0.04, 0.5, f"{var.replace('_', ' ').title()}", 
                          va='center', rotation='vertical', 
                          fontsize=14, fontweight='bold')
        else:
            figs[var].text(0.04, 0.5, "Motor Neuron Activity", 
                          va='center', rotation='vertical', 
                          fontsize=14, fontweight='bold')
        
        # Adjust layout
        figs[var].tight_layout(rect=[0.08, 0.08, 0.98, 0.95])
        
        # Create a meaningful filename
        var_name = var.replace('_', '-')
        param_range = f"{param_name}_{min(param_values)}to{max(param_values)}"
        filename = f"{var_name}_{param_range}_{timestamp}_{seed}.png"
        filepath = os.path.join(save_dir, filename)
        
        # Save the figure with high resolution
        figs[var].savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    
    # Return activities if this was a coactivation analysis
    if num_muscles == 2:
        return activities
    
    # Display all figures
    plt.show()

    print(f"Simulation and plotting complete! All plots saved to '{save_dir}' directory.")
    
    # Co-activation analysis if we have 2 muscles 
    if num_muscles == 2:
        # ===== Flexor-Extensor Activation Analysis =====
        print("\nPerforming flexor-extensor activation analysis...")
        
        # Define activation threshold
        activation_threshold = 0.1  # Threshold to consider a muscle as "active"
        
        flexor_idx = 0  # tib_ant_r (tibialis anterior - flexor)
        extensor_idx = 1  # med_gas_r (medial gastrocnemius - extensor)
        
        # Calculate grid layout (e.g., 2 rows if there are more than 3 parameter values)
        n_cols = 2  # or choose based on space
        n_rows = math.ceil(len(param_values) / n_cols)
        
        # Create scatter plot grid with multiple rows
        fig_scatter, axs_scatter = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        fig_scatter.suptitle("Flexor vs Extensor Activity", fontsize=16)
        
        # Ensure axs_scatter is 2D array
        axs_scatter = np.atleast_2d(axs_scatter)
        
        # Create a figure for coactivation metrics
        fig_coact, axs_coact = plt.subplots(1, 2, figsize=(15, 6))
        fig_coact.suptitle("Coactivation Analysis", fontsize=16)
        
        # Create a figure for activation time analysis
        fig_time, axs_time = plt.subplots(1, 2, figsize=(15, 6))
        fig_time.suptitle("Muscle Activation Time Analysis", fontsize=16)
        
        # Arrays to store metrics across parameter values
        min_coactivation = np.zeros(len(param_values))
        product_coactivation = np.zeros(len(param_values))
        flexor_active_time = np.zeros(len(param_values))
        extensor_active_time = np.zeros(len(param_values))
        
        # Get total simulation time
        time_array = main_data['Time']
        total_time = time_array.iloc[-1] if hasattr(time_array, 'iloc') else time_array[-1]
        
        # Analyze each parameter value
        for i, value in enumerate(param_values):
            # Get flexor and extensor activation data
            flexor_activation = activities[flexor_idx, i, :]
            extensor_activation = activities[extensor_idx, i, :]
            
            # Calculate time step for integration
            dt = time_array[1] - time_array[0] if len(time_array) > 1 else 0.001
            
            # 1. Flexor vs Extensor Scatter Plot
            row = i // n_cols
            col = i % n_cols
            ax = axs_scatter[row, col]
        
            # Plot scatter
            ax.scatter(flexor_activation, extensor_activation, alpha=0.6, s=10)
            ax.set_xlabel("Flexor Activation")
            ax.set_ylabel("Extensor Activation")
            ax.set_title(f"{param_label}: {value}")
            ax.grid(True, linestyle='--', alpha=0.7)
        
            # Diagonal reference line
            max_val = max(np.max(flexor_activation), np.max(extensor_activation))
            ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
            
            # 2. Calculate coactivation metrics
            # Minimum-based coactivation: integral(min(flexor, extensor)dt)/total_time
            min_coact = np.sum(np.minimum(flexor_activation, extensor_activation)) * dt / total_time
            min_coactivation[i] = min_coact
            
            # Product-based coactivation: integral(flexor*extensor dt)/total_time
            prod_coact = np.sum(flexor_activation * extensor_activation) * dt / total_time
            product_coactivation[i] = prod_coact
            
            # 3. Calculate activation time (time above threshold)
            flexor_active = np.sum(flexor_activation > activation_threshold) * dt / total_time
            extensor_active = np.sum(extensor_activation > activation_threshold) * dt / total_time
            
            flexor_active_time[i] = flexor_active
            extensor_active_time[i] = extensor_active
        
        # Plot coactivation metrics vs parameter value
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
        
        # Plot activation time metrics vs parameter value
        axs_time[0].plot(param_values, flexor_active_time, 'o-', linewidth=2, color='blue', label='Flexor')
        axs_time[0].plot(param_values, extensor_active_time, 'o-', linewidth=2, color='green', label='Extensor')
        axs_time[0].set_xlabel(param_label)
        axs_time[0].set_ylabel("Fraction of Time Active")
        axs_time[0].set_title(f"Time Active (threshold = {activation_threshold})")
        axs_time[0].legend()
        axs_time[0].grid(True)
        
        # Plot activation time ratio (flexor/extensor)
        ratio = np.divide(flexor_active_time, extensor_active_time, 
                         out=np.ones_like(flexor_active_time), 
                         where=extensor_active_time!=0)
        axs_time[1].plot(param_values, ratio, 'o-', linewidth=2, color='purple')
        axs_time[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)  # Reference line at ratio=1
        axs_time[1].set_xlabel(param_label)
        axs_time[1].set_ylabel("Flexor/Extensor Ratio")
        axs_time[1].set_title("Balance of Activation")
        axs_time[1].grid(True)
        
        # Hide any unused subplots
        for j in range(len(param_values), n_rows * n_cols):
            row = j // n_cols
            col = j % n_cols
            axs_scatter[row, col].axis('off')
        
        # Adjust layout for all figures
        for fig in [fig_scatter, fig_coact, fig_time]:
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
        
        # Save these new figures
        for fig, name in zip([fig_scatter, fig_coact, fig_time], 
                            ["flexor_vs_extensor", "coactivation_metrics", "activation_time"]):
            filename = f"{name}_{param_name}_{min(param_values)}to{max(param_values)}_{timestamp}_{seed}.png"
            filepath = os.path.join(save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved analysis plot: {filename}")
        
        # Display all figures
        plt.show()

        print("Flexor-extensor activation analysis complete!")
