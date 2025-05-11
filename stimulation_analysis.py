def EES_stim_analysis(
    param_dict,
    vary_param,
    NUM_ITERATIONS,
    REACTION_TIME, 
    TIME_STEP, 
    NEURON_COUNTS, 
    CONNECTIONS,
    equation_Ia, 
    equation_II, 
    BIOPHYSICAL_PARAMS, 
    MUSCLE_NAMES_STR,
    sto_path_base, 
    seed=42
):
    """
    Generalized EES stimulation analysis that can vary any parameter of interest.
    
    Parameters:
    -----------
    param_dict : dict
        Dictionary containing all EES parameters with their default values
        Expected keys: 'ees_freq', 'Ia_recruited', 'II_recruited', 'eff_recruited'
    vary_param : dict
        Dictionary specifying which parameter to vary with its range of values
        Format: {'param_name': [values_to_test], 'label': 'Display Label'}
        Example: {'param_name': 'ees_freq', 'values': [10, 20, 30], 'label': 'EES Frequency (Hz)'}
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create a directory for saving plots if it doesn't exist
    save_dir = "parameter_analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory '{save_dir}' for saving plots")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    muscles_names = MUSCLE_NAMES_STR.split(',')
    num_muscles = len(muscles_names)
    
    # Define a custom color palette for muscles
    # Using a colorblind-friendly palette
    muscle_colors = {
        muscles_names[i]: plt.cm.tab10(i % 10) for i in range(len(muscles_names))
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
        fig, axs = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True)
        if n_rows == 1:
            axs = [axs]  # Ensure axs is always a list
        figs[var] = fig
        axs_dict[var] = axs
      
    if (num_muscles == 2):  # coactivation analysis
        # Preallocate activities array
        activities = None  # Will initialize inside loop
    
    # Run simulations for each parameter value
    for i, value in enumerate(param_values):
        # Create a copy of the base parameters
        current_params = param_dict.copy()
        
        # Update the parameter we're varying
        current_params[param_name] = value
        
        # Create a descriptive name for the output file
        param_str_parts = []
        for key in ['Ia_recruited', 'II_recruited', 'eff_recruited', 'ees_freq']:
            if key == 'ees_freq':
                param_str_parts.append(f"{key}_{current_params[key]}")
            else:
                param_str_parts.append(f"{key}_{current_params[key]}")
        
        param_str = '_'.join(param_str_parts)
        sto_name = f'All_opensim_{param_str}_{timestamp}_{seed}.sto'
        sto_path = os.path.join(save_dir, sto_name)
    
        # --- Run simulation ---
        spikes, main_data = closed_loop(
            NUM_ITERATIONS, REACTION_TIME, TIME_STEP, current_params, NEURON_COUNTS, CONNECTIONS,
            equation_Ia, equation_II, BIOPHYSICAL_PARAMS,
            MUSCLE_NAMES_STR, sto_path, seed=seed
        )
        
        # Get time length for preallocation on first iteration
        if num_muscles == 2 and activities is None:
            T = len(main_data[0]['Time'])
            activities = np.zeros((len(muscles_names), n_rows, T))
    
        # --- Plot each variable ---
        for var in time_series_to_plot:
            # Get the axis for this variable and parameter value
            ax = axs_dict[var][i]
            
            # Set title with parameter information
            value_display = f"{value} Hz" if param_name == 'ees_freq' else f"{value}"
            ax.set_title(f"{param_label}: {value_display}, "
                        f"Ia: {current_params['Ia_recruited']}, "
                        f"II: {current_params['II_recruited']}, "
                        f"Motoneurons: {current_params['eff_recruited']}", 
                        fontweight='bold')
            
            ax.set_xlabel("Time (s)", fontweight='bold')
            if "rate" in var:
                ax.set_ylabel(var.replace('_', ' ').title() + " (hertz)", fontweight='bold')
            else:
                ax.set_ylabel(var.replace('_', ' ').title() + " (dimless)", fontweight='bold')
            
            # Add a light background grid for better readability
            ax.grid(True, linestyle='--', alpha=0.3)
    
            if var == 'Joints':
                joints = read_sto(sto_path, ['ankle_angle_r'])
                ax.plot(joints['time'], joints['ankle_angle_r/value']*180/np.pi, color='darkred', 
                       label='Ankle Angle (degree)', linewidth=2.5)
    
            elif var == 'Raster_MN':
                # Add different colors for each muscle in the raster plot
                for idx, muscle_name in enumerate(muscles_names):
                    if muscle_name in spikes:
                        color = muscle_colors[muscle_name]
                        
                        # Plot spikes for this muscle with a distinct color
                        for neuron_id, neuron_spikes in spikes[muscle_name]['MN'].items():
                            if neuron_spikes:
                                ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), 
                                       '.', markersize=4, color=color)
                        
                        # Add a label for this muscle at its position
                        ax.text(0.01, 10, muscle_name, 
                               transform=ax.get_xaxis_transform(), color=color,
                               fontweight='bold', verticalalignment='center')
                
                # Add a more descriptive y-axis label for raster plot
                ax.set_ylabel("Neuron ID ", fontweight='bold')
                    
            else:
                # Plot data for each muscle with consistent colors
                for idx, muscle_name in enumerate(muscles_names):
                    t = main_data[idx]['Time']
                    y = main_data[idx][var]
                    ax.plot(t, y, label=muscle_name, color=muscle_colors[muscle_name], 
                           linewidth=2.0, alpha=0.8)
                    
                    # Store mean activation
                    if var == 'Activation' and num_muscles == 2:
                        activities[idx, i, :] = y
                
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
    
    # Display all figures
    plt.show()
    
    print(f"Simulation and plotting complete! All plots saved to '{save_dir}' directory.")
    
    # Return activities if co-activation analysis was performed
    if num_muscles == 2:
        return activities
    return None


# Example usage for the three scenarios:

# 1. Varying EES frequency with fixed afferent and efferent recruitment
def analyze_frequency_effects(freq_range, Ia_recruited, II_recruited, eff_recruited, **kwargs):
    """Analyze the effects of varying EES frequency."""
    base_params = {
        'ees_freq': 20,  # Default value, will be overridden
        'Ia_recruited': Ia_recruited,
        'II_recruited': II_recruited,
        'eff_recruited': eff_recruited
    }
    
    vary_param = {
        'param_name': 'ees_freq',
        'values': freq_range,
        'label': 'EES Frequency (Hz)'
    }
    
    return EES_stim_analysis(base_params, vary_param, **kwargs)


# 2. Varying afferent recruitment with fixed EES frequency and efferent recruitment
def analyze_afferent_effects(afferent_range, ees_freq, II_recruited, eff_recruited, afferent_type='Ia', **kwargs):
    """Analyze the effects of varying afferent recruitment (either Ia or II)."""
    base_params = {
        'ees_freq': ees_freq,
        'Ia_recruited': II_recruited if afferent_type == 'II' else 0.5,  # Default
        'II_recruited': II_recruited if afferent_type != 'II' else 0.5,  # Default
        'eff_recruited': eff_recruited
    }
    
    param_name = f'{afferent_type}_recruited'
    
    vary_param = {
        'param_name': param_name,
        'values': afferent_range,
        'label': f'{afferent_type} Fiber Recruitment (%)'
    }
    
    return EES_stim_analysis(base_params, vary_param, **kwargs)


# 3. Varying efferent recruitment with fixed EES frequency and afferent recruitment
def analyze_efferent_effects(efferent_range, ees_freq, Ia_recruited, II_recruited, **kwargs):
    """Analyze the effects of varying efferent (motoneuron) recruitment."""
    base_params = {
        'ees_freq': ees_freq,
        'Ia_recruited': Ia_recruited,
        'II_recruited': II_recruited,
        'eff_recruited': 0.5  # Default value, will be overridden
    }
    
    vary_param = {
        'param_name': 'eff_recruited',
        'values': efferent_range,
        'label': 'Motoneuron Recruitment (%)'
    }
    
    return EES_stim_analysis(base_params, vary_param, **kwargs)


