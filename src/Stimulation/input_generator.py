import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from brian2 import *

def transform_torque_params_in_array(time_points, TORQUE):
    validate_torque(TORQUE)
    if TORQUE['type'] == "bump":
        torque = bump(time_points*second, TORQUE['t_peak'], TORQUE['sigma'], 
                     TORQUE['max_amplitude'], TORQUE['sustained_amplitude'])
    elif TORQUE['type'] == "ramp":
        torque = ramp(time_points*second, TORQUE['t_stop'], TORQUE['max_amplitude'], TORQUE['sustained_amplitude'])
    else:
        raise ValueError(f"{TORQUE['type']} is not implemented yet, existing types are bump or ramp")
    return torque

def ramp(time_array, t_stop, max_amplitude, sustained_amplitude=0):
    num_points = len(time_array)
    torque_profile = np.zeros((num_points))
    # Create the torque profile
    for i, t in enumerate(time_array):
        if t < t_stop:
            # Rapid increase in torque (quick stretch phase)
            progress = t / t_stop
            torque_profile[i] = max_amplitude * progress
        else:
            # Reduced holding torque (to allow natural oscillation)
            torque_profile[i] = sustained_amplitude
    return torque_profile

def bump(time_array, t_peak, sigma, max_amplitude, sustained_amplitude=0):
    # Gaussian torque
    gaussian = max_amplitude * np.exp(-0.5 * ((time_array - t_peak) / sigma) ** 2)

    # Find index where the torque drops below threshold after the peak
    i_peak = np.argmax(gaussian)
    i_hold_start = np.where(gaussian[i_peak:] <= sustained_amplitude)[0]
    i_hold_start = i_hold_start[0] + i_peak if len(i_hold_start) > 0 else time_array.shape[0]

    # Construct the final torque profile
    torque = np.copy(gaussian)
    torque[i_hold_start:] = sustained_amplitude
    return torque
    
def plot_recruitment_curves(site, muscle_name, current_current=None, 
                           base_output_path=None):
    """
    Plot recruitment curves for all fiber types using the threshold-based sigmoid.
    Only shows fractions of population, not absolute counts.

    Parameters:
    -----------
    site : str
        Electrode position 
    muscle_name: str
        Muscle to consider
    current_current : float, optional
        Current intensity value to highlight
    ees_recruitment_profile : dict, optional
        Dictionary with threshold and saturation values
    base_output_path : str, optional
        Path to save the plot
    """  
    df = pd.read_csv(f"data/Human_{site}.csv")  # Fixed: added pd.read_csv

    fig, ax = plt.subplots(figsize=(10, 6))
    
    style_map = {'Ia': 'g-', 'II': 'b-', 'Ib': 'c-', 'MN': 'r-'}
    muscle_col = [col for col in df.columns if muscle_name in col]
    for col in muscle_col:
        style = style_map.get(col, '-')  # default to '-' if fiber type not in map
        ax.plot(df['current_uA'].values, df[col].values, style, label=col)
        fraction = np.interp(current_current/uA, df['current_uA'], df[col], left=0, right=0)
        print(f'The current current recruits: {fraction} {col}')

    if current_current is not None:  
        ax.axvline(x=current_current/uA, color='r', linestyle='--', label='Current')


    ax.set_xlabel('Current Amplitude (uA)')
    ax.set_ylabel('Fraction of Fibers Recruited')
    ax.set_title(f'Fiber Recruitment at {site} - {muscle_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)


    filename = f'Recruitment_Curve_{site}_{muscle_name}.png'
    
    if base_output_path:
        os.makedirs(base_output_path, exist_ok=True)
        fig_path = os.path.join(base_output_path, filename)
    else:
        os.makedirs("Results", exist_ok=True)
        fig_path = os.path.join("Results", filename)
    plt.tight_layout()

    fig.savefig(fig_path)
    plt.show()

def transform_intensity_site_in_recruitment(ees_stimulation_params, neuron_population, muscle_names):
    site = ees_stimulation_params.get('site', 'S1')
    current = ees_stimulation_params.get('current', 0.0)

    recruitment = {}
    df_site = pd.read_csv(f"data/Human_{site}.csv")
    if df_site.empty:
        raise ValueError(f"No data found for site '{site}'. Ensure the CSV file exists and is correctly formatted.")
    
    muscles_cleaned = [m[:-2] if m.endswith(('_r', '_l')) else m for m in muscle_names]  # Fixed: muscle_names
    
    if len(muscle_names) == 2:
        for neuron, neuron_count in neuron_population.items():
            if neuron.startswith('I') or neuron.startswith('MN'):
                name = neuron.replace('flexor', muscles_cleaned[0]).replace('extensor', muscles_cleaned[1])
                
                if name not in df_site.columns:
                    raise ValueError(f"Neuron type '{neuron}' not found in recruitment profile for site '{site}'. "
                                f"Check the CSV file or the neuron_population dictionary.")
                frac = np.interp(current/uA, df_site['current_uA'], df_site[name], left=0, right=0)  
                recruitment[neuron] = int(frac * neuron_count)
    else:
        for neuron, neuron_count in neuron_population.items():
            if neuron.startswith('I') or neuron.startswith('MN'):
                name = f"{neuron}_{muscles_cleaned[0]}"
                if name not in df_site.columns:
                    raise ValueError(f"Neuron type '{neuron}' not found in recruitment profile for site '{site}'. "
                                f"Check the CSV file or the neuron_population dictionary.")
                frac = np.interp(current/uA, df_site['current_uA'], df_site[name], left=0, right=0) 
                recruitment[neuron] = int(frac * neuron_count)
    return {'recruitment': recruitment, 
            'frequency': ees_stimulation_params['frequency']}

def validate_torque(torque):
    issues = {"warnings": [], "errors": []}
    if torque is not None:
        # Check if torque_profile has a type field
        if 'type' not in torque:
            issues["errors"].append("Torque must contain a 'type' parameter")
        else:
            profile_type = torque['type']
            
            # Common parameters for all types
            if 'max_amplitude' not in torque:
                issues["errors"].append("Torque must contain 'max_amplitude' parameter")
            
            if 'sustained_amplitude' not in torque:
                issues["errors"].append("Torque must contain 'sustained_amplitude' parameter")
            
            # Type-specific parameter validation
            if profile_type == "ramp":
                # Required parameters for ramp type
                if 't_stop' not in torque:
                    issues["errors"].append("Torque with type 'ramp' must contain 't_stop' parameter")
                else:
                    # Check that t_stop has units
                    t_stop_str = str(torque['t_stop'])
                    if not (t_stop_str.endswith('ms') or t_stop_str.endswith('s')):
                        issues["errors"].append("'t_stop' must have units (ms or s)")  # Fixed: 'second' to 's'
                
                # Check for invalid parameters for this type
                invalid_params = set(torque.keys()) - {'type', 'max_amplitude', 'sustained_amplitude', 't_stop'}
                if invalid_params:
                    issues["warnings"].append(f"Torque with type 'ramp' has unexpected parameters: {invalid_params}")
                    
            elif profile_type == "bump":
                # Required parameters for bump type
                if 't_peak' not in torque:
                    issues["errors"].append("Torque with type 'bump' must contain 't_peak' parameter")
                else:
                    # Check that t_peak has units
                    t_peak_str = str(torque['t_peak'])
                    if not (t_peak_str.endswith('ms') or t_peak_str.endswith('s')):
                        issues["errors"].append("'t_peak' must have units (ms or s)")  # Fixed: 'second' to 's'
                        
                if 'sigma' not in torque:
                    issues["errors"].append("Torque with type 'bump' must contain 'sigma' parameter")
                else:
                    # Check that sigma has units
                    sigma_str = str(torque['sigma'])
                    if not (sigma_str.endswith('ms') or sigma_str.endswith('s')):
                        issues["errors"].append("'sigma' must have units (ms or s)")  # Fixed: 'second' to 's'
                
                # Check for invalid parameters for this type
                invalid_params = set(torque.keys()) - {'type', 'max_amplitude', 'sustained_amplitude', 't_peak', 'sigma'}
                if invalid_params:
                    issues["warnings"].append(f"Torque with type 'bump' has unexpected parameters: {invalid_params}")
                    
            else:
                issues["errors"].append(f"For torque, type must be 'ramp' or 'bump', got '{profile_type}'")
                
    if issues["errors"]:
        error_messages = "\n".join(issues["errors"])
        raise ValueError(f"Configuration errors found:\n{error_messages}")
        
    if issues["warnings"]:
        warning_messages = "\n".join(issues["warnings"])
        print(f"WARNING: Configuration issues detected:\n{warning_messages}")

def validate_ees(ees_stimulation_params, number_muscle):
    issues = {"warnings": [], "errors": []}
    if ees_stimulation_params is not None:
        # Check if freq has hertz unit
        if 'frequency' in ees_stimulation_params:
            # Check if frequency is a tuple (i.e., different EES frequencies for two muscles)
            frequency = ees_stimulation_params['frequency']
         
            if isinstance(frequency, tuple):
                if number_muscle != 2:
                    issues["errors"].append("When EES frequency is a tuple, exactly two muscles must be defined.")

                for i, f in enumerate(frequency):
                    if not str(f).endswith('Hz'):
                        issues["errors"].append(f"The frequency of EES for muscle {i+1} must have 'Hz' as unit.")
                    # Fixed: removed brian2 unit check since it may not work with all values
                    try:
                        freq_value = float(str(f).replace('Hz', ''))
                        if freq_value < 0:
                            issues["errors"].append(f"EES frequency for muscle {i+1} must be positive, got {f}.")
                    except ValueError:
                        issues["errors"].append(f"Invalid frequency format for muscle {i+1}: {f}")
            
            elif isinstance(frequency, list):
                for i, f in enumerate(frequency):
                    if not str(f).endswith('Hz'):
                        issues["errors"].append(f"The frequency of EES must have 'Hz' as unit.")
                    try:
                        freq_value = float(str(f).replace('Hz', ''))
                        if freq_value < 0:
                            issues["errors"].append(f"EES frequency must be positive, got {f}.")
                    except ValueError:
                        issues["errors"].append(f"Invalid frequency format: {f}")
    
            else:
                if not str(frequency).endswith('Hz'):
                    issues["errors"].append("The frequency of EES must have 'Hz' as unit.")
                try:
                    freq_value = float(str(frequency).replace('Hz', ''))
                    if freq_value < 0:
                        issues["errors"].append(f"EES frequency must be positive, got {frequency}.")
                except ValueError:
                    issues["errors"].append(f"Invalid frequency format: {frequency}")

        else:
            issues["errors"].append("EES parameters must contain 'frequency' parameter")
        
        if 'current' in ees_stimulation_params:
            current = ees_stimulation_params["current"]
            if isinstance(current,Quantity):
                unit= current.dim 
                if unit != 'A':
                    issues["errors"].append(f"EES current must have 'A' as unit, got {unit}.")
                else:
                    if (current>570*uA):
                        issues["errors"].append(f"EES current must be smaller than 570 uA, recruitement curve not defined for higher current.")
            else:
                    issues["errors"].append("EES current must be a Quantity with 'A' as unit.")

                
        else:
            issues["errors"].append("EES parameters must contain 'intensity' parameter")

        if 'site' not in ees_stimulation_params:  # Fixed: not 'site' in to 'site' not in
            issues["errors"].append("You should specify the EES stimulation site!")
            
    if issues["errors"]:
        error_messages = "\n".join(issues["errors"])
        raise ValueError(f"Configuration errors found:\n{error_messages}")
    
    if issues["warnings"]:
        warning_messages = "\n".join(issues["warnings"])
        print(f"WARNING: Configuration issues detected:\n{warning_messages}")
