
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
from brian2 import*

def transform_torque_params_in_array(time_points,TORQUE):
    validate_torque(TORQUE)
    if TORQUE['type']=="bump":
        torque=bump(time_points*second,TORQUE['t_peak'], TORQUE['sigma'], 
        TORQUE['max_amplitude'], TORQUE['sustained_amplitude'])
    elif TORQUE['type']=="ramp":
        torque=ramp(time_points*second, TORQUE['t_stop'], TORQUE['max_amplitude'], TORQUE['sustained_amplitude'])
    else:
        raise ValueError(f"{TORQUE['type']} is not implemented yet, existing type are bump or ramp")
    return torque

def ramp (time_array,t_stop, max_amplitude, sustained_amplitude=0):

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
    

def plot_recruitment_curves(site, muscle_name, current_current=None, ees_recruitment_profile=None,
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
    if ees_recruitment_profile is None:
        with open('ees_recruitment.json', 'r') as f:
            ees_recruitment_profile = json.load(f)

    currents = np.linspace(0, 1, 100)
    fraction_results = []

    for current in currents:
        fractions = calculate_full_recruitment(
            current,
            site,
            ees_recruitment_profile, 
            [muscle_name]
        )
        fraction_results.append(fractions[muscle_name])

    df = pd.DataFrame(fraction_results)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    style_map = {'Ia': 'g-', 'II': 'b-', 'Ib': 'c-', 'MN': 'r-'}
    for col in df.columns:
        style = style_map.get(col, '-')  # default to '-' if fiber type not in map
        ax.plot(currents, df[col], style, label=col)

    if current_current is not None:  
        ax.axvline(x=current_current, color='r', linestyle='--', label='Current')

    ax.set_xlabel('Normalized Current Amplitude')
    ax.set_ylabel('Fraction of Fibers Recruited')
    ax.set_title(f'Fiber Recruitment at {site} - {muscle_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'Recruitment_Curve_{site}_{muscle_name}_{timestamp}.png'
    
    if base_output_path:
        fig_path = os.path.join(base_output_path, filename)
    else:
        os.makedirs("Results", exist_ok=True)
        fig_path = os.path.join("Results", filename)

    fig.savefig(fig_path)
    plt.show()

        
def sigmoid_recruitment(current_amplitude, threshold, saturation, slope):
    """
    Calculate recruitment fraction using a sigmoid function with user-defined parameters.

    Parameters:
    -----------
    current_amplitude : float
        Normalized stimulation amplitude [0–1]
    threshold : float
        Activation threshold [0–1]
    saturation : float
        Maximum recruitment level [0–1]
    slope : float
        Slope of the sigmoid curve

    Returns:
    --------
    float
        Recruitment fraction (0 to saturation)
    """
    if not (0 <= threshold <= 1) or not (0 <= saturation <= 1):
        raise ValueError(f"saturation and threshold must be between 0 and 1. "
                         f"Got: threshold={threshold}, saturation={saturation}")
    if not isinstance(slope, (int, float)) or slope <= 0:
        raise ValueError(f"Slope must be a positive number. Got: slope={slope}")
    
    return saturation / (1 + np.exp(-slope * (current_amplitude - threshold)))



def transform_intensity_balance_in_recruitment(ees_recruitment_profile, ees_stimulation_params, neurons_population, muscles_names):
    """
    Transform intensity and balance parameters into recruitment counts
    """
    validate_ees(ees_stimulation_params, len(muscles_names))
    muscles_cleaned = [m[:-2] if m.endswith(('_r', '_l')) else m for m in muscles_names]
    fractions = calculate_full_recruitment(
            ees_stimulation_params['intensity'],
            ees_stimulation_params['site'],
            ees_recruitment_profile,
            muscles_cleaned         
    )   

    # Convert fractions to counts
    counts = {}
    for key, fraction in fractions.items():
        if key in neurons_population:
            counts[key] = int(fraction * neurons_population[key])
    
    return {
        "recruitment": counts,
        "frequency": ees_stimulation_params['frequency']
    }


def calculate_full_recruitment(normalized_current, site, ees_recruitment_profile, muscle_names):
    """
    Calculate recruitment fractions for all fiber types based on normalized current and site.

    Parameters
    ----------
    normalized_current : float
        The current intensity (0 to 1) for which to calculate fiber recruitment.
    site : str
        Stimulation site (e.g. 'L4', 'S1', etc.)
    ees_recruitment_profile : dict
        Recruitment profile from the JSON.
    muscle_names : list of str
        List of muscle names to include (e.g. ["tib_ant_r", "med_gas_r"]).
        
    Returns
    -------
    dict
        Recruitment fractions per fiber type and muscle group.
    """
    if site not in ees_recruitment_profile:
        raise ValueError(f"The stimulation site '{site}' does not exist in the EES recruitment profile. "
                         f"Please add it to the EES recruitment JSON.")

    # Optional: auto-assign flexor/extensor suffix if exactly two muscles are provided
    label_suffixes = ['_flexor', '_extensor'] if len(muscle_names) == 2 else ['']

    fractions = {}

    for muscle_name, suffix in zip(muscle_names, label_suffixes):
        if muscle_name not in ees_recruitment_profile[site]:
            raise ValueError(f"Recruitment profile for muscle '{muscle_name}' is missing at site '{site}'. "
                             f"Add it to the EES profile.")

        for neuron_type, neuron_params in ees_recruitment_profile[site][muscle_name].items():
            try:
                slope = neuron_params['slope']
                threshold = neuron_params['threshold']
                saturation = neuron_params['saturation']
            except KeyError as e:
                raise ValueError(f"Missing parameter {e} in recruitment profile for '{neuron_type}' at site '{site}', "
                                 f"muscle '{muscle_name}'.")

            key = f"{neuron_type}{suffix}"
            fractions[key] = sigmoid_recruitment(normalized_current, threshold, saturation, slope)

    return fractions



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
                            issues["errors"].append("'t_stop' must have units (ms or second)")
                    
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
                            issues["errors"].append("'t_peak' must have units (ms or second)")
                            
                    if 'sigma' not in torque:
                        issues["errors"].append("Torque with type 'bump' must contain 'sigma' parameter")
                    else:
                        # Check that sigma has units
                        sigma_str = str(torque['sigma'])
                        if not (sigma_str.endswith('ms') or sigma_str.endswith('s')):
                            issues["errors"].append("'sigma' must have units (ms or second)")
                    
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
                        if not (0 * hertz <= f):
                            issues["errors"].append(f"EES frequency for muscle {i+1} must be positive, got {f}.")

                else:
                    if not str(frequency).endswith('Hz'):
                        issues["errors"].append("The frequency of EES must have 'Hz' as unit.")
                    if not (0 * hertz <= frequency):
                        issues["errors"].append(f"EES frequency must be positive, got {frequency}.")


            else:
                issues["errors"].append("EES parameters must contain 'frequency' parameter")
            
       
            if 'intensity' in ees_stimulation_params:
                    if not (0 <= ees_stimulation_params["intensity"] <= 1):
                        issues["errors"].append(f"'intensity paramter ' in ees stimulation must contains values between 0 and 1, got {val}")
            else:
                issues["errors"].append("EES parameters must contain 'intensity' parameter")

            if not 'site' in ees_stimulation_params:
                issues["errors"].append(f"you should specify the EES stimulation site! ")
                
        

        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"Configuration errors found:\n{error_messages}")
        
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: Configuration issues detected:\n{warning_messages}")
