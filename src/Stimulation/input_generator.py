
import pandas as pd
import numpy as np
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


def sigmoid_recruitment(current_amplitude, threshold_10pct, saturation_90pct):
    """
    Calculate recruitment fraction using sigmoid function precisely calibrated 
    to match given threshold and saturation points.
    
    Parameters:
    - current_amplitude: Current stimulation amplitude (same units as threshold/saturation)
    - threshold_10pct: Current amplitude at which 10% of fibers are recruited
    - saturation_90pct: Current amplitude at which 90% of fibers are recruited
    
    Returns:
    - Fraction of fibers recruited (0-1)
    """
    # Calculate sigmoid parameters from the 10% and 90% points
    # For sigmoid function: f(x) = 1/(1 + exp(-k*(x-x0)))
    # where x0 is the midpoint (50% recruitment)
    # and k determines the steepness
    
    x0 = (threshold_10pct + saturation_90pct) / 2
    k = np.log(9) / (saturation_90pct - x0)
    
    # Apply the sigmoid function
    fraction = 1 / (1 + np.exp(-k * (current_amplitude - x0)))
    
    return fraction
  
def transform_intensity_balance_in_recruitment(ees_recruitment_profile, ees_stimulation_params, neurons_population, num_muscles):
    """
    Transform intensity and balance parameters into recruitment counts
    
    Parameters:
    - ees_recruitment_params: Dictionary with threshold and saturation values
    - ees_stimulation_params: Dictionary with stimulation parameters (intensity, freq)
    - neurons_population: Dictionary with neuron counts
    - balance: Balance parameter (-1 to 1)
    - num_muscles: Number of muscles (default 2 for flexor/extensor model)
    
    Returns:
    - Dictionary with recruitment counts and frequency
    """
    validate_ees(ees_stimulation_params,ees_recruitment_profile, num_muscles, neurons_population)
    
    # Get fractions first
    if 'balance' in ees_stimulation_params:
        fractions = calculate_full_recruitment(
            ees_stimulation_params['intensity'], 
            ees_recruitment_profile,
            num_muscles,
            ees_stimulation_params['balance']   
        )
    else:
        fractions = calculate_full_recruitment(
            ees_stimulation_params['intensity'], 
            ees_recruitment_profile,
            num_muscles
        )   
    # Convert fractions to counts
    counts = {}
    for key, fraction in fractions.items():
            
        counts[key] = int(fraction * neurons_population[key])
    
    return {
        "recruitment": counts,
        "frequency": ees_stimulation_params['frequency']
    }

def calculate_full_recruitment(normalized_current, ees_recruitment_profile,num_muscles, balance=0):
    """
    Calculate recruitment fractions for all fiber types based on normalized current and balance.
    
    Parameters:
    - normalized_current: float (0-1), normalized stimulation intensity
    - ees_recruitment_params: Dictionary with threshold and saturation values
    - balance: float (-1 to 1), electrode position bias
    - num_muscles: Number of muscles (2 for flexor/extensor or 1 for single muscle)
    
    Returns:
    - Dictionary with recruitment fractions (0-1)
    """
    fractions = {}

    if num_muscles == 2:
        for fiber_type in ees_recruitment_profile.keys():
            for muscle_type in ['flexor', 'extensor']:
                key = f"{fiber_type}_{muscle_type}"
              
                # Positive balance favors extensors, negative balance favors flexors
                shift = 0.2 * balance  # 0.2 scaling factor determines strength of balance effect
                if muscle_type == 'flexor':
                    shift = -shift  # Reverse effect for flexors
    
                # Apply shifts to threshold and saturation
                threshold = ees_recruitment_profile[fiber_type]['threshold_10pct'] + shift
                saturation = ees_recruitment_profile[fiber_type]['saturation_90pct'] + shift
    
                # Ensure values stay in reasonable range
                threshold = max(0.1, min(0.7, threshold))
                saturation = max(0.3, min(0.9, saturation))
                
                # Calculate recruitment fraction
                fractions[key] = sigmoid_recruitment(normalized_current, threshold, saturation)
    else:
        # Single muscle case
        for fiber_type in ees_recruitment_profile.keys():
            threshold = ees_recruitment_profile[fiber_type]['threshold_10pct'] 
            saturation = ees_recruitment_profile[fiber_type]['saturation_90pct'] 
            
            # Calculate recruitment fraction directly
            fractions[fiber_type] = sigmoid_recruitment(normalized_current, threshold, saturation)
    
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

def validate_ees(ees_stimulation_params,ees_recruitment_params, number_muscle, neurons_population):
        issues = {"warnings": [], "errors": []}
        if ees_stimulation_params is not None:
            # Check if freq has hertz unit
            if 'frequency' in ees_stimulation_params:
                if not str(ees_stimulation_params['frequency']).endswith('Hz'):
                    issues["errors"].append("The frequency of EES must have hertz as unit")
                if not (0*hertz <= ees_stimulation_params['frequency'] ):
                    issues["errors"].append(f"ees frequency must be positive, , got {val}")
                # Check if freq is a tuple and if so, ensure we have exactly two muscles
                if isinstance(ees_stimulation_params['frequency'], tuple):
                    if number_muscle != 2:
                        issues["errors"].append("When EES frequency is a tuple, exactly two muscles must be defined")
            else:
                issues["errors"].append("EES parameters must contain 'frequency' parameter")
            
       
            if 'intensity' in ees_stimulation_params:
                    if not (0 <= ees_stimulation_params["intensity"] <= 1):
                        issues["errors"].append(f"'intensity paramter ' in ees stimulation must contains values between 0 and 1, got {val}")
            else:
                issues["errors"].append("EES parameters must contain 'intensity' parameter")

            if 'balance' in ees_stimulation_params:
                    if not (-1 <= ees_stimulation_params["balance"] <= 1):
                        issues["errors"].append(f"'balance parameter ' in ees stimulation must contains values between -1 and 1, got {val}")
                    if number_muscle==1:
                        issues["warnings"].append(f"'balance parameter ' in ees stimulation is for two muscles simulation only, it will be not be considered")
            else:
                if number_muscle==2:
                     issues["warnings"].append(f"you should specify a 'balance paramater' for two muscles simulation with ees stimulation,'balance' parameter is set to zero")
                
        

        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"Configuration errors found:\n{error_messages}")
        
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: Configuration issues detected:\n{warning_messages}")
