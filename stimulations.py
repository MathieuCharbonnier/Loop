

import numpy as np
import matplotlib.pyplot as plt


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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
  
def transform_intensity_balance_in_recruitment(ees_recruitment_params, ees_stimulation_params, neurons_population, balance=0, num_muscles=2):
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
    # Get fractions first
    fractions = calculate_full_recruitment(
        ees_stimulation_params['intensity'], 
        ees_recruitment_params,
        balance, 
        num_muscles
    )
    
    # Convert fractions to counts
    counts = {}
    for key, fraction in fractions.items():
        if "_" in key:
            fiber_type = key.split("_")[0]
        else:
            fiber_type = key
            
        counts[key] = int(fraction * neurons_population[fiber_type])
    
    return {
        "recruitment": counts,
        "freq": ees_stimulation_params['freq']
    }

def calculate_full_recruitment(normalized_current, ees_recruitment_params, balance=0, num_muscles=2):
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
        for fiber_type in ees_recruitment_params.keys():
            for muscle_type in ['flexor', 'extensor']:
                key = f"{fiber_type}_{muscle_type}"
              
                # Positive balance favors extensors, negative balance favors flexors
                shift = 0.2 * balance  # 0.2 scaling factor determines strength of balance effect
                if muscle_type == 'flexor':
                    shift = -shift  # Reverse effect for flexors
    
                # Apply shifts to threshold and saturation
                threshold = ees_recruitment_params[fiber_type]['threshold_10pct'] + shift
                saturation = ees_recruitment_params[fiber_type]['saturation_90pct'] + shift
    
                # Ensure values stay in reasonable range
                threshold = max(0.1, min(0.7, threshold))
                saturation = max(0.3, min(0.9, saturation))
                
                # Calculate recruitment fraction
                fractions[key] = sigmoid_recruitment(normalized_current, threshold, saturation)
    else:
        # Single muscle case
        for fiber_type in ees_recruitment_params.keys():
            threshold = ees_recruitment_params[fiber_type]['threshold_10pct'] 
            saturation = ees_recruitment_params[fiber_type]['saturation_90pct'] 
            
            # Calculate recruitment fraction directly
            fractions[fiber_type] = sigmoid_recruitment(normalized_current, threshold, saturation)
    
    return fractions

def plot_recruitment_curves(ees_recruitment_params, balance=0, num_muscles=2):
    """
    Plot recruitment curves for all fiber types using the threshold-based sigmoid.
    Only shows fractions of population, not absolute counts.
    
    Parameters:
    - ees_recruitment_params: Dictionary with threshold and saturation values
    - balance: float (-1 to 1), electrode position bias
    - num_muscles: Number of muscles (2 for flexor/extensor or 1 for single muscle)
    """
    currents = np.linspace(0, 1, 100)
    
    # Calculate recruitment fractions at each intensity
    fraction_results = []
    
    for current in currents:
        # Get fractions directly
        fractions = calculate_full_recruitment(
            current, 
            ees_recruitment_params, 
            balance, 
            num_muscles
        )
        fraction_results.append(fractions)
      
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(fraction_results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Define colors and styles for different fiber types
    style_map = {
        'Ia_flexor': 'r-', 'II_flexor': 'r--', 'MN_flexor': 'r-.',
        'Ia_extensor': 'b-', 'II_extensor': 'b--', 'MN_extensor': 'b-.'
    }
    
    # For non-muscle-specific case
    single_style_map = {'Ia': 'g-', 'II': 'g--', 'MN': 'g-.'}
    
    for col in df.columns:
        # Choose appropriate style
        if col in style_map:
            line_style = style_map[col]
        elif col in single_style_map:
            line_style = single_style_map[col]
        else:
            # Default styling
            if "extensor" in col:
                line_style = 'b-'  # Blue for extensors
            else:
                line_style = 'r-'  # Red for flexors
        
        plt.plot(currents, df[col], line_style, label=col)
    
    plt.xlabel('Normalized Current Amplitude')
    plt.ylabel('Fraction of Fibers Recruited')
    plt.title(f'Fiber Recruitment (Balance = {balance})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines at 10% and 90% recruitment
    plt.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7)
    plt.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


