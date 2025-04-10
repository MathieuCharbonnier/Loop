import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_times_series(file):
    data = pd.read_csv(file)
    
    # Plot Muscle properties
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['stretch'], label='stretch')
    plt.plot(data['time'], data['velocity'], label='stretch velocity')
    plt.plot(data['time'], data['fiber_length'], label='fiber_length')
    plt.plot(data['time'], data['fiber_velocity'], label='fiber_velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Muscle properties')
    plt.legend()
    plt.savefig('Muscle.png')
    plt.show()

    # Create a figure with three subplots: Ia, II, Efferent fibers
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Raster Plots for Ia afferent fibers
    for i in range(n['afferent']):
        spikes_time = data[f'spikes_times_Ia_{i}']
        axs[0].plot(spikes_time, np.ones_like(spikes_time == 1) * i, 'k.', markersize=3)  # Raster dots
    
    # Raster Plots for II afferent fibers
    for i in range(n['afferent']):
        spikes_time = data[f'spikes_times_II_{i}']
        axs[1].plot(spikes_time, np.ones_like(spikes_time == 1) * i, 'b.', markersize=3)  # Raster dots
    
    # Raster Plots for motoneurons
    for i in range(n['motor']):
        spikes_time = data[f'spikes_times_motor_{i}']
        axs[2].plot(spikes_time, np.ones_like(spikes_time) * i, 'r.', markersize=3)  # Raster dots

    axs[0].set(title="Ia Spike Raster Plot", ylabel="Neuron Index")
    axs[1].set(title="II Spike Raster Plot", ylabel="Neuron Index")
    axs[2].set(title="Motoneurons Spike Raster Plot", ylabel="Neuron Index")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('Spikings.png')
    plt.show()

    # Plot activations
    fig, axs = plt.subplots(n['motor'] + 1, 1, figsize=(10, 10), sharex=True)
    for i in range(n['motor']):
        axs[i].plot(data['time'], data[f'activation_motor_{i}'], label=f'activation of motoneuron number {i}')
    
    axs[-1].plot(data['time'], data['mean_activation'], label='mean activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.legend()
    plt.savefig('Activation.png')
    plt.show()

def load_sto_file(filepath):
    """
    Load a .sto file (OpenSim Storage file) into a pandas DataFrame.
    Skips header lines starting with 'header' or until it reaches the column names.
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Find where the actual data starts (usually marked by "endheader")
    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start_idx = i + 1
            break

    # Now read the actual data using pandas
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=data_start_idx)
    return df


