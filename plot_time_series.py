import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_times_series(file):
    data = pd.read_csv(file)
    
    # Plot Muscle properties
    plt.figure(figsize=(10, 5))
    plt.plot(data['Time'], data['stretch'], label='stretch')
    plt.plot(data['Time'], data['velocity'], label='stretch velocity')
    plt.plot(data['Time'], data['fiber_length'], label='fiber_length')
    plt.plot(data['Time'], data['fiber_velocity'], label='fiber_velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Muscle properties')
    plt.legend()
    plt.savefig('Muscle.png')
    plt.show()

    # Create a figure with three subplots: Ia, II, Efferent fibers
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Raster Plots for Ia afferent fibers
    dt=data.loc[1,'Time']-data.loc[0, 'Time']
    for i, type_fiber in enumerate(['Ia', 'II', 'motoneuron']):
      spike_matrix = data.filter(like=f'spikes_times_{type_fiber}').to_numpy()
      print('spike_matrix', spike_matrix.shape)
      rows, _ = np.nonzero(spike_matrix)
      axs[i].plot(rows * dt, i*np.ones((len(rows))), 'k.', markersize=3)
      axs[i].set(title=f"{type_fiber} Spike Raster Plot", ylabel="Neuron Index")
      axs[i].grid(True)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('Spikings.png')
    plt.show()

    # Plot activations
    # Automatically find all activation columns for motor neurons
    activation_cols = [col for col in data.columns if col.startswith('activation_')]
  
    # Create subplots: one for each motor neuron + one for mean activation
    fig, axs_ = plt.subplots(len(activation_cols), 1, figsize=(10, 1.5 * (len(activation_cols))), sharex=True)

    # Plot individual motor neuron activations
    for i, col in enumerate(activation_cols):
        axs_[i].plot(data['Time'], data[col], label=col.replace('_', ' ').capitalize())
        axs_[i].set_ylabel('Activation')
        axs_[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('Activation.png')
    plt.show()



