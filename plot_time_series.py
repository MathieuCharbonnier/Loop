import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def plot_times_series(file_spikes, file_muscle):

    #Raster Plots
    with open(file_spikes, "r") as f:
      spikes = json.load(f)

    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)
    
    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs) : 
      for neuron_id, neuron_spikes in fiber_spikes.items():
        ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), 'k.', markersize=3)  # Raster dots
      ax.set(title=f" {fiber_type} Spikes Raster Plot", ylabel="Neuron Index")
      ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    
    df = pd.read_csv(file_muscle)
    
    # Plot Muscle properties
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['stretch'], label='stretch (a.u)')
    plt.plot(df['Time'], df['velocity'], label='stretch velocity (s-1)')
    plt.plot(df['Time'], df['fiber_length'], label='fiber_length (m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Muscle states')
    plt.legend()
    plt.savefig('Muscle.png')
    plt.show()

    # Plot activations
    activation_cols = [col for col in df.columns if col.startswith('activation_')]
    fig, axs_ = plt.subplots(len(activation_cols), 1, figsize=(10, 1. * (len(activation_cols))), sharex=True)

    # Plot individual motor neuron activations
    for i, col in enumerate(activation_cols):
        axs_[i].plot(df['Time'], df[col], label=col.replace('_', ' ').capitalize())
        axs_[i].set_ylabel('a')
        axs_[i].legend(loc='upper right')
    axs_[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('Activation.png')
    plt.show()



