import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json

def plot_times_series(initial_time, initial_stretch, file_spikes, file_muscle):

    plt.plot(initial_time, initial_stretch)
    plt.xlabel('Time (s)')
    plt.ylabel('Stretch (a.u)')
    plt.title('Initial Profile')
    plt.show()

    #load files
    df = pd.read_csv(file_muscle)
    with open(file_spikes, "r") as f:
      spikes = json.load(f)

    #Raster Plots

    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)
    
    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs) : 
      for neuron_id, neuron_spikes in fiber_spikes.items():
        ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), 'k.', markersize=3)  # Raster dots
      ax.set(title=f" {fiber_type} Spikes Raster Plot", ylabel="Neuron Index")
      ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    plt.savefig('Raster_plots.png')
    plt.show()

    #Firing rate plots
    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)

    time = df['Time'].values
    stretch=np.concatenate([initial_stretch,df["stretch"].values])
    stretch=stretch[:len(time)]
    velocity=np.gradient(stretch)

    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs):

        # Concatenate all spike times for this specific fiber type
        all_spike_times = np.concatenate(list(fiber_spikes.values()))

        # Estimate density using KDE
        kde = gaussian_kde(all_spike_times, bw_method=0.5)
        
        firing_rate = kde(time) * len(all_spike_times) / len(fiber_spikes)  # Normalize by number of neurons in that fiber type

        ax.plot(time, firing_rate, label="Observed", color='blue')

        if fiber_type == 'Ia':
            theory = 50 + 2 * stretch + 4.3 * np.sign(velocity) * np.abs(velocity) ** 0.6
            ax.plot(time, theory, label="Theory", linestyle='--', color='orange')
        elif fiber_type == 'II':
            theory = 80 + 13.5 * stretch
            ax.plot(time, theory, label="Theory", linestyle='--', color='orange')

        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'Fiber type: {fiber_type}')
        ax.legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Smoothed Instantaneous Firing Rate (KDE)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
    plt.savefig('Firing_plots.png')
    plt.show()


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



