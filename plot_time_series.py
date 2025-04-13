import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json

def plot_times_series(initial_time, initial_stretch, file_spikes, file_muscle):
    """
    plt.plot(initial_time, initial_stretch)
    plt.xlabel('Time (s)')
    plt.ylabel('Stretch (a.u)')
    plt.title('Initial Profile')
    plt.show()
    """
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
        kde = gaussian_kde(all_spike_times, bw_method=0.1)
        
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


    # Plot activations
    """
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
    """
    fig, axs=plt.subplots(5,1, figsize=(10, 10), sharex=True)
    axs[0].plot(df['Time'], df['mean_e'], label='mean e')
    axs[0].legend()
    axs[1].plot(df['Time'], df['mean_u'], label='mean u')
    axs[1].legend()
    axs[2].plot(df['Time'], df['mean_c'], label= 'mean c')
    axs[2].legend()
    axs[3].plot(df['Time'], df['mean_P'], label= 'mean P')
    axs[3].legend()
    axs[4].plot(df['Time'], df['mean_activation'], label= "mean activation")
    axs[4].set_xlabel('Time(s)')
    axs[4].legend()

    fig.suptitle("Mean Activation Dynamic", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig('activation.png')
    plt.show()

    # Plot Muscle properties
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(df['Time'], df['fiber_length'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Fiber length (m)')
    axs[1].plot(df['Time'], df['stretch'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Stretch (a.u)')
    axs[2].plot(df.iloc[20:]['Time'], df.iloc[20:]['velocity'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Stretch Velocity (s-1)')
    plt.savefig('Muscle.png')
    plt.show()
"""
#other methods maybe useful at some point
def RampHold(T_reaction, dt, v=0.2, t_ramp=0.1*second, t_hold=0.3*second):

  time = np.arange(0, T_reaction, dt)

  stretch = np.piecewise(
      time,
      [time < t_ramp/second,
      (time >= t_ramp/second) & (time < t_hold/second),
      time >= t_hold/second],
      [lambda t: v * t,
      lambda t: v * t_ramp/second,
      lambda t: v * t_ramp/second - v * (t - t_hold/second)]
  )

  return stretch

def SinusoidalStretch(dt,T_reaction, A=0.01, f=2*hertz):

  time = np.arange(0, T_reaction, dt)
  stretch = A * np.sin(2 * np.pi * f/hertz * time)

  return stretch
"""
