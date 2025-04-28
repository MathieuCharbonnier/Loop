import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json
import os

# Colorblind-friendly palette
colorblind_friendly_colors = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7"
}
color_keys = list(colorblind_friendly_colors.keys())

def plot_times_series(file_spikes, muscle_data, muscle_names, folder, ees_freq, aff_recruited, eff_recruited):

    
    # Load spike data
    with open(file_spikes, "r") as f:
        spikes = json.load(f)

    num_muscles = len(spikes)
    num_fiber_types = len(next(iter(spikes.values())))  # assuming all muscles have the same fiber types
    
    # Raster Plots
    fig, axs = plt.subplots(num_fiber_types, num_muscles, figsize=(10, 10), sharex=True)

    # Iterate over muscles and fiber types to plot spikes
    for i, (muscle, spikes_muscle) in enumerate(spikes.items()):
        for j, (fiber_type, fiber_spikes) in enumerate(spikes_muscle.items()):
            # Plot individual neuron spikes
            for neuron_id, neuron_spikes in fiber_spikes.items():
                if neuron_spikes:  # Check if neuron_spikes is not empty
                    axs[j, i].plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), '.', markersize=3, color='black')

            # Set plot properties
            axs[j, i].set(title=f"{muscle}_{fiber_type}", ylabel="Neuron Index")
            axs[j, i].grid(True)

    # Set common x-axis label
    axs[-1, 0].set_xlabel("Time (s)")  # Set xlabel for the bottom-most plot
    fig.suptitle('Spikes Raster Plot', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ensure titles and labels do not overlap

    # Save and display the plot
    fig_path = os.path.join(folder, f'Raster_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()

    # Firing rate plots
    fig, axs = plt.subplots(num_fiber_types, 1, figsize=(10, 10), sharex=True)
    time = muscle_data[0]['Time'].values 

    for i, (muscle, spikes_muscle) in enumerate(spikes.items()):
        for j, (fiber_type, fiber_spikes) in enumerate(spikes_muscle.items()):
            all_spike_times = np.concatenate(list(fiber_spikes.values()))
            
            if len(all_spike_times)>0:
                kde = gaussian_kde(all_spike_times, bw_method=0.1)
                firing_rate = kde(time) * len(all_spike_times) / len(fiber_spikes)
                axs[j].plot(time, firing_rate, label=f"{muscle}", color=colorblind_friendly_colors[color_keys[i]])
            else:
                axs[j].plot(time, np.zeros_like(time), label=f"{muscle}", color=colorblind_friendly_colors[color_keys[i]])
            axs[j].set_ylabel(f'{fiber_type} firing rate (Hz)')

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Smoothed Instantaneous Firing Rate (KDE)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Firing_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()

    # Mean activation dynamics - 
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    labels = ['mean_e', 'mean_u', 'mean_c', 'mean_P', 'mean_activation']
    for i, label in enumerate(labels):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            color_idx = j % len(color_keys)  
            axs[i].plot(df['Time'], df[label], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[color_idx]])
        axs[i].set_ylabel(label)
        axs[i].legend()


    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Mean Activation Dynamics: Muscle Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Activation_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()

    # Muscle properties - for all muscles
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    props = ['fiber_length', 'stretch', 'velocity']
    ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s-1)']

    for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            color_idx = j % len(color_keys)
            axs[i].plot(df['Time'], df[prop], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[color_idx]])
        axs[i].set_ylabel(ylabel)
        axs[i].legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Muscle Properties: Muscle Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Muscle_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()



def plot_joint_angle_from_sto_file(filepath, columns_wanted, folder, aff_recruited, eff_recruited, ees_freq):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start_idx = i + 1
            break

    df = pd.read_csv(filepath, sep='\t', skiprows=data_start_idx)
    df.columns = ["/".join(col.split("/")[-2:]) for col in df.columns]

    fig, axs = plt.subplots(len(columns_wanted), 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Joint Angles and Speeds", fontsize=16)

    for i, column in enumerate(columns_wanted):
        axs[i, 0].plot(df['time'], df[column + '/value']*180/np.pi, label=f"{column} value", color=colorblind_friendly_colors["blue"])
        axs[i, 0].set_ylabel("Angle (degree)")
        axs[i, 0].set_title(f"{column} - Value")
        axs[i, 0].grid(True)

        axs[i, 1].plot(df['time'], df[column + '/speed']*180/np.pi, label=f"{column} speed", color=colorblind_friendly_colors["orange"])
        axs[i, 1].set_ylabel("Speed (degree/s)")
        axs[i, 1].set_title(f"{column} - Speed")
        axs[i, 1].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'Joint_angles_and_speed_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()
