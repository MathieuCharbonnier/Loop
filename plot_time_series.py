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

def plot_times_series(initial_time, initial_stretch, file_spikes, file_muscle, folder, ees_freq, aff_recruited, eff_recruited):
    # Load files
    df = pd.read_csv(file_muscle)
    with open(file_spikes, "r") as f:
        spikes = json.load(f)

    # Raster Plots
    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)
    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs):
        for neuron_id, neuron_spikes in fiber_spikes.items():
            if neuron_spikes:
                ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), '.', markersize=3, color='black')
        ax.set(title=f" {fiber_type} Spikes Raster Plot", ylabel="Neuron Index")
        ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    fig_path = os.path.join(folder, f'Raster_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()

    # Firing rate plots
    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)
    time = df['Time'].values
    stretch = np.concatenate([initial_stretch, df["stretch"].values])
    stretch = stretch[:len(time)]
    velocity = np.gradient(stretch)

    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs):

        if fiber_type == 'Ia':
            theory = 50 + 2 * stretch + 4.3 * np.sign(velocity) * np.abs(velocity) ** 0.6
            ax.plot(time, theory, label="calculated from muscle stretch", linestyle='--',
                    color=colorblind_friendly_colors["orange"])
        elif fiber_type == 'II':
            theory = 80 + 13.5 * stretch
            ax.plot(time, theory, label="calculated from muscle stretch", linestyle='--',
                    color=colorblind_friendly_colors["orange"])
        if len(fiber_spikes)>0:
            all_spike_times = np.concatenate(list(fiber_spikes.values()))
            kde = gaussian_kde(all_spike_times, bw_method=0.1)
            firing_rate = kde(time) * len(all_spike_times) / len(fiber_spikes)

            ax.plot(time, firing_rate, label="with EES and refractory", color=colorblind_friendly_colors["blue"])
        else:

            ax.plot(time, np.zeros_like(time),label="with EES and refractory", color=colorblind_friendly_colors["blue"] )
        

        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'Fiber type: {fiber_type}')
        ax.legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Smoothed Instantaneous Firing Rate (KDE)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Firing_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()

    # Mean activation dynamics
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(df['Time'], df['mean_e'], label='mean e', color=colorblind_friendly_colors["blue"])
    axs[0].set_ylabel("mean e")
    axs[0].legend()

    axs[1].plot(df['Time'], df['mean_u'], label='mean u', color=colorblind_friendly_colors["green"])
    axs[1].set_ylabel("mean u")
    axs[1].legend()

    axs[2].plot(df['Time'], df['mean_c'], label='mean c', color=colorblind_friendly_colors["orange"])
    axs[2].set_ylabel('mean c')
    axs[2].legend()

    axs[3].plot(df['Time'], df['mean_P'], label='mean P', color=colorblind_friendly_colors["purple"])
    axs[3].set_ylabel('mean P')
    axs[3].legend()

    axs[4].plot(df['Time'], df['mean_activation'], label='mean activation', color=colorblind_friendly_colors["red"])
    axs[4].set_ylabel('mean a')
    axs[4].set_xlabel('Time(s)')
    axs[4].legend()

    fig.suptitle("Mean Activation Dynamic", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Activation_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()

    # Muscle properties
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(df['Time'], df['fiber_length'], color=colorblind_friendly_colors["blue"])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Fiber length (m)')

    axs[1].plot(df['Time'], df['stretch'], color=colorblind_friendly_colors["orange"])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Stretch (dimless)')

    axs[2].plot(df.iloc[20:]['Time'], df.iloc[20:]['velocity'], color=colorblind_friendly_colors["green"])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Stretch Velocity (s-1)')

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
        axs[i, 0].plot(df['time'], df[column + '/value'], label=f"{column} value", color=colorblind_friendly_colors["blue"])
        axs[i, 0].set_ylabel("Angle (rad)")
        axs[i, 0].set_title(f"{column} - Value")
        axs[i, 0].grid(True)

        axs[i, 1].plot(df['time'], df[column + '/speed'], label=f"{column} speed", color=colorblind_friendly_colors["orange"])
        axs[i, 1].set_ylabel("Speed (rad/s)")
        axs[i, 1].set_title(f"{column} - Speed")
        axs[i, 1].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'Joint_angles_and_speed_aff_{aff_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()
