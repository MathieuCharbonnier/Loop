import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json
import os

def plot_times_series(initial_time, initial_stretch, file_spikes, file_muscle, folder):
    
    #load files
    df = pd.read_csv(file_muscle)
    with open(file_spikes, "r") as f:
      spikes = json.load(f)

    #Raster Plots

    fig, axs = plt.subplots(len(spikes), 1, figsize=(10, 10), sharex=True)
    
    for (fiber_type, fiber_spikes), ax in zip(spikes.items(), axs) : 
      for neuron_id, neuron_spikes in fiber_spikes.items():
        if neuron_spikes:
            ax.plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), 'k.', markersize=3)  # Raster dots
      ax.set(title=f" {fiber_type} Spikes Raster Plot", ylabel="Neuron Index")
      ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    fig_path=os.path.join(folder, 'Raster_plots.png')
    plt.savefig(fig_path)
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
    path_fig=os.path.join(folder, 'Firing_plots.png')
    plt.savefig(path_fig)
    plt.show()


    fig, axs=plt.subplots(5,1, figsize=(10, 10), sharex=True)
    axs[0].plot(df['Time'], df['mean_e'], label='mean e')
    axs[0].set_ylabel("mean e")
    axs[0].legend()
    axs[1].plot(df['Time'], df['mean_u'], label='mean u')
    axs[1].set_ylabel("mean u")
    axs[1].legend()
    axs[2].plot(df['Time'], df['mean_c'], label= 'mean c')
    axs[2].set_ylabel('mean c')
    axs[2].legend()
    axs[3].plot(df['Time'], df['mean_P'], label= 'mean P')
    axs[3].set_ylabel('mean P')
    axs[3].legend()
    axs[4].plot(df['Time'], df['mean_activation'], label= "mean activation")
    axs[4].set_ylabel('mean a')
    axs[4].set_xlabel('Time(s)')
    axs[4].legend()

    fig.suptitle("Mean Activation Dynamic", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig=os.path.join(folder,'activation.png' ) 
    plt.savefig(path_fig)
    plt.show()

    # Plot Muscle properties
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(df['Time'], df['fiber_length'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Fiber length (m)')
    axs[1].plot(df['Time'], df['stretch'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Stretch (dimless)')
    axs[2].plot(df.iloc[20:]['Time'], df.iloc[20:]['velocity'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Stretch Velocity (s-1)')
    path_fig=os.path.join(folder,'Muscle.png' )
    plt.savefig(path_fig)
    plt.show()


def plot_joint_angle_from_sto_file(filepath, columns_wanted, folder ):
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
    df = pd.read_csv(filepath, sep='\t', skiprows=data_start_idx)
    df.columns = ["/".join(col.split("/")[-2:]) for col in df.columns]

    # Create subplots
    fig, axs = plt.subplots(len(columns_wanted), 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Joint Angles and Speeds", fontsize=16)

    # Plot value and speed for each joint
    for i, column in enumerate(columns_wanted):
        axs[i, 0].plot(df['time'], df[column + '/value'], label=f"{column} value")
        axs[i, 0].set_ylabel("Angle (rad)")
        axs[i, 0].set_title(f"{column} - Value")
        axs[i, 0].grid(True)

        axs[i, 1].plot(df['time'], df[column + '/speed'], label=f"{column} speed", color='orange')
        axs[i, 1].set_ylabel("Speed (rad/s)")
        axs[i, 1].set_title(f"{column} - Speed")
        axs[i, 1].grid(True)

    # Common X label
    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    # Improve layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    fig_path=os.path.join(folder,"joint_angles_and_speeds.png" )
    plt.savefig(fig_path)
    plt.show()




