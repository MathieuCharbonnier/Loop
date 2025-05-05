import argparse
from jinja2.runtime import F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from brian2.units import second, hertz
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

def plot_times_series(initial_stretch,spikes, muscle_data, muscle_names, folder, ees_freq, Ia_recruited,II_recruited, eff_recruited, T_refr):

    num_muscles = len(spikes)
    num_fiber_types = len(next(iter(spikes.values())))  
    
    time = muscle_data[0]['Time'].values
    stretch=np.zeros((num_muscles, len(time)))
    velocity=np.zeros((num_muscles, len(time)))
    for i,muscle in enumerate(muscle_names):
        stretch_init = np.append(initial_stretch[i], muscle_data[i]['stretch'].values)
        stretch_init = stretch_init[:len(time)] 
        stretch[i]=stretch_init
        velocity[i] = np.gradient(stretch_init, time)
    Ia_rates=10 + 0.4 * stretch + 0.86 * np.sign(velocity) * np.abs(velocity) ** 0.6
    II_rates=20 + 3.375*stretch

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for i,muscle in enumerate(muscle_names):
        axs[0].plot(time, stretch[i], label=f'Stretch, {muscle} ')
        axs[1].plot(time, velocity[i], label=f'Velocity, {muscle} ')
        axs[2].plot(time, Ia_rates[i], label=f'Ia rate, {muscle} ')
        axs[3].plot(time, II_rates[i], label=f'II rate, {muscle} ')

    axs[2].plot(time, np.ones_like(time) * 10, 'k--', label='Base rate')
    axs[3].plot(time, np.ones_like(time) * 20, 'k--', label='Base rate')
    axs[3].set_xlabel('Time (s)')
    axs[0].set_ylabel('Stretch (dimless)')
    axs[1].set_ylabel('Velocity (s-1)')
    axs[2].set_ylabel('Ia rate (Hz)')
    axs[3].set_ylabel('II rate (Hz)')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    fig.suptitle('Impact of stretching on the afferent firing rate')
    plt.show()

        # Plot voltages
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Voltage")
    for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):  
            axs[0].plot(df['Time'], df['v_exc'], label=f' exc {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
            axs[1].plot(df['Time'], df['v_inh'], label=f' inh {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
            axs[2].plot(df['Time'], df['v_moto'], label=f' moto {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
    axs[2].set_xlabel('time (ms)')
    axs[0].set_ylabel('v exc (mV)')
    axs[1].set_ylabel('v inh (mV)')
    axs[2].set_ylabel('v moto (mV)')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()

    # Plot conductances
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Inhibitory Conductances")
    for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
        axs[0].plot(df['Time'],df['gIa_inh'], label=f'gIa {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[1].plot(df['Time'],df['gII_inh'], label=f'gII {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[2].plot(df['Time'],df['gi_inh'], label=f'gi {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
    
    axs[2].set_xlabel('time (ms)')
    axs[0].set_ylabel('gIa (nS)')
    axs[1].set_ylabel('gII (nS)')
    axs[2].set_ylabel('gi (nS)')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.title("Excitatory Conductances")
    for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
        plt.plot(df['Time'],df['gII_exc'], label=f'gII {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])  
    plt.xlabel('time (ms)')
    plt.ylabel('gII (nS)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.title("Excitatory Vpsp")
    for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
        plt.plot(df['Time'],df['Vpsp'], label=f'Vpsp {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        print('mean Vpsp : ', np.mean(df['Vpsp'] )
    plt.xlabel('time (ms)')
    plt.ylabel('Vpsp (mV)')
    plt.legend()
    plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Motoneurons Conductances")
    for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
        axs[0].plot(df['Time'],df['gIa_moto'], label=f'gIa {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[1].plot(df['Time'],df['gex_moto'], label=f'gex {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[2].plot(df['Time'],df['gi_moto'], label=f'gi {muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
    
    axs[2].set_xlabel('time (ms)')
    axs[0].set_ylabel('gIa (nS)')
    axs[1].set_ylabel('gex (nS)')
    axs[2].set_ylabel('gi (nS)')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()

    # Raster Plots
    fig, axs = plt.subplots(num_fiber_types, num_muscles, figsize=(10, 10), sharex=True)
    
    if num_muscles == 1:
        axs = np.expand_dims(axs, axis=1)

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
    fig_path = os.path.join(folder, f'Raster_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()

    # Firing rate plots
    fig, axs = plt.subplots(num_fiber_types, 1, figsize=(10, 10), sharex=True)
    time = muscle_data[0]['Time'].values 

    for i, (muscle, spikes_muscle) in enumerate(spikes.items()):
        for j, (fiber_type, fiber_spikes) in enumerate(spikes_muscle.items()):
            all_spike_times = np.concatenate(list(fiber_spikes.values()))
            firing_rate=np.zeros_like(time)
            if len(all_spike_times)>1:

                kde = gaussian_kde(all_spike_times, bw_method=0.3)
                firing_rate = kde(time) * len(all_spike_times) / len(fiber_spikes)

            axs[j].plot(time, firing_rate, label=f"{muscle}", color=colorblind_friendly_colors[color_keys[i]])
            
            if (fiber_type=="MN0"):
                lambda_=firing_rate+eff_recruited/len(fiber_spikes)*ees_freq/hertz
                axs[j+1].plot(time, (lambda_**(-1)+ T_refr/second)**(-1), label="theoretical MN rate {muscle}", color=colorblind_friendly_colors[color_keys[i+2]])

            axs[j].set_ylabel(f'{fiber_type} firing rate (Hz)')
            axs[j].legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Smoothed Instantaneous Firing Rate (KDE)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Firing_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()


    # Mean activation dynamics - 
    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    labels = ['mean_e', 'mean_u', 'mean_c', 'mean_P', 'mean_activation']
    for i, label in enumerate(labels):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            axs[i].plot(df['Time'], df[label], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[i].set_ylabel(label)
        axs[i].legend()


    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Mean Activation Dynamics: Muscle Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Activation_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()


    # Muscle properties - for all muscles
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    props = ['fiber_length', 'stretch', 'velocity']
    ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s-1)']

    for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            axs[i].plot(df['Time'], df[prop], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[i].set_ylabel(ylabel)
        axs[i].legend()

    axs[-1].set_xlabel('Time (s)')
    fig.suptitle("Muscle Properties: Muscle Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    path_fig = os.path.join(folder, f'Muscle_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()



def plot_joint_angle_from_sto_file(filepath, columns_wanted, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):
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
    fig_path = os.path.join(folder, f'Joint_angles_and_speed_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()


def plot_act_length_from_sto_file(filepath, muscle_names, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start_idx = i + 1
            break

    df = pd.read_csv(filepath, sep='\t', skiprows=data_start_idx)
    df.columns = ["/".join(col.split("/")[-2:]) for col in df.columns]

    fig, axs = plt.subplots(len(muscle_names), 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Activations and fiber lengths", fontsize=16)

    for i, muscle_name in enumerate(muscle_names):
        axs[i, 1].plot(df['time'], df[muscle_name + '/fiber_length'], label=f"{muscle_name}", color=colorblind_friendly_colors["blue"])
        axs[i, 1].set_ylabel("L (m)")
        axs[i, 1].set_title("Fiber length")
        axs[i, 1].legend()
        axs[i, 1].grid(True)

        axs[i, 0].plot(df['time'], df[muscle_name + '/activation'], label=f"{muscle_name} ", color=colorblind_friendly_colors["orange"])
        axs[i, 0].set_ylabel("a (dimless)")
        axs[i, 0].set_title("Activation")
        axs[i, 0].legend()
        axs[i, 0].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'act_length_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()
