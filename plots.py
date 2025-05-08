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

def raster_plots(spikes, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq ):

    # Raster Plots
    num_muscles = len(spikes)
    num_fiber_types = len(next(iter(spikes.values())))
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

def plot_times_series( muscle_data, muscle_names, folder, ees_freq, Ia_recruited,II_recruited, eff_recruited, T_refr):

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

    # Firing rate plots
    firing_rates_columns=['Ia_FR_theoretical', 'Ia_FR_monitored',
     'II_FR_theoretical', 'II_FR_monitored',
     'exc_FR_monitored', 'inh_FR_monitored', 
     'MN0_FR_Monitored','MN_FR_theoretical','MN_FR_monitored', 'MN_recruited']
    fig, axs = plt.subplots(len(firing_rates_columns), 1, figsize=(10, 10), sharex=True)
    time = muscle_data[0]['Time'].values 

 

    axs[-1].set_xlabel('Time (s)')
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


  
def plot_joints(df, columns_wanted, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):


    fig, axs = plt.subplots(len(columns_wanted), 2, figsize=(12, 10), sharex=True)
    fig.suptitle("Joint Angles and Speeds", fontsize=16)

    for i, column in enumerate(columns_wanted):
        axs[i, 0].plot(df['time'], df[column + '/value']*180/np.pi, label=f"{column} value", color=colorblind_friendly_colors["green"])
        axs[i, 0].set_ylabel("Angle (degree)")
        axs[i, 0].set_title(f"{column} - Value")
        axs[i, 0].grid(True)

        axs[i, 1].plot(df['time'], df[column + '/speed']*180/np.pi, label=f"{column} speed", color=colorblind_friendly_colors["red"])
        axs[i, 1].set_ylabel("Speed (degree/s)")
        axs[i, 1].set_title(f"{column} - Speed")
        axs[i, 1].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'Joint_angles_and_speed_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()


def plot_act_length_from_sto_file(df, muscle_names, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):
  

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
