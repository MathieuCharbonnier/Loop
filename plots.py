import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import second, hertz
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


def plot_raster(spikes, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):
    num_muscles = len(spikes)
    num_fiber_types = len(next(iter(spikes.values())))
    fig, axs = plt.subplots(num_fiber_types, num_muscles, figsize=(12, 10), sharex=True)

    if num_muscles == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, (muscle, spikes_muscle) in enumerate(spikes.items()):
        for j, (fiber_type, fiber_spikes) in enumerate(spikes_muscle.items()):
            for neuron_id, neuron_spikes in fiber_spikes.items():
                if neuron_spikes:
                    axs[j, i].plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), '.', markersize=3, color='black')
            axs[j, i].set(title=f"{muscle}_{fiber_type}", ylabel="Neuron Index")
            axs[j, i].tick_params(labelsize=11)
            axs[j, i].grid(True)

    axs[-1, 0].set_xlabel("Time (s)", fontsize=11)
    fig.suptitle('Spikes Raster Plot', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(folder, f'Raster_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()


def plot_neural_dynamic(muscle_data, muscle_names, folder, ees_freq, Ia_recruited, II_recruited, eff_recruited):
    rate_columns = [(col, "FR (Hz)") for col in ["Ia_rate", "II_rate"]]
    IPSP_columns = [(col, "IPSP (nA)") for col in muscle_data[0].columns if "IPSP" in col]
    columns = rate_columns + IPSP_columns + [("MN_rate", "FR (Hz)")]

    fig, axs = plt.subplots(len(columns), 1, figsize=(12, 15), sharex=True)
    time = muscle_data[0]['Time'].values

    for i, (col, ylabel) in enumerate(columns):
        ax = axs[i]
        ax.set_title(col, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        for idx, muscle_name in enumerate(muscle_names):
            ax.plot(muscle_data[idx]['Time'], muscle_data[idx][col], label=muscle_name)
        ax.legend(fontsize=10)
        ax.tick_params(labelsize=11)

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Neural Dynamics', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path_fig = os.path.join(folder, f'Dynamic_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()


def plot_activation(muscle_data, muscle_names, folder, ees_freq, Ia_recruited, II_recruited, eff_recruited):
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    labels = ['mean_e', 'mean_u', 'mean_c', 'mean_P', 'Activation']

    for i, label in enumerate(labels):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            axs[i].plot(df['Time'], df[label], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[i].set_ylabel(label, fontsize=11)
        axs[i].legend(fontsize=11)
        axs[i].tick_params(labelsize=11)

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle("Mean Activation Dynamics: Muscle Comparison", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path_fig = os.path.join(folder, f'Activation_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()


def plot_muscle_length(muscle_data, muscle_names, folder, ees_freq, Ia_recruited, II_recruited, eff_recruited):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    props = ['Fiber_length', 'Stretch', 'Velocity']
    ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s⁻¹)']

    for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            axs[i].plot(df['Time'], df[prop], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[i].set_ylabel(ylabel, fontsize=11)
        axs[i].legend(fontsize=11)
        axs[i].tick_params(labelsize=10)

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle("Muscles Properties", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path_fig = os.path.join(folder, f'Muscle_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(path_fig)
    plt.show()

def read_sto(filepath, columns):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start_idx = i + 1
            break

    df = pd.read_csv(filepath, sep='\t', skiprows=data_start_idx)
    df.columns = ["/".join(col.split("/")[-2:]) for col in df.columns]
    cols = ['time']+[f"{c}/{suffix}" for c in columns for suffix in ("value", "speed")]

    return df[cols]


  
def plot_joints(filepath, columns_wanted, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):

    df=read_sto(filepath, columns_wanted)
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


def plot_act_length (filepath, muscle_names, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq):
  
    df=read_sto(filepath, columns_wanted)
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
