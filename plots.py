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
    rate_columns = [(col, "FR (Hz)") for col in muscle_data[0].columns if "rate" in col and "I" in col]
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


def plot_mouvement(muscle_data,muscle_names, joint_value, joint_name folder, ees_freq, Ia_recruited, II_recruited, eff_recruited):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    props = ['Fiber_length', 'Stretch', 'Velocity']
    ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s⁻¹)']

    for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
        for j, (muscle_name, df) in enumerate(zip(muscle_names, muscle_data)):
            axs[i].plot(df['Time'], df[prop], label=f'{muscle_name}', color=colorblind_friendly_colors[color_keys[j]])
        axs[i].set_ylabel(ylabel, fontsize=11)
        axs[i].legend(fontsize=11)
        axs[i].tick_params(labelsize=10)
    axs[-1].plot(muscle_data[0]['Time'], joint_value, label=joint_name)
    axs[-1].set_ylabel("angle (degree)", fontsize=11)
    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle("Mouvement", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    path_fig = os.path.join(folder, f'Mouvement_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
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
 
def plot_from_sto(filepath, columns_wanted, folder, Ia_recruited, II_recruited, eff_recruited, ees_freq, title=None):

    df=read_sto(filepath, columns_wanted.keys())
    fig, axs = plt.subplots(len(columns_wanted), 1, figsize=(10, 3*len(columns_wanted), sharex=True)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    for i, (name_df, name_label) in enumerate(columns_wanted):
        axs[i].plot(df['time'], df[name_df], label=f"name_label", color=colorblind_friendly_colors["green"])
        axs[i].set_ylabel(name_label)
        axs[i].grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'Supplement_sto_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()


  
    

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(folder, f'act_length_Ia_{Ia_recruited}_II_{II_recruited}_eff_{eff_recruited}_freq_{ees_freq}.png')
    plt.savefig(fig_path)
    plt.show()
