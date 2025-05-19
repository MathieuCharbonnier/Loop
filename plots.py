import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import second, hertz
import os
from datetime import datetime

# Colorblind-friendly palette
colorblind_friendly_colors = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7"
}
color_keys = list(colorblind_friendly_colors.keys())
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_raster(spikes, base_output_path):
    """
    Plot raster plot of spikes for different neuron types and muscles.
    
    Parameters:
    -----------
    spikes : dict
        Dictionary of spike data organized by muscle and fiber type
    base_output_path : str
        Path to save the plot

    """
    num_muscles = len(spikes)
    num_fiber_types = len(next(iter(spikes.values())))
    fig, axs = plt.subplots(num_fiber_types, num_muscles, figsize=(12, 3.5*num_fiber_types), sharex=True)

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

    fig_path = base_output_path+ f'RASTER_{timestamp}.png'
    plt.savefig(fig_path)
    plt.show()



def plot_neural_dynamic(df, muscle_names, base_output_path):
    """
    Plot neural dynamics from a combined dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined dataframe containing data for all muscles with muscle name suffixes
    muscle_names : list
        List of muscle names
    base_output_path : str
        Path to save the plot (e.g., "./figures/")
    """
    base_labels = []
    #take the first muscle and recover the variables to plot:
    muscle= muscle_names[0]
    # Ia/II rate
    ia_cols = [col.replace(f"_{muscle}", "") for col in df.columns if "rate" in col.lower() and "I" in col and muscle in col]
    # IPSP
    ipsp_cols = [col.replace(f"_{muscle}", "") for col in df.columns if "IPSP" in col and muscle in col]
    # Membrane potential
    v_cols = [col.replace(f"_{muscle}", "") for col in df.columns if "potential" in col and muscle in col]
    # Motoneuron rate
    mn_cols = [col.replace(f"_{muscle}", "") for col in df.columns if "MN_rate" in col and muscle in col]

    base_labels.extend(ia_cols + ipsp_cols + v_cols + mn_cols)

    # Create subplots
    fig, axs = plt.subplots(len(base_labels), 1, figsize=(12, 3.5 * len(base_labels)), sharex=True)
    if len(base_labels) == 1:
        axs = [axs]

    time = df['Time'].values

    for i, base_label in enumerate(base_labels):
        ax = axs[i]

        # Determine y-label based on feature type
        if "rate" in base_label.lower():
            ylabel = "FR (Hz)"
        elif "potential" in base_label.lower():
            ylabel = "v (mV)"
        elif "IPSP" in base_label:
            ylabel = "IPSP (nA)"
        else:
            ylabel = base_label  # fallback

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(base_label, fontsize=13)

        for muscle in muscle_names:
            full_col = f"{base_label}_{muscle}"
            if full_col in df.columns:
                ax.plot(time, df[full_col], label=muscle)

        ax.legend(fontsize=9)
        ax.tick_params(labelsize=11)

    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle('Neural Dynamics', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = f"{base_output_path}NEURONS_DYNAMICS_{timestamp}.png"
    plt.savefig(fig_path)
    plt.show()


def plot_activation(df, muscle_names, base_output_path):
    """
    Plot activation dynamics from a combined dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined dataframe containing data for all muscles with muscle name suffixes
    muscle_names : list
        List of muscle names
    base_output_path: str
        Path to save the plot
    """
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    labels = ['mean_e', 'mean_u', 'mean_c', 'mean_P', 'Activation']
    time = df['Time'].values
    
    for i, base_label in enumerate(labels):
        for j, muscle_name in enumerate(muscle_names):
            column_name = f"{base_label}_{muscle_name}"
            if column_name in df.columns:
                axs[i].plot(time, df[column_name], 
                           label=f'{muscle_name}', 
                           color=colorblind_friendly_colors[color_keys[j % len(color_keys)]])
        
        axs[i].set_ylabel(base_label, fontsize=11)
        axs[i].legend(fontsize=11)
        axs[i].tick_params(labelsize=11)
    
    axs[-1].set_xlabel('Time (s)', fontsize=11)
    fig.suptitle("Activation Dynamics ", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = base_output_path+ f'ACTIVATIONS_{timestamp}.png'
    plt.savefig(fig_path)
    plt.show()


def plot_mouvement(df, muscle_names, joint_name, base_output_path):
    """
    Plot movement dynamics from a combined dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined dataframe containing data for all muscles with muscle name suffixes
    muscle_names : list
        List of muscle names
    joint_name : str
        Name of the joint
    base_output_path : str
        Path to save the plot
    """
    torque_column = f'Torque'
    has_torque = torque_column in df.columns

    n_subplots = 6 if has_torque else 5
    fig, axs = plt.subplots(n_subplots, 1, figsize=(12,3*n_subplots ), sharex=True)

    time = df['Time'].values
    current_axis = 0

    # Plot torque first, if available
    if has_torque:
        axs[current_axis].plot(time, df[torque_column], label=f'Torque {joint_name}', color='tab:red')
        axs[current_axis].set_ylabel("Torque (Nm)", fontsize=11)
        axs[current_axis].legend(fontsize=11)
        axs[current_axis].tick_params(labelsize=10)
        current_axis += 1

    # Plot joint angle
    joint_column = f"Joint_{joint_name}"
    axs[current_axis].plot(time, df[joint_column], label=joint_name)
    axs[current_axis].set_ylabel("Joint Angle (°)", fontsize=11)
    axs[current_axis].set_xlabel('Time (s)', fontsize=11)
    axs[current_axis].legend(fontsize=11)
    current_axis += 1

    # Plot joint Velocity
    joint_column = f"Joint_Velocity_{joint_name}"
    axs[current_axis].plot(time, df[joint_column], label=joint_name+ " velocity")
    axs[current_axis].set_ylabel("Joint Velocity (°/s)", fontsize=11)
    axs[current_axis].set_xlabel('Time (s)', fontsize=11)
    axs[current_axis].legend(fontsize=11)
    current_axis += 1


    # Plot fiber properties: Fiber_length, Stretch, Velocity
    props = ['Fiber_length', 'Stretch', 'Stretch_Velocity']
    ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s⁻¹)']

    for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
        for j, muscle_name in enumerate(muscle_names):
            column_name = f"{prop}_{muscle_name}"
            if column_name in df.columns:
                axs[current_axis].plot(time, df[column_name],
                                       label=f'{muscle_name}',
                                       color=colorblind_friendly_colors[color_keys[j % len(color_keys)]])
        axs[current_axis].set_ylabel(ylabel, fontsize=11)
        axs[current_axis].legend(fontsize=11)
        axs[current_axis].tick_params(labelsize=10)
        current_axis+=1
    fig.suptitle("Movement", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = base_output_path+ f'MOUVEMENT_{timestamp}.png'
    plt.savefig(fig_path)
    plt.show()




def read_sto(filepath, columns):
    """
    Read OpenSim .sto file and extract specified columns.
    
    Parameters:
    -----------
    filepath : str
        Path to the .sto file
    columns : dict
        Dictionary mapping column names to labels
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the extracted columns
    """
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
 

def plot_from_sto(filepath, columns_wanted, base_output_path, title=None):
    """
    Plot data from an OpenSim .sto file.
    
    Parameters:
    -----------
    filepath : str
        Path to the .sto file
    columns_wanted : dict
        Dictionary mapping column names to labels
    output_base_path : str
        Path to save the plot

    title : str, optional
        Title for the plot
    """
    df = read_sto(filepath, columns_wanted.keys())
    fig, axs = plt.subplots(len(columns_wanted), 1, figsize=(10, 3*len(columns_wanted)), sharex=True)
    
    # Handle single subplot case
    if len(columns_wanted) == 1:
        axs = [axs]
    
    if title is not None:
        fig.suptitle(title, fontsize=16)

    for i, (name_df, name_label) in enumerate(columns_wanted.items()):
        col_name = f"{name_df}/value"
        if col_name in df.columns:
            axs[i].plot(df['time'], df[col_name], label=name_label, color=colorblind_friendly_colors["green"])
            axs[i].set_ylabel(name_label)
            axs[i].grid(True)
            axs[i].legend()

    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = base_output_path+  f'Supplement_sto_{timestamp}.png'
    plt.savefig(fig_path)
    plt.show()
