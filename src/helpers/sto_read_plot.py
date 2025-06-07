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


def read_sto(filepath):
    """
    Read OpenSim .sto file and extract specified columns.
    
    Parameters:
    -----------
    filepath : str
        Path to the .sto file

    
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

    return df
 

def plot_from_sto(filepath, columns_wanted, title=None):
    """
    Plot data from an OpenSim .sto file.
    
    Parameters:
    -----------
    filepath : str
        Path to the .sto file
    columns_wanted : dict
        Dictionary mapping column names to labels

    title : str, optional
        Title for the plot
    """
    df = read_sto(filepath)
    fig, axs = plt.subplots(len(columns_wanted), 1, figsize=(10, 3*len(columns_wanted)), sharex=True)
    
    # Handle single subplot case
    if len(columns_wanted) == 1:
        axs = [axs]
    
    if title is not None:
        fig.suptitle(title)

    for i, (col_name, label_name) in enumerate(columns_wanted.items()):
        if col_name in df.columns:
            axs[i].plot(df['time'], df[col_name], label=label_name, color=colorblind_friendly_colors["green"])
            axs[i].set_ylabel(label_name)
            axs[i].grid(True)
            axs[i].legend()

    axs[-1].set_xlabel("Time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = 'figure_sto.png'
    plt.savefig(fig_path)
    plt.show()
