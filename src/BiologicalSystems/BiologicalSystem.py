from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
from scipy.signal import find_peaks
from datetime import datetime

from ..Loop.closed_loop import closed_loop, get_sto_file
from ..Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array
from ..helpers.copy_brian_dict import copy_brian_dict

class BiologicalSystem(ABC):
    """
    
    This abstract class provides the common framework for different types of reflex systems,
    handling the core simulation.
    """
    
    # Colorblind-friendly palette (moved to class level)
    colorblind_friendly_colors = {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "red": "#D55E00",
        "purple": "#CC79A7"
    }
    color_keys = list(colorblind_friendly_colors.keys())
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names,
                 associated_joint, fast_type_mu, neurons_population, connections, spindle_model, seed,
                 initial_state_neurons, initial_condition_spike_activation, initial_state_opensim,
                 activation_func, stretch_history_func=None):
        """
        Initialize the biological system with common parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity
            Reaction time of the system (with time units)
        ees_recruitment_profile : dict
            Dictionary containing EES recruitment parameters for different neuron types
        biophysical_params : dict
            Dictionary containing biophysical parameters for neurons
        muscles_names : list
            List of muscle names involved in the system
        associated_joint : str
            Name of the joint associated with the system
        fast_type_mu : bool
            If True, use fast twitch motor units
        neurons_population : dict
            Dictionary containing neuron populations
        connections : dict
            Dictionary containing neural connections
        spindle_model : object
            Spindle model object
        initial_state_neurons : dict
            Initial state of neurons
        initial_condition_spike_activation : dict
            Initial spike activation conditions
        initial_state_opensim : dict
            Initial OpenSim state
        activation_func : function
            Activation function
        """
        self.reaction_time = reaction_time
        self.ees_recruitment_profile = ees_recruitment_profile
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        
        with open("data/muscle_resting_lengths.json", "r") as f:
            fiber_length_dict = json.load(f)
        
        self.resting_lengths = []
        
        for muscle_name in self.muscles_names:
            if muscle_name not in fiber_length_dict:
                raise ValueError(f"The muscle '{muscle_name}' does not exist")
            self.resting_lengths.append(fiber_length_dict[muscle_name])
      
        self.associated_joint = associated_joint
        self.fast_type_mu = fast_type_mu
        self.neurons_population = neurons_population
        self.connections = connections
        self.spindle_model = spindle_model
        
        # Initial conditions
        self.initial_state_neurons = initial_state_neurons
        self.initial_state_opensim = initial_state_opensim
        self.initial_condition_spike_activation = initial_condition_spike_activation
        self.activation_function = activation_func
        self.stretch_history_function=stretch_history_func
                     
        #To fix the connection probability
        self.seed=seed
                     
        # Store the results:
        self.spikes = None
        self.time_series = None
        self.final_state = None

        #rapid check of type:
        type_checks = {
            "resting_lengths": (list, "Resting lengths should be a list."),
            "biophysical_params": (dict, "Biophysical parameters should be a dict."),
            "muscles_names": (list, "Muscle names should be in a list."),
            "associated_joint": (str, "Joint name should be a string."),
            "neurons_population": (dict, "Neuron population must be a dict."),
            "connections": (dict, "Connections should be a dict."),
            "spindle_model": (dict, "Spindle model should be a dict."),
            "ees_recruitment_profile": (dict, "EES recruitment parameters should be a dict."),
            "fast_type_mu": (bool, "The 'fast_type_mu' flag must be a boolean (True for fast, False for slow)."),
            "initial_state_neurons": (dict, "Initial state of neurons should be a dict."),
            "initial_condition_spike_activation": (list, "Initial conditions for activation dynamics should be a list."),
            "initial_state_opensim": ((dict, type(None)), "Initial state of the musculoskeletal model should be a dict or None."),
        }
        for attr, (expected_type, error_msg) in type_checks.items():
            value = getattr(self, attr)
            if not isinstance(value, expected_type):
                raise TypeError(f"Invalid type for '{attr}': {error_msg} Got {type(value).__name__} instead.")
    

        
    @abstractmethod
    def validate_input(self):
        """
        Validate input parameters and ensure the system is properly configured.
        Must be implemented by subclasses to check their specific requirements.
        """
        pass


    def run_simulation(self, n_iterations, time_step=0.1*ms, 
                      ees_stimulation_params=None, torque_profile=None,
                       base_output_path=None):
        """
        Run simulations and generate plots.
        
        Parameters:
        -----------
        n_iterations : int
            Number of iterations to run
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for the simulation
        ees_stimulation_params : dict, optional
            Parameters for epidural electrical stimulation
        torque_profile : dict, optional
            External torque applied to the joint
        seed : int
            Random seed for reproducibility
        base_output_path : str
            Base path for saving output files
        
        Returns:
        --------
        tuple
            (spikes, time_series) containing simulation results
        """

        if not isinstance(time_step, Quantity):
            raise TypeError("time_step must be a Quantity with physical units.")

        if not time_step.dimensions == second.dimensions:
            raise ValueError(f"Time step has incorrect unit! Got {time_step.unit}, expected a time unit.")

        self.torque_array = None
        if torque_profile is not None:
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            self.torque_array = transform_torque_params_in_array(time_points, torque_profile)
  
        self.ees_params = None
        if ees_stimulation_params is not None:
            self.ees_params = transform_intensity_balance_in_recruitment(
                self.ees_recruitment_profile, ees_stimulation_params, 
                self.neurons_population, self.muscles_names)
        
        self.spikes, self.time_series, self.final_state = closed_loop(
            n_iterations, 
            self.reaction_time, 
            time_step,
            self.neurons_population,
            self.connections,
            self.spindle_model, 
            self.biophysical_params, 
            self.muscles_names, 
            self.number_muscles,
            self.resting_lengths,
            self.associated_joint, 
            self.fast_type_mu,
            copy_brian_dict(self.initial_state_neurons), 
            deepcopy(self.initial_condition_spike_activation), 
            deepcopy(self.initial_state_opensim),
            self.activation_function, 
            self.stretch_history_function,
            torque_array=self.torque_array, 
            ees_params=self.ees_params,
            seed=self.seed, 
            base_output_path=base_output_path)
        
        return self.spikes, self.time_series

    def get_sto_file(self,base_output_path):

        return get_sto_file(self.time_series.loc[1,"Time"]-self.time_series.loc[0,"Time"],
            self.time_series.iloc[-1]['Time'], self.muscles_names, self.associated_joint,
             np.array([self.time_series[f'Activation_{muscle_name}'] for muscle_name in self.muscles_names]), 
             self.torque_array, base_output_path)

    def plot(self, base_output_path=None, prefix=None):
        """
        Generate plots of simulation results.
        
        Parameters:
        -----------
        base_output_path : str, optional
            Base path for saving output files
        ees_stimulation_params : dict, optional
            EES parameters used in simulation (for recruitment curve plotting)
        """
        if self.spikes is None or self.time_series is None:
            raise ValueError("You should first launch a simulation!")
        
                
        self.plot_mouvement(base_output_path, prefix)
        self.plot_neural_dynamic(base_output_path, prefix)
        self.plot_raster(base_output_path, prefix)
        self.plot_activation(base_output_path, prefix)
        

    def plot_frequency_spectrum(self, col_name):
        """
        Plot frequency spectrum of a time series column.
        
        Parameters:
        -----------
        col_name : str
            Name of the column to analyze
        """
        if self.time_series is None:
            raise ValueError("You should first launch a simulation!")
            
        time_serie = self.time_series[col_name]
        dt = self.time_series.iloc[2, 'Time'] - self.time_series.iloc[1, 'Time']
        n = len(time_serie)
        freqs = np.fft.fftfreq(n, d=dt)
        fft_values = np.fft.fft(time_serie)
    
        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_values[pos_mask])
        plt.plot(freqs, magnitudes)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title(f"Frequency Spectrum of {col_name}")
        plt.grid()
        plt.show()

    def compute_period_from_peaks(self, col_name):
        """
        Compute period from peaks in a time series column.
        
        Parameters:
        -----------
        col_name : str
            Name of the column to analyze
            
        Returns:
        --------
        float
            Mean period between peaks
        """
        if self.time_series is None:
            raise ValueError("You should first launch a simulation!")
            
        time_serie = self.time_series[col_name]
        dt = self.time_series.iloc[2, 'Time'] - self.time_series.iloc[1, 'Time']
        peaks, _ = find_peaks(time_serie)
        if len(peaks) < 2:
            return 0.0  # Not enough peaks to estimate frequency
        peak_times = peaks * dt
        periods = np.diff(peak_times)
        return np.mean(periods)

    
    def plot_raster(self, base_output_path=None, prefix=None):
        """
        Plot raster plot of spikes for different neuron types and muscles.
        
        Parameters:
        -----------
        base_output_path : str
            Path to save the plot
        """
        if self.spikes is None:
            raise ValueError("You should first launch a simulation!")
            
        num_muscles = len(self.spikes)
        num_fiber_types = len(next(iter(self.spikes.values())))
        fig, axs = plt.subplots(num_fiber_types, num_muscles, figsize=(12, 3.5*num_fiber_types), sharex=True)
    
        if num_muscles == 1:
            axs = np.expand_dims(axs, axis=1)
    
        for i, (muscle, spikes_muscle) in enumerate(self.spikes.items()):
            for j, (fiber_type, fiber_spikes) in enumerate(spikes_muscle.items()):
                for neuron_id, neuron_spikes in fiber_spikes.items():
                    if neuron_spikes:
                        axs[j, i].plot(neuron_spikes, np.ones_like(neuron_spikes) * int(neuron_id), '.', markersize=3, color='black')
                axs[j, i].set(title=f"{muscle}_{fiber_type}", ylabel="Neuron Index")
                axs[j, i].grid(True)
    
        axs[-1, 0].set_xlabel("Time (s)")
        fig.suptitle('Spikes Raster Plot')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix is not None:
            filename = f'{prefix}_Raster_Plot_{timestamp}.png'
        else:
            filename = f'Raster_Plot_{timestamp}.png'
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_neural_dynamic(self, base_output_path=None, prefix=None):
        """
        Plot neural dynamics from a combined dataframe.
    
        Parameters:
        -----------
        base_output_path : str
            Path to save the plot (e.g., "./figures/")
        """
        if self.time_series is None:
            raise ValueError("You should first launch a simulation!")
            
        df = self.time_series
        muscle_names = self.muscles_names
        
        base_labels = []
        # Take the first muscle and recover the variables to plot:
        muscle = muscle_names[0]
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
    
            ax.set_ylabel(ylabel)
            ax.set_title(base_label)
    
            for muscle in muscle_names:
                full_col = f"{base_label}_{muscle}"
                if full_col in df.columns:
                    ax.plot(time, df[full_col], label=muscle)
    
            ax.legend()
    
        axs[-1].set_xlabel('Time (s)')
        fig.suptitle('Neural Dynamics')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix is not None:
            filename = f'{prefix}Neurons_Dynamics_{timestamp}.png'
        else:
            filename = f'Neurons_Dynamics_{timestamp}.png'
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_activation(self, base_output_path=None, prefix=None):
        """
        Plot activation dynamics from a combined dataframe.
        
        Parameters:
        -----------
        base_output_path: str
            Path to save the plot
        """
        if self.time_series is None:
            raise ValueError("You should first launch a simulation!")
            
        df = self.time_series
        muscle_names = self.muscles_names
        
        fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
        labels = ['mean_e', 'mean_u', 'mean_c', 'mean_P', 'Activation']
        time = df['Time'].values
        
        for i, base_label in enumerate(labels):
            for j, muscle_name in enumerate(muscle_names):
                column_name = f"{base_label}_{muscle_name}"
                if column_name in df.columns:
                    axs[i].plot(time, df[column_name], 
                               label=f'{muscle_name}', 
                               color=self.colorblind_friendly_colors[self.color_keys[j % len(self.color_keys)]])
            
            axs[i].set_ylabel(base_label)
            axs[i].legend()
        
        axs[-1].set_xlabel('Time (s)')
        fig.suptitle("Activation Dynamics ")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix is not None:
            filename = f'{prefix}_Activation_{timestamp}.png'
        else:
            filename = f'Activation_{timestamp}.png'
        if base_output_path:
            os.makedirs(base_output_path, exist_ok=True)
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_mouvement(self, base_output_path=None, prefix=None):
        """
        Plot joint and muscle dynamics from dataframe.
    
        Parameters:
        -----------
        base_output_path : str, optional
            Path to save the plots.
        """
        if self.time_series is None:
            raise ValueError("You should first launch a simulation!")
            
        df = self.time_series
        muscle_names = self.muscles_names
        joint_name = self.associated_joint
        
        def save_figure(fig, filename, prefix):
            """Helper to save figures."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if prefix is not None:
                filename = f'{prefix}_{filename}_{timestamp}.png'
            else:
                filename = f'{filename}_{timestamp}.png'
                
            if base_output_path:
                os.makedirs(base_output_path, exist_ok=True)
                fig_path = os.path.join(base_output_path, filename)
            else:
                os.makedirs("Results", exist_ok=True)
                fig_path = os.path.join("Results", filename)
            fig.savefig(fig_path)
            plt.show()
    
        time = df['Time'].values
        torque_column = 'Torque'
        has_torque = torque_column in df.columns
    
        # ---------- JOINT PLOT ----------
        fig_joint, axs_joint = plt.subplots(3 if has_torque else 2, 1, figsize=(12, 9), sharex=True)
        axs_joint = axs_joint if isinstance(axs_joint, (list, np.ndarray)) else [axs_joint]
    
        axis_idx = 0
        if has_torque:
            axs_joint[axis_idx].plot(time, df[torque_column], label=f'Torque {joint_name}', color='tab:red')
            axs_joint[axis_idx].set_ylabel("Torque (Nm)")
            axs_joint[axis_idx].legend()
            axis_idx += 1
    
        axs_joint[axis_idx].plot(time, df[f"Joint_{joint_name}"], label=f"{joint_name} angle")
        axs_joint[axis_idx].set_ylabel("Joint Angle (°)")
        axs_joint[axis_idx].legend()
        axis_idx += 1
    
        axs_joint[axis_idx].plot(time, df[f"Joint_Velocity_{joint_name}"], label=f"{joint_name} velocity")
        axs_joint[axis_idx].set_ylabel("Joint Velocity (°/s)")
        axs_joint[axis_idx].set_xlabel("Time (s)")
        axs_joint[axis_idx].legend()
    
        fig_joint.suptitle("Joint Dynamics")
        fig_joint.tight_layout(rect=[0, 0.03, 1, 0.95])
            
        save_figure(fig_joint, f'Joint_{joint_name}', prefix)
    
        # ---------- MUSCLE DYNAMICS PLOT ----------
        props = ['Fiber_length', 'Stretch', 'Stretch_Velocity', 'Force']
        ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s⁻¹)', 'Force (dimless)']
    
        fig_muscle, axs_muscle = plt.subplots(len(props), 1, figsize=(12, 3 * len(props)), sharex=True)
    
        for i, (prop, ylabel) in enumerate(zip(props, ylabels)):
            for j, muscle_name in enumerate(muscle_names):
                column = f"{prop}_{muscle_name}"
                if column in df.columns:
                    color_key = self.color_keys[j % len(self.color_keys)]
                    color = self.colorblind_friendly_colors[color_key]
                    axs_muscle[i].plot(time, df[column], label=muscle_name, color=color)
            axs_muscle[i].set_ylabel(ylabel)
            axs_muscle[i].legend()
        axs_muscle[-1].set_xlabel("Time (s)")
    
        fig_muscle.suptitle("Muscle Dynamics")
        fig_muscle.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_figure(fig_muscle, "Muscle_dynamic", prefix)
    

    def get_system_state(self):
        """Return the current state of the biological system for transfer"""
        if self.final_state is not None:
            return copy_brian_dict(self.final_state)
        else:
            return {'neurons': copy_brian_dict(self.initial_state_neurons),
                    'spikes_activations':self.initial_condition_spike_activation,
                    'opensim': self.initial_state_opensim ,
                    'last_activations': self.activation_function,
                    'stretch_history' : self.stretch_history_function}
                    
        
    def set_system_state(self, state):
        """Set the biological system to a specific state"""
        self.initial_state_neurons = state['neurons']
        self.initial_condition_spike_activation = state['spikes_activations']
        self.initial_state_opensim = state['opensim']
        self.activation_function = state['last_activations']
        self.stretch_history_function=state['stretch_history']
        self.final_state=None
        
    def update_system_state(self):
        """
        Update the system state with the results from the last simulation.
        This allows for chaining simulations.
        """
        if self.final_state is None:
            raise ValueError("You should first launch a simulation!")
        
        self.initial_state_neurons = copy_brian_dict(self.final_state['neurons'])
        self.initial_condition_spike_activation = self.final_state['spikes_activations']
        self.initial_state_opensim = self.final_state['opensim']
        self.activation_function = self.final_state['last_activations']
        self.stretch_history_func=self.final_state.get("stretch_history", None)
        self.final_state=None


    def clone_with(self, **params): 
        """
        Create a clone of the system with modified parameters.
        
        Parameters:
        -----------
        **params : dict
            Parameters to override in the clone
            
        Returns:
        --------
        BiologicalSystem
            Cloned system with modified parameters
        """
        cls = self.__class__
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if hasattr(self, name):
                attr = getattr(self, name)
                # Use deep copy or custom logic for specific types
                if isinstance(attr, dict):
                    # Safe copy for Brian2 quantities
                    kwargs[name] = copy_brian_dict(attr)
                else:
                    kwargs[name] = deepcopy(attr)
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default

        kwargs.update(params)
        return cls(**kwargs)


