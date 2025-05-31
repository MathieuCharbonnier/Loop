from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import inspect
from scipy.signal import find_peaks
from datetime import datetime

from ..Loop.closed_loop import closed_loop
from ..Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array, calculate_full_recruitment


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
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, resting_lengths,
                 associated_joint, fast_type_mu, neurons_population, connections, spindle_model, 
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
        self.resting_lengths=resting_lengths
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

                     
        # Store the results:
        self.spikes = None
        self.time_series = None
        self.final_state = None
        

    @abstractmethod
    def validate_input(self):
        """
        Validate input parameters and ensure the system is properly configured.
        Must be implemented by subclasses to check their specific requirements.
        """
        pass


    def run_simulation(self, n_iterations, time_step=0.1*ms, 
                      ees_stimulation_params=None, torque_profile=None, 
                      seed=42, base_output_path=None):
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

        torque_array = None
        if torque_profile is not None:
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            torque_array = transform_torque_params_in_array(time_points, torque_profile)
        
        ees_params = None
        if ees_stimulation_params is not None:
            ees_params = transform_intensity_balance_in_recruitment(
                self.ees_recruitment_profile, ees_stimulation_params, 
                self.neurons_population, self.number_muscles)
        
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
            BiologicalSystem.copy_brian_dict(self.initial_state_neurons), 
            deepcopy(self.initial_condition_spike_activation), 
            deepcopy(self.initial_state_opensim),
            self.activation_function, 
            self.stretch_history,
            torque_array=torque_array, 
            ees_params=ees_params,
            seed=seed, 
            base_output_path=base_output_path)
        
        return self.spikes, self.time_series

    def plot(self, base_output_path=None, ees_stimulation_params=None):
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
        
        if ees_stimulation_params is not None:
            self.plot_recruitment_curves(
                self.ees_recruitment_profile, 
                current_current=ees_stimulation_params.get('intensity'),
                base_output_path=base_output_path, 
                balance=ees_stimulation_params.get('balance', 0), 
                num_muscles=self.number_muscles
            )
                
        self.plot_mouvement(base_output_path)
        self.plot_neural_dynamic(base_output_path)
        self.plot_raster(base_output_path)
        self.plot_activation(base_output_path)
        

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

    
    def plot_raster(self, base_output_path=None):
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
        filename = f'Raster_Plot_{timestamp}.png'
        if base_output_path:
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_neural_dynamic(self, base_output_path=None):
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
        filename = f'Neurons_Dynamics_{timestamp}.png'
        if base_output_path:
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_activation(self, base_output_path=None):
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
        filename = f'Activation_{timestamp}.png'
        if base_output_path:
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()
    
    
    def plot_mouvement(self, base_output_path=None):
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
        
        def save_figure(fig, filename):
            """Helper to save figures."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{filename}_{timestamp}.png'
            if base_output_path:
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
        save_figure(fig_joint, f'Joint_{joint_name}')
    
        # ---------- MUSCLE DYNAMICS PLOT ----------
        props = ['Fiber_length', 'Stretch', 'Stretch_Velocity', 'Normalized_Force']
        ylabels = ['Fiber length (m)', 'Stretch (dimless)', 'Stretch Velocity (s⁻¹)', 'Normalized Force (N)']
    
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
        save_figure(fig_muscle, "Muscle_dynamic")
    
    
    def plot_recruitment_curves(self, ees_recruitment_params, current_current, balance=0, 
                               base_output_path=None, num_muscles=None):
        """
        Plot recruitment curves for all fiber types using the threshold-based sigmoid.
        Only shows fractions of population, not absolute counts.
        
        Parameters:
        -----------
        ees_recruitment_params : dict
            Dictionary with threshold and saturation values
        current_current : float
            Current intensity value to highlight
        balance : float
            Electrode position bias (-1 to 1)
        base_output_path : str, optional
            Path to save the plot
        num_muscles : int, optional
            Number of muscles (defaults to self.number_muscles)
        """
        if num_muscles is None:
            num_muscles = self.number_muscles
            
        currents = np.linspace(0, 1, 100)
        
        # Calculate recruitment fractions at each intensity
        fraction_results = []
        
        for current in currents:
            # Get fractions directly
            fractions = calculate_full_recruitment(
                current, 
                ees_recruitment_params, 
                balance, 
                num_muscles
            )
            fraction_results.append(fractions)
          
        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(fraction_results)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors and styles for different fiber types
        style_map = {
            'Ia_flexor': 'r-', 'II_flexor': 'r--', 'MN_flexor': 'r-.',
            'Ia_extensor': 'b-', 'II_extensor': 'b--', 'MN_extensor': 'b-.'
        }
        
        # For non-muscle-specific case
        single_style_map = {'Ia': 'g-', 'II': 'g--', 'MN': 'g-.'}
        
        for col in df.columns:
            # Choose appropriate style
            if col in style_map:
                line_style = style_map[col]
            elif col in single_style_map:
                line_style = single_style_map[col]
            else:
                # Default styling
                if "extensor" in col:
                    line_style = 'b-'  # Blue for extensors
                else:
                    line_style = 'r-'  # Red for flexors
            
            ax.plot(currents, df[col], line_style, label=col)
            
        ax.axvline(x=current_current, color='r', linestyle='--', label='Current current')
        ax.set_xlabel('Normalized Current Amplitude')
        ax.set_ylabel('Fraction of Fibers Recruited')
        if num_muscles == 2:
            ax.set_title(f'Fiber Recruitment (Balance = {balance})')
        else:
            ax.set_title('Fiber Recruitment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines at 10% and 90% recruitment
        ax.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'Recruitment_Curve_{timestamp}.png'
        if base_output_path:
            fig_path = os.path.join(base_output_path, filename)
        else:
            os.makedirs("Results", exist_ok=True)
            fig_path = os.path.join("Results", filename)
        fig.savefig(fig_path)
        plt.show()

    def get_system_state(self):
        """Return the current state of the biological system for transfer"""
        if self.final_state is not None:
            return BiologicalSystem.copy_brian_dict(self.final_state)
        else:
            return {'neurons': BiologicalSystem.copy_brian_dict(self.nitial_state_neurons),
                    'spikes_activations':self.initial_condition_spike_activation,
                    'opensim': self.initial_state_opensim ,
                    'last_activations': self.activation_function,
                    'stretch_history' : self.stretch_history}
                    
        
    def set_system_state(self, state):
        """Set the biological system to a specific state"""
        self.initial_state_neurons = state['neurons']
        self.initial_condition_spike_activation = state['spikes_activations']
        self.initial_state_opensim = state['opensim']
        self.activation_function = state['last_activations']
        self.stretch_history_func=self.final.get("stretch_history", None)
        self.final_state=None
        
    def update_system_state(self):
        """
        Update the system state with the results from the last simulation.
        This allows for chaining simulations.
        """
        if self.final_state is None:
            raise ValueError("You should first launch a simulation!")
        
        self.initial_state_neurons = BiologicalSystem.copy_brian_dict(self.final_state['neurons'])
        self.initial_condition_spike_activation = self.final_state['spikes_activations']
        self.initial_state_opensim = self.final_state['opensim']
        self.activation_function = self.final_state['last_activations']
        self.stretch_history_func=self.final.get("stretch_history", None)
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
                    kwargs[name] = BiologicalSystem.copy_brian_dict(attr)
                else:
                    kwargs[name] = deepcopy(attr)
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default

        kwargs.update(params)
        return cls(**kwargs)

    @staticmethod 
    def copy_brian_dict(d):
        """
        Safely copy dictionaries that may contain Brian2 quantities.
        
        Parameters:
        -----------
        d : dict or other
            Dictionary or value to copy
            
        Returns:
        --------
        dict or other
            Copied dictionary or value
        """
        if isinstance(d, dict):
            # If d is a dictionary, apply the function to each value
            return {k: BiologicalSystem.copy_brian_dict(v) for k, v in d.items()}
        elif hasattr(d, 'copy') and callable(getattr(d, 'copy', None)):
            # If the object has a callable `.copy()` method (e.g., Brian2 Quantity), use it
            return d.copy()
        else:
            # Otherwise, return the value as-is (int, float, string, bool, etc.)
            return d
