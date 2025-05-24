from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

from .Loop.closed_loop import closed_loop
from .Visualization.plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves
from .Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array


class BiologicalSystem:
    """
    Base class for neural reflex systems.
    
    This class provides the common framework for different types of reflex systems,
    handling the core simulation and analysis functionality.
    """
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, associated_joint, fast_type_mu, initial_state_opensim):
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
        initial_state_opensim : dict
            Initial state for OpenSim simulation
        """
        self.reaction_time = reaction_time
        self.ees_recruitment_profile = ees_recruitment_profile
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        self.associated_joint = associated_joint
        self.fast_type_mu = fast_type_mu
        self.initial_state_opensim = initial_state_opensim
        
        # These will be set by subclasses
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
        self.initial_potentials = {}
        self.initial_condition_spike_activation = {}
    
    def _initialize_activation_history(self, time_step):
        """
        Initialize activation history based on initial_condition_spike_activation and time parameters.
        
        Parameters:
        -----------
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for the simulation
            
        Returns:
        --------
        numpy.ndarray
            Activation history array of shape (number_muscles, int(reaction_time/time_step))
        """
        n_time_steps = int(self.reaction_time / time_step)
        activation_history = np.zeros((self.number_muscles, n_time_steps))
        
        # Initialize with values from initial_condition_spike_activation
        for muscle_idx in range(self.number_muscles):
            if muscle_idx < len(self.initial_condition_spike_activation):
                # Get the initial activation value for this muscle
                muscle_initial_conditions = self.initial_condition_spike_activation[muscle_idx]
                if muscle_initial_conditions and len(muscle_initial_conditions) > 0:
                    # Use the activation value from the first motor neuron as baseline
                    initial_activation = muscle_initial_conditions[0].get('a0', 0.0)
                    activation_history[muscle_idx, :] = initial_activation
        
        return activation_history

    def validate_input(self):
        """
        Base validation method - should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate_input method")

    def run_simulation(self, n_iterations, time_step=0.1*ms, 
                      ees_stimulation_params=None, torque_profile=None, 
                      seed=42, base_output_path=None, plot=True):
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
        plot : bool
            Whether to generate plots
        
        Returns:
        --------
        tuple
            (spikes, time_series) containing simulation results
        """
        
        # Initialize activation history
        activation_history = self._initialize_activation_history(time_step)
        
        torque_array = None
        if torque_profile is not None:
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            torque_array = transform_torque_params_in_array(time_points, torque_profile)
        
        ees_params = None
        if ees_stimulation_params is not None:
            ees_params = transform_intensity_balance_in_recruitment(
                self.ees_recruitment_profile, ees_stimulation_params, 
                self.neurons_population, self.number_muscles)
        
        spikes, time_series = closed_loop(
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
            self.initial_potentials, self.initial_condition_spike_activation, self.initial_state_opensim,
            activation_history, torque_array=torque_array, ees_params=ees_params,
            fast=self.fast_type_mu, seed=seed, base_output_path=base_output_path)
        
        # Generate standard plots
        if plot:
            if ees_stimulation_params is not None:
                plot_recruitment_curves(self.ees_recruitment_profile, current_current=ees_stimulation_params.get('intensity'),
                    base_output_path=base_output_path, balance=ees_stimulation_params.get('balance', 0), num_muscles=self.number_muscles)
                    
            plot_mouvement(time_series, self.muscles_names, self.associated_joint, base_output_path)
            plot_neural_dynamic(time_series, self.muscles_names, base_output_path)
            plot_raster(spikes, base_output_path)
            plot_activation(time_series, self.muscles_names, base_output_path)
        
        return spikes, time_series
