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
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, 
                 associated_joint, fast_type_mu, initial_state_opensim, activation_function=None):
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
        activation_function : callable, optional
            Function f(muscle_idx, t) -> activation_value where t is in [0, reaction_time]
            If None, uses values from initial_condition_spike_activation
            If provided, automatically updates a0 in initial_condition_spike_activation for consistency
        """
        self.reaction_time = reaction_time
        self.ees_recruitment_profile = ees_recruitment_profile
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        self.associated_joint = associated_joint
        self.fast_type_mu = fast_type_mu
        self.initial_state_opensim = initial_state_opensim
        self.activation_function = activation_function
        
        # Cache for activation history
        self._activation_cache = {}
        
        # These will be set by subclasses
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
        self.initial_potentials = {}
        self.initial_condition_spike_activation = {}
    
    def _ensure_activation_consistency(self):
        """
        Ensure consistency between activation_function and initial_condition_spike_activation.
        If activation_function is provided, update a0 values to match activation at t=reaction_time.
        """
        if self.activation_function is not None and self.initial_condition_spike_activation:
            T_reaction_seconds = float(self.reaction_time / second)
            
            for muscle_idx in range(self.number_muscles):
                if muscle_idx < len(self.initial_condition_spike_activation):
                    muscle_conditions = self.initial_condition_spike_activation[muscle_idx]
                    if muscle_conditions:
                        for condition in muscle_conditions:
                            # Update a0 to match activation function at end of reaction time
                            condition['a0'] = self.activation_function(muscle_idx, T_reaction_seconds)
    
    def get_activation_history(self, time_step):
        """
        Get activation history for a given time step.
        
        Parameters:
        -----------
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for the simulation
            
        Returns:
        --------
        numpy.ndarray
            Activation history array of shape (number_muscles, int(reaction_time/time_step))
        """
        cache_key = float(time_step / ms)
        
        if cache_key not in self._activation_cache:
            n_steps = int(self.reaction_time / time_step)
            activation_history = np.zeros((self.number_muscles, n_steps))
            
            if self.activation_function is not None:
                # Sample the user function
                times = np.linspace(0, float(self.reaction_time/second), n_steps)
                for muscle_idx in range(self.number_muscles):
                    for i, t in enumerate(times):
                        activation_history[muscle_idx, i] = self.activation_function(muscle_idx, t)
            else:
                # Use default from initial_condition_spike_activation
                for muscle_idx in range(self.number_muscles):
                    if muscle_idx < len(self.initial_condition_spike_activation):
                        muscle_conditions = self.initial_condition_spike_activation[muscle_idx]
                        if muscle_conditions and len(muscle_conditions) > 0:
                            a0 = muscle_conditions[0].get('a0', 0.0)
                            activation_history[muscle_idx, :] = a0
            
            self._activation_cache[cache_key] = activation_history
            
        return self._activation_cache[cache_key].copy()

    def validate_input(self):
        """Base validation method - should be overridden by subclasses."""
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
        
        # Ensure consistency between activation function and initial conditions
        self._ensure_activation_consistency()
        
        activation_history = self.get_activation_history(time_step)
        
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
        
        if plot:
            if ees_stimulation_params is not None:
                plot_recruitment_curves(self.ees_recruitment_profile, current_current=ees_stimulation_params.get('intensity'),
                    base_output_path=base_output_path, balance=ees_stimulation_params.get('balance', 0), num_muscles=self.number_muscles)
                    
            plot_mouvement(time_series, self.muscles_names, self.associated_joint, base_output_path)
            plot_neural_dynamic(time_series, self.muscles_names, base_output_path)
            plot_raster(spikes, base_output_path)
            plot_activation(time_series, self.muscles_names, base_output_path)
        
        return spikes, time_series
