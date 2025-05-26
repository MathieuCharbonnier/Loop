from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from copy import deepcopy

from ..Loop.closed_loop import closed_loop
from ..Visualization.plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves
from ..Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array


class BiologicalSystem(ABC):
    """
    
    This abstract class provides the common framework for different types of reflex systems,
    handling the core simulation.
    """
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, 
                 associated_joint, fast_type_mu):
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

        """
        self.reaction_time = reaction_time
        self.ees_recruitment_profile = ees_recruitment_profile
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        self.associated_joint = associated_joint
        self.fast_type_mu = fast_type_mu
        
        # These will be set by subclasses in their constructors
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
        
        # Initial conditions
        self.initial_potentials = {}
        self.initial_state_opensim = {}
        self.initial_condition_spike_activation = {}
        self.activation_function = None
                     
        # Store the results:
        self.spikes = None
        self.time_series = None
        self.final_state = None
        
        # Call validation after initialization
        self.validate_input()

    @abstractmethod
    def validate_input(self):
        """
        Validate input parameters and ensure the system is properly configured.
        Must be implemented by subclasses to check their specific requirements.
        """
        # Base validation that all subclasses should perform
        if not self.muscles_names:
            raise ValueError("muscles_names cannot be empty")
        if self.number_muscles <= 0:
            raise ValueError("Number of muscles must be positive")
        if self.reaction_time <= 0:
            raise ValueError("Reaction time must be positive")


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
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
            self.initial_potentials, self.initial_condition_spike_activation, self.initial_state_opensim,
            self.activation_function, torque_array=torque_array, ees_params=ees_params,
            fast=self.fast_type_mu, seed=seed, base_output_path=base_output_path)
        
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
            plot_recruitment_curves(
                self.ees_recruitment_profile, 
                current_current=ees_stimulation_params.get('intensity'),
                base_output_path=base_output_path, 
                balance=ees_stimulation_params.get('balance', 0), 
                num_muscles=self.number_muscles
            )
                
        plot_mouvement(self.time_series, self.muscles_names, self.associated_joint, base_output_path)
        plot_neural_dynamic(self.time_series, self.muscles_names, base_output_path)
        plot_raster(self.spikes, base_output_path)
        plot_activation(self.time_series, self.muscles_names, base_output_path)

    def update_system_state(self):
        """
        Update the system state with the results from the last simulation.
        This allows for chaining simulations.
        """
        if self.final_state is None:
            raise ValueError("You should first launch a simulation!")
        
        self.initial_potentials = self.final_state['potentials']
        self.initial_condition_spike_activation = self.final_state['spikes_activation']
        self.initial_state_opensim = self.final_state['opensim']
        self.activation_function=self.final_state['last_activation']

            
    def clone_with(self, **params):
        """
        Create a copy of the current system with modified parameters.
        
        Parameters:
        -----------
        **params : dict
            Parameters to modify in the cloned system
            
        Returns:
        --------
        BiologicalSystem
            A new instance with modified parameters
        """
        # Create a deep copy of the current instance
        new_instance = deepcopy(self)
        
        # Update parameters directly
        for param_name, param_value in params.items():
            if hasattr(new_instance, param_name):
                setattr(new_instance, param_name, param_value)
            else:
                raise AttributeError(f"Parameter '{param_name}' does not exist")
        
        # Update number_muscles if muscles_names was changed
        if 'muscles_names' in params:
            new_instance.number_muscles = len(params['muscles_names'])
        
        # Reset simulation results and neural network
        new_instance.spikes = None
        new_instance.time_series = None
        new_instance.final_state = None
        new_instance.neurons_population = {}
        new_instance.connections = {}
        new_instance.spindle_model = {}
        
        # Validate the new configuration
        new_instance.validate_input()
        
        return new_instance
