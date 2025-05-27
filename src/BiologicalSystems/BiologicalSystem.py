from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from copy import deepcopy
import inspect

from ..Loop.closed_loop import closed_loop
from ..Visualization.plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves
from ..Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array


class BiologicalSystem(ABC):
    """
    
    This abstract class provides the common framework for different types of reflex systems,
    handling the core simulation.
    """
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, 
                 associated_joint, fast_type_mu, neurons_population, connections, spindle_model, 
                 initial_state_neurons, initial_condition_spike_activation,initial_state_opensim,
                 activation_func):
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
        
        self.neurons_population = neurons_population
        self.connections = connections
        self.spindle_model = spindle_model
        
        # Initial conditions
        self.initial_state_neurons = initial_state_neurons
        self.initial_state_opensim = initial_state_opensim
        self.initial_condition_spike_activation = initial_condition_spike_activation
        self.activation_function = activation_func
                     
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
        if not time_step.dim == second.dim:
            raise ValueError(f"Time step has incorrect unit! ")

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
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,self.fast_type_mu,
            self.copy_brian_dict(self.initial_state_neurons), deepcopy(self.initial_condition_spike_activation), deepcopy(self.initial_state_opensim),
            self.activation_function, torque_array=torque_array, ees_params=ees_params,
            seed=seed, base_output_path=base_output_path)
        
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
        
        self.initial_state_neurons = self.final_state['neurons']
        self.initial_condition_spike_activation = self.final_state['spikes_activations']
        self.initial_state_opensim = self.final_state['opensim']
        self.activation_function=self.final_state['last_activations']



    def clone_with(self, **params):
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
                    kwargs[name] = self.copy_brian_dict(attr)
                else:
                    kwargs[name] = deepcopy(attr)
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default

        kwargs.update(params)
        return cls(**kwargs)


    @staticmethod 
    def copy_brian_dict(self,d):
        if isinstance(d, dict):
            # If d is a dictionary, apply the function to each value
            return {k: self.copy_brian_dict(v) for k, v in d.items()}
        elif hasattr(d, 'copy'):
            # If the object has a `.copy()` method (e.g., Brian2 Quantity), use it
            return d.copy()
        else:
            # Otherwise, return the value as-is (int, float, string, etc.)
            return d

