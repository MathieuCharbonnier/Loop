
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from closed_loop import closed_loop
from analysis import delay_excitability_MU_type_analysis, EES_stim_analysis
from controller import HierarchicalAnkleController
from plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves

class BiologicalSystem:
    """
    Base class for neural reflex systems.
    
    This class provides the common framework for different types of reflex systems,
    handling the core simulation and analysis functionality.
    """
    
    def __init__(self, reaction_time, ees_recruitment_params, biophysical_params, muscles_names, associated_joint):
        """
        Initialize the biological system with common parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity
            Reaction time of the system (with time units)
        ees_recruitment_params : dict
            Dictionary containing EES recruitment parameters for different neuron types
        biophysical_params : dict
            Dictionary containing biophysical parameters for neurons
        muscles_names : list
            List of muscle names involved in the system
        associated_joint : str
            Name of the joint associated with the system
        """
        self.reaction_time = reaction_time
        self.ees_recruitment_params = ees_recruitment_params
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        self.associated_joint = associated_joint
        
        # These will be set by subclasses
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
    
    def validate_parameters(self):
        """
        Validates the configuration parameters for the neural model.
        
        Checks for consistency between neurons, connections, spindle models,
        and biophysical parameters.
        
        Raises:
            ValueError: If critical errors are found in the configuration
        """
        issues = {"warnings": [], "errors": []}
        
        # Check if neuron types in neuron population match with those in connections and spindle_model
        defined_neurons = set(self.neurons_population.keys())
        
        # Validate muscle count
        if self.number_muscles > 2:
            issues["errors"].append("This pipeline supports only 1 or 2 muscles!")
        
        # If there are Ia or II neurons, check if their equations are properly defined
        neuron_types = {n.split('_')[0] if '_' in n else n for n in defined_neurons}
        
        # Check for II neurons and related conditions
        if "II" in neuron_types:
            # Check if II equations are defined in spindle model
            has_ii_equation = False
            for key in self.spindle_model:
                if key == "II" or key.startswith("II_"):
                    has_ii_equation = True
                    break
                    
            if not has_ii_equation:
                issues["errors"].append("II neurons are defined in neuron population but no equation found in spindle model")
                
            if "exc" not in neuron_types:
                issues["warnings"].append("When II neurons are defined, exc neurons are typically also defined")
        else:
            # Check if II equation is defined but II neurons are not
            for key in self.spindle_model:
                if key == "II" or key.startswith("II_"):
                    issues["warnings"].append("Equation for II defined in spindle model but II neurons not defined in the neurons population")
                    break
        
        # Check for inhibitory neurons and related parameters
        if "inh" in neuron_types:
            if "E_inh" not in self.biophysical_params or "tau_i" not in self.biophysical_params:
                issues["errors"].append("You defined inhibitory neurons, but you forgot to specify one or both inhibitory synapse parameters (E_inh and tau_i)")
        else:
            if "E_inh" in self.biophysical_params or "tau_i" in self.biophysical_params:
                issues["warnings"].append("Inhibitory neuron parameters (E_inh or tau_i) present but no inhibitory neurons defined")
        
        # Check for all mandatory neuron types when multiple muscles are defined
        if self.number_muscles == 2:
            required_types = {"Ia", "MN"}  # Minimum required types
            recommended_types = {"Ia", "II", "inh", "exc", "MN"}  # Recommended for full reciprocal inhibition
            
            defined_types = neuron_types
            missing_required = required_types - defined_types
            
            if missing_required:
                issues["errors"].append(f"For two muscles, at minimum the neuron types {required_types} must be defined. Missing: {missing_required}")
            
            missing_recommended = recommended_types - defined_types
            if missing_recommended:
                issues["warnings"].append(f"For full reciprocal inhibition, all neuron types {recommended_types} are recommended. Missing: {missing_recommended}")
                
            # Check spindle model completeness for two muscles
            if "Ia" in defined_types:
                has_ia_equation = False
                for key in self.spindle_model:
                    if key == "Ia" or key.startswith("Ia_"):
                        has_ia_equation = True
                        break
                
                if not has_ia_equation:
                    issues["errors"].append("Ia neurons defined but no Ia equation found in spindle model")
        
        # Check if all neurons used in connections are defined in neurons_population
        for connection_pair in self.connections:
            pre_neuron, post_neuron = connection_pair
            
            # For two muscles, check if connection neurons have proper muscle suffix
            if self.number_muscles == 2:
                if not any(muscle in pre_neuron for muscle in self.muscles_names) and '_' not in pre_neuron:
                    issues["warnings"].append(f"With two muscles, pre-neuron '{pre_neuron}' in connection {connection_pair} should typically specify which muscle it belongs to")
                
                if not any(muscle in post_neuron for muscle in self.muscles_names) and '_' not in post_neuron:
                    issues["warnings"].append(f"With two muscles, post-neuron '{post_neuron}' in connection {connection_pair} should typically specify which muscle it belongs to")
            
            # Check if neuron types exist in the population
            pre_type = pre_neuron.split('_')[0] if '_' in pre_neuron else pre_neuron
            post_type = post_neuron.split('_')[0] if '_' in post_neuron else post_neuron
            
            if pre_neuron not in self.neurons_population and pre_type not in self.neurons_population:
                issues["errors"].append(f"Neuron '{pre_neuron}' used in connection {connection_pair} but not defined in the neurons population")
            
            if post_neuron not in self.neurons_population and post_type not in self.neurons_population:
                issues["errors"].append(f"Neuron '{post_neuron}' used in connection {connection_pair} but not defined in the neurons population")

        # Validate EES recruitment parameters
        if self.ees_recruitment_params:
            # Check if all required neuron types have recruitment parameters
            for neuron_type in neuron_types:
                if neuron_type in ["Ia", "II", "MN"] and neuron_type not in self.ees_recruitment_params:
                    issues["errors"].append(f"Missing EES recruitment parameters for neuron type '{neuron_type}'")

            
            # Check each recruitment parameter set
            for neuron_type, params in self.ees_recruitment_params.items():
                required_params = ["threshold_10pct", "saturation_90pct"]
                for param in required_params:
                    if param not in params:
                        issues["errors"].append(f"Missing '{param}' in EES recruitment parameters for '{neuron_type}'")
                
                # Check if threshold is less than saturation
                if "threshold_10pct" in params and "saturation_90pct" in params:
                    threshold = params['threshold_10pct']
                    saturation = params['saturation_90pct']
                    
                    # Check values are between 0 and 1
                    if not (0 <= threshold <= 1) or not (0 <= saturation <= 1):
                        raise ValueError(
                            f"Values for '{fiber}' must be between 0 and 1. Got: threshold={threshold}, saturation={saturation}"
                        )
                    if threshold >= saturation:
                        issues["errors"].append(f"Threshold (10%) must be less than saturation (90%) for '{neuron_type}'")

        # Define expected units for each parameter
        expected_units = {
            'T_refr': second,
            'Eleaky': volt,
            'gL': siemens,  
            'Cm': farad,
            'E_ex': volt,
            'tau_e': second,
            'threshold_v': volt
        }
        
        # Check all expected parameters are defined
        for param, expected_unit in expected_units.items():
            if param not in self.biophysical_params:
                issues["errors"].append(f"Missing mandatory biophysical parameter: '{param}'")
                continue
        
            value = self.biophysical_params[param]

            # Check unit compatibility
            if not value.dim == expected_unit.dim:

                issues["errors"].append(
                    f"Parameter '{param}' has incorrect unit. "
                    f"Expected unit compatible with {expected_unit}, but got {value.unit}"
                )
        
        # Check inhibitory parameters 
        if 'tau_i' in self.biophysical_params:
            value = self.biophysical_params['tau_i']
            if not hasattr(value, 'unit') or not value.unit.is_compatible_with(second):
                issues["errors"].append(
                    f"Parameter 'tau_i' has incorrect unit. "
                    f"Expected unit compatible with second, but got {value.unit if hasattr(value, 'unit') else 'no unit'}"
                )
        
        if 'E_inh' in self.biophysical_params:
            value = self.biophysical_params['E_inh']
            if not hasattr(value, 'unit') or not value.unit.is_compatible_with(volt):
                issues["errors"].append(
                    f"Parameter 'E_inh' has incorrect unit. "
                    f"Expected unit compatible with volt, but got {value.unit if hasattr(value, 'unit') else 'no unit'}"
                )

        # Raise error if there are critical issues
        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"Configuration errors found:\n{error_messages}")
        
        # Print warnings if any
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: Configuration issues detected:\n{warning_messages}")
            
        return True  # Return True if validation passes
    
    def run_simulation(self, base_output_path, n_iterations, time_step=0.1*ms, ees_stimulation_params=None,
                   torque=None, fast_type_mu=True, seed=42, save=True):
        """
        Run simulations and generate plots.
        
        Parameters:
        -----------
        base_output_path : str
            Base path for saving output files
        n_iterations : int
            Number of iterations to run
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for the simulation
        ees_params : dict, optional
            Parameters for epidural electrical stimulation
        torque : dict, optional
            External torque applied to the joint
        fast_type_mu : bool
            If True, use fast twitch motor units
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (spikes, time_series) containing simulation results
        """
        spikes, time_series = closed_loop(
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
            base_output_path=base_output_path, TORQUE=torque,EES_RECRUITMENT_PARAMS=self.ees_recruitment_params, EES_STIMULATION_PARAMS=ees_stimulation_params, fast=fast_type_mu, seed=seed
        )
        
        # Generate standard plots
        if ees_params is not None:
            plot_recruitment_curves(self.ees_recruitment_params, current_current=ees_params.get('intensity'), balance=ees_params.get('balance'), num_muscles=self.number_muscles, save=save)
            
        plot_mouvement(time_series, self.muscles_names, self.associated_joint, base_output_path, save=save)
        plot_neural_dynamic(time_series, self.muscles_names, base_output_path, save=save)
        plot_raster(spikes, base_output_path,save=save)
        plot_activation(time_series, self.muscles_names, base_output_path, save=save)
        
        return spikes, time_series

    def analyze_frequency_effects(self, freq_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying EES frequency with fixed afferent and efferent recruitment.
        
        Parameters:
        -----------
        freq_range : array-like
            Range of frequencies to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results
        """
        vary_param = {
            'param_name': 'ees_freq',
            'values': freq_range,
            'label': 'EES Frequency '
        }
        
        return EES_stim_analysis(base_ees_params, vary_param, n_iterations, self.reaction_time, 
                               self.neurons_population, self.connections, self.spindle_model, 
                               self.biophysical_params, self.muscles_names, time_step, seed)
    
    def analyze_co_recruitment_effects(self, afferent_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying afferent recruitment.
        
        Parameters:
        -----------
        afferent_range : array-like
            Range of afferent recruitment values to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results
        """
        vary_param = {
            'param_name': 'afferent_recruited',
            'values': afferent_range,
            'label': 'Afferent Fiber Co-Recruitment '
        }
        
        return EES_stim_analysis(base_ees_params, vary_param, n_iterations, self.reaction_time, 
                               self.neurons_population, self.connections, self.spindle_model, 
                               self.biophysical_params, self.muscles_names, time_step, seed)
    
    def analyze_efferent_recruitment_effects(self, mn_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying efferent (motoneuron) recruitment.
        
        Parameters:
        -----------
        mn_range : array-like
            Range of motoneuron recruitment values to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results
        """
        vary_param = {
            'param_name': 'MN_recruited',
            'values': mn_range,
            'label': 'Motoneuron Recruitment '
        }
        
        return EES_stim_analysis(base_ees_params, vary_param, n_iterations, self.reaction_time, 
                               self.neurons_population, self.connections, self.spindle_model, 
                               self.biophysical_params, self.muscles_names, time_step, seed)

      
    def clonus_analysis(self, base_output_path, delay_values=[10, 25, 50, 75, 100]*ms, 
                        threshold_values=[-45, -50, -55]*mV, duration=1*second, 
                        time_step=0.1*ms, fast_type_mu=True, torque_profile=None, 
                        ees_stimulations_params=None, seed=41):
        """
        Analyze clonus behavior by varying one parameter at a time.
        
        Parameters:
        -----------
        base_output_path : str
            Base path for saving output files
        delay_values : list
            List of delay values to test
        threshold_values : list
            List of threshold values to test
        duration : brian2.units.fundamentalunits.Quantity
            Duration of each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        fast_type_mu : bool
            If True, use fast twitch motor units
        torque_profile : dict
            Dictionary with torque profile parameters
        ees_stimulations_params : dict
            Parameters for EES stimulation
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Analysis results
        """
        return delay_excitability_MU_type_analysis(
            duration, self.reaction_time, self.neurons_population, self.connections, 
            self.spindle_model, self.biophysical_params, self.muscles_names,
            torque_profile, ees_stimulations_params, time_step, seed)

    def find_ees_protocol(self, target_amplitude=15, target_period=2*second, 
                        update_interval=200*ms, prediction_horizon=1000*ms, 
                        simulation_time=10*second):
        """
        Find an optimal EES protocol for a given target movement.
        
        Parameters:
        -----------
        target_amplitude : float
            Target amplitude of joint movement
        target_period : brian2.units.fundamentalunits.Quantity
            Target period of oscillation
        update_interval : brian2.units.fundamentalunits.Quantity
            Update interval for controller
        prediction_horizon : brian2.units.fundamentalunits.Quantity
            Prediction horizon for controller
        simulation_time : brian2.units.fundamentalunits.Quantity
            Total simulation time
            
        Returns:
        --------
        dict
            Optimal EES protocol parameters
        """
        # Initial state
        initial_state = {
            'joint_angle': 0.0,
            'joint_velocity': 0.0,
            'flexor_activation': 0.0,
            'extensor_activation': 0.0
        }
        
        # Create controller
        controller = HierarchicalAnkleController(
            target_amplitude=target_amplitude,
            target_period=target_period, 
            update_interval=update_interval,   
            prediction_horizon=prediction_horizon  
        )
        
        # Initialize and run simulation
        controller.initialize_simulation(initial_state)
        controller.run_simulation(simulation_time)
        
        # Plot results
        controller.plot_results()
        
        return controller.get_optimal_protocol()

        
class Monosynaptic(BiologicalSystem):
    """
    Specialized class for monosynaptic reflexes.
    
    Monosynaptic reflexes involve a direct connection from afferent (Ia) neurons
    to motor neurons (MN) without intermediate interneurons.
    """
    
    def __init__(self, reaction_time=25*ms, biophysical_params=None, muscles_names=None, 
                associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                custom_spindle=None, custom_ees_recruitment_profile=None):
        """
        Initialize a monosynaptic reflex system with default or custom parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity, optional
            Reaction time of the system (default: 25ms)
        biophysical_params : dict, optional
            Custom biophysical parameters for neurons (if None, use defaults)
        muscles_names : list, optional
            List of muscle names (default: ["soleus_r"])
        associated_joint : str, optional
            Name of the associated joint (default: "ankle_angle_r")
        custom_neurons : dict, optional
            Custom neuron population counts (if None, use defaults)
        custom_connections : dict, optional
            Custom neural connections (if None, use defaults)
        custom_spindle : dict, optional
            Custom spindle model equations (if None, use defaults)
        custom_ees_recruitment_params : dict, optional
            Custom EES recruitment parameters (if None, use defaults)
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["soleus_r"]
            
        if biophysical_params is None:
            biophysical_params = {
                'T_refr': 5 * ms,  # Refractory period
                'Eleaky': -70*mV,
                'gL': 10*nS,
                'Cm': 0.3*nF,  
                'E_ex': 0*mV,
                'tau_e': 0.5*ms,
                'threshold_v': -50*mV
            }
            
        if custom_ees_recruitment_params is None:
            ees_recruitment_profile = {
                'Ia': {
                    'threshold_10pct': 0.3,  # Normalized current for 10% recruitment
                    'saturation_90pct': 0.7  # Normalized current for 90% recruitment
                },
                'MN': {
                    'threshold_10pct': 0.7,  # Motoneurons are recruited at high intensity
                    'saturation_90pct': 0.9  
                }
            }
        else:
            ees_recruitment_profile = custom_ees_recruitment_profile
            
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, associated_joint)
        
        # Set default neuron populations
        self.neurons_population = {
            "Ia": 410,       # Type Ia afferent neurons
            "MN": 500       # Motor neurons
        }
        
        # Override with custom values if provided
        if custom_neurons is not None:
            self.neurons_population.update(custom_neurons)
            
        # Set default connections
        self.connections = {
            ("Ia", "MN"): {"w": 2*2.1*nS, "p": 0.9}
        }
        
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model
        self.spindle_model = {
            "Ia": "clip(0.1*(1.6*joint+joint_velocity),0, 100)"
        }
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
            
        # Validate parameters
        self.validate_parameters()


class Trisynaptic(BiologicalSystem):
    """
    Specialized class for trisynaptic reflexes.
    
    Trisynaptic reflexes involve connections from afferent neurons (Ia and II)
    to motor neurons (MN) through excitatory interneurons.
    """
    
    def __init__(self, reaction_time=40*ms, biophysical_params=None, muscles_names=None, 
                associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                custom_spindle=None, custom_ees_recruitment_profile=None):
        """
        Initialize a trisynaptic reflex system with default or custom parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity, optional
            Reaction time of the system (default: 25ms)
        biophysical_params : dict, optional
            Custom biophysical parameters for neurons (if None, use defaults)
        muscles_names : list, optional
            List of muscle names (default: ["tib_ant_r"])
        associated_joint : str, optional
            Name of the associated joint (default: "ankle_angle_r")
        custom_neurons : dict, optional
            Custom neuron population counts (if None, use defaults)
        custom_connections : dict, optional
            Custom neural connections (if None, use defaults)
        custom_spindle : dict, optional
            Custom spindle model equations (if None, use defaults)
        custom_ees_recruitment_params : dict, optional
            Custom EES recruitment parameters (if None, use defaults)
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["tib_ant_r"]
            
        if biophysical_params is None:
            biophysical_params = {
                'T_refr': 5 * ms,
                'Eleaky': -70*mV,
                'gL': 10*nS,
                'Cm': 0.3*nF,
                'E_ex': 0*mV,
                'tau_e': 0.5*ms,
                'threshold_v': -50*mV
            }
            
        if custom_ees_recruitment_profile is None:
            ees_recruitment_profile = {
                'Ia': {
                    'threshold_10pct': 0.3,  # Normalized current for 10% recruitment
                    'saturation_90pct': 0.7  # Normalized current for 90% recruitment
                },
                'II': {
                    'threshold_10pct': 0.4,  # Type II fibers have higher threshold
                    'saturation_90pct': 0.8  # and higher saturation point
                },
                'MN': {
                    'threshold_10pct': 0.7,  # Motoneurons are recruited at high intensity
                    'saturation_90pct': 0.9  
                }
            }
        else:
            ees_recruitment_params = custom_ees_recruitment_profile
            
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_params, biophysical_params, muscles_names, associated_joint)
        
        # Set default neuron populations
        self.neurons_population = {
            "Ia": 280,       # Type Ia afferent neurons
            "II": 280,       # Type II afferent neurons
            "exc": 500,     # Excitatory interneurons
            "MN": 450       # Motor neurons
        }
        
        # Override with custom values if provided
        if custom_neurons is not None:
            self.neurons_population.update(custom_neurons)
            
        # Set default connections
        self.connections = {
            ("Ia", "MN"): {"w": 2*2.1*nS, "p": 0.9},
            ("II", "exc"): {"w": 2*3.64*nS, "p": 0.9},
            ("exc", "MN"): {"w": 2*2.1*nS, "p": 0.9}
        }
        
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model
        self.spindle_model = {
            "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
            "II": "20 + 13.5*stretch"
        }
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
            
        # Validate parameters
        self.validate_parameters()


class ReciprocalInhibition(BiologicalSystem):
    """
    Specialized class for reciprocal inhibition reflexes.
    
    Reciprocal inhibition reflexes involve complex connections between two antagonistic
    muscle systems, with both excitatory and inhibitory connections.
    """
    
    def __init__(self, reaction_time=50*ms, biophysical_params=None, muscles_names=None, 
                associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                custom_spindle=None, custom_ees_recruitment_profile=None):
        """
        Initialize a reciprocal inhibition system with default or custom parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity, optional
            Reaction time of the system (default: 25ms)
        biophysical_params : dict, optional
            Custom biophysical parameters for neurons (if None, use defaults)
        muscles_names : list, optional
            List of muscle names (default: ["tib_ant_r", "med_gas_r"])
        associated_joint : str, optional
            Name of the associated joint (default: "ankle_angle_r")
        custom_neurons : dict, optional
            Custom neuron population counts (if None, use defaults)
        custom_connections : dict, optional
            Custom neural connections (if None, use defaults)
        custom_spindle : dict, optional
            Custom spindle model equations (if None, use defaults)
        custom_ees_recruitment_params : dict, optional
            Custom EES recruitment parameters (if None, use defaults)
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["tib_ant_r", "med_gas_r"]
        elif len(muscles_names) != 2:
            raise ValueError("Reciprocal inhibition requires exactly 2 muscles")
            
        if biophysical_params is None:
            biophysical_params = {
                'T_refr': 5 * ms,
                'Eleaky': -70*mV,
                'gL': 10*nS,
                'Cm': 0.3*nF,
                'E_ex': 0*mV,
                'tau_e': 0.5*ms,
                'E_inh': -75*mV,
                'tau_i': 5*ms,
                'threshold_v': -50*mV
            }
            
        if custom_ees_recruitment_profile is None:
            ees_recruitment_profile = {
                'Ia': {
                    'threshold_10pct': 0.3,  # Normalized current for 10% recruitment
                    'saturation_90pct': 0.7  # Normalized current for 90% recruitment
                  },
                  'II': {
                      'threshold_10pct': 0.4,  # Type II fibers have higher threshold
                      'saturation_90pct': 0.8  # and higher saturation point
                  },
                  'MN':{
                      'threshold_10pct': 0.7,  # Motoneuron are recruited at high intensity
                      'saturation_90pct': 0.9  
              }  
            }
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, associated_joint)
        
        
        # Setup specialized neuron populations for reciprocal inhibition
        self.neurons_population = {
            # Afferents for each muscle
            f"Ia_flexor": 280,
            f"II_flexor": 280,
            f"Ia_extensor": 160,
            f"II_extensor": 160,
            
            # Interneurons
            f"exc_flexor": 500,
            f"exc_extensor": 500,
            f"inh_flexor": 500,
            f"inh_extensor": 500,
            
            # Motor neurons
            f"MN_flexor": 450,
            f"MN_extensor": 580
        }
        
        # Override with custom values if provided
        if custom_neurons is not None:
            self.neurons_population.update(custom_neurons)
            
        # Set default connections with reciprocal inhibition pattern
        self.connections = {
            # Direct pathways
            (f"Ia_flexor", f"MN_flexor"): {"w": 2*2.1*nS, "p": 0.9},
            (f"Ia_extensor", f"MN_extensor"): {"w": 2*2.1*nS, "p": 0.9},
            
            # Ia inhibition pathways
            (f"Ia_flexor", f"inh_flexor"): {"w": 2*3.64*nS, "p": 0.9},
            (f"Ia_extensor", f"inh_extensor"): {"w": 2*3.64*nS, "p": 0.9},
            
            # Type II excitation pathways
            (f"II_flexor", f"exc_flexor"): {"w": 2*1.65*nS, "p": 0.9},
            (f"II_extensor", f"exc_extensor"): {"w": 2*1.65*nS, "p": 0.9},
            
            # Type II inhibition pathways
            (f"II_flexor", f"inh_flexor"): {"w": 2*2.19*nS, "p": 0.9},
            (f"II_extensor", f"inh_extensor"): {"w": 2*2.19*nS, "p": 0.9},
            
            # Excitatory interneuron to motoneuron pathways
            (f"exc_flexor", f"MN_flexor"): {"w": 2*0.7*nS, "p": 0.6},
            (f"exc_extensor", f"MN_extensor"): {"w": 2*0.7*nS, "p": 0.6},
            
            # Reciprocal inhibition pathways
            (f"inh_flexor", f"MN_extensor"): {"w": 2*0.2*nS, "p": 0.8},
            (f"inh_extensor", f"MN_flexor"): {"w": 2*0.2*nS, "p": 0.8},
            
            # Inhibitory interneuron interactions
            (f"inh_flexor", f"inh_extensor"): {"w": 2*0.76*nS, "p": 0.3},
            (f"inh_extensor", f"inh_flexor"): {"w": 2*0.76*nS, "p": 0.3}
        }
        
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model - need to handle specific muscle names
        self.spindle_model = {}
        self.spindle_model[f"Ia"] = "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6"
        self.spindle_model[f"II"] = "20 + 13.5*stretch"
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
    
    def analyse_unbalanced_recruitment_effects(self, b_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of unbalanced afferent recruitment between antagonistic muscles.
        
        Parameters:
        -----------
        b_range : array-like
            Range of balance values to analyze (0-1 where 0.5 is balanced)
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results
        """
        vary_param = {
            'param_name': 'balance',
            'values': b_range,
            'label': 'Afferent Fiber Unbalanced Recruitment'
        }
        
        return EES_stim_analysis(base_ees_params, vary_param, n_iterations, self.reaction_time, 
                               self.neurons_population, self.connections, self.spindle_model, 
                               self.biophysical_params, self.muscles_names, time_step, seed)
                    
