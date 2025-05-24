
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os

from .helpers.checker import validate_parameters
from .Loop.closed_loop import closed_loop
from .Visualization.plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves
from .Stimulation.input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array


class BiologicalSystem:
    """
    Base class for neural reflex systems.
    
    This class provides the common framework for different types of reflex systems,
    handling the core simulation and analysis functionality.
    """
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, associated_joint, fast_type_mu):
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
        """
        self.reaction_time = reaction_time
        self.ees_recruitment_profile = ees_recruitment_profile
        self.biophysical_params = biophysical_params
        self.muscles_names = muscles_names
        self.number_muscles = len(muscles_names)
        self.associated_joint = associated_joint
        self.fast_type_mu=fast_type_mu
        
        # These will be set by subclasses
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
    
  
    
def run_simulation(self, n_iterations, time_step=0.1*ms, 
                  ees_stimulation_params=None,torque_profile=None, 
                  seed=42,base_output_path=None, plot=True):
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
        torque_profile : dict, optional
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
        
        torque_array = None
        if torque_profile is not None:
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            torque_array = transform_torque_params_in_array(time_points, torque_profile)
        
        ees_params = None
        if ees_stimulation_params is not None:
            ees_params = transform_intensity_balance_in_recruitment(
            self.ees_recruitment_profile, ees_stimulation_params, 
            self.neurons_population, self.num_muscles)
        
        spikes, time_series = closed_loop(
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
             torque_array=torque_array,ees_params=ees_params,
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
            
        if custom_ees_recruitment_profile is None:
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
            ("Ia", "MN"): {"w": 2.1*nS, "p": 0.7}
        }
        
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model
        self.spindle_model = {
            "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6"
        }
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
            
        # Validate parameters
        validate_parameters(self.neurons_population, self.connections, self.spindle_model, 
        self.biophysical_params, self.muscles_names, self.number_muscles,self.ees_recruitment_profile )


class Disynaptic(BiologicalSystem):
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
            "II": "20 + 13.5*stretch",
            "II_Ia_delta_delay": 15*ms
        }
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
            
        # Validate parameters
        validate_parameters(self.neurons_population, self.connections, self.spindle_models, 
        self.biophysical_parameters, self.muscles_names, self.number_muscles,self.ees_recruitment_profile)



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
            "Ia_flexor": 280,
            "II_flexor": 280,
            "Ia_extensor": 160,
            "II_extensor": 160,
            
            # Interneurons
            "exc_flexor": 500,
            "exc_extensor": 500,
            "inh_flexor": 500,
            "inh_extensor": 500,
            
            # Motor neurons
            "MN_flexor": 450,
            "MN_extensor": 580
        }
        
        # Override with custom values if provided
        if custom_neurons is not None:
            self.neurons_population.update(custom_neurons)
            
        # Set default connections with reciprocal inhibition pattern
        self.connections = {
            # Direct pathways
            ("Ia_flexor", "MN_flexor"): {"w": 2*2.1*nS, "p": 0.9},
            ("Ia_extensor", "MN_extensor"): {"w": 2*2.1*nS, "p": 0.9},
            
            # Ia inhibition pathways
            ("Ia_flexor", "inh_flexor"): {"w": 2*3.64*nS, "p": 0.9},
            ("Ia_extensor", "inh_extensor"): {"w": 2*3.64*nS, "p": 0.9},
            
            # Type II excitation pathways
            ("II_flexor", "exc_flexor"): {"w": 2*1.65*nS, "p": 0.9},
            ("II_extensor", "exc_extensor"): {"w": 2*1.65*nS, "p": 0.9},
            
            # Type II inhibition pathways
            ("II_flexor", "inh_flexor"): {"w": 2*2.19*nS, "p": 0.9},
            ("II_extensor", "inh_extensor"): {"w": 2*2.19*nS, "p": 0.9},
            
            # Excitatory interneuron to motoneuron pathways
            ("exc_flexor", "MN_flexor"): {"w": 2*0.7*nS, "p": 0.6},
            ("exc_extensor", "MN_extensor"): {"w": 2*0.7*nS, "p": 0.6},
            
            # Reciprocal inhibition pathways
            ("inh_flexor", "MN_extensor"): {"w": 2*0.2*nS, "p": 0.8},
            ("inh_extensor", "MN_flexor"): {"w": 2*0.2*nS, "p": 0.8},
            
            # Inhibitory interneuron interactions
            ("inh_flexor", "inh_extensor"): {"w": 2*0.76*nS, "p": 0.3},
            ("inh_extensor", "inh_flexor"): {"w": 2*0.76*nS, "p": 0.3}
        }
        
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model - need to handle specific muscle names
        self.spindle_model = {}
        self.spindle_model["Ia"] = "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6"
        self.spindle_model["II"] = "20 + 13.5*stretch"
        self.spindle_model["II_Ia_delta_delay"]= 15*ms
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)

        validate_parameters(self.neurons_population, self.connections, self.spindle_models, 
        self.biophysical_parameters, self.muscles_names, self.number_muscles,self.ees_recruitment_profile)
    
 

class SpinalCircuitWithIb(BiologicalSystem):
    """
    Specialized class that integrate Ib fibers in the realistic biological neural network between two ntagonistic muscles systems.
    We only consider known di-synaptic pathways.
    
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
                'Ib': {
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
            "Ia_flexor": 280,
            "II_flexor": 280,
            "Ib_flexor": 280,
            "Ia_extensor": 160,
            "II_extensor": 160,
            "Ib_extensor": 160,
       
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
                            
        self.connections =  {
            # Direct pathways
            ("Ia_flexor", "MN_flexor"): {"w": 2*2.1*nS, "p": 0.9},
            ("Ia_extensor", "MN_extensor"): {"w": 2*2.1*nS, "p": 0.9},
                                           
            # Ia inhibition pathways
            ("Ia_flexor", "inh_flexor"): {"w": 2*3.64*nS, "p": 0.9},
            ("Ia_extensor", "inh_extensor"): {"w": 2*3.64*nS, "p": 0.9},
            ("Ia_flexor", "inhb_flexor"): {"w": 3.0*nS, "p": 0.6},  # e14
            ("Ia_extensor", "inhb_extensor"): {"w": 3.0*nS, "p": 0.6},  # e14
            
            # Type II excitation pathways
            ("II_flexor", "exc_flexor"): {"w": 2*1.65*nS, "p": 0.9},
            ("II_extensor", "exc_extensor"): {"w": 2*1.65*nS, "p": 0.9},
            
            # Type II inhibition pathways
            ("II_flexor", "inh_flexor"): {"w": 2*2.19*nS, "p": 0.9},
            ("II_extensor", "inh_extensor"): {"w": 2*2.19*nS, "p": 0.9},

            # Type Ib pathways e21 → IN i3 → MN (negative, within population)
            ("Ib_flexor", "inhb_flexor"): {"w": 3.5*nS, "p": 0.6},  # e21
            ("Ib_extensor", "inhb_extensor"): {"w": 3.5*nS, "p": 0.6},  # e21
                                
            # Excitatory interneuron to motoneuron pathways
            ("exc_flexor", "MN_flexor"): {"w": 2*0.7*nS, "p": 0.6},
            ("exc_extensor", "MN_extensor"): {"w": 2*0.7*nS, "p": 0.6},
                                
            # inhb interneuron to motoneuron pathways
            ("inhb_flexor", "MN_flexor"): {"w": 3.0*nS, "p": 0.6},  # i3 (inhibitory)
            ("inhb_extensor", "MN_extensor"): {"w": 3.0*nS, "p": 0.6},  # i3 (inhibitory)
            
            # Reciprocal inhibition pathways
            ("inh_flexor", "MN_extensor"): {"w": 2*0.2*nS, "p": 0.8},
            ("inh_extensor", "MN_flexor"): {"w": 2*0.2*nS, "p": 0.8},
            
            # Inhibitory interneuron interactions
            ("inh_flexor", "inh_extensor"): {"w": 2*0.76*nS, "p": 0.3},
            ("inh_extensor", "inh_flexor"): {"w": 2*0.76*nS, "p": 0.3}
        } 
                
        # Override with custom connections if provided
        if custom_connections is not None:
            self.connections.update(custom_connections)
            
        # Set default spindle model - need to handle specific muscle names
        self.spindle_model = {}
        self.spindle_model["Ia"] = "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6"
        self.spindle_model["II"] = "20 + 13.5*stretch"
        self.spindle_model["Ib"] = "10 + 1*force_normalized**0.2"
        self.spindle_model["II_Ia_delta_delay"]= 15*ms
                    
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)

        validate_parameters(self.neurons_population, self.connections, self.spindle_models, 
        self.biophysical_parameters, self.muscles_names, self.number_muscles,self.ees_recruitment_profile)
    
    
                    
