from brian2 import *
from .helpers.checker import validate_parameters
from .BiologicalSystem import BiologicalSystem
from .Visualization.plot_parameters_variations import plot_ees_analysis_results

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

        validate_parameters(self.neurons_population, self.connections, self.spindle_models, 
        self.biophysical_parameters, self.muscles_names, self.number_muscles,self.ees_recruitment_profile)
    
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

        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="balance_analysis", seed=seed)

