from brian2 import *
import json
from .BiologicalSystem import BiologicalSystem

class BiMuscles(BiologicalSystem):
    """
    Specialized class for reciprocal inhibition reflexes.
    
    Reciprocal inhibition reflexes involve complex connections between two antagonistic
    muscle systems, with both excitatory and inhibitory connections.
    """
    
    def __init__(self, reaction_time=150*ms, biophysical_params=None, muscles_names=None,
             associated_joint="ankle_angle_r", neurons_population=None, connections=None, 
             spindle_model=None, ees_recruitment_profile=None, fast_type_mu=True, 
             initial_state_neurons=None, initial_condition_spike_activation=None, 
             initial_state_opensim=None, activation_funct=None, stretch_history_func=None, seed=42):
        """
        Initialize a reciprocal inhibition system with default or custom parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity, optional
            Reaction time of the system (default: 150ms)
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
        ees_recruitment_profile : dict, optional
            Custom EES recruitment parameters (if None, use defaults)
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["tib_ant_r", "med_gas_r"]
        elif len(muscles_names) != 2:
            raise ValueError("Reciprocal inhibition requires exactly 2 muscles")
            
        if biophysical_params is None:
            biophysical_params = {
                'T_refr': 10 * ms,
                'Eleaky': -70*mV,
                'gL': 10*nS,
                'Cm': 0.3*nF,
                'E_ex': 0*mV,
                'tau_e': 0.5*ms,
                'E_inh': -75*mV,
                'tau_i': 2.8*ms,
                'threshold_v': -50*mV
            }
            
        if ees_recruitment_profile is None:
            with open('data/ees_recruitment.json', 'r') as f:
                ees_recruitment_profile = json.load(f)
                
        if neurons_population is None:
        
            # Setup specialized neuron populations for reciprocal inhibition
            neurons_population = {
                # Afferents for each muscle
                "Ia_flexor": 200, #280,
                "II_flexor": 200,#280,
                "Ia_extensor": 200,#160,
                "II_extensor": 200,#160,
                
                # Interneurons
                "exc_flexor": 400,
                "exc_extensor": 400,
                "inh_flexor": 400,
                "inh_extensor": 400,
                
                # Motor neurons
                "MN_flexor": 300,#450,
                "MN_extensor":300,# 580
            }
        if connections is None:
            connections = {
                # Direct pathways
                ("Ia_flexor", "MN_flexor"): {"w": 2.1*nS, "p": 0.7},
                ("Ia_extensor", "MN_extensor"): {"w": 2.1*nS, "p": 0.7},
                
                # Ia inhibition pathways
                ("Ia_flexor", "inh_flexor"): {"w": 3.64*nS, "p": 0.7},
                ("Ia_extensor", "inh_extensor"): {"w": 3.64*nS, "p": 0.7},
                
                # Type II excitation pathways
                ("II_flexor", "exc_flexor"): {"w": 1.65*nS, "p": 0.7},
                ("II_extensor", "exc_extensor"): {"w": 1.65*nS, "p": 0.7},
                
                # Type II inhibition pathways
                ("II_flexor", "inh_flexor"): {"w": 2.19*nS, "p": 0.7},
                ("II_extensor", "inh_extensor"): {"w": 2.19*nS, "p": 0.7},
                
                # Excitatory interneuron to motoneuron pathways
                ("exc_flexor", "MN_flexor"): {"w": 0.7*nS, "p": 0.5},
                ("exc_extensor", "MN_extensor"): {"w": 0.7*nS, "p": 0.5},
                
                # Reciprocal inhibition pathways
                ("inh_flexor", "MN_extensor"): {"w": 0.2*nS, "p": 0.6},
                ("inh_extensor", "MN_flexor"): {"w": 0.2*nS, "p": 0.6},
                
                # Inhibitory interneuron interactions
                ("inh_flexor", "inh_extensor"): {"w": 0.76*nS, "p": 0.3},
                ("inh_extensor", "inh_flexor"): {"w": 0.76*nS, "p": 0.3}
            }
            
        if spindle_model is None:
            spindle_model = {
                "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
                "II": "20 + 13.5*stretch_delay",
                "Ia_II_delta_delay": 20*ms
            }
            
        if initial_state_neurons is None:
            initial_state_neurons = {
                "inh":{'v': biophysical_params['Eleaky'],
                      'gIa':0*nS,
                      'gII':0*nS,
                      'gi':0*nS,
                      'ginh':0*nS
                      },
                "exc": {'v':biophysical_params['Eleaky'],
                        'gII':0*nS},
                "MN": {'v':biophysical_params['Eleaky'],
                      'gexc':0*nS,
                      'gIa':0*nS,
                      'ginh':0*nS,
                      'gi':0*nS}
            }
            
        if initial_condition_spike_activation is None:  
        # Initialize parameters for each motoneuron
            initial_condition_spike_activation = [
                [{
                    'u0': [0.0, 0.0],    # Initial fiber AP state
                    'c0': [0.0, 0.0],    # Initial calcium concentration state
                    'P0': 0.0,           # Initial calcium-troponin binding state
                    'a0': 0.0            # Initial activation state
                } for _ in range(neurons_population['MN_flexor' if i == 0 else 'MN_extensor'])] 
                for i in range(2)  # two muscles
            ]

        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, 
                        muscles_names, associated_joint, fast_type_mu,
                        neurons_population, connections, spindle_model,seed, 
                        initial_state_neurons, initial_condition_spike_activation, 
                        initial_state_opensim, activation_funct, stretch_history_func)  

        # Validate the configuration
        self.validate_input()

    def validate_input(self):
        """
        Validates the configuration parameters for reciprocal inhibition reflex system.
        
        Raises:
            ValueError: If critical errors are found in the configuration
        """
        issues = {"warnings": [], "errors": []}
                
        # Check muscle count (should be exactly 2 for reciprocal inhibition)
        if self.number_muscles != 2:
            issues["errors"].append("Reciprocal inhibition reflex should have exactly 2 muscles")
        if len(self.resting_lengths)!=2:
            issues["errors"].append("Your should specify the resting length for the muscles {self.muscles_names[0]} and {self.muscles_names[1]}, incorrect input")
        # Check required neuron types for reciprocal inhibition
        required_neurons = {
            "Ia_flexor", "II_flexor", "Ia_extensor", "II_extensor",
            "exc_flexor", "exc_extensor", "inh_flexor", "inh_extensor",
            "MN_flexor", "MN_extensor"
        }
        defined_neurons = set(self.neurons_population.keys())
        
        missing_neurons = required_neurons - defined_neurons
        if missing_neurons:
            issues["errors"].append(f"Missing required neuron types for reciprocal inhibition: {missing_neurons}")
        
        # Check for unexpected neuron types
        unexpected_neurons = defined_neurons - required_neurons
        if unexpected_neurons:
            issues["warnings"].append(f"Unexpected neuron types for reciprocal inhibition: {unexpected_neurons}")
        
        # Check required connections for reciprocal inhibition
        required_connections = {
            # Direct pathways
            ("Ia_flexor", "MN_flexor"),
            ("Ia_extensor", "MN_extensor"),
            
            # Ia inhibition pathways
            ("Ia_flexor", "inh_flexor"),
            ("Ia_extensor", "inh_extensor"),
            
            # Type II excitation pathways
            ("II_flexor", "exc_flexor"),
            ("II_extensor", "exc_extensor"),
            
            # Type II inhibition pathways
            ("II_flexor", "inh_flexor"),
            ("II_extensor", "inh_extensor"),
            
            # Excitatory interneuron to motoneuron pathways
            ("exc_flexor", "MN_flexor"),
            ("exc_extensor", "MN_extensor"),
            
            # Reciprocal inhibition pathways (the key feature)
            ("inh_flexor", "MN_extensor"),
            ("inh_extensor", "MN_flexor")
        }
        
        defined_connections = set(self.connections.keys())
        
        missing_connections = required_connections - defined_connections
        if missing_connections:
            issues["errors"].append(f"Missing required connections for reciprocal inhibition: {missing_connections}")
        
        # Check for the presence of key reciprocal inhibition connections
        reciprocal_connections = {("inh_flexor", "MN_extensor"), ("inh_extensor", "MN_flexor")}
        if not reciprocal_connections.issubset(defined_connections):
            missing_reciprocal = reciprocal_connections - defined_connections
            issues["errors"].append(f"Missing critical reciprocal inhibition connections: {missing_reciprocal}")
        
        # Check spindle model
        required_spindle_equations = ["Ia", "II"]
        for eq in required_spindle_equations:
            if eq not in self.spindle_model:
                issues["errors"].append(f"Missing {eq} equation in spindle model for reciprocal inhibition")
        if "Ia_II_delta_delay" in self.spindle_model and not "stretch_delay" in self.spindle_model.get("II"):
            issues["errors"].append("You define a delay in the spindle model, but you use the 'stretch' variable. Use 'stretch_delay', to model delayed II pathway, otherwise, don't specify a delay! ")
            
        
        # Check mandatory biophysical parameters (including inhibitory ones)
        required_params = ['T_refr', 'Eleaky', 'gL', 'Cm', 'E_ex', 'tau_e', 'E_inh', 'tau_i', 'threshold_v']
        for param in required_params:
            if param not in self.biophysical_params:
                issues["errors"].append(f"Missing mandatory biophysical parameter: '{param}'")
        
        # Check units
        expected_units = {
            'T_refr': second,
            'Eleaky': volt,
            'gL': siemens,  
            'Cm': farad,
            'E_ex': volt,
            'tau_e': second,
            'E_inh': volt,
            'tau_i': second,
            'threshold_v': volt
        }
        
        for param, expected_unit in expected_units.items():
            if param in self.biophysical_params:
                value = self.biophysical_params[param]
                if not value.dim == expected_unit.dim:
                    issues["errors"].append(
                        f"Parameter '{param}' has incorrect unit. "
                        f"Expected unit compatible with {expected_unit}, but got {value.unit}"
                    )
        
        
        # Check connection weights and probabilities
        for connection, params in self.connections.items():
            if "w" not in params:
                issues["errors"].append(f"Missing weight 'w' for connection {connection}")
            if "p" not in params:
                issues["errors"].append(f"Missing probability 'p' for connection {connection}")
            
            if "p" in params:
                prob = params["p"]
                if not (0 <= prob <= 1):
                    issues["errors"].append(f"Connection probability for {connection} must be between 0 and 1, got {prob}")
            
            if "w" in params:
                weight = params["w"]
                # Check if weight has proper siemens units
                if hasattr(weight, 'dim') and not weight.dim == siemens.dim:
                    issues["errors"].append(f"Connection weight for {connection} should have siemens units, got {weight.unit}")
   
        

        # Raise error if there are critical issues
        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"Reciprocal inhibition configuration errors:\n{error_messages}")
        
        # Print warnings if any
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: Reciprocal inhibition configuration issues:\n{warning_messages}")
            
        return True
        
        
