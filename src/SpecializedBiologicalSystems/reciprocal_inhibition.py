from brian2 import *
from .BiologicalSystem import BiologicalSystem

class ReciprocalInhibition(BiologicalSystem):
    """
    Specialized class for reciprocal inhibition reflexes.
    
    Reciprocal inhibition reflexes involve complex connections between two antagonistic
    muscle systems, with both excitatory and inhibitory connections.
    """
    
    def __init__(self, reaction_time=50*ms, biophysical_params=None, muscles_names=None, 
                 associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                 custom_spindle=None, ees_recruitment_profile=None, fast_type_mu=True):
        """
        Initialize a reciprocal inhibition system with default or custom parameters.
        
        Parameters:
        -----------
        reaction_time : brian2.units.fundamentalunits.Quantity, optional
            Reaction time of the system (default: 50ms)
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
            
        if ees_recruitment_profile is None:
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
            
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, 
                        associated_joint, fast_type_mu, initial_state_opensim)
        
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
        self.spindle_model = {
            "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
            "II": "20 + 13.5*stretch",
            "II_Ia_delta_delay": 15*ms
        }
        
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)

        self.initial_potentials = {
            "inh": self.biophysical_params['Eleaky'],
            "exc": self.biophysical_params['Eleaky'],
            "MN": self.biophysical_params['Eleaky']
        }
        
            
        # Initialize parameters for each motoneuron
        self.initial_condition_spike_activation = [
            [{
                'u0': [0.0, 0.0],    # Initial fiber AP state
                'c0': [0.0, 0.0],    # Initial calcium concentration state
                'P0': 0.0,           # Initial calcium-troponin binding state
                'a0': 0.0            # Initial activation state
            } for _ in range(self.neurons_population['MN_flexor' if i == 0 else 'MN_extensor'])] 
            for i in range(2)  # two muscles
        ]
                    
        # Validate the configuration
        self.validate_reciprocal_inhibition_parameters()

    def validate_reciprocal_inhibition_parameters(self):
        """
        Validates the configuration parameters for reciprocal inhibition reflex system.
        
        Raises:
            ValueError: If critical errors are found in the configuration
        """
        issues = {"warnings": [], "errors": []}
        
        # Check muscle count (should be exactly 2 for reciprocal inhibition)
        if self.number_muscles != 2:
            issues["errors"].append("Reciprocal inhibition reflex should have exactly 2 muscles")
        
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
        
        # Check EES recruitment parameters
        for neuron_type in ["Ia", "II", "MN"]:
            if neuron_type not in self.ees_recruitment_profile:
                issues["errors"].append(f"Missing EES recruitment parameters for neuron type '{neuron_type}'")
        
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
        
        # Validate EES parameters
        for neuron_type, params in self.ees_recruitment_profile.items():
            if neuron_type in ["Ia", "II", "MN"]:
                required_ees_params = ["threshold_10pct", "saturation_90pct"]
                for param in required_ees_params:
                    if param not in params:
                        issues["errors"].append(f"Missing '{param}' in EES recruitment parameters for '{neuron_type}'")
                
                if "threshold_10pct" in params and "saturation_90pct" in params:
                    threshold = params['threshold_10pct']
                    saturation = params['saturation_90pct']
                    
                    if not (0 <= threshold <= 1) or not (0 <= saturation <= 1):
                        issues["errors"].append(
                            f"EES parameters for '{neuron_type}' must be between 0 and 1. "
                            f"Got: threshold={threshold}, saturation={saturation}"
                        )
                    if threshold >= saturation:
                        issues["errors"].append(f"Threshold must be less than saturation for '{neuron_type}'")
        
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
        
        
