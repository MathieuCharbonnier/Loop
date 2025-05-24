from brian2 import *
from .base_system import BiologicalSystem


class Disynaptic(BiologicalSystem):
    """
    Specialized class for disynaptic reflexes.
    
    Disynaptic reflexes involve connections from afferent neurons (Ia and II)
    to motor neurons (MN) through excitatory interneurons.
    """
    
    def __init__(self, reaction_time=40*ms, biophysical_params=None, muscles_names=None, 
                associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                custom_spindle=None, ees_recruitment_profile=None, fast_type_mu=False,
                custom_initial_potentials=None, custom_initial_condition_spike_activation=None, 
                initial_state_opensim=None):
        """
        Initialize a disynaptic reflex system with default or custom parameters.
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
            
        if ees_recruitment_profile is None:
            ees_recruitment_profile = {
                'Ia': {
                    'threshold_10pct': 0.3,
                    'saturation_90pct': 0.7
                },
                'II': {
                    'threshold_10pct': 0.4,
                    'saturation_90pct': 0.8
                },
                'MN': {
                    'threshold_10pct': 0.7,
                    'saturation_90pct': 0.9  
                }
            }
            
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, 
                        muscles_names, associated_joint, fast_type_mu, initial_state_opensim)
        
        # Set default neuron populations
        self.neurons_population = {
            "Ia": 280,       # Type Ia afferent neurons
            "II": 280,       # Type II afferent neurons
            "exc": 500,      # Excitatory interneurons
            "MN": 450        # Motor neurons
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

        self.initial_potentials = {
            "exc": self.biophysical_params['Eleaky'],
            "MN": self.biophysical_params['Eleaky']
        }
        if custom_initial_potentials is not None:
            self.initial_potentials = custom_initial_potentials
            
        # Initialize parameters for each motoneuron
        self.initial_condition_spike_activation = [
            [{
                'u0': [0.0, 0.0],    # Initial fiber AP state
                'c0': [0.0, 0.0],    # Initial calcium concentration state
                'P0': 0.0,           # Initial calcium-troponin binding state
                'a0': 0.0            # Initial activation state
            } for _ in range(self.neurons_population['MN'])]
        ]
        if custom_initial_condition_spike_activation is not None:   
            self.initial_condition_spike_activation = custom_initial_condition_spike_activation

        # Validate parameters
        self.validate_input()

    def validate_input(self):
        """
        Validates the configuration parameters for disynaptic reflex system.
        
        Raises:
            ValueError: If critical errors are found in the configuration
        """
        issues = {"warnings": [], "errors": []}
        
        # Check muscle count (should be 1 for disynaptic)
        if self.number_muscles != 1:
            issues["errors"].append("Disynaptic reflex should have exactly 1 muscle")
        
        # Check required neuron types
        required_neurons = {"Ia", "II", "exc", "MN"}
        defined_neurons = set(self.neurons_population.keys())
        
        missing_neurons = required_neurons - defined_neurons
        if missing_neurons:
            issues["errors"].append(f"Missing required neuron types for disynaptic reflex: {missing_neurons}")
        
        # Check for unexpected neuron types (especially inhibitory ones)
        unexpected_neurons = defined_neurons - required_neurons
        if unexpected_neurons:
            issues["warnings"].append(f"Unexpected neuron types for disynaptic reflex: {unexpected_neurons}")
        
        # Check connections
        required_connections = {("Ia", "MN"), ("II", "exc"), ("exc", "MN")}
        defined_connections = set(self.connections.keys())
        
        missing_connections = required_connections - defined_connections
        if missing_connections:
            issues["errors"].append(f"Missing required connections for disynaptic reflex: {missing_connections}")
        
        # Check spindle model
        required_spindle_equations = ["Ia", "II"]
        for eq in required_spindle_equations:
            if eq not in self.spindle_model:
                issues["errors"].append(f"Missing {eq} equation in spindle model for disynaptic reflex")
        
        # Check EES recruitment parameters
        for neuron_type in ["Ia", "II", "MN"]:
            if neuron_type not in self.ees_recruitment_profile:
                issues["errors"].append(f"Missing EES recruitment parameters for neuron type '{neuron_type}'")
        
        # Check biophysical parameters (no inhibitory parameters should be present)
        if "E_inh" in self.biophysical_params or "tau_i" in self.biophysical_params:
            issues["warnings"].append("Inhibitory parameters present but no inhibitory neurons in disynaptic reflex")
        
        # Check mandatory biophysical parameters
        required_params = ['T_refr', 'Eleaky', 'gL', 'Cm', 'E_ex', 'tau_e', 'threshold_v']
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
        
        # Raise error if there are critical issues
        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"Disynaptic configuration errors:\n{error_messages}")
        
        # Print warnings if any
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: Disynaptic configuration issues:\n{warning_messages}")
            
        return True