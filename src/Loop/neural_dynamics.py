from brian2 import *
import json
from .BiologicalSystem import BiologicalSystem


class DisynapticIb(BiologicalSystem):
    """
    Disynaptic reflex system with Ib pathway (inhibitory feedback).
    
    This system includes:
    - Ia afferents → Motor neurons (monosynaptic excitation)
    - II afferents → Excitatory interneurons → Motor neurons (disynaptic excitation)
    - Ib afferents → Inhibitory interneurons → Motor neurons (disynaptic inhibition)
    """
    
    def __init__(self, reaction_time=50*ms, biophysical_params=None, muscles_names=None, 
                 associated_joint="ankle_angle_r", neurons_population=None, connections=None, 
                 spindle_model=None, ees_recruitment_profile=None, fast_type_mu=True, 
                 initial_state_neurons=None, initial_condition_spike_activation=None, 
                 initial_state_opensim=None, activation_funct=None, stretch_history_func=None, seed=41):
        """
        Initialize a disynaptic reflex system with Ib pathway.
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["soleus_r"]
            
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
            # Include all neuron types including Ib pathway
            neurons_population = {
                "Ia": 400,       # Type Ia afferent neurons
                "II": 400,       # Type II afferent neurons
                "Ib": 400,       # Type Ib afferent neurons (force feedback)
                "exc": 500,      # Excitatory interneurons
                "inhb": 500,     # Inhibitory interneurons
                "MN": 500        # Motor neurons
            }
        
        if connections is None:
            # Include all connections including Ib pathway
            connections = {
                ("Ia", "MN"): {"w": 2.1*nS, "p": 0.7},      # Direct excitation
                ("II", "exc"): {"w": 1.65*nS, "p": 0.7},    # Stretch feedback
                ("exc", "MN"): {"w": 0.7*nS, "p": 0.5},     # Excitatory interneuron
                ("Ib", "inhb"): {"w": 1.65*nS, "p": 0.5},   # Force feedback
                ("inhb", "MN"): {"w": 0.2*nS, "p": 0.5},     # Inhibitory interneuron
                ("Ia", "inhb"): {"w": 1.65*nS, "p": 0.5}
            }

        if spindle_model is None:
            # Include Ib equation for force feedback
             spindle_model = {
                "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
                "II": "20 + 13.5*stretch_delay",
                "Ib": " 57*force_normalized**0.2",
                "Ia_II_delta_delay": 20*ms
            }
                 
        if initial_state_neurons is None:
            # Include all neuron types in initial states
            initial_state_neurons = {
                "exc": {'v': biophysical_params['Eleaky'], 'gII': 0*nS},
                "inhb": {'v': biophysical_params['Eleaky'], 'gIb': 0*nS, 'gIa':0*nS},
                "MN": {
                    'v': biophysical_params['Eleaky'],
                    'gIa': 0*nS,
                    'gexc': 0*nS,
                    'ginhb': 0*nS,
                    'gi': 0*nS
                }
            }
            
        if initial_condition_spike_activation is None:   
            initial_condition_spike_activation = [
                [{
                    'u0': [0.0, 0.0],
                    'c0': [0.0, 0.0],
                    'P0': 0.0,
                    'a0': 0.0
                } for _ in range(neurons_population['MN'])]
            ]
            
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, 
                        muscles_names, associated_joint, fast_type_mu,
                        neurons_population, connections, spindle_model, seed,
                        initial_state_neurons, initial_condition_spike_activation, 
                        initial_state_opensim, activation_funct, stretch_history_func)
        
        self.validate_input()

    def validate_input(self):
        """Validates the configuration for disynaptic reflex with Ib pathway."""
        issues = {"warnings": [], "errors": []}

        # Check muscle count
        if self.number_muscles != 1:
            issues["errors"].append("Disynaptic reflex should have exactly 1 muscle")
        if len(self.resting_lengths) != 1:
            issues["errors"].append(
                f"You should specify the resting length for the muscle {self.muscles_names}, "
                f"got an array of size {len(self.resting_lengths)}"
            )

        # Check required neuron types (including Ib pathway)
        required_neurons = {"Ia", "II", "Ib", "exc", "inhb", "MN"}
        defined_neurons = set(self.neurons_population.keys())
        missing_neurons = required_neurons - defined_neurons
        if missing_neurons:
            issues["errors"].append(f"Missing required neuron types: {missing_neurons}")

        # Check required connections (including Ib pathway)
        required_connections = {
            ("Ia", "MN"), ("II", "exc"), ("exc", "MN"),
            ("Ib", "inhb"), ("inhb", "MN"), ("Ia", "inhb")
        }
        defined_connections = set(self.connections.keys())
        missing_connections = required_connections - defined_connections
        if missing_connections:
            issues["errors"].append(f"Missing required connections: {missing_connections}")

        # Check spindle model (including Ib)
        required_spindle_equations = ["Ia", "II", "Ib"]
        for eq in required_spindle_equations:
            if eq not in self.spindle_model:
                issues["errors"].append(f"Missing {eq} equation in spindle model")

        if "Ia_II_delta_delay" in self.spindle_model and "stretch_delay" not in self.spindle_model.get("II", ""):
            issues["errors"].append(
                "You define a delay in the spindle model, but you use the 'stretch' variable. "
                "Use 'stretch_delay' to model delayed II pathway!"
            )

        # Check mandatory biophysical parameters
        required_params = ['T_refr', 'Eleaky', 'gL', 'Cm', 'E_ex', 'tau_e', 'E_inh', 'tau_i', 'threshold_v']
        for param in required_params:
            if param not in self.biophysical_params:
                issues["errors"].append(f"Missing mandatory biophysical parameter: '{param}'")

        # Unit validation
        expected_units = {
            'T_refr': second, 'Eleaky': volt, 'gL': siemens, 'Cm': farad,
            'E_ex': volt, 'tau_e': second, 'E_inh': volt, 'tau_i': second, 'threshold_v': volt
        }

        for param, expected_unit in expected_units.items():
            if param in self.biophysical_params:
                value = self.biophysical_params[param]
                if not value.dim == expected_unit.dim:
                    issues["errors"].append(
                        f"Parameter '{param}' has incorrect unit. "
                        f"Expected unit compatible with {expected_unit}, but got {value.unit}"
                    )

        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"DisynapticWithIb configuration errors:\n{error_messages}")

        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: DisynapticWithIb configuration issues:\n{warning_messages}")

        return True


