from brian2 import *
import json
from .BiologicalSystem import BiologicalSystem


class Disynaptic(BiologicalSystem):
    """
    Disynaptic reflex system without Ib pathway (no inhibitory feedback).
    
    This system includes:
    - Ia afferents → Motor neurons (monosynaptic excitation)
    - II afferents → Excitatory interneurons → Motor neurons (disynaptic excitation)
    """
    
    def __init__(self, reaction_time=100*ms, biophysical_params=None, muscles_names=None, 
                 associated_joint="ankle_angle_r", neurons_population=None, connections=None, 
                 spindle_model=None, fast_type_mu=True, 
                 initial_state_neurons=None, initial_condition_spike_activation=None, 
                 initial_state_opensim=None, activation_funct=None, stretch_history_func=None, seed=41):
        """
        Initialize a disynaptic reflex system without Ib pathway.
        """
        # Set default parameters if not provided
        if muscles_names is None:
            muscles_names = ["tib_ant_r"]
            
        if biophysical_params is None:
            biophysical_params = {
                'T_refr': 2 * ms,
                'Eleaky': -70*mV,
                'gL': 10*nS,
                'Cm': 0.3*nF,
                'E_ex': 0*mV,
                'tau_e': 0.5*ms,
                'threshold_v': -50*mV
            }
            
        if ees_recruitment_profile is None:
            with open('data/ees_recruitment.json', 'r') as f:
                ees_recruitment_profile = json.load(f)
            
        if neurons_population is None:
            # Exclude Ib and inhb neurons
            neurons_population = {
                "Ia": 200,       # Type Ia afferent neurons
                "II": 200,       # Type II afferent neurons
                "exc": 400,      # Excitatory interneurons
                "MN": 300        # Motor neurons
            }
        
        if connections is None:
            connections = {
                ("Ia", "MN"): {"w": 2.1*nS, "p": 0.7},      # Direct excitation
                ("II", "exc"): {"w": 1.65*nS, "p": 0.7},    # Stretch feedback
                ("exc", "MN"): {"w": 0.7*nS, "p": 0.5},     # Excitatory interneuron
            }

        if spindle_model is None:
            # Exclude Ib equation
            spindle_model = {
                "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
                "II": "20 + 13.5*stretch_delay",
                "Ia_II_delta_delay": 20*ms
            }
                 
        if initial_state_neurons is None:
            # Exclude inhb neurons from initial states
            initial_state_neurons = {
                "exc": {'v': biophysical_params['Eleaky'], 'gII': 0*nS},
                "MN": {
                    'v': biophysical_params['Eleaky'],
                    'gIa': 0*nS,
                    'gexc': 0*nS
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
            
        super().__init__(reaction_time, biophysical_params, 
                        muscles_names, associated_joint, fast_type_mu,
                        neurons_population, connections, spindle_model, seed,
                        initial_state_neurons, initial_condition_spike_activation, 
                        initial_state_opensim, activation_funct, stretch_history_func)
        
        self.validate_input()

    def validate_input(self):
        """Validates the configuration for disynaptic reflex without Ib pathway."""
        issues = {"warnings": [], "errors": []}

        # Check muscle count
        if self.number_muscles != 1:
            issues["errors"].append("Disynaptic reflex should have exactly 1 muscle")
        if len(self.resting_lengths) != 1:
            issues["errors"].append(
                f"You should specify the resting length for the muscle {self.muscles_names}, "
                f"got an array of size {len(self.resting_lengths)}"
            )

        # Check required neuron types (excluding Ib pathway)
        required_neurons = {"Ia", "II", "exc", "MN"}
        defined_neurons = set(self.neurons_population.keys())
        missing_neurons = required_neurons - defined_neurons
        if missing_neurons:
            issues["errors"].append(f"Missing required neuron types: {missing_neurons}")

        # Warn if Ib pathway components are present
        ib_neurons = {"Ib", "inhb"}
        present_ib_neurons = ib_neurons & defined_neurons
        if present_ib_neurons:
            issues["warnings"].append(
                f"Ib pathway neurons {present_ib_neurons} are present but not expected in this configuration"
            )

        # Check required connections (excluding Ib pathway)
        required_connections = {("Ia", "MN"), ("II", "exc"), ("exc", "MN")}
        defined_connections = set(self.connections.keys())
        missing_connections = required_connections - defined_connections
        if missing_connections:
            issues["errors"].append(f"Missing required connections: {missing_connections}")

        # Warn if Ib pathway connections are present
        ib_connections = {("Ib", "inhb"), ("inhb", "MN")}
        present_ib_connections = ib_connections & defined_connections
        if present_ib_connections:
            issues["warnings"].append(
                f"Ib pathway connections {present_ib_connections} are present but not expected"
            )

        # Check spindle model (excluding Ib)
        required_spindle_equations = ["Ia", "II"]
        for eq in required_spindle_equations:
            if eq not in self.spindle_model:
                issues["errors"].append(f"Missing {eq} equation in spindle model")

        # Warn if Ib equation is present
        if "Ib" in self.spindle_model:
            issues["warnings"].append("Ib equation is present in spindle model but not expected")

        if "Ia_II_delta_delay" in self.spindle_model and "stretch_delay" not in self.spindle_model.get("II", ""):
            issues["errors"].append(
                "You define a delay in the spindle model, but you use the 'stretch' variable. "
                "Use 'stretch_delay' to model delayed II pathway!"
            )

        # Check mandatory biophysical parameters
        required_params = ['T_refr', 'Eleaky', 'gL', 'Cm', 'E_ex', 'tau_e', 'threshold_v']
        for param in required_params:
            if param not in self.biophysical_params:
                issues["errors"].append(f"Missing mandatory biophysical parameter: '{param}'")

        # Unit validation
        expected_units = {
            'T_refr': second, 'Eleaky': volt, 'gL': siemens, 'Cm': farad,
            'E_ex': volt, 'tau_e': second, 'threshold_v': volt
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
            raise ValueError(f"DisynapticWithoutIb configuration errors:\n{error_messages}")

        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: DisynapticWithoutIb configuration issues:\n{warning_messages}")

        return True
