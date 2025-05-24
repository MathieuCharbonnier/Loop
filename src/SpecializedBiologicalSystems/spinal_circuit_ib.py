from brian2 import *

class SpinalCircuitWithIb(BiologicalSystem):
    """
    Specialized class that integrates Ib fibers in the realistic biological neural network 
    between two antagonistic muscle systems. We only consider known di-synaptic pathways.
    """
    
    def __init__(self, reaction_time=50*ms, biophysical_params=None, muscles_names=None, 
                 associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                 custom_spindle=None, custom_ees_recruitment_profile=None, fast_type_mu=True,
                 custom_initial_potentials=None, custom_initial_condition_spike_activation=None,
                 initial_state_opensim=None):
        """
        Initialize a reciprocal inhibition system with Ib fibers with default or custom parameters.
        
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
        custom_ees_recruitment_profile : dict, optional
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
                'MN': {
                    'threshold_10pct': 0.7,  # Motoneurons are recruited at high intensity
                    'saturation_90pct': 0.9  
                }  
            }
        else:
            ees_recruitment_profile = custom_ees_recruitment_profile
                
        # Initialize the base class
        super().__init__(reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, 
                        associated_joint, fast_type_mu, initial_state_opensim)
        
        # Setup specialized neuron populations for reciprocal inhibition with Ib fibers
        self.neurons_population = {
            # Afferents for each muscle
            "Ia_flexor": 280,
            "II_flexor": 280,
            "Ib_flexor": 280,
            "Ia_extensor": 160,
            "II_extensor": 160,
            "Ib_extensor": 160,
       
            # Interneurons
            "exc_flexor": 500,
            "exc_extensor": 500,
            "inh_flexor": 500,
            "inh_extensor": 500,
            "inhb_flexor": 300,  # Ib interneurons
            "inhb_extensor": 300,
            
            # Motor neurons
            "MN_flexor": 450,
            "MN_extensor": 580
        }
        
        # Override with custom values if provided
        if custom_neurons is not None:
            self.neurons_population.update(custom_neurons)
            
        # Set default connections with reciprocal inhibition pattern including Ib pathways
        self.connections = {
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
                                
            # inhb interneuron to motoneuron pathways (Ib inhibition)
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
        self.spindle_model = {
            "Ia": "10+ 2*stretch + 4.3*sign(stretch_velocity)*abs(stretch_velocity)**0.6",
            "II": "20 + 13.5*stretch",
            "Ib": "10 + 1*force_normalized**0.2",
            "II_Ia_delta_delay": 15*ms
        }
                    
        # Override with custom spindle model if provided
        if custom_spindle is not None:
            self.spindle_model.update(custom_spindle)
        
        self.initial_potentials = {
            "inh": self.biophysical_params['Eleaky'],
            "inhb": self.biophysical_params['Eleaky'],
            "exc": self.biophysical_params['Eleaky'],
            "MN": self.biophysical_params['Eleaky']
        }
        
        if custom_initial_potentials is not None:
            self.initial_potentials.update(custom_initial_potentials)
            
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
                    
        if custom_initial_condition_spike_activation is not None:   
            self.initial_condition_spike_activation = custom_initial_condition_spike_activation
        
        # Validate the configuration
        self.validate_ib_circuit_parameters()

    def validate_ib_circuit_parameters(self):
        """
        Validates the configuration parameters specifically for the SpinalCircuitWithIb class.
        
        This method extends the base validation with specific checks for Ib fiber integration
        and di-synaptic pathways.
        
        Raises:
            ValueError: If critical errors are found in the Ib circuit configuration
        """
        issues = {"warnings": [], "errors": []}
        
        # Get neuron types from population
        defined_neurons = set(self.neurons_population.keys())
        neuron_types = {n.split('_')[0] if '_' in n else n for n in defined_neurons}
        
        # Specific validation for Ib circuit requirements
        self._validate_ib_specific_requirements(issues, defined_neurons, neuron_types)
        self._validate_ib_connections(issues)
        self._validate_ib_spindle_model(issues, neuron_types)
        self._validate_ib_ees_recruitment(issues, neuron_types)
        
        # Call the base validation function
        try:
            from your_module import validate_parameters  # Import the original function
            validate_parameters(
                self.neurons_population, 
                self.connections, 
                self.spindle_model, 
                self.biophysical_params, 
                self.muscles_names, 
                len(self.muscles_names),
                self.ees_recruitment_profile
            )
        except ValueError as e:
            issues["errors"].append(f"Base validation failed: {str(e)}")
        
        # Raise error if there are critical issues
        if issues["errors"]:
            error_messages = "\n".join(issues["errors"])
            raise ValueError(f"SpinalCircuitWithIb configuration errors found:\n{error_messages}")
        
        # Print warnings if any
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: SpinalCircuitWithIb configuration issues detected:\n{warning_messages}")
            
        return True
    
    def _validate_ib_specific_requirements(self, issues, defined_neurons, neuron_types):
        """Validate Ib-specific circuit requirements."""
        
        # Check for mandatory Ib fiber presence
        if "Ib" not in neuron_types:
            issues["errors"].append("SpinalCircuitWithIb requires Ib neurons to be defined")
        
        # Check for required Ib interneurons (inhb)
        if "inhb" not in neuron_types:
            issues["errors"].append("SpinalCircuitWithIb requires inhb (Ib interneurons) to be defined")
        
        # For two muscles, check muscle-specific Ib neurons
        if len(self.muscles_names) == 2:
            required_ib_neurons = ["Ib_flexor", "Ib_extensor", "inhb_flexor", "inhb_extensor"]
            missing_ib = [neuron for neuron in required_ib_neurons if neuron not in defined_neurons]
            
            if missing_ib:
                issues["errors"].append(f"Missing required Ib neurons for two-muscle system: {missing_ib}")
        
        # Check for balanced populations between flexor and extensor
        if "Ib_flexor" in defined_neurons and "Ib_extensor" in defined_neurons:
            flexor_count = self.neurons_population["Ib_flexor"]
            extensor_count = self.neurons_population["Ib_extensor"]
            
            ratio = max(flexor_count, extensor_count) / min(flexor_count, extensor_count)
            if ratio > 2.0:  # Allow up to 2:1 ratio
                issues["warnings"].append(
                    f"Unbalanced Ib populations: flexor={flexor_count}, extensor={extensor_count}"
                )
    
    def _validate_ib_connections(self, issues):
        """Validate Ib-specific connections."""
        
        connection_pairs = list(self.connections.keys())
        
        # Check for essential Ib pathways
        required_ib_connections = [
            ("Ib_flexor", "inhb_flexor"),
            ("Ib_extensor", "inhb_extensor"),
            ("inhb_flexor", "MN_flexor"),
            ("inhb_extensor", "MN_extensor")
        ]
        
        for connection in required_ib_connections:
            if connection not in connection_pairs:
                issues["errors"].append(f"Missing required Ib connection: {connection}")
        
        # Check for proper inhibitory nature of Ib connections to MN
        ib_to_mn_connections = [conn for conn in connection_pairs 
                               if conn[0].startswith('inhb') and conn[1].startswith('MN')]
        
        for connection in ib_to_mn_connections:
            # These should be inhibitory connections (typically checked by connection strength)
            weight = self.connections[connection].get('w', 0)
            if not hasattr(weight, 'unit'):
                issues["warnings"].append(f"Ib-MN connection {connection} weight should have units")
    
    def _validate_ib_spindle_model(self, issues, neuron_types):
        """Validate Ib-specific spindle model requirements."""
        
        if "Ib" in neuron_types:
            if "Ib" not in self.spindle_model:
                issues["errors"].append("Ib neurons defined but no Ib equation found in spindle model")
            else:
                # Check if Ib equation includes force dependency
                ib_equation = self.spindle_model["Ib"]
                if "force" not in ib_equation.lower():
                    issues["warnings"].append(
                        "Ib spindle equation should typically depend on force/tension"
                    )
    
    def _validate_ib_ees_recruitment(self, issues, neuron_types):
        """Validate Ib-specific EES recruitment parameters."""
        
        if self.ees_recruitment_profile and "Ib" in neuron_types:
            if "Ib" not in self.ees_recruitment_profile:
                issues["errors"].append("Missing EES recruitment parameters for Ib neurons")
            else:
                ib_params = self.ees_recruitment_profile["Ib"]
                
                # Ib fibers typically have different recruitment characteristics
                if "threshold_10pct" in ib_params and "saturation_90pct" in ib_params:
                    threshold = ib_params["threshold_10pct"]
                    saturation = ib_params["saturation_90pct"]
                    
                    # Ib fibers typically have higher thresholds than Ia
                    if "Ia" in self.ees_recruitment_profile:
                        ia_threshold = self.ees_recruitment_profile["Ia"]["threshold_10pct"]
                        if threshold < ia_threshold:
                            issues["warnings"].append(
                                "Ib threshold is typically higher than Ia threshold for EES recruitment"
                            )