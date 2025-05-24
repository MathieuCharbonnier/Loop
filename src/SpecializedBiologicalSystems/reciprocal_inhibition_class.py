from brian2 import *

class ReciprocalInhibition(BiologicalSystem):
    """
    Specialized class for reciprocal inhibition reflexes.
    
    Reciprocal inhibition reflexes involve complex connections between two antagonistic
    muscle systems, with both excitatory and inhibitory connections.
    """
    
    def __init__(self, reaction_time=50*ms, biophysical_params=None, muscles_names=None, 
                 associated_joint="ankle_angle_r", custom_neurons=None, custom_connections=None, 
                 custom_spindle=None, ees_recruitment_profile=None, fast_type_mu=True,
                 custom_initial_potentials=None, custom_initial_condition_spike_activation=None, 
                 initial_state_opensim=None):
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
        self.validate_reciprocal_inhibition_parameters()

    def validate_reciprocal_inhibition_parameters(self):
        """
        Validates the configuration parameters specifically for the ReciprocalInhibition class.
        
        This method extends the base validation with specific checks for reciprocal inhibition
        circuits and their required pathways.
        
        Raises:
            ValueError: If critical errors are found in the reciprocal inhibition configuration
        """
        issues = {"warnings": [], "errors": []}
        
        # Get neuron types from population
        defined_neurons = set(self.neurons_population.keys())
        neuron_types = {n.split('_')[0] if '_' in n else n for n in defined_neurons}
        
        # Specific validation for reciprocal inhibition requirements
        self._validate_reciprocal_inhibition_requirements(issues, defined_neurons, neuron_types)
        self._validate_reciprocal_connections(issues)
        self._validate_reciprocal_spindle_model(issues, neuron_types)
        self._validate_reciprocal_ees_recruitment(issues, neuron_types)
        self._validate_circuit_symmetry(issues, defined_neurons)
        
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
            raise ValueError(f"ReciprocalInhibition configuration errors found:\n{error_messages}")
        
        # Print warnings if any
        if issues["warnings"]:
            warning_messages = "\n".join(issues["warnings"])
            print(f"WARNING: ReciprocalInhibition configuration issues detected:\n{warning_messages}")
            
        return True
    
    def _validate_reciprocal_inhibition_requirements(self, issues, defined_neurons, neuron_types):
        """Validate core reciprocal inhibition circuit requirements."""
        
        # Essential neuron types for reciprocal inhibition
        essential_types = {"Ia", "inh", "MN"}
        recommended_types = {"Ia", "II", "exc", "inh", "MN"}
        
        missing_essential = essential_types - neuron_types
        if missing_essential:
            issues["errors"].append(
                f"ReciprocalInhibition requires essential neuron types: {essential_types}. "
                f"Missing: {missing_essential}"
            )
        
        missing_recommended = recommended_types - neuron_types
        if missing_recommended:
            issues["warnings"].append(
                f"For complete reciprocal inhibition, recommended neuron types are: {recommended_types}. "
                f"Missing: {missing_recommended}"
            )
        
        # Check for muscle-specific neurons (flexor/extensor pairs)
        required_muscle_neurons = [
            ("Ia_flexor", "Ia_extensor"),
            ("inh_flexor", "inh_extensor"),
            ("MN_flexor", "MN_extensor")
        ]
        
        for flexor, extensor in required_muscle_neurons:
            if flexor not in defined_neurons:
                issues["errors"].append(f"Missing required flexor neuron: {flexor}")
            if extensor not in defined_neurons:
                issues["errors"].append(f"Missing required extensor neuron: {extensor}")
        
        # Check if both excitatory and inhibitory interneurons are present
        if "exc" in neuron_types and "inh" not in neuron_types:
            issues["warnings"].append("Excitatory interneurons defined without inhibitory interneurons")
        if "inh" in neuron_types and "exc" not in neuron_types:
            issues["warnings"].append("Inhibitory interneurons defined without excitatory interneurons")
    
    def _validate_reciprocal_connections(self, issues):
        """Validate reciprocal inhibition specific connections."""
        
        connection_pairs = list(self.connections.keys())
        
        # Essential reciprocal inhibition pathways
        essential_reciprocal_connections = [
            # Direct monosynaptic pathways
            ("Ia_flexor", "MN_flexor"),
            ("Ia_extensor", "MN_extensor"),
            
            # Reciprocal inhibition pathways (cross-inhibition)
            ("inh_flexor", "MN_extensor"),
            ("inh_extensor", "MN_flexor"),
            
            # Ia to inhibitory interneuron pathways
            ("Ia_flexor", "inh_flexor"),
            ("Ia_extensor", "inh_extensor")
        ]
        
        missing_connections = []
        for connection in essential_reciprocal_connections:
            if connection not in connection_pairs:
                missing_connections.append(connection)
        
        if missing_connections:
            issues["errors"].append(f"Missing essential reciprocal inhibition connections: {missing_connections}")
        
        # Check for proper reciprocal inhibition pattern
        reciprocal_pairs = [
            (("inh_flexor", "MN_extensor"), ("inh_extensor", "MN_flexor")),
            (("Ia_flexor", "inh_flexor"), ("Ia_extensor", "inh_extensor"))
        ]
        
        for pair1, pair2 in reciprocal_pairs:
            has_pair1 = pair1 in connection_pairs
            has_pair2 = pair2 in connection_pairs
            
            if has_pair1 and not has_pair2:
                issues["warnings"].append(f"Asymmetric reciprocal connection: {pair1} present but {pair2} missing")
            elif has_pair2 and not has_pair1:
                issues["warnings"].append(f"Asymmetric reciprocal connection: {pair2} present but {pair1} missing")
        
        # Validate Type II connections if present
        if "II_flexor" in [conn[0] for conn in connection_pairs] or "II_extensor" in [conn[0] for conn in connection_pairs]:
            recommended_ii_connections = [
                ("II_flexor", "exc_flexor"),
                ("II_extensor", "exc_extensor"),
                ("II_flexor", "inh_flexor"),
                ("II_extensor", "inh_extensor")
            ]
            
            missing_ii = [conn for conn in recommended_ii_connections if conn not in connection_pairs]
            if missing_ii:
                issues["warnings"].append(f"Type II neurons present but missing typical connections: {missing_ii}")
    
    def _validate_reciprocal_spindle_model(self, issues, neuron_types):
        """Validate spindle model for reciprocal inhibition."""
        
        required_spindle_types = []
        if "Ia" in neuron_types:
            required_spindle_types.append("Ia")
        if "II" in neuron_types:
            required_spindle_types.append("II")
        
        for spindle_type in required_spindle_types:
            if spindle_type not in self.spindle_model:
                issues["errors"].append(f"Missing spindle model equation for {spindle_type} neurons")
        
        # Check for proper stretch dependency in Ia equation
        if "Ia" in self.spindle_model:
            ia_equation = self.spindle_model["Ia"]
            if "stretch" not in ia_equation.lower():
                issues["warnings"].append("Ia spindle equation should typically depend on stretch")
            if "velocity" not in ia_equation.lower():
                issues["warnings"].append("Ia spindle equation should typically include velocity dependency")
        
        # Check for proper stretch dependency in II equation
        if "II" in self.spindle_model:
            ii_equation = self.spindle_model["II"]
            if "stretch" not in ii_equation.lower():
                issues["warnings"].append("Type II spindle equation should typically depend on stretch")
        
        # Check for delay parameter
        if "II" in neuron_types and "Ia" in neuron_types:
            if "II_Ia_delta_delay" not in self.spindle_model:
                issues["warnings"].append("Consider adding II_Ia_delta_delay for realistic timing differences")
    
    def _validate_reciprocal_ees_recruitment(self, issues, neuron_types):
        """Validate EES recruitment parameters for reciprocal inhibition."""
        
        if not self.ees_recruitment_profile:
            return
        
        # Check recruitment parameters for all neuron types
        for neuron_type in ["Ia", "II", "MN"]:
            if neuron_type in neuron_types and neuron_type not in self.ees_recruitment_profile:
                issues["errors"].append(f"Missing EES recruitment parameters for {neuron_type}")
        
        # Validate recruitment thresholds make biological sense
        if "Ia" in self.ees_recruitment_profile and "MN" in self.ees_recruitment_profile:
            ia_threshold = self.ees_recruitment_profile["Ia"]["threshold_10pct"]
            mn_threshold = self.ees_recruitment_profile["MN"]["threshold_10pct"]
            
            if ia_threshold >= mn_threshold:
                issues["warnings"].append(
                    "Ia afferents typically have lower EES thresholds than motoneurons"
                )
        
        if "II" in self.ees_recruitment_profile and "Ia" in self.ees_recruitment_profile:
            ii_threshold = self.ees_recruitment_profile["II"]["threshold_10pct"]
            ia_threshold = self.ees_recruitment_profile["Ia"]["threshold_10pct"]
            
            if ii_threshold <= ia_threshold:
                issues["warnings"].append(
                    "Type II afferents typically have higher EES thresholds than Ia afferents"
                )
    
    def _validate_circuit_symmetry(self, issues, defined_neurons):
        """Validate symmetry between flexor and extensor circuits."""
        
        # Check for balanced neuron populations
        neuron_pairs = [
            ("Ia_flexor", "Ia_extensor"),
            ("II_flexor", "II_extensor"),
            ("exc_flexor", "exc_extensor"),
            ("inh_flexor", "inh_extensor"),
            ("MN_flexor", "MN_extensor")
        ]
        
        for flexor, extensor in neuron_pairs:
            if flexor in defined_neurons and extensor in defined_neurons:
                flexor_count = self.neurons_population[flexor]
                extensor_count = self.neurons_population[extensor]
                
                # Allow some asymmetry (up to 2:1 ratio) as it's biologically realistic
                ratio = max(flexor_count, extensor_count) / min(flexor_count, extensor_count)
                if ratio > 2.5:  # Warning for high asymmetry
                    issues["warnings"].append(
                        f"High asymmetry in neuron populations: {flexor}={flexor_count}, "
                        f"{extensor}={extensor_count} (ratio: {ratio:.1f}:1)"
                    )
        
        # Check for connection weight symmetry
        connection_pairs = [
            (("Ia_flexor", "MN_flexor"), ("Ia_extensor", "MN_extensor")),
            (("inh_flexor", "MN_extensor"), ("inh_extensor", "MN_flexor"))
        ]
        
        for conn1, conn2 in connection_pairs:
            if conn1 in self.connections and conn2 in self.connections:
                weight1 = self.connections[conn1].get("w", 0)
                weight2 = self.connections[conn2].get("w", 0)
                
                # Compare weights if they have the same units
                if hasattr(weight1, 'magnitude') and hasattr(weight2, 'magnitude'):
                    if abs(weight1.magnitude - weight2.magnitude) / max(weight1.magnitude, weight2.magnitude) > 0.5:
                        issues["warnings"].append(
                            f"Asymmetric connection weights: {conn1}={weight1}, {conn2}={weight2}"
                        )