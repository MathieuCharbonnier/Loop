 def validate_parameters(neurons_population, connections, spindle_models, biophysical_parameters):
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
        if self.ees_recruitment_profile:
            # Check if all required neuron types have recruitment parameters
            for neuron_type in neuron_types:
                if neuron_type in ["Ia", "II", "MN"] and neuron_type not in self.ees_recruitment_profile:
                    issues["errors"].append(f"Missing EES recruitment parameters for neuron type '{neuron_type}'")

            
            # Check each recruitment parameter set
            for neuron_type, params in self.ees_recruitment_profile.items():
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
