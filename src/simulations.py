
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from closed_loop import closed_loop
from analysis import delay_excitability_MU_type_analysis, EES_stim_analysis
from controller import HierarchicalAnkleController
from plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves

class BiologicalSystem:
    """
    Base class for neural reflex systems.
    
    This class provides the common framework for different types of reflex systems,
    handling the core simulation and analysis functionality.
    """
    
    def __init__(self, reaction_time, ees_recruitment_profile, biophysical_params, muscles_names, associated_joint):
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
        
        # These will be set by subclasses
        self.neurons_population = {}
        self.connections = {}
        self.spindle_model = {}
    
    def validate_parameters(self):
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
    
    def run_simulation(self, base_output_path, n_iterations, time_step=0.1*ms, ees_stimulation_params=None,
                   torque_profile=None, fast_type_mu=True, seed=42, save=True):
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
        spikes, time_series = closed_loop(
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
             torque=torque_profile,ees_recruitment_profile=self.ees_recruitment_profile,ees_stimulation_params=ees_stimulation_params,
             fast=fast_type_mu, seed=seed, base_output_path=base_output_path)
        
        # Generate standard plots
        if ees_stimulation_params is not None:
            plot_recruitment_curves(self.ees_recruitment_profile, current_current=ees_stimulation_params.get('intensity'),
            base_output_path=base_output_path, balance=ees_stimulation_params.get('balance', 0), num_muscles=self.number_muscles)
            
        plot_mouvement(time_series, self.muscles_names, self.associated_joint, base_output_path)
        plot_neural_dynamic(time_series, self.muscles_names, base_output_path)
        plot_raster(spikes, base_output_path)
        plot_activation(time_series, self.muscles_names, base_output_path)
        
        return spikes, time_series

    def analyze_frequency_effects(self, freq_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying EES frequency with fixed afferent and efferent recruitment.
        
        Parameters:
        -----------
        freq_range : array-like
            Range of frequencies to analyze
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
            'param_name': 'ees_freq',
            'values': freq_range,
            'label': 'EES Frequency'
        }
        
        return EES_stim_analysis(
            base_ees_params,
            vary_param,
            n_iterations,
            self.reaction_time, 
            self.neurons_population, 
            self.connections,
            self.spindle_model, 
            self.biophysical_params, 
            self.muscles_names,
            self.number_muscles,
            self.associated_joint,
            self.ees_recruitment_profile,
            time_step, 
            seed
        )

    def analyze_intensity_effects(self, intensity_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
        """
        Analyze the effects of varying afferent recruitment.
        
        Parameters:
        -----------
        intensity_range : array-like
            Range of normalized intensity values to analyze
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
            'param_name': 'intensity',
            'values': intensity_range,
            'label': 'Increase stimulation amplitude'
        }
        
        return EES_stim_analysis(
            base_ees_params,
            vary_param, 
            n_iterations,
            self.reaction_time, 
            self.neurons_population, 
            self.connections,
            self.spindle_model, 
            self.biophysical_params, 
            self.muscles_names,
            self.number_muscles,
            self.associated_joint,
            self.ees_recruitment_profile,
            time_step, 
            seed
        )
      
    def clonus_analysis(self, delay_values=[10, 25, 50, 75, 100]*ms, 
                        threshold_values=[-45, -50, -55]*mV, duration=1*second, 
                        time_step=0.1*ms, fast_type_mu=True, torque_profile=None, 
                        ees_stimulation_params=None, seed=41):
        """
        Analyze clonus behavior by varying one parameter at a time.
        
        Parameters:
        -----------
        delay_values : list
            List of delay values to test
        threshold_values : list
            List of threshold values to test
        duration : brian2.units.fundamentalunits.Quantity
            Duration of each simulation
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        fast_type_mu : bool
            If True, use fast twitch motor units
        torque_profile : dict
            Dictionary with torque profile parameters
        ees_stimulation_params : dict
            Parameters for EES stimulation
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Analysis results
        """
       
        return delay_excitability_MU_type_analysis(
            delay_values,
            threshold_values,
            duration,
            time_step, 
            self.reaction_time, 
            self.neurons_population, 
            self.connections, 
            self.spindle_model, 
            self.biophysical_params, 
            self.muscles_names,
            self.number_muscles, 
            self.associated_joint,
            torque_profile, 
            ees_stimulation_params,
            self.ees_recruitment_profile,
            fast_type_mu,
            seed
        )

    def find_ees_protocol(self, target_amplitude=15, target_period=2*second, 
                        update_interval=200*ms, prediction_horizon=1000*ms, 
                        simulation_time=10*second):
        """
        Find an optimal EES protocol for a given target movement.
        
        Parameters:
        -----------
        target_amplitude : float
            Target amplitude of joint movement
        target_period : brian2.units.fundamentalunits.Quantity
            Target period of oscillation
        update_interval : brian2.units.fundamentalunits.Quantity
            Update interval for controller
        prediction_horizon : brian2.units.fundamentalunits.Quantity
            Prediction horizon for controller
        simulation_time : brian2.units.fundamentalunits.Quantity
            Total simulation time
            
        Returns:
        --------
        dict
            Optimal EES protocol parameters
        """
        # Initial state
        initial_state = {
            'joint_angle': 0.0,
            'joint_velocity': 0.0,
            'flexor_activation': 0.0,
            'extensor_activation': 0.0
        }
        
        # Create controller
        controller = HierarchicalAnkleController(
            target_amplitude=target_amplitude,
            target_period=target_period, 
            update_interval=update_interval,   
            prediction_horizon=prediction_horizon  
        )
        
        # Initialize and run simulation
        controller.initialize_simulation(initial_state)
        controller.run_simulation(simulation_time)
        
        # Plot results
        controller.plot_results()
        
        return controller.get_optimal_protocol()

    def sensitivity_analysis(self, base_output_path, n_iterations=20, n_samples=50, time_step=0.1*ms, 
                      base_ees_params=None, torque_profile=None, method='morris', seed=42):
        """
        Perform sensitivity analysis on the biological system, focusing on joint angle dynamics.
        
        Parameters:
        -----------
        base_output_path : str
            Base path for saving output files
        n_iterations : int
            Number of iterations for each simulation
        n_samples : int
            Number of parameter samples to generate
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for the simulation
        base_ees_params : dict, optional
            Base parameters for epidural electrical stimulation
        torque_profile : dict, optional
            External torque applied to the joint
        method : str
            Sensitivity analysis method ('morris', 'sobol', or 'fast')
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Sensitivity analysis results containing sensitivity indices and feature importances
        """

        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
        from SALib.sample import saltelli
        from SALib.analyze import sobol
        from SALib.sample import fast_sampler
        from SALib.analyze import fast
        from copy import deepcopy
        import pickle
        from tqdm import tqdm
        
        # Define parameter ranges for sensitivity analysis
        # These would need to be adjusted based on your specific model parameters
        problem = {
            'num_vars': 0,  # Will be updated based on parameters
            'names': [],    # Will be populated with parameter names
            'bounds': []    # Will be populated with parameter bounds
        }
        
        # Add biophysical parameters to the problem
        for param, value in self.biophysical_params.items():
            # Skip parameters that shouldn't be varied
            if param in ['gL', 'Cm']:  # These are typically fixed physical constraints
                continue
                
            problem['names'].append(f'biophysical_{param}')
            
            # Set bounds based on parameter type (with appropriate units)
            if param == 'Eleaky':  # Resting membrane potential
                problem['bounds'].append([-80*mV, -60*mV])
            elif param == 'T_refr':    # Leak conductance
                problem['bounds'].append([0*ms, 20*ms])
            elif param == 'E_ex':  # Excitatory reversal potential
                problem['bounds'].append([0*mV, 20*mV])
            elif param == 'E_inh': # Inhibitory reversal potential
                problem['bounds'].append([-80*mV, -60*mV])
            elif param == 'tau_e': # Excitatory time constant
                problem['bounds'].append([3*ms, 7*ms])
            elif param == 'tau_i': # Inhibitory time constant
                problem['bounds'].append([5*ms, 15*ms])
            elif param == 'threshold_v': # Threshold voltage
                problem['bounds'].append([-55*mV, -45*mV])
            else:
                # For other parameters, use ±30% around nominal value
                lower_bound = float(value) * 0.7
                upper_bound = float(value) * 1.3
                problem['bounds'].append([lower_bound, upper_bound])

        
        # Add connection strengths
        for connection_pair, weight in self.connections.items():
            param_name = f'conn_{connection_pair[0]}_{connection_pair[1]}'
            problem['names'].append(param_name)
            # Connection weights typically vary by ±50%
            if hasattr(weight, 'value'):
                lower_bound = float(weight) * 0.5
                upper_bound = float(weight) * 1.5
            else:
                lower_bound = weight * 0.5
                upper_bound = weight * 1.5
            problem['bounds'].append([lower_bound, upper_bound])
        
        # Update number of variables
        problem['num_vars'] = len(problem['names'])
        
        print(f"Performing sensitivity analysis with {problem['num_vars']} parameters")
        
        # Create output directory for sensitivity analysis
        sa_output_path = os.path.join(base_output_path, 'sensitivity_analysis')
        os.makedirs(sa_output_path, exist_ok=True)
        
        # Generate parameter samples based on the selected method
        if method == 'morris':
            param_values = morris_sample.sample(problem, N=n_samples, num_levels=4, optimal_trajectories=None)
        elif method == 'sobol':
            param_values = saltelli.sample(problem, n_samples)
        elif method == 'fast':
            param_values = fast_sampler.sample(problem, n_samples)
        else:
            raise ValueError(f"Unknown sensitivity analysis method: {method}")
        
        # Extract joint name from associated_joint
        joint_name = self.associated_joint
        
        # Initialize array to store features extracted from joint angle time series
        features = {
            'max_angle': [],           # Maximum joint angle
            'min_angle': [],           # Minimum joint angle
            'range_of_motion': [],     # Range of motion (max - min)
            'mean_angle': [],          # Mean joint angle
            'std_angle': [],           # Standard deviation of angle
            'time_to_max': [],         # Time to reach maximum angle
            'time_to_steady': [],      # Time to reach steady state
            'steady_state_angle': [],  # Steady state angle
            'oscillation_freq': [],    # Frequency of oscillation if any
            'oscillation_amp': []      # Amplitude of oscillation if any
        }
        
        # Function to extract features from joint angle time series
        def extract_angle_features(angle_series, time_vector):
            import numpy as np
            from scipy import signal
            
            features_dict = {}
            
            # Basic statistics
            features_dict['max_angle'] = np.max(angle_series)
            features_dict['min_angle'] = np.min(angle_series)
            features_dict['range_of_motion'] = features_dict['max_angle'] - features_dict['min_angle']
            features_dict['mean_angle'] = np.mean(angle_series)
            features_dict['std_angle'] = np.std(angle_series)
            
            # Time to max angle
            max_idx = np.argmax(angle_series)
            features_dict['time_to_max'] = time_vector[max_idx]
            
            # Steady state analysis
            # Use the last 20% of the signal as steady state
            steady_start_idx = int(0.8 * len(angle_series))
            steady_state = angle_series[steady_start_idx:]
            features_dict['steady_state_angle'] = np.mean(steady_state)
            
            # Time to reach steady state (within 5% of final value)
            final_value = features_dict['steady_state_angle']
            steady_threshold = 0.05 * features_dict['range_of_motion']
            for i, val in enumerate(angle_series):
                if abs(val - final_value) <= steady_threshold:
                    features_dict['time_to_steady'] = time_vector[i]
                    break
            else:
                features_dict['time_to_steady'] = time_vector[-1]  # Never reached steady state
            
            # Frequency analysis (for oscillations)
            if len(angle_series) > 10:  # Need sufficient data points
                # Detrend the signal
                detrended = signal.detrend(angle_series)
                
                # Compute power spectral density
                freqs, psd = signal.welch(detrended, fs=1/float(time_vector[1]-time_vector[0]), 
                                         nperseg=min(256, len(detrended)//2))
                
                # Find dominant frequency excluding DC component
                if len(freqs) > 1:
                    peak_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                    features_dict['oscillation_freq'] = freqs[peak_idx]
                    
                    # Estimate oscillation amplitude using Fourier transform
                    fft_vals = np.abs(np.fft.rfft(detrended))
                    features_dict['oscillation_amp'] = np.max(fft_vals[1:]) * 2 / len(detrended)
                else:
                    features_dict['oscillation_freq'] = 0
                    features_dict['oscillation_amp'] = 0
            else:
                features_dict['oscillation_freq'] = 0
                features_dict['oscillation_amp'] = 0
                
            return features_dict
        
        # Save problem definition
        with open(os.path.join(sa_output_path, 'problem_definition.pkl'), 'wb') as f:
            pickle.dump(problem, f)
        
        # Run simulations for each parameter combination
        print(f"Running {len(param_values)} simulations...")
        
        # Store failed simulations
        failed_sims = []
        
        # Create a directory to store time series data
        time_series_dir = os.path.join(sa_output_path, 'time_series')
        os.makedirs(time_series_dir, exist_ok=True)
        
        for i, params in enumerate(tqdm(param_values)):
            # Update parameters for this simulation
            modified_system = deepcopy(self)
            
            # Apply parameter values
            param_idx = 0
            
            # Update biophysical parameters
            for orig_param in list(self.biophysical_params.keys()):
                if f'biophysical_{orig_param}' in problem['names']:
                    idx = problem['names'].index(f'biophysical_{orig_param}')
                    # Need to preserve units
                    if hasattr(self.biophysical_params[orig_param], 'unit'):
                        unit = self.biophysical_params[orig_param].unit
                        modified_system.biophysical_params[orig_param] = params[idx] * unit
                    else:
                        modified_system.biophysical_params[orig_param] = params[idx]
   
            
            # Update connection strengths
            for conn_pair in self.connections:
                param_name = f'conn_{conn_pair[0]}_{conn_pair[1]}'
                if param_name in problem['names']:
                    idx = problem['names'].index(param_name)
                    if hasattr(self.connections[conn_pair], 'unit'):
                        unit = self.connections[conn_pair].unit
                        modified_system.connections[conn_pair] = params[idx] * unit
                    else:
                        modified_system.connections[conn_pair] = params[idx]
            
            try:
                # Run simulation with modified parameters
                sim_output_path = os.path.join(sa_output_path, f'sim_{i}')
                
                # Run with minimal output (no plotting)
                spikes, time_series = modified_system.run_simulation(
                    sim_output_path, 
                    n_iterations, 
                    time_step=time_step, 
                    ees_stimulation_params=base_ees_params,
                    torque_profile=torque_profile,
                    seed=seed
                )
                
                # Extract joint angle time series
                joint_col = f'joint_{joint_name}'
                if joint_col in time_series.columns:
                    joint_angle = time_series[joint_col].values
                    time_vector = time_series['time'].values
                    
                    # Extract features from joint angle
                    angle_features = extract_angle_features(joint_angle, time_vector)
                    
                    # Store features
                    for feature_name, value in angle_features.items():
                        features[feature_name].append(value)
                    
                    # Save time series for this simulation
                    time_series[[joint_col, 'time']].to_csv(os.path.join(time_series_dir, f'joint_angle_{i}.csv'))
                else:
                    print(f"Warning: Joint column '{joint_col}' not found in time series")
                    failed_sims.append(i)
                    # Add NaN values for this simulation
                    for feature_name in features:
                        features[feature_name].append(np.nan)
                        
            except Exception as e:
                print(f"Simulation {i} failed: {str(e)}")
                failed_sims.append(i)
                # Add NaN values for this simulation
                for feature_name in features:
                    features[feature_name].append(np.nan)
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(features)
        
        # Save features
        features_df.to_csv(os.path.join(sa_output_path, 'angle_features.csv'))
        
        # Analyze sensitivity using selected method
        sensitivity_results = {}
        
        # Remove rows with NaN values
        valid_rows = ~features_df.isna().any(axis=1)
        
        if sum(valid_rows) < 10:
            print(f"Warning: Only {sum(valid_rows)} valid simulations out of {len(param_values)}")
            return {
                'error': 'Too few valid simulations for sensitivity analysis',
                'features': features_df
            }
        
        for feature in features_df.columns:
            feature_values = features_df[feature].values[valid_rows]
            valid_params = param_values[valid_rows]
            
            # Perform sensitivity analysis using selected method
            try:
                if method == 'morris':
                    Si = morris_analyze.analyze(
                        problem, 
                        valid_params, 
                        feature_values, 
                        print_to_console=False
                    )
                    sensitivity_results[feature] = {
                        'mu': Si['mu'],
                        'mu_star': Si['mu_star'],
                        'sigma': Si['sigma'],
                        'mu_star_conf': Si['mu_star_conf'],
                        'parameter_names': problem['names']
                    }
                elif method == 'sobol':
                    Si = sobol.analyze(
                        problem, 
                        feature_values, 
                        print_to_console=False
                    )
                    sensitivity_results[feature] = {
                        'S1': Si['S1'],
                        'S1_conf': Si['S1_conf'],
                        'ST': Si['ST'],
                        'ST_conf': Si['ST_conf'],
                        'parameter_names': problem['names']
                    }
                elif method == 'fast':
                    Si = fast.analyze(
                        problem, 
                        feature_values, 
                        print_to_console=False
                    )
                    sensitivity_results[feature] = {
                        'S1': Si['S1'],
                        'S1_conf': Si['S1_conf'],
                        'parameter_names': problem['names']
                    }
            except Exception as e:
                print(f"Sensitivity analysis failed for feature {feature}: {str(e)}")
                sensitivity_results[feature] = {'error': str(e)}
        
        # Save sensitivity results
        with open(os.path.join(sa_output_path, 'sensitivity_results.pkl'), 'wb') as f:
            pickle.dump(sensitivity_results, f)
        
        # Create plots for each feature
        plot_dir = os.path.join(sa_output_path, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        for feature, results in sensitivity_results.items():
            if 'error' in results:
                continue
                
            plt.figure(figsize=(12, 8))
            
            if method == 'morris':
                # Morris method: plot mu* (mean absolute elementary effects)
                y_pos = np.arange(len(problem['names']))
                plt.barh(y_pos, results['mu_star'], xerr=results['mu_star_conf'], align='center')
                plt.yticks(y_pos, problem['names'])
                plt.xlabel('μ* (Mean Absolute Elementary Effects)')
            elif method == 'sobol':
                # Sobol method: plot total effects
                y_pos = np.arange(len(problem['names']))
                plt.barh(y_pos, results['ST'], xerr=results['ST_conf'], align='center')
                plt.yticks(y_pos, problem['names'])
                plt.xlabel('Total Effects Sensitivity Index')
            elif method == 'fast':
                # FAST method: plot first-order effects
                y_pos = np.arange(len(problem['names']))
                plt.barh(y_pos, results['S1'], xerr=results['S1_conf'], align='center')
                plt.yticks(y_pos, problem['names'])
                plt.xlabel('First-Order Sensitivity Index')
            
            plt.title(f'Sensitivity Analysis for {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'sensitivity_{feature}.png'))
            plt.close()
        
        # Create summary plot with top 5 parameters for each feature
        plt.figure(figsize=(15, 10))
        n_features = len(sensitivity_results)
        cols = 2
        rows = (n_features + 1) // 2
        
        feature_idx = 0
        for feature, results in sensitivity_results.items():
            if 'error' in results:
                continue
                
            plt.subplot(rows, cols, feature_idx + 1)
            
            if method == 'morris':
                # Sort parameters by mu*
                sorted_indices = np.argsort(results['mu_star'])[-5:]  # Top 5 parameters
                param_names = [problem['names'][i] for i in sorted_indices]
                sensitivity = results['mu_star'][sorted_indices]
                
                plt.barh(np.arange(len(sorted_indices)), sensitivity, align='center')
                plt.yticks(np.arange(len(sorted_indices)), param_names)
                plt.xlabel('μ*')
            elif method == 'sobol':
                # Sort parameters by total effects
                sorted_indices = np.argsort(results['ST'])[-5:]  # Top 5 parameters
                param_names = [problem['names'][i] for i in sorted_indices]
                sensitivity = results['ST'][sorted_indices]
                
                plt.barh(np.arange(len(sorted_indices)), sensitivity, align='center')
                plt.yticks(np.arange(len(sorted_indices)), param_names)
                plt.xlabel('Total Effects')
            elif method == 'fast':
                # Sort parameters by first-order effects
                sorted_indices = np.argsort(results['S1'])[-5:]  # Top 5 parameters
                param_names = [problem['names'][i] for i in sorted_indices]
                sensitivity = results['S1'][sorted_indices]
                
                plt.barh(np.arange(len(sorted_indices)), sensitivity, align='center')
                plt.yticks(np.arange(len(sorted_indices)), param_names)
                plt.xlabel('First-Order Effects')
            
            plt.title(f'Top Parameters for {feature}')
            feature_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'sensitivity_summary.png'))
        plt.close()
        
        # Create time series visualization for a few representative samples
        # Select a few samples with diverse parameter values
        if len(features_df) > 0:
            plt.figure(figsize=(12, 8))
            
            # Sort simulations based on range of motion
            if 'range_of_motion' in features_df.columns:
                sorted_indices = features_df['range_of_motion'].sort_values().index
                # Take a few simulations from different parts of the distribution
                sample_indices = [
                    sorted_indices[0],  # Min range
                    sorted_indices[len(sorted_indices)//4],  # 25th percentile
                    sorted_indices[len(sorted_indices)//2],  # Median
                    sorted_indices[3*len(sorted_indices)//4],  # 75th percentile
                    sorted_indices[-1]  # Max range
                ]
                
                for idx in sample_indices:
                    if idx < len(param_values):
                        try:
                            ts_df = pd.read_csv(os.path.join(time_series_dir, f'joint_angle_{idx}.csv'))
                            joint_col = f'joint_{joint_name}'
                            if joint_col in ts_df.columns:
                                plt.plot(ts_df['time'], ts_df[joint_col], label=f'Sim {idx}')
                        except Exception as e:
                            print(f"Could not plot time series for simulation {idx}: {str(e)}")
                
                plt.xlabel('Time (s)')
                plt.ylabel(f'{joint_name} Joint Angle')
                plt.title('Representative Joint Angle Time Series')
                plt.legend()
                plt.savefig(os.path.join(plot_dir, 'representative_time_series.png'))
                plt.close()
        
        # Return results
        return {
            'sensitivity_results': sensitivity_results,
            'features': features_df,
            'problem_definition': problem,
            'output_path': sa_output_path
        }
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
            ("Ia", "MN"): {"w": 2*2.1*nS, "p": 0.9}
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
        self.validate_parameters()


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
        self.validate_parameters()


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
        
        return EES_stim_analysis(base_ees_params, vary_param, n_iterations, self.reaction_time, 
                               self.neurons_population, self.connections, self.spindle_model, 
                               self.biophysical_params, self.muscles_names, time_step, seed)
                    
