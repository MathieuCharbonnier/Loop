import copy
import numpy as np
from tqdm import tqdm
from brian2 import *
from .Visualization.plot_parameters_variations import plot_delay_results, plot_excitability_results, plot_twitch_results, plot_ees_analysis_results
from .helpers.sensitivity_analysis import sensitivity_analysis_func

class Analyzer:
  
    def __init__(self, system):
        self.original_system = system
    
          
    def analyze_frequency_effects(self, freq_range, base_ees_params, n_iterations=20, time_step=0.1, seed=42):
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
        time_step : float
            Time step for simulations (in ms, will be converted to brian2 units)
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        vary_param = {
            'param_name': 'ees_freq',
            'values': freq_range,
            'label': 'EES Frequency (Hz)'
        }

        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param,
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="frequency_analysis", seed=seed)
        return results
        

    def analyze_intensity_effects(self, intensity_range, base_ees_params, n_iterations=20, time_step=0.1, seed=42):
        """
        Analyze the effects of varying stimulation intensity.
        
        Parameters:
        -----------
        intensity_range : array-like
            Range of normalized intensity values to analyze
        base_ees_params : dict
            Base parameters for EES
        n_iterations : int
            Number of iterations for each simulation
        time_step : float
            Time step for simulations (in ms, will be converted to brian2 units)
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Analysis results containing simulation data and computed metrics
        """
        vary_param = {
            'param_name': 'intensity',
            'values': intensity_range,
            'label': 'Stimulation Intensity'
        }
        
        # Compute parameter sweep
        results = self._compute_ees_parameter_sweep(
            base_ees_params,
            vary_param, 
            n_iterations,
            time_step, 
            seed
        )
        
        plot_ees_analysis_results(results, save_dir="intensity_analysis", seed=seed)
        return results
        

    def _compute_ees_parameter_sweep(self, param_dict, vary_param, n_iterations, time_step=0.1, seed=42):
        """
        Compute EES stimulation analysis by varying a parameter of interest.
        
        Parameters:
        -----------
        param_dict : dict
            Dictionary containing all EES parameters with their default values
        vary_param : dict
            Dictionary specifying which parameter to vary with its range of values
            Format: {'param_name': [values_to_test], 'label': 'Display Label'}
        n_iterations : int
            Number of iterations for each simulation
        time_step : float
            Time step for simulation (in ms)
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'param_values': list of parameter values tested
            - 'param_name': name of the varied parameter
            - 'param_label': display label for the parameter
            - 'time_data': time array from simulations
            - 'simulation_data': list of main_data dictionaries for each parameter value
            - 'spikes_data': list of spikes dictionaries for each parameter value
            - 'activities': numpy array of muscle activities (if num_muscles == 2)
            - 'muscle_names': list of muscle names
            - 'associated_joint': joint name
        """
        
        # Get parameter info
        param_name = vary_param['param_name']
        param_values = vary_param['values']
        param_label = vary_param['label']
        
        # Initialize storage for results
        simulation_data = []
        spikes_data = []
        time_data = None
        activities = None
        
        print(f"Running parameter sweep for {param_label}...")
        
        # Run simulations for each parameter value
        for i, value in enumerate(param_values):
            print(f"  Computing {param_label} = {value} ({i+1}/{len(param_values)})")
            
            # Create a copy of the base parameters
            current_params = param_dict.copy()
            
            # Update the parameter we're varying
            current_params[param_name] = value
            
            # Run simulation
            spikes, main_data = self.original_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=current_params,
                torque_profile=None,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            # Store results
            simulation_data.append(main_data)
            spikes_data.append(spikes)
            
            # Extract time data (same for all simulations)
            if time_data is None:
                time_data = main_data['Time']
                
        
        # Return comprehensive results dictionary
        results = {
            'param_values': param_values,
            'param_name': param_name,
            'param_label': param_label,
            'time_data': time_data,
            'simulation_data': simulation_data,
            'spikes_data': spikes_data,
            'muscle_names': self.original_system.muscles_names,
            'associated_joint': self.original_system.associated_joint,
            'num_muscles': self.original_system.num_muscles
        }
        
        return results
      

    def sensitivity_analysis(self, base_output_path, n_iterations=20, n_samples=50, 
                           time_step=0.1, base_ees_params=None, torque_profile=None, 
                           method='morris', seed=42):
        """
        Wrapper method to perform sensitivity analysis on the biological system.
        
        Parameters:
        -----------
        base_output_path : str
            Base path for saving output files
        n_iterations : int
            Number of iterations for each simulation
        n_samples : int
            Number of parameter samples to generate
        time_step : float
            Time step for the simulation (in ms)
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

        
        # Call the function with self as the first parameter and all other parameters
        return sensitivity_analysis_func(
            biological_system=self.original_system,
            base_output_path=base_output_path,
            n_iterations=n_iterations,
            n_samples=n_samples,
            time_step=time_step,
            base_ees_params=base_ees_params,
            torque_profile=torque_profile,
            method=method,
            seed=seed
        )

        
    def clonus_analysis(self, torque_profile, delay_values=None, 
                    threshold_values=None, duration=1, 
                    time_step=0.1, fast_type_mu=True, seed=41):
        """
        Analyze clonus behavior by varying one parameter at a time.
        
        Parameters:
        -----------
        torque_profile : dict
            Dictionary with torque profile parameters
        delay_values : list, optional
            List of delay values to test (in ms)
        threshold_values : list, optional
            List of threshold values to test (in mV)
        duration : float
            Duration of each simulation (in seconds)
        time_step : float
            Time step for simulations (in ms)
        fast_type_mu : bool
            If True, use fast twitch motor units
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (delay_results, fast_twitch_results, threshold_results)
            Each containing list of (parameter_value, spikes, time_series) tuples
        """
        
        # Set default values if not provided
        if delay_values is None:
            delay_values = [10, 25, 50, 75, 100]  # in ms
        if threshold_values is None:
            threshold_values = [-45, -50, -55]  # in mV
                        
        delay_results = []
        fast_twitch_results = []
        threshold_results = []
        
        # 1. Vary delay (reaction time)
        print("Running delay variation analysis...")
        for delay in tqdm(delay_values, desc="Varying delay"):
            delay_seconds = delay / 1000.0  # Convert ms to seconds
            n_iterations = int(duration / delay_seconds)
           
            new_system = self.original_system.deepcopy()
            new_system.reaction_time = delay_seconds
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            delay_results.append((delay, spikes, time_series))

        plot_delay_results(delay_results, delay_values, self.original_system.muscles_names, self.original_system.associated_joint)
                        
        n_iterations = int(duration / self.original_system.reaction_time)
        time_points = np.arange(0, self.original_system.reaction_time * n_iterations, time_step / 1000.0)
        torque_array = transform_torque_params_in_array(time_points, torque_profile)
                        
        # 2. Vary fast twitch parameter
        print("Running fast twitch variation analysis...")
        fast_twitch_values = [False, True]
        
        for fast in tqdm(fast_twitch_values, desc="Varying fast twitch parameter"):
            new_system = self.original_system.deepcopy()
            new_system.fast = fast
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            fast_twitch_results.append((fast, spikes, time_series))
            
        plot_twitch_result(fast_twitch_results, self.original_system.muscles_names, self.original_system.associated_joint)
                        
        # 3. Vary threshold voltage
        print("Running threshold variation analysis...")
        for threshold in tqdm(threshold_values, desc="Varying threshold voltage"):
            new_system = self.original_system.deepcopy()
            new_system.biophysical_params['v_threshold'] = threshold / 1000.0  # Convert mV to V
            spikes, time_series = new_system.run_simulation(
                n_iterations, 
                time_step,
                ees_stimulation_params=None,
                torque_profile=torque_profile,
                seed=seed, 
                base_output_path=None,
                plot=False)
            
            threshold_results.append((threshold, spikes, time_series))

        plot_excitability_results(threshold_results, threshold_values, self.original_system.muscles_names, self.original_system.associated_joint)
        
        return delay_results, fast_twitch_results, threshold_results

class ReciprocalInhibitoryAnalyzer(Analyzer):
    """
    Specialized analyzer for ReciprocalInhibitorySystem.
    
    This analyzer provides specific analysis methods for reciprocal inhibition
    patterns, cross-inhibition dynamics, and antagonist muscle coordination.
    """
    
    def __init__(self, reciprocal_system: 'ReciprocalInhibitorySystem'):
        """
        Initialize the specialized analyzer.
        
        Parameters:
        -----------
        reciprocal_system : ReciprocalInhibitorySystem
            The reciprocal inhibitory biological system to analyze
        """
        
        super().__init__(reciprocal_system)
        
    
    def analyze_inhibitory_strength_effects(self, strength_range, base_ees_params=None, 
                                          n_iterations=20, time_step=0.1, seed=42):
        """
        Analyze the effects of varying inhibitory connection strength.
        
        Parameters:
        -----------
        strength_range : array-like
            Range of inhibitory strength values to test
        base_ees_params : dict, optional
            Base parameters for EES stimulation
        n_iterations : int
            Number of iterations per simulation
        time_step : float
            Time step in ms
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Analysis results including cross-inhibition metrics
        """
        print("Analyzing inhibitory strength effects...")
        
        results = []
        original_strength = self.original_system.inhibitory_strength
        
        for strength in tqdm(strength_range, desc="Varying inhibitory strength"):
            # Temporarily modify the system
            self.original_system.inhibitory_strength = strength
            
            # Run simulation
            spikes, time_series = self.original_system.run_simulation(
                n_iterations=n_iterations,
                time_step=time_step,
                ees_stimulation_params=base_ees_params,
                torque_profile=None,
                seed=seed,
                base_output_path=None,
                plot=False
            )
            
            # Compute reciprocal inhibition metrics
            inhibition_metrics = self._compute_inhibition_metrics(spikes, time_series)
            
            results.append({
                'inhibitory_strength': strength,
                'spikes': spikes,
                'time_series': time_series,
                'inhibition_metrics': inhibition_metrics
            })
        
        # Restore original strength
        self.original_system.inhibitory_strength = original_strength
        
        # Generate specialized plots
        self._plot_inhibitory_strength_analysis(results, strength_range)
        
        return results
    
    
        return metrics
    
    def _compute_alternation_frequency(self, flexor_activity, extensor_activity, time_points):
        """Compute the frequency of alternation between flexor and extensor."""
        # Simple threshold-based detection of alternations
        threshold = 0.1
        flexor_active = flexor_activity > threshold
        extensor_active = extensor_activity > threshold
        
        # Count transitions
        transitions = 0
        for i in range(1, len(flexor_active)):
            if (flexor_active[i] != flexor_active[i-1]) or (extensor_active[i] != extensor_active[i-1]):
                transitions += 1
        
        total_time = (time_points[-1] - time_points[0]) / 1000.0  # Convert ms to seconds
        return transitions / (2 * total_time)  # Divide by 2 for full cycles
    

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
