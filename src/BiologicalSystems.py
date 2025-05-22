
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from closed_loop import closed_loop
from analysis import delay_excitability_MU_type_analysis, EES_stim_analysis
from controller import HierarchicalAnkleController
from plots import plot_mouvement, plot_neural_dynamic, plot_raster, plot_activation, plot_recruitment_curves
from input_generator import transform_intensity_balance_in_recruitment, transform_torque_params_in_array

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
        
        torque_array = None
        if torque_profile is not None:
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            torque_array = transform_torque_params_in_array(time_points, torque_profile)
        
        ees_params = None
        if ees_stimulation_params is not None:
            ees_params = transform_intensity_balance_in_recruitment(
            self.ees_recruitment_profile, ees_stimulation_params, 
            self.neurons_population, self.num_muscles)
        
        spikes, time_series = closed_loop(
            n_iterations, self.reaction_time, time_step, self.neurons_population, self.connections,
            self.spindle_model, self.biophysical_params, self.muscles_names, self.number_muscles, self.associated_joint,
             torque_array=torque_array,ees_params=ees_params,
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
        plot_results : bool
            Whether to automatically generate plots
        
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

    def analyze_intensity_effects(self, intensity_range, base_ees_params, n_iterations=20, time_step=0.1*ms, seed=42):
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
        time_step : brian2.units.fundamentalunits.Quantity
            Time step for simulations
        seed : int
            Random seed for reproducibility
        plot_results : bool
            Whether to automatically generate plots
        
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

    def _compute_ees_parameter_sweep(self, param_dict, vary_param, n_iterations, time_step=0.1*ms, seed=42):
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
        time_step : brian2.unit
            Time step for simulation
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
            
            ees_params = transform_intensity_balance_in_recruitment(
            self.ees_recruitment_profile, current_params, 
            self.neurons_population, self.num_muscles)
            
            # Run simulation
            spikes, main_data = closed_loop(
                n_iterations, 
                self.reaction_time, 
                time_step, 
                self.neurons_population, 
                self.connections,
                self.spindle_model, 
                self.biophysical_params, 
                self.muscles_names,
                self.number_muscles, 
                self.associated_joint,
                ees_params, # EES_STIMULATION_PARAMS
                None,  # TORQUE 
                True, 
                seed,
                None
            )
            
            # Store results
            simulation_data.append(main_data)
            spikes_data.append(spikes)
            
            # Extract time data (same for all simulations)
            if time_data is None:
                time_data = main_data['Time']
                
                # Initialize activities array for coactivation analysis
                if self.number_muscles == 2:
                    T = len(time_data)
                    activities = np.zeros((self.number_muscles, len(param_values), T))
            
            # Store activation data for coactivation analysis
            if self.number_muscles == 2:
                for muscle_idx, muscle_name in enumerate(self.muscles_names):
                    col_name = f"Activation_{muscle_name}"
                    if col_name in main_data.columns:
                        activities[muscle_idx, i, :] = main_data[col_name].values
        
        
        # Return comprehensive results dictionary
        results = {
            'param_values': param_values,
            'param_name': param_name,
            'param_label': param_label,
            'time_data': time_data,
            'simulation_data': simulation_data,
            'spikes_data': spikes_data,
            'activities': activities,
            'muscle_names': self.muscles_names,
            'associated_joint': self.associated_joint,
            'num_muscles': self.number_muscles
        }
        
        return results
        
    def clonus_analysis(self, torque_profile, delay_values=[10, 25, 50, 75, 100]*ms, 
                    threshold_values=[-45, -50, -55]*mV, duration=1*second, 
                    time_step=0.1*ms, fast_type_mu=True, , seed=41):
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
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Analysis results containing:
            - delay_results: list of (delay_value, spikes, time_series) tuples
            - fast_twitch_results: list of (fast_twitch_bool, spikes, time_series) tuples
            - threshold_results: list of (threshold_value, spikes, time_series) tuples
        """

        results = {
            'delay_results': [],
            'fast_twitch_results': [],
            'threshold_results': [],
            'parameters': {
                'delay_values': delay_values,
                'threshold_values': threshold_values,
                'duration': duration,
                'time_step': time_step,
                'reaction_time': self.reaction_time,
                'muscles_names': self.muscles_names,
                'associated_joint': self.associated_joint,
                'fast_type_mu': fast_type_mu,
                'seed': seed
            }
        }
        
        # 1. Vary delay (reaction time)
        print("Running delay variation analysis...")
        for delay in tqdm(delay_values, desc="Varying delay"):
            current_reaction_time = delay
            n_iterations = int(duration/current_reaction_time)
 
            time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
            torque_array = transform_torque_params_in_array(time_points, torque_profile)
            
            spikes, time_series = closed_loop(
                n_iterations, 
                current_reaction_time, 
                time_step, 
                self.neurons_population, 
                self.connections,
                self.spindle_model, 
                self.biophysical_params, 
                self.muscles_names,
                self.number_muscles, 
                self.associated_joint,
                None,
                torque_array,
                fast_type_mu, 
                seed,
                None
            )
            
            results['delay_results'].append((delay, spikes, time_series))
            
        n_iterations = int(duration/self.reaction_time)
        time_points = np.arange(0, self.reaction_time*n_iterations, time_step)
        torque_array = transform_torque_params_in_array(time_points, torque_profile)
                        
        # 2. Vary fast twitch parameter
        print("Running fast twitch variation analysis...")
        fast_twitch_values = [False, True]
        
        for fast in tqdm(fast_twitch_values, desc="Varying fast twitch parameter"):
            spikes, time_series = closed_loop(
                n_iterations, 
                self.reaction_time, 
                time_step, 
                self.neurons_population, 
                self.connections,
                self.spindle_model, 
                self.biophysical_params, 
                self.muscles_names,
                self.number_muscles, 
                self.associated_joint,
                None,
                torque_profile,
                fast, 
                seed,
                None
            )
            
            results['fast_twitch_results'].append((fast, spikes, time_series))
        
        # 3. Vary threshold voltage
        print("Running threshold variation analysis...")
        for threshold in tqdm(threshold_values, desc="Varying threshold voltage"):
            current_biophysical_params = self.biophysical_params.copy()
            current_biophysical_params['threshold_v'] = threshold
            
            n_iterations = int(duration/self.reaction_time)
            spikes, time_series = closed_loop(
                n_iterations, 
                self.reaction_time, 
                time_step, 
                self.neurons_population, 
                self.connections,
                self.spindle_model, 
                current_biophysical_params, 
                self.muscles_names,
                self.number_muscles, 
                self.associated_joint,
                None,
                torque_profile,
                fast_type_mu, 
                seed,
                None
            )
            
            results['threshold_results'].append((threshold, spikes, time_series))
        
        return results

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
        time_step : float or brian2.units
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
        # Import the function from the separate module
        from sensitivity_analysis import sensitivity_analysis as sensitivity_analysis_func
        
        # Call the function with self as the first parameter and all other parameters
        return sensitivity_analysis_func(
            biological_system=self,
            base_output_path=base_output_path,
            n_iterations=n_iterations,
            n_samples=n_samples,
            time_step=time_step,
            base_ees_params=base_ees_params,
            torque_profile=torque_profile,
            method=method,
            seed=seed
        )

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
            ("Ia", "MN"): {"w": 2.1*nS, "p": 0.7}
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
                    
