import numpy as np
from tqdm import tqdm
from BiologicalSystemAnalyzer import BiologicalSystemAnalyzer

class ReciprocalInhibitoryAnalyzer(BiologicalSystemAnalyzer):
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
