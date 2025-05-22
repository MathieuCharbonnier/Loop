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
