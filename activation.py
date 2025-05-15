import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from functools import partial
import joblib
from joblib import Parallel, delayed

def decode_spikes_to_activation(spikes_times, dt, T, initial_params, fast=True, f1_l=1.0, f2_l=1.0, f3_l=0.54, f4_l=1.34, f5_l=0.87, n_jobs=-1):
    """
    Decode spike times to muscle activation signals using a biophysical model.
    
    Parameters:
    -----------
    spikes_times : list of arrays
        List containing spike times for each motoneuron
    dt : float
        Time step for simulation
    T : float
        Total simulation time
    initial_params : list of dict
        Initial parameters for each motoneuron containing 'u0', 'c0', 'P0', and 'a0'
    f1_l, f2_l, f3_l, f4_l, f5_l : float
        Scaling parameters for the model
    n_jobs : int, default=-1
        Number of parallel jobs to run. -1 means using all processors.
        
    Returns:
    --------
    u_all : ndarray
        Fiber action potential for each motoneuron
    c_all : ndarray
        Calcium concentration for each motoneuron
    P_all : ndarray
        Calcium-troponin binding for each motoneuron
    a_all : ndarray
        Activation signals for each motoneuron
    final_values : dict
        Final state values for each motoneuron
    """
    # Define parameter values 
    params_all = {
        'slow':{
                # Eq 13 coefficients (fibre AP dynamics)
                'a1': 7e7, 'a2': 5e7, 'a3': 2e4,
                # Eq 14 coefficients (calcium dynamics)
                'b1': 0.4, 'b2': 1.5e5, 'b3': 2.5e3,
                # Eq 17 coefficients (calcium-troponin binding)
                'c1': 6e12, 'c2': 21, 'P0': 1.7e-4,
                # Eq 18 coefficients (MU activation)
                'd1': 5e4, 'd2': 0.024, 'd3': 270,
                # AP generation parameters
                'Ve': 90, 't_ap': 0.0014,
               }
        'fast':{
                # Eq 13 coefficients (fibre AP dynamics)
                'a1': 7e7, 'a2': 5e7, 'a3': 2e4,
                # Eq 14 coefficients (calcium dynamics)
                'b1': 0.9, 'b2': 4.3e5, 'b3': 2.4e3,
                # Eq 17 coefficients (calcium-troponin binding)
                'c1': 1e12, 'c2': 41, 'P0': 3.8e-4,
                # Eq 18 coefficients (MU activation)
                'd1': 5e4, 'd2': 0.024, 'd3': 270,
                # AP generation parameters
                'Ve': 90, 't_ap': 0.0014,
               }
    }
    if fast:
        params=params_all['fast']
    else:
        params=params_all['slow']
        
    time = np.arange(0, T, dt)
    num_motoneurons = len(spikes_times)
    num_timesteps = len(time)
    
    # Pre-compute sine values for window to avoid repetitive calculations
    t_ap = params['t_ap']
    Ve = params['Ve']
    sin_wave_template = Ve * np.sin(2 * np.pi / t_ap * np.arange(0, t_ap/2, dt))
    sin_wave_len = len(sin_wave_template)
    
    # Initialize output arrays
    e_t_all = np.zeros((num_motoneurons, num_timesteps))
    u_all = np.zeros((num_motoneurons, num_timesteps))
    c_all = np.zeros((num_motoneurons, num_timesteps))
    P_all = np.zeros((num_motoneurons, num_timesteps))
    a_all = np.zeros((num_motoneurons, num_timesteps))
    final_values = []
    
    # Precompute e(t) for all motoneurons
    for i, spike_times_moto in enumerate(spikes_times):
        for spike_time in spike_times_moto:
            idx_start = int(spike_time / dt)
            if idx_start < num_timesteps:
                # Calculate how many samples we can place
                samples_to_copy = min(sin_wave_len, num_timesteps - idx_start)
                e_t_all[i, idx_start:idx_start + samples_to_copy] = sin_wave_template[:samples_to_copy]
    
    # Define a consolidated ODE system that computes all equations at once
    def consolidated_ode_system(t, y, e_t_func, params, f_params):
        """
        Consolidated ODE system that solves all equations at once
        y = [u, du/dt, c, dc/dt, P, a]
        """
        u, u_dot, c, c_dot, P, a = y
        
        # Extract parameters
        a1, a2, a3 = params['a1'], params['a2'], params['a3']
        b1, b2, b3 = params['b1'], params['b2'], params['b3']
        c1, c2, P0 = params['c1'], params['c2'], params['P0']
        d1, d2, d3 = params['d1'], params['d2'], params['d3']
        f1_l, f2_l, f3_l, f4_l, f5_l = f_params
        
        # Get input signal at time t
        e_t = e_t_func(t)
        
        # Fibre AP dynamics equations
        du_dt = u_dot
        du_dot_dt = a1 * e_t - a2 * u - a3 * u_dot
        
        # Calcium dynamics equations
        dc_dt = c_dot
        dc_dot_dt = b1 * u - (1/f1_l) * (b2 * f2_l * c + b3 * c_dot)
        
        # Calcium-troponin binding equation
        dP_dt = (c1/f3_l) * (P0/f4_l - P) * (c**2) - (c2/f5_l) * P
        
        # Activation dynamics equation
        da_dt = d1 * P - a / (d2 + d3 * P)
        
        return [du_dt, du_dot_dt, dc_dt, dc_dot_dt, dP_dt, da_dt]
    
    # Optimize solver parameters for this specific problem
    solver_kwargs = {
        'method': 'RK45',  
        'rtol': 1e-3,
        'atol': 1e-5,      
        'max_step': 1e-2,  
        'dense_output': True
    }
    
    # Function to process a single motoneuron
    def process_motoneuron(i, e_t_i, initial_param_i):
        # Create interpolator for e_t
        e_t_interp = interp1d(time, e_t_i, kind='linear', bounds_error=False, fill_value=0.0)
        
        # Extract initial conditions
        u_init = initial_param_i['u0']  # Assuming this is [u, du/dt]
        c_init = initial_param_i['c0']  # Assuming this is [c, dc/dt]
        P_init = initial_param_i['P0']  # This is a scalar
        a_init = initial_param_i['a0']  # This is a scalar
   
        # Combine all initial conditions as a flat array of scalars
        y0 = np.array([u_init[0], u_init[1], c_init[0], c_init[1], P_init, a_init], dtype=float)
        # Pack f parameters 
        f_params = (f1_l, f2_l, f3_l, f4_l, f5_l)
        
        # Solve the consolidated ODE system
        sol = solve_ivp(
            partial(consolidated_ode_system, e_t_func=e_t_interp, params=params, f_params=f_params),
            [0, T], y0, t_eval=time, **solver_kwargs
        )
        
        # Extract results
        u_i = sol.y[0]
        c_i = sol.y[2]
        P_i = sol.y[4]
        a_i = sol.y[5]
        
        # Prepare final values
        final_vals = {
            'u0': [sol.y[0, -1], sol.y[1, -1]],
            'c0': [sol.y[2, -1], sol.y[3, -1]],
            'P0': sol.y[4, -1],
            'a0': sol.y[5, -1]
        }
        
        return u_i, c_i, P_i, a_i, final_vals
    
    # Use parallelization to process all motoneurons
    # Determine optimal number of jobs based on machine availability
    n_jobs = n_jobs if n_jobs > 0 else joblib.cpu_count()
    
    # Cap the number of jobs at the number of motoneurons
    n_jobs = min(n_jobs, num_motoneurons)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_motoneuron)(i, e_t_all[i], initial_params[i]) 
        for i in range(num_motoneurons)
    )
    
    # Unpack results
    for i, (u_i, c_i, P_i, a_i, final_vals) in enumerate(results):
        u_all[i] = u_i
        c_all[i] = c_i
        P_all[i] = P_i
        a_all[i] = a_i
        final_values.append(final_vals) 

    return e_t_all, u_all, c_all, P_all, a_all, final_values
