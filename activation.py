import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from functools import partial
from scipy.signal import convolve

def decode_spikes_to_activation(spikes_times, dt, T, initial_params, f1_l=1.0, f2_l=1.0, f3_l=0.54, f4_l=1.34, f5_l=0.87):
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
    # Define parameter values (from Table 1)
    # Model parameters as a dictionary for better maintainability
    params = {
        # Eq 13 coefficients (fibre AP dynamics)
        'a1': 7e7, 'a2': 5e7, 'a3': 2e4,
        # Eq 14 coefficients (calcium dynamics)
        'b1': 0.9, 'b2': 4.3e5, 'b3': 2.4e3,
        # Eq 17 coefficients (calcium-troponin binding)
        'c1': 1e12, 'c2': 41, 'P0': 3.8e-4,
        # Eq 18 coefficients (MU activation)
        'd1': 5e4, 'd2': 0.02, 'd3': 200,
        # AP generation parameters
        'Ve': 90, 't_ap': 0.0014,
    }

    def generate_action_potentials(spike_times, dt, T, Ve=params['Ve'], t_ap=params['t_ap']):
        """
        Generate action potentials at spike times.
        
        Parameters:
        -----------
        spike_times : array-like
            Times at which spikes occur
        dt : float
            Time step
        T : float
            Total simulation time
        Ve : float
            Amplitude of action potential
        t_ap : float
            Duration of action potential 
            
        Returns:
        --------
        e_t : ndarray
            Action potential values at each time point
        """
        time_points = np.arange(0, T, dt)
        e_t = np.zeros_like(time_points, dtype=float)
        
        # For each spike time, create action potential waveform
        for spike_time in spike_times:
            # Find indices of time points within action potential duration after spike
            spike_idx = np.where((time_points >= spike_time) & (time_points < spike_time + t_ap))[0]
            
            # Calculate sine wave for action potential
            if len(spike_idx) > 0:
                e_t[spike_idx] += Ve * np.sin(2 * np.pi / t_ap * (time_points[spike_idx] - spike_time))
        
        return e_t

    # Precompute e(t) for all motoneurons
    e_t_all = np.array([generate_action_potentials(spikes, dt, T) for spikes in spikes_times])
    
    # Create ODE system functions for each stage
    def fibre_ap_dynamics(t, u, e_t_func, a1=params['a1'], a2=params['a2'], a3=params['a3']):
        """ODE system for fiber action potential dynamics"""
        du_dt = u[1]
        dv_dt = a1 * e_t_func(t) - a2 * u[0] - a3 * u[1]
        return [du_dt, dv_dt]
    
    def calcium_dynamics(t, c, u_func, b1=params['b1'], b2=params['b2'], b3=params['b3']):
        """ODE system for free calcium concentration dynamics"""
        dc_dt = c[1]
        dg_dt = b1 * u_func(t) - (1/f1_l) * (b2 * f2_l * c[0] + b3 * c[1])
        return [dc_dt, dg_dt]
    
    def ca_troponin_dynamics(t, P, c_func, c1=params['c1'], c2=params['c2'], P0=params['P0']):
        """ODE system for calcium-troponin binding dynamics"""
        dP_dt = (c1/f3_l) * (P0/f4_l - P) * (c_func(t))**2 - (c2/f5_l) * P
        return dP_dt
    
    def activation_dynamics(t, a, P_func, d1=params['d1'], d2=params['d2'], d3=params['d3']):
        """ODE system for MU activation dynamics"""
        da_dt = d1 * P_func(t) - a / (d2 + d3 * P_func(t))
        return da_dt
    
    # Configure ODE solver
    solver_kwargs = {
        'method': 'RK45',        # Runge-Kutta 4(5)
        'rtol': 1e-4,           # Relative tolerance
        'atol': 1e-6,           # Absolute tolerance
        'max_step': 1e-3,       # Maximum step size
        'dense_output': True    # Enable dense output for efficient interpolation
    }
    
    # Process each motoneuron
    time = np.arange(0, T, dt)
    u_all = np.zeros((len(spikes_times), len(time)))
    c_all = np.zeros((len(spikes_times), len(time)))
    P_all = np.zeros((len(spikes_times), len(time)))
    a_all = np.zeros((len(spikes_times), len(time)))
    final_values = {i: {} for i in range(len(spikes_times))}  # Initialize final_values dictionary
    
    for i, e_t in enumerate(e_t_all):
        # Create interpolation function for e(t)
        e_t_interp = interp1d(time, e_t, kind='linear', bounds_error=False, fill_value=0.0)
     
        # Solve fiber AP dynamics
        u_init = initial_params[i]['u0']
        sol_u = solve_ivp(
            partial(fibre_ap_dynamics, e_t_func=e_t_interp),
            [0, T], u_init, t_eval=time, **solver_kwargs
        )
        final_values[i]['u0'] = sol_u.y[:, -1]
        u_interp = interp1d(sol_u.t, sol_u.y[0], kind='linear', bounds_error=False, fill_value=0.0)
      
        # Solve calcium dynamics
        c_init = initial_params[i]['c0']
        sol_c = solve_ivp(
            partial(calcium_dynamics, u_func=u_interp),
            [0, T], c_init, t_eval=time, **solver_kwargs
        )
        final_values[i]['c0'] = sol_c.y[:, -1]  # Fixed key name to cf
        c_interp = interp1d(sol_c.t, sol_c.y[0], kind='linear', bounds_error=False, fill_value=0.0)
   
        # Solve calcium-troponin binding
        P_init = initial_params[i]['P0'] 
        sol_P = solve_ivp(
            partial(ca_troponin_dynamics, c_func=c_interp),
            [0, T], P_init, t_eval=time, **solver_kwargs
        )
        final_values[i]['P0'] = sol_P.y[:, -1]  # Fixed key name to Pf
        P_interp = interp1d(sol_P.t, sol_P.y[0], kind='linear', bounds_error=False, fill_value=0.0)

        # Solve activation dynamics
        a_init = initial_params[i]['a0']  
        sol_a = solve_ivp(
            partial(activation_dynamics, P_func=P_interp),
            [0, T], a_init, t_eval=time, **solver_kwargs
        )
        final_values[i]['a0'] = sol_a.y[:, -1]  # Fixed key name to af
        
        # Store results
        u_all[i, :] = sol_u.y[0]
        c_all[i, :] = sol_c.y[0]
        P_all[i, :] = sol_P.y[0]
        a_all[i, :] = sol_a.y[0]

    return u_all, c_all, P_all, a_all, final_values
