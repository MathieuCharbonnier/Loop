import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from functools import partial
from scipy.signal import convolve

def decode_spikes_to_activation(spikes_times, time, dt, f1_l=1.0, f2_l=1.0, f3_l=0.54, f4_l=1.34, f5_l=0.87):
    """
    Decode spike times to muscle activation signals using a biophysical model.
    
    Parameters:
    -----------
    spikes_times : list of arrays
        List containing spike times for each motoneuron
    time : array-like
        Time points at which to evaluate the activation
    f1_l, f2_l, f3_l, f4_l, f5_l : float
        Scaling parameters for the model
        
    Returns:
    --------
    a_all_interp : ndarray
        Activation signals for each motoneuron interpolated to the provided time grid
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
        'Ve': 90, 'T': 14,
    }
    t_min=time[0]
    t_max=time[-1]
    
    def generate_action_potentials(spike_times, time_points, dt, Ve=params['Ve'], t_ap=params['T']):
        """
        Generate action potentials at spike times.
        
        Parameters:
        -----------
        spike_times : array-like
            Times at which spikes occur
        time_points : array-like
            Time points at which to evaluate e(t)
        Ve : float
            Amplitude of action potential
        t_ap : float
            Duration of action potential 
            
        Returns:
        --------
        e_t : ndarray
            Action potential values at each time point
        """
        e_t = np.zeros_like(time_points, dtype=float)
        n_ap=int(t_ap/dt)
        kernel = np.ones(n_ap + 1)  # 1 for the spike + n after it
        extended = convolve(spike_times, kernel, mode='full')[:len(spike_times)]
        e_t[extended] += Ve * np.sin(2 * np.pi / T * (time_points[extended] - spike_times))
        return e_t
    
    # Precompute e(t) for all motoneurons
    e_t_all = np.array([generate_action_potentials(spikes, time) for spikes in spikes_times])
    
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
    a_all_interp = np.zeros((len(spikes_times), len(time)))
    
    for i, e_t in enumerate(e_t_all):
        # Create interpolation function for e(t)
        e_t_interp = interp1d(time, e_t, kind='linear', bounds_error=False, fill_value=0.0)
     
        # Solve fiber AP dynamics
        u_init = [0.0, 0.0]
        sol_u = solve_ivp(
            partial(fibre_ap_dynamics, e_t_func=e_t_interp),
            [t_min, t_max], u_init, **solver_kwargs
        )
        u_interp = interp1d(sol_u.t, sol_u.y[0], kind='linear', bounds_error=False, fill_value=0.0)
      
        # Solve calcium dynamics
        c_init = [0.0, 0.0]
        sol_c = solve_ivp(
            partial(calcium_dynamics, u_func=u_interp),
            [t_min, t_max], c_init, **solver_kwargs
        )
        c_interp = interp1d(sol_c.t, sol_c.y[0], kind='linear', bounds_error=False, fill_value=0.0)
   
        # Solve calcium-troponin binding
        P_init = [0.0]
        sol_P = solve_ivp(
            partial(ca_troponin_dynamics, c_func=c_interp),
            [t_min, t_max], P_init, **solver_kwargs
        )
        P_interp = interp1d(sol_P.t, sol_P.y[0], kind='linear', bounds_error=False, fill_value=0.0)

        # Solve activation dynamics
        a_init = [0.0]
        sol_a = solve_ivp(
            partial(activation_dynamics, P_func=P_interp),
            [t_min, t_max], a_init, **solver_kwargs
        )
     
        # Interpolate activation to the original time grid
        a_all_interp[i, :] = np.interp(time, sol_a.t, sol_a.y[0])
    
    return a_all_interp
