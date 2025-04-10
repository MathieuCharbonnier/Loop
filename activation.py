import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

def decode_spikes_to_activation(spikes_times, time, f1_l=1.0, f2_l=1.0, f3_l=0.54, f4_l=1.34, f5_l=0.87):
    # Define parameter values (from Table 1)
    a1, a2, a3 = 7e7, 5e7, 2e4  # Eq 13 coefficients
    b1, b2, b3 = 0.9, 4.3e5, 2.4e3  # Eq 14 coefficients
    c1, c2, P0 = 1e12, 41, 3.8e-4  # Eq 17 coefficients
    d1, d2, d3 = 5e4, 0.02, 200  # Eq 18 coefficients

    time = np.array(time)  # Convert to numpy array if not already
    t_max = time[-1]

    def e_t_preprocessed(spikes_times, time, Ve=90, T=1.4e-3):
        e_t_all = np.zeros((len(spikes_times), len(time)))
        for i, spike_times_moto in enumerate(spikes_times):
            for spike_time in spike_times_moto:
                t_end = spike_time + T / 2
                # Precompute sine values for the window
                t_range = (time >= spike_time) & (time <= t_end)
                e_t_all[i, t_range] = Ve * np.sin(2 * np.pi / T * (time[t_range] - spike_time))
        return e_t_all

    # Preprocess e(t) for all motoneurons
    e_t_all = e_t_preprocessed(spikes_times, time)

    # Define impulse response for fibre AP dynamics
    def fibre_ap_dynamics(t, u):
        du_dt = u[1]
        dv_dt = a1 * e_t_interpolate(t) - a2 * u[0] - a3 * u[1]
        return [du_dt, dv_dt]

    # Define free calcium concentration dynamics
    def calcium_dynamics(t, c):
        dc_dt = c[1]
        dg_dt = b1 * sol_u_interpolate(t) - (1 / f1_l) * (b2 * f2_l * c[0] + b3 * c[1])
        return [dc_dt, dg_dt]

    # Define calcium-troponin binding dynamics
    def ca_troponin_dynamics(t, P):
        dP_dt = (c1 / f3_l) * (P0 / f4_l - P) * (sol_c_interpolate(t)) ** 2 - (c2 / f5_l) * P
        return dP_dt

    # Define MU activation dynamics
    def activation_dynamics(t, a):
        da_dt = d1 * sol_P_interpolate(t) - a / (d2 + d3 * sol_P_interpolate(t))
        return da_dt

    t_results = []
    a_t_results = []
    # Process motoneurons
    for i, e_t in enumerate(e_t_all):
        e_t_interpolate = interp1d(time, e_t, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Solve each ODE after defining interpolation
        u_init = [0, 0]
        sol_u = solve_ivp(fibre_ap_dynamics, [0, t_max], u_init, max_step=1e-3)
        sol_u_interpolate = interp1d(sol_u.t, sol_u.y[0], kind='linear', bounds_error=False, fill_value='extrapolate')

        c_init = [0, 0]
        sol_c = solve_ivp(calcium_dynamics, [0, t_max], c_init, max_step=1e-3)
        sol_c_interpolate = interp1d(sol_c.t, sol_c.y[0], kind='linear', bounds_error=False, fill_value='extrapolate')

        P_init = [0]
        sol_P = solve_ivp(ca_troponin_dynamics, [0, t_max], P_init, max_step=1e-3)
        sol_P_interpolate = interp1d(sol_P.t, sol_P.y[0], kind='linear', bounds_error=False, fill_value='extrapolate')

        a_init = [0]
        a_t = solve_ivp(activation_dynamics, [0, t_max], a_init, max_step=1e-3)
        t_results.append(sol_u.t)
        a_t_results.append(a_t)

    # Interpolate and stack a(t) results on the common time grid
    a_all_interp = np.array([
        np.interp(time, a_t.t, a_t.y[0])
        for a_t in a_t_results
    ])
    return a_all_interp


