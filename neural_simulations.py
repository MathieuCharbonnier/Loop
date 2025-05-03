from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional

import matplotlib.pyplot as plt

def run_flexor_extensor_neuron_simulation(stretch, velocity, 
                                          neuron_pop, dt_run, T, initial_potentials=None, Eleaky=-70*mV,
                                          gL=0.1*mS, Cm=1*uF, E_ex=0*mV, E_inh=-75*mV, 
                                          tau_exc=0.5*ms, tau_1=1.5*ms, tau_2=2*ms, threshold_v=-55*mV, 
                                          ees_freq=0*hertz, Ia_recruited=0, II_recruited=0, eff_recruited=0, T_refr=10*ms):
    """
    Run a simulation of flexor-extensor neuron dynamics.
    
    Parameters:
    ----------
    stretch : list of arrays
        Stretch inputs for flexor [0] and extensor [1].
    velocity : list of arrays
        Velocity inputs for flexor [0] and extensor [1].
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'II', 'motor', 'exc', 'inh').
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    initial_potentials : dict, optional
        Initial membrane potentials for neuron groups.
    Eleaky : volt, optional
        Leaky potential.
    gL : siemens, optional
        Leak conductance.
    Cm : farad, optional
        Membrane capacitance.
    E_ex : volt, optional
        Excitatory reversal potential.
    E_inh : volt, optional
        Inhibitory reversal potential.
    tau_exc : time, optional
        Excitatory time constant.
    tau_1 : time, optional
        First inhibitory time constant.
    tau_2 : time, optional
        Second inhibitory time constant.
    threshold_v : volt, optional
        Voltage threshold.
    ees_freq : hertz, optional
        EES frequency.
    aff_recruited : int, optional
        Number of afferent neurons recruited.
    eff_recruited : float, optional
        Number of efferent neurons recruited.
    T_refr : time, optional
        Refractory period.
    
    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data for flexor and extensor pathways,
        and a dictionary with final membrane potentials.
    """
    # Set up random seeds for reproducibility
    np.random.seed(42)
    seed(42)
    defaultclock.dt = dt_run

    net = Network()

    # Input arrays
    stretch_flexor_array = TimedArray(stretch[0], dt=dt_run)
    velocity_flexor_array = TimedArray(velocity[0], dt=dt_run)
    stretch_extensor_array = TimedArray(stretch[1], dt=dt_run)
    velocity_extensor_array = TimedArray(velocity[1], dt=dt_run)

    # Extract neuron counts from dictionary
    n_Ia = neuron_pop['Ia']
    n_II = neuron_pop['II']
    n_exc = neuron_pop['exc']
    n_inh = neuron_pop['inh']
    n_motor = neuron_pop['motor']  

    # Afferent neuron equations

    ia_eq = '''
    is_flexor = (i < n_Ia) : boolean
    stretch_array = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    velocity_array = velocity_flexor_array(t) * int(is_flexor) + velocity_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < Ia_recruited) or (not is_flexor and i < n_Ia + Ia_recruited)) : boolean
    rate = 10*hertz + 0.4*hertz*stretch_array + 0.86*hertz*sign(velocity_array)*abs(velocity_array)**0.6 + ees_freq * int(is_ees) : Hz
    '''
 
    ii_eq = '''
    is_flexor = (i < n_II) : boolean
    stretch_array = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < II_recruited) or (not is_flexor and i < n_II + II_recruited)) : boolean
    rate = 20*hertz + 3.375*hertz*stretch_array + ees_freq * int(is_ees) : Hz
    '''
    
    # Create afferent neurons
    Ia = NeuronGroup(2*n_Ia, ia_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II = NeuronGroup(2*n_II, ii_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    net.add([Ia, II])

    # LIF neuron equations
    ex_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn = gII*(E_ex - v) :amp
    dgII/dt = -gII / tau_exc : siemens 
    '''
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = gIa*(E_ex - v) + gex*(E_ex-v) + gi*(E_inh - v) :amp
    dgIa/dt = -gIa / tau_exc : siemens 
    dgex/dt = -gex / tau_exc : siemens
    dgi/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-gi)/tau_1 : siemens
    dx/dt = -x/tau_2: siemens
    '''
    inh_eq = '''
    dv/dt = (gL*(Eleaky - v)+Isyn ) / Cm : volt
    Isyn = gi*(E_inh - v) + gIa*(E_ex-v) + gII*(E_ex - v) :amp
    dgIa/dt = -gIa / tau_exc : siemens 
    dgII/dt = -gII / tau_exc : siemens 
    dgi/dt = ((tau_2 / tau_1) ** (tau_1 / (tau_2 - tau_1))*x-gi)/tau_1 : siemens
    dx/dt = -x/tau_2: siemens
    '''
    
    # Create neuron groups
    inh = NeuronGroup(2*n_inh, inh_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', refractory=T_refr, method='euler')
  
    exc = NeuronGroup(2*n_exc, ex_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', refractory=T_refr, method='euler')
                                         
    moto = NeuronGroup(2*n_motor, mn_eq, threshold='v > threshold_v', 
                       reset='v = Eleaky', refractory=T_refr, method='euler')
                       
    # Initialize membrane potentials
    inh.v = initial_potentials['inh']
    exc.v = initial_potentials['exc']
    moto.v = initial_potentials['moto']

    # Add neuron groups to the network
    net.add([inh, exc, moto])
    
    # Define neural connections
    connections = {
        ("Ia_flexor", "moto_flexor"): {"pre": Ia[:n_Ia], "post": moto[:n_motor], "model": "gIa_post += 2.1*nS", "p": 1},
        ("Ia_flexor", "inh_flexor"): {"pre": Ia[:n_Ia], "post": inh[:n_inh], "model": "gIa_post += 3.64*nS", "p": 1},
        ("Ia_extensor", "moto_extensor"): {"pre": Ia[n_Ia:], "post": moto[n_motor:], "model": "gIa_post += 2.1*nS", "p": 1},
        ("Ia_extensor", "inh_extensor"): {"pre": Ia[n_Ia:], "post": inh[n_inh:], "model": "gIa_post += 3.64*nS", "p": 1},
        
        ("II_flexor", "exc_flexor"): {"pre": II[:n_II], "post": exc[:n_exc], "model": "gII_post += 1.65*nS", "p": 1},
        ("II_flexor", "inh_flexor"): {"pre": II[:n_II], "post": inh[:n_inh], "model": "gII_post += 2.9*nS", "p": 1},
        ("II_extensor", "exc_extensor"): {"pre": II[n_II:], "post": exc[n_exc:], "model": "gII_post += 1.65*nS", "p": 1},
        ("II_extensor", "inh_extensor"): {"pre": II[n_II:], "post": inh[n_inh:], "model": "gII_post += 2.9*nS", "p": 1},
        
        ("exc_flexor", "moto_flexor"): {"pre": exc[:n_exc], "post": moto[:n_motor], "model": "gex_post += 0.7*nS", "p": 0.6},
        ("exc_extensor", "moto_extensor"): {"pre": exc[n_exc:], "post": moto[n_motor:], "model": "gex_post += 0.7*nS", "p": 0.6},
        
        ("inh_flexor", "moto_extensor"): {"pre": inh[:n_inh], "post": moto[n_motor:], "model": "gi_post += 0.2*nS", "p": 1},
        ("inh_extensor", "moto_flexor"): {"pre": inh[n_inh:], "post": moto[:n_motor], "model": "gi_post += 0.2*nS", "p": 1},
        ("inh_flexor", "inh_extensor"): {"pre": inh[:n_inh], "post": inh[n_inh:], "model": "gi_post += 0.76*nS", "p": 0.5},
        ("inh_extensor", "inh_flexor"): {"pre": inh[n_inh:], "post": inh[:n_inh], "model": "gi_post += 0.76*nS", "p": 0.5}
    }
    
    # Create synaptic connections
    synapses = {}
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = conn_info["pre"]
        post = conn_info["post"]
        model = conn_info["model"]
        p = conn_info["p"]
  
        syn = Synapses(pre, post, on_pre=model, method='exact')
        syn.connect(p=p)
          
        net.add(syn)
        synapses[key] = syn
          
    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)
    mon_exc = SpikeMonitor(exc)
    mon_inh = SpikeMonitor(inh)
    mon_motoneuron = SpikeMonitor(moto)
    
    mon_exc_flexor=StateMonitor(exc, ['v', 'gII'], 20)
    mon_inh_flexor=StateMonitor(inh, ['v','gIa','gII','gi'], 20)
    mon_moto_flexor=StateMonitor(moto, ['v','gIa','gex','gi'], 20)
    
    mon_exc_extensor=StateMonitor(exc, ['v', 'gII'], 80)
    mon_inh_extensor=StateMonitor(inh, ['v','gIa','gII','gi'], 80)
    mon_moto_extensor=StateMonitor(moto, ['v','gIa','gex','gi'], 80)
    
    
    # Add all monitors to the network
    monitors = [
        mon_Ia, mon_II, mon_exc, mon_inh, mon_motoneuron, 
        mon_exc_flexor, mon_inh_flexor, mon_moto_flexor,
        mon_exc_extensor, mon_inh_extensor, mon_moto_extensor
    ]
                 
    net.add(monitors)
    
    # Variables for EES monitors
    mon_ees_moto_flexor = None
    mon_ees_moto_extensor = None
                                            
    # Handle EES stimulation if enabled
    if ees_freq > 0 and eff_recruited > 0:
        ees_motoneuron = PoissonGroup(N=2*eff_recruited, rates=ees_freq)
        mon_ees_moto = SpikeMonitor(ees_motoneuron)
        net.add([ees_motoneuron, mon_ees_moto])

    # Run simulation
    net.run(T)
    
    # Extract motoneuron spikes
    moto_flexor_spikes ={i: mon_motoneuron.spike_trains()[i] for i in range(n_motor)} 
    moto_extensor_spikes = {i: mon_motoneuron.spike_trains()[i] for i in range(n_motor, 2*n_motor)} 

    # Process motoneuron spikes by adding EES effect if enabled
    motor_flexor_spikes = moto_flexor_spikes
    motor_extensor_spikes = moto_extensor_spikes
    
    if ees_freq > 0 and eff_recruited > 0:
        ees_spikes = mon_ees_moto.spike_trains()
        motor_flexor_spikes = process_motoneuron_spikes(
        neuron_pop, moto_flexor_spikes, {i: ees_spikes[i] for i in range(eff_recruited)}, T_refr)
        motor_extensor_spikes = process_motoneuron_spikes(
        neuron_pop, moto_extensor_spikes, {i: ees_spikes[i+eff_recruited] for i in range(eff_recruited)}, T_refr)
   
    # Final membrane potentials
    final_potentials = {
        'inh': inh.v[:],
        'exc': exc.v[:],
        'moto': moto.v[:]
    } 
    # Store state monitors for plotting
    state_monitors = [ {
            'v_exc': mon_exc_flexor.v[0]/mV,
            'gII_exc': mon_exc_flexor.gII[0]/nS,
            'v_inh': mon_inh_flexor.v[0]/mV,
            'gIa_inh': mon_inh_flexor.gIa[0]/nS,
            'gII_inh': mon_inh_flexor.gII[0]/nS,
            'gi_inh': mon_inh_flexor.gi[0]/nS,
            'v_moto': mon_moto_flexor.v[0]/mV,
            'gIa_moto': mon_moto_flexor.gIa[0]/nS,
            'gex_moto': mon_moto_flexor.gex[0]/nS,
            'gi_moto': mon_moto_flexor.gi[0]/nS
        },{
            'v_exc': mon_exc_extensor.v[0]/mV,
            'gII_exc': mon_exc_extensor.gII[0]/nS,
            'v_inh': mon_inh_extensor.v[0]/mV,
            'gIa_inh': mon_inh_extensor.gIa[0]/nS,
            'gII_inh': mon_inh_extensor.gII[0]/nS,
            'gi_inh': mon_inh_extensor.gi[0]/nS,
            'v_moto': mon_moto_extensor.v[0]/mV,
            'gIa_moto': mon_moto_extensor.gIa[0]/nS,
            'gex_moto': mon_moto_extensor.gex[0]/nS,
            'gi_moto': mon_moto_extensor.gi[0]/nS
        }
    ]

    # Return results
    return [{"Ia": {i: mon_Ia.spike_trains()[i] for i in range(n_Ia)},
            "II": {i: mon_II.spike_trains()[i] for i in range(n_II)},
            "exc": {i: mon_exc.spike_trains()[i] for i in range(n_exc)},
            "inh": {i: mon_inh.spike_trains()[i] for i in range(n_inh)},
            "MN": motor_flexor_spikes},
            {"Ia": {i: mon_Ia.spike_trains()[i] for i in range(n_Ia, 2*n_Ia)},
            "II": {i: mon_II.spike_trains()[i] for i in range(n_II, 2*n_II)},
            "exc": {i: mon_exc.spike_trains()[i] for i in range(n_exc, 2*n_exc)},
            "inh": {i: mon_inh.spike_trains()[i] for i in range(n_inh, 2*n_inh)},
            "MN": motor_extensor_spikes}], final_potentials, state_monitors


def run_neural_simulations(stretch, velocity, neuron_pop, dt_run, T, w=500*uS, p=0.4,Eleaky= -70*mV,
    gL= 0.1 * mS,Cm= 1 * uF,E_ex= 0 * mV,E_inh= -75 * mV,tau_exc= 0.5 * ms,tau_inh= 3 * ms,
    threshold_v= -55 * mV,ees_freq=0*hertz, aff_recruited=0, eff_recruited=0, T_refr=10*ms):
    """
    Run neural simulations with stretch and velocity inputs.
    
    Parameters:
    ----------
    stretch : array-like
        Stretch signal for muscle spindles
    velocity : array-like
        Velocity signal for muscle spindles
    neuron_pop : dict
        Dictionary with neuron population sizes {"Ia": int, "II": int, "exc": int, "motor": int}
    dt_run : Quantity
        Time step for the simulation
    T : Quantity
        Total simulation time
    w_run : Quantity, optional
        Synaptic weight (default: 500*uS)
    p_run : float, optional
        Connection probability (default: 0.4)
    ees_freq : Quantity, optional
        Frequency of electrical stimulation (default: 0*hertz)
    aff_recruited : float, optional
        Proportion of afferent neurons recruited by EES (0-1)
    eff_recruited : float, optional
        Proportion of efferent neurons recruited by EES (0-1)
    T_refr : Quantity, optional
        Refractory period (default: 10*ms)
        
    Returns:
    -------
    dict
        Spike trains for Ia, II, and motor neurons
    """
    # Set up random seeds for reproducibility
    np.random.seed(42)
    seed(42)
    
    # Setting the simulation time step
    defaultclock.dt = dt_run
    
    # Prepare input arrays
    stretch_array = TimedArray(stretch[0], dt=dt_run)
    velocity_array = TimedArray(velocity[0], dt=dt_run)
    
    ia_recruited=aff_recruited*neuron_pop['Ia']
    # Dynamic response (Ia) depends on stretch and velocity
    ia_eq = '''
        is_ees = (i < aff_recruited): boolean
        rate = 10*hertz + 0.4*hertz*stretch_array(t) + 0.86*hertz*sign(velocity_array(t))*abs(velocity_array(t))**0.6 + ees_freq * int(is_ees) : Hz
    '''
    Ia = NeuronGroup(neuron_pop["Ia"], ia_eq, threshold='rand() < rate*dt',reset='v=Eleaky', refractory=T_refr)

    ii_recruited=aff_recruited*neuron_pop['II']
    # Static response (II) depends primarily on stretch
    ii_eq = '''
        is_ees = (i < aff_recruited) : boolean
        rate = 20*hertz + 3.375*hertz*stretch_array(t) + ees_freq * int(is_ees) : Hz
    '''
    II = NeuronGroup(neuron_pop["II"], ii_eq, threshold='rand() < rate*dt',reset='v=Eleaky', refractory=T_refr)
    
    # Create neuron model
    eqs = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = (ge + ge2) * (E_ex - v) : amp
    ge : siemens
    ge2 : siemens
    '''
 
    # Creating the interneuron and motoneuron groups
    Excitatory = NeuronGroup(
        neuron_pop["exc"], 
        eqs, 
        threshold=f"v > threshold_v", 
        reset="v = Eleaky", 
        method="exact"
    )
    Excitatory.v = Eleaky  # Set initial voltage

    Motoneuron = NeuronGroup(
        neuron_pop["motor"], 
        eqs, 
        threshold=f"v > threshold_v", 
        reset="v = Eleaky", 
        method="exact"
    )
    Motoneuron.v = Eleaky # Set initial voltage
    
    # Define synapse models
    synapse_models = {
        "II_Ex": """
        dx/dt = -x / tau_exc : siemens (clock-driven)
        ge_post = x : siemens (summed)
        w_: siemens # Synaptic weight
        """,
        
        "Ia_Motoneuron": """
        dy/dt = -y / tau_exc : siemens (clock-driven)
        ge_post = y : siemens (summed)
        w_: siemens # Synaptic weight
        """,
        
        "Ex_Motoneuron": """
        dz/dt = -z / tau_exc : siemens (clock-driven)
        ge2_post = z : siemens (summed)
        w_: siemens # Synaptic weight
        """
    }
    
    # Create and connect synapses
    II_Ex = Synapses(II, Excitatory, model=synapse_models["II_Ex"], on_pre='x += w_', method='exact')
    II_Ex.connect(p=p)
    II_Ex.w_ = w

    II_ees_Ex=Synapses(II, Excitatory, model=synapse_models["II_Ex"], on_pre='x += w_', method='exact')
    II_Ex.connect(p=p)
    II_Ex.w_ = w

    
    Ia_Motoneuron = Synapses(Ia, Motoneuron, model=synapse_models["Ia_Motoneuron"], on_pre='y += w_', method='exact')
    Ia_Motoneuron.connect(p=p)
    Ia_Motoneuron.w_ = w
    
    Ex_Motoneuron = Synapses(Excitatory, Motoneuron, model=synapse_models["Ex_Motoneuron"], on_pre='z += w_', method='exact')
    Ex_Motoneuron.connect(p=p)
    Ex_Motoneuron.w_ = w

    # Set up monitoring for main simulation
    mon_motor = SpikeMonitor(Motoneuron)
    
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)

    # Create and run main network
    net = Network()
    net.add([
        Ia, II, Excitatory, Motoneuron,
        II_Ex, Ia_Motoneuron, Ex_Motoneuron,
        mon_Ia, mon_II, mon_motor])

    if (ees_freq>0 and eff_recruited>0):
        ees_motoneuron=PoissonGroup(N=eff_recruited*neuron_pop['motor'], rates= ees_freq)
        mon_ees_motor=SpikeMonitor(ees_motoneuron)
        net.add([ees_motoneuron, mon_ees_motor])

    net.run(T)

    if (ees_freq>0 and eff_recruited>0):
        # Process motoneuron spikes by adding EES effect
        moto_spike_dict = process_motoneuron_spikes(
        neuron_pop, mon_motor.spike_trains(), mon_ees_motor.spike_trains(), T_refr
        )
    else:
        moto_spike_dict=mon_motor.spike_trains()
  
    # Return final results
    return [{
        "Ia": mon_Ia.spike_trains(), 
        "II": mon_II.spike_trains(), 
        "MN": moto_spike_dict
    }]
        
def merge_and_filter_spikes(natural_spikes: np.ndarray, ees_spikes: np.ndarray, T_refr: Quantity) -> np.ndarray:
    """
    Merge natural and EES-induced spikes, filtering based on refractory period.
    Natural spikes are preserved unless they violate the refractory period.
    
    Parameters:
    ----------
    natural_spikes : array-like
        Array of natural spike times
    ees_spikes : array-like
        Array of EES-induced spike times
    T_refr : Quantity
        Refractory period
        
    Returns:
    -------
    array
        Filtered spike times
    """
    # Handle empty arrays efficiently
    if len(natural_spikes) == 0 and len(ees_spikes) == 0:
        return np.array([], dtype=float)*second
    
    if len(natural_spikes) == 0:
        natural_spikes = np.array([], dtype=float)*second
    if len(ees_spikes) == 0:
        ees_spikes = np.array([], dtype=float)*second
    
    # Create structured array for better memory efficiency
    dtype = [('time', float), ('is_natural', bool)]
    spikes = np.empty(len(natural_spikes) + len(ees_spikes), dtype=dtype)
  
    # Fill the structured array
    spikes['time'][:len(natural_spikes)] = natural_spikes
    spikes['time'][len(natural_spikes):] = ees_spikes
    spikes['is_natural'][:len(natural_spikes)] = True
    spikes['is_natural'][len(natural_spikes):] = False
    # Sort by time
    spikes.sort(order='time')
    
    if len(spikes) == 0:
        return np.array([], dtype=float)
    
    # Pre-allocate result array for better performance (assume worst case size)
    final_spikes = np.zeros(len(spikes), dtype=float)
    final_count = 0
    
    # Add first spike
    final_spikes[0] = spikes[0]['time']*second
    final_count = 1
    last_spike_time = spikes[0]['time']*second
    
    # Process remaining spikes more efficiently
    for i in range(1, len(spikes)):
        current_time = spikes[i]['time']*second
        is_natural = spikes[i]['is_natural']

        if is_natural:
            if current_time - last_spike_time >= T_refr:
                final_spikes[final_count] = current_time
                final_count += 1
                last_spike_time = current_time
        else:  # EES spike
            if current_time - last_spike_time >= T_refr:
                final_spikes[final_count] = current_time
                final_count += 1
                last_spike_time = current_time

    # Return only the filled part of the array
    return final_spikes[:final_count]*second

  
 
def process_motoneuron_spikes(neuron_pop: Dict[str, int], motor_spikes: Dict, 
                            ees_spikes: Dict, 
                             T_refr: Quantity) -> Dict:
    """
    Process motoneuron spike trains, combining natural and EES-induced spikes.
    
    Parameters:
    ----------
    neuron_pop : dict
        Dictionary with neuron population sizes
    motor_spikes : dict
        Dictionary with natural motoneuron spike trains
    ees_spikes : dict
        Dictionary with EES-induced spikes
    n_poisson : dict
        Dictionary of recruited neurons
    T_refr : Quantity
        Refractory period
        
    Returns:
    -------
    dict
        Dictionary with processed motoneuron spike times
    """
    moto_spike_dict = {}
    
    for i in range(neuron_pop["motor"]):
        nat_spikes = motor_spikes[i] 
        
        # Get EES-induced spikes for this neuron if it's in the recruited population
        ees_i_spikes = np.array([])
        if i < len(ees_spikes):
            ees_i_spikes = ees_spikes[i]
        
        # Merge and filter spikes
        moto_spike_dict[i] = merge_and_filter_spikes(nat_spikes, ees_i_spikes, T_refr)
    
    return moto_spike_dict



