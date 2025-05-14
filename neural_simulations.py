from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional


def run_one_muscle_neuron_simulation(stretch_input, velocity_input, neuron_pop, connections, dt_run, T,
                                          spindle_model, seed_run, initial_potentials, 
                                          Eleaky, gL, Cm, E_ex, tau_e, threshold_v, T_refr,
                                          ees_freq, Ia_recruited, II_recruited, eff_recruited):
    """
    Run a simulation of flexor-extensor neuron dynamics.
    
    Parameters:
    ----------
    stretch : list of arrays
        Stretch inputs.
    velocity : list of arrays
        Velocity inputs.
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'II', 'MN', 'exc', 'inh').
    connections: dict
        Dictionnary with the weights and probability of connection all synapse of the network
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    T_refr : time
        Refractory period.
    Spindle_model: dict
        equations that relate afferent firing rate with stretch or joint
    initial_potentials : dict
        Initial membrane potentials for neuron groups.
    Eleaky : volt
        Leaky potential.
    gL : siemens
        Leak conductance.
    Cm : farad
        Membrane capacitance.
    E_ex : volt
        Excitatory reversal potential.
    E_inh : volt
        Inhibitory reversal potential.
    tau_e : time
        Excitatory time constant.
    tau_i : time
        Inhibitory time constant.
    threshold_v : volt
        Voltage threshold.
    ees_freq : hertz
        EES frequency.
    aff_recruited : int
        Number of afferent neurons recruited.
    eff_recruited : int
        Number of efferent neurons recruited.

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data for flexor and extensor pathways,
        a dictionary with final membrane potentials and post_synapstic current recorded in interesting neurons type.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()

    # Input arrays
    stretch_array = TimedArray(stretch_input[0], dt=dt_run)
    velocity_array = TimedArray(velocity_input[0], dt=dt_run)

    # Extract neuron counts from dictionary
    n_Ia = neuron_pop['Ia']
    n_MN = neuron_pop['MN']  

    # Afferent neuron equations
    equation_Ia = spindle_model['Ia']
    ia_eq = f'''
    is_ees = ( i < Ia_recruited): boolean
    stretch = stretch_array(t):1
    velocity = velocity_array(t):1
    rate = ({equation_Ia})*hertz + ees_freq * int(is_ees) : Hz
    '''
    Ia = NeuronGroup(n_Ia, ia_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    net.add([Ia])

    group_map = {
    "Ia": Ia[:],
    }
                                            
    #If II is specify in the spindle model, then the network contain II and excitatory neurons                                    
    if 'II' in spindle_model and 'II' in neuron_pop and 'exc' in neuron_pop:
      
        n_II = neuron_pop['II']
        n_exc = neuron_pop['exc']

        equation_II = spindle_model['II']
        ii_eq = f'''
        is_ees = (i < II_recruited) : boolean
        stretch = stretch_array(t):1
        velocity = velocity_array(t):1
        rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
        '''
        II = NeuronGroup(n_II, ii_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
        group_map["II"] = II[:]
  
        # LIF neuron equations
        ex_eq = '''
        dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
        Isyn = gII*(E_ex - v) :amp
        dgII/dt = -gII / tau_e : siemens
        '''
        exc = NeuronGroup(n_exc, ex_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', refractory=T_refr, method='euler')
        exc.v = initial_potentials['exc']
        group_map['exc'] = exc[:]
        net.add([II, exc])
        
        mn_eq = '''
        dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
        Isyn = gIa*(E_ex - v) + gexc*(E_ex-v) :amp
        dgIa/dt = -gIa / tau_e : siemens 
        dgexc/dt = -gexc / tau_e : siemens  
        '''
                                      
        MN = NeuronGroup(n_MN, mn_eq, threshold='v > threshold_v', 
                       reset='v = Eleaky', refractory=T_refr, method='euler')
        MN.v = initial_potentials['MN']
        net.add([MN])
        group_map["MN"] = MN[:] 

    else:  # Monosynaptic Reflex
                
        mn_eq = '''
        dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
        Isyn = gIa*(E_ex - v)  :amp
        dgIa/dt = -gIa / tau_e : siemens 
        '''
                                      
        MN = NeuronGroup(n_MN, mn_eq, threshold='v > threshold_v', 
                       reset='v = Eleaky', refractory=T_refr, method='euler')
                      
        MN.v = initial_potentials['MN']
        net.add([MN])
        group_map["MN"] = MN[:] 
    
    # Create synaptic connections
    synapses = {}
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
  
        syn = Synapses(pre, post, model="w : siemens", on_pre=f"g{pre_name.split('_')[0]}_post += w", method='exact')
        syn.connect(p=p)
        syn.w = np.clip(weight + 0.2 * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        net.add(syn)
        synapses[key] = syn
          
    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    if 'II' in spindle_model and 'II' in neuron_pop and 'exc' in neuron_pop:
        mon_II = SpikeMonitor(II)
        mon_exc = SpikeMonitor(exc)
        net.add([mon_II, mon_exc])
    mon_MN = SpikeMonitor(MN)
    
    mon_MN_state = StateMonitor(MN, ['Isyn'], n_MN/2)
    
    # Add all monitors to the network
    monitors = [
        mon_Ia, mon_MN, mon_MN_state
    ]
                 
    net.add(monitors)
    
    # Variables for EES monitors
    mon_ees_MN = None
                                            
    # Handle EES stimulation if enabled
    if ees_freq > 0 and eff_recruited > 0:
        ees_MN = PoissonGroup(N=eff_recruited, rates=ees_freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])

    # Run simulation
    net.run(T)
    
    # Extract motoneuron spikes
    MN_spikes = mon_MN.spike_trains()
 
    if ees_freq > 0 and eff_recruited > 0:
        ees_spikes = mon_ees_MN.spike_trains()
        before_MN_spikes = MN_spikes.copy()
        MN_spikes = process_motoneuron_spikes(
        neuron_pop, MN_spikes, ees_spikes, T_refr)
       
    # Count spiking neurons
    recruited_MN = sum(1 for spikes in MN_spikes.values() if len(spikes) > 0)
    print(f"Number of recruited motoneuron: {recruited_MN}/{n_MN}")


    # Final membrane potentials
    final_potentials = {
        'MN': MN.v[:]
    } 
    # Store state monitors for plotting
    state_monitors = [{
        'IPSP_MN': mon_MN_state.Isyn[0]/nA
    }]
    
    result = {
        "Ia": mon_Ia.spike_trains()
    }

    if 'II' in spindle_model and 'II' in neuron_pop and 'exc' in neuron_pop:
        result['II'] = mon_II.spike_trains()
        result['exc'] = mon_exc.spike_trains()
        final_potentials['exc'] = exc.v[:]
        
    if ees_freq > 0 and eff_recruited > 0:
        result["MN0"] = before_MN_spikes

    result["MN"] = MN_spikes

    return [result], final_potentials, state_monitors


def run_flexor_extensor_neuron_simulation(stretch_input, velocity_input, neuron_pop, connections, dt_run, T,
                                          spindle_model, seed_run, initial_potentials, 
                                          Eleaky, gL, Cm, E_ex, E_inh, tau_e, tau_i, threshold_v, T_refr,
                                          ees_freq, Ia_recruited, II_recruited, eff_recruited):
    """
    Run a simulation of flexor-extensor neuron dynamics.
    
    Parameters:
    ----------
    stretch : list of arrays
        Stretch inputs for flexor [0] and extensor [1].
    velocity : list of arrays
        Velocity inputs for flexor [0] and extensor [1].
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'II', 'MN', 'exc', 'inh').
    connections: dict
        Dictionnary with the weights and probability of connection all synapse of the network
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    T_refr : time
        Refractory period.
    initial_potentials : dict
        Initial membrane potentials for neuron groups.
    Eleaky : volt
        Leaky potential.
    gL : siemens
        Leak conductance.
    Cm : farad
        Membrane capacitance.
    E_ex : volt
        Excitatory reversal potential.
    E_inh : volt
        Inhibitory reversal potential.
    tau_e : time
        Excitatory time constant.
    tau_i : time
        Inhibitory time constant.
    threshold_v : volt
        Voltage threshold.
    ees_freq : hertz
        EES frequency.
    aff_recruited : int
        Number of afferent neurons recruited.
    eff_recruited : int
        Number of efferent neurons recruited.

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data for flexor and extensor pathways,
        a dictionary with final membrane potentials and post_synapstic current recorded in interesting neurons type.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()

    # Input arrays
    stretch_flexor_array = TimedArray(stretch_input[0], dt=dt_run)
    velocity_flexor_array = TimedArray(velocity_input[0], dt=dt_run)
    stretch_extensor_array = TimedArray(stretch_input[1], dt=dt_run)
    velocity_extensor_array = TimedArray(velocity_input[1], dt=dt_run)

    # Extract neuron counts from dictionary
    n_Ia = neuron_pop['Ia']
    n_II = neuron_pop['II']
    n_exc = neuron_pop['exc']
    n_inh = neuron_pop['inh']
    n_MN = neuron_pop['MN']  

    # Afferent neuron equations
    equation_Ia = spindle_model['Ia']
    ia_eq = f'''
    is_flexor = (i < n_Ia) : boolean
    stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    velocity = velocity_flexor_array(t) * int(is_flexor) + velocity_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < Ia_recruited) or (not is_flexor and i < n_Ia + Ia_recruited)) : boolean
    rate = ({equation_Ia})*hertz + ees_freq * int(is_ees) : Hz
    '''
    equation_II = spindle_model['II']
    ii_eq = f'''
    is_flexor = (i < n_II) : boolean
    stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < II_recruited) or (not is_flexor and i < n_II + II_recruited)) : boolean
    rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
    '''
    
    # Create afferent neurons
    Ia = NeuronGroup(2*n_Ia, ia_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II = NeuronGroup(2*n_II, ii_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    net.add([Ia, II])

    # LIF neuron equations
    ex_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn = gII*(E_ex - v) :amp
    dgII/dt = -gII / tau_e : siemens
    '''
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = gIa*(E_ex - v) + gexc*(E_ex-v) + gi*(E_inh - v) :amp
    dgIa/dt = -gIa / tau_e : siemens 
    dgexc/dt = -gexc / tau_e : siemens
    dgi/dt = (ginh-gi)/tau_i : siemens
    dginh/dt = -ginh/tau_i :siemens   
    '''
    inh_eq = '''
    dv/dt = (gL*(Eleaky - v)+Isyn ) / Cm : volt
    Isyn = gi*(E_inh - v) + gIa*(E_ex-v) + gII*(E_ex - v) :amp
    dgIa/dt = -gIa / tau_e : siemens 
    dgII/dt = -gII / tau_e : siemens
    dgi/dt = (ginh-gi)/tau_i : siemens
    dginh/dt = -ginh/tau_i :siemens                                           
    ''' 
  
    # Create neuron groups
    inh = NeuronGroup(2*n_inh, inh_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', refractory=T_refr, method='euler')
  
    exc = NeuronGroup(2*n_exc, ex_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', refractory=T_refr, method='euler')
                                         
    MN = NeuronGroup(2*n_MN, mn_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky', refractory=T_refr, method='euler')
                       
    # Initialize membrane potentials
    inh.v = initial_potentials['inh']
    exc.v = initial_potentials['exc']
    MN.v = initial_potentials['MN']

    # Add neuron groups to the network
    net.add([inh, exc, MN])
                                            
    group_map = {
        "Ia_flexor": Ia[:n_Ia],
        "Ia_extensor": Ia[n_Ia:],
        "II_flexor": II[:n_II],
        "II_extensor": II[n_II:],
        "exc_flexor": exc[:n_exc],
        "exc_extensor": exc[n_exc:],
        "inh_flexor": inh[:n_inh],
        "inh_extensor": inh[n_inh:],
        "MN_flexor": MN[:n_MN],
        "MN_extensor": MN[n_MN:],
    }
    
    # Create synaptic connections
    synapses = {}
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
  
        syn = Synapses(pre, post, model="w : siemens", on_pre=f"g{pre_name.split('_')[0]}_post += w", method='exact')
        syn.connect(p=p)
        syn.w = np.clip(weight + 0.2 * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        net.add(syn)
        synapses[key] = syn
          
    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)
    mon_exc = SpikeMonitor(exc)
    mon_inh = SpikeMonitor(inh)
    mon_MN = SpikeMonitor(MN)
    
    mon_inh_flexor = StateMonitor(inh, ['Isyn'], n_inh/2)
    mon_MN_flexor = StateMonitor(MN, ['Isyn'], n_MN/2)
    
    mon_inh_extensor = StateMonitor(inh, ['Isyn'], 3*n_inh/2)
    mon_MN_extensor = StateMonitor(MN, ['Isyn'], 3*n_MN/2)
    
    # Add all monitors to the network
    monitors = [
        mon_Ia, mon_II, mon_exc, mon_inh, mon_MN, 
        mon_inh_flexor, mon_MN_flexor, mon_inh_extensor, mon_MN_extensor
    ]
                 
    net.add(monitors)
    
    # Variables for EES monitors
    mon_ees_MN_flexor = None
    mon_ees_MN_extensor = None
                                            
    # Handle EES stimulation if enabled
    if ees_freq > 0 and eff_recruited > 0:
        ees_MN = PoissonGroup(N=2*eff_recruited, rates=ees_freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])

    # Run simulation
    net.run(T)
    
    # Extract motoneuron spikes
    MN_flexor_spikes = {i: mon_MN.spike_trains()[i] for i in range(n_MN)} 
    MN_extensor_spikes = {i%n_MN: mon_MN.spike_trains()[i] for i in range(n_MN, 2*n_MN)} 

    if ees_freq > 0 and eff_recruited > 0:
        ees_spikes = mon_ees_MN.spike_trains()
        before_MN_flexor_spikes = MN_flexor_spikes.copy()
        before_MN_extensor_spikes = MN_extensor_spikes.copy()
        MN_flexor_spikes = process_motoneuron_spikes(
            neuron_pop, MN_flexor_spikes, {i: ees_spikes[i] for i in range(eff_recruited)}, T_refr)
        MN_extensor_spikes = process_motoneuron_spikes(
            neuron_pop, MN_extensor_spikes, {i%eff_recruited: ees_spikes[i+eff_recruited] for i in range(eff_recruited)}, T_refr)
    
    # Count spiking neurons
    recruited_MN_flexor = sum(1 for spikes in MN_flexor_spikes.values() if len(spikes) > 0)
    print(f"Number of flexor recruited motoneuron: {recruited_MN_flexor}/{n_MN}")
    recruited_MN_extensor = sum(1 for spikes in MN_extensor_spikes.values() if len(spikes) > 0)
    print(f"Number of extensor recruited motoneuron: {recruited_MN_extensor}/{n_MN}")

    # Final membrane potentials
    final_potentials = {
        'inh': inh.v[:],
        'exc': exc.v[:],
        'MN': MN.v[:]
    } 
    # Store state monitors for plotting
    state_monitors = [{
            'IPSP_inh': mon_inh_flexor.Isyn[0]/nA,
            'IPSP_MN': mon_MN_flexor.Isyn[0]/nA
        },{
            'IPSP_inh': mon_inh_extensor.Isyn[0]/nA,
            'IPSP_MN': mon_MN_extensor.Isyn[0]/nA,
        }
    ]
    result_flexor = {
        "Ia": {i: mon_Ia.spike_trains()[i] for i in range(n_Ia)},
        "II": {i: mon_II.spike_trains()[i] for i in range(n_II)},
        "exc": {i: mon_exc.spike_trains()[i] for i in range(n_exc)},
        "inh": {i: mon_inh.spike_trains()[i] for i in range(n_inh)}
    }
    result_extensor = {
        "Ia": {i%n_Ia: mon_Ia.spike_trains()[i] for i in range(n_Ia, 2*n_Ia)},
        "II": {i%n_II: mon_II.spike_trains()[i] for i in range(n_II, 2*n_II)},
        "exc": {i%n_exc: mon_exc.spike_trains()[i] for i in range(n_exc, 2*n_exc)},
        "inh": {i%n_inh: mon_inh.spike_trains()[i] for i in range(n_inh, 2*n_inh)}
    }

    if ees_freq > 0 and eff_recruited > 0:
        result_flexor["MN0"] = before_MN_flexor_spikes
        result_extensor["MN0"] = before_MN_extensor_spikes

    result_flexor["MN"] = MN_flexor_spikes
    result_extensor["MN"] = MN_extensor_spikes

    return [result_flexor, result_extensor], final_potentials, state_monitors

        
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

  
def process_motoneuron_spikes(neuron_pop: Dict[str, int], MN_spikes: Dict, 
                            ees_spikes: Dict, 
                             T_refr: Quantity) -> Dict:
    """
    Process motoneuron spike trains, combining natural and EES-induced spikes.
    
    Parameters:
    ----------
    neuron_pop : dict
        Dictionary with neuron population sizes
    MN_spikes : dict
        Dictionary with natural motoneuron spike trains
    ees_spikes : dict
        Dictionary with EES-induced spikes
    T_refr : Quantity
        Refractory period
        
    Returns:
    -------
    dict
        Dictionary with processed motoneuron spike times
    """
    MN_spike_dict = {}
    
    for i, nat_spikes in MN_spikes.items():
        
        # Get EES-induced spikes for this neuron if it's in the recruited population
        ees_i_spikes = np.array([])
        if i < len(ees_spikes):
            ees_i_spikes = ees_spikes[i]
        
        # Merge and filter spikes
        MN_spike_dict[i] = merge_and_filter_spikes(nat_spikes, ees_i_spikes, T_refr)
    
    return MN_spike_dict
