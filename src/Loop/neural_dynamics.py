from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional
import time

def run_monosynaptic_simulation(stretch_input, stretch_velocity_input,  
                                neuron_pop, connections, dt_run, T, spindle_model, seed_run, 
                                initial_state_neurons, Eleaky, gL, Cm, E_ex, tau_e, threshold_v, T_refr,
                                ees_params=None):
    """
    Run a simulation of monosynaptic reflex pathway (Ia to MN only).
    
    Parameters:
    ----------
    stretch_input : list of arrays
        Stretch inputs.
    stretch_velocity_input : list of arrays
        Velocity inputs.
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'MN').
    connections: dict
        Dictionary with the weights and probability of connection all synapse of the network
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    T_refr : time
        Refractory period.
    spindle_model: dict
        Equations that relate afferent firing rate with stretch or joint
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
    tau_e : time
        Excitatory time constant.
    threshold_v : volt
        Voltage threshold.
    ees_params : dict, optional
        EES frequency and number of afferent/efferent neurons recruited for each neuron.

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data,
        a dictionary with final membrane potentials and post_synaptic current recorded in motoneurons.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()
    group_map = {}
    monitors = []
    
    # Create TimedArray inputs
    stretch_array = TimedArray(stretch_input[0], dt=dt_run)
    stretch_velocity_array = TimedArray(stretch_velocity_input[0], dt=dt_run)
  
    # Extract EES parameters
    Ia_recruited = 0
    freq = 0
    if ees_params is not None:
        freq = ees_params['frequency']
        Ia_recruited = ees_params['recruitment']['Ia']
    
    
    # Create Ia afferent neurons
    n_Ia = neuron_pop['Ia']
    equation = spindle_model['Ia']
    equation_baseline = """
        stretch = stretch_array(t): 1
        stretch_velocity = stretch_velocity_array(t): 1
        """

    if freq > 0 and Ia_recruited > 0:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline + f'''
            is_ees = (i < {Ia_recruited}): boolean
            rate = ({equation})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline + f'''
            rate = ({equation})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    Ia_spike_mon = SpikeMonitor(Ia_neurons)
    net.add([Ia_neurons, Ia_spike_mon])
    group_map['Ia'] = Ia_neurons
    monitors.append(Ia_spike_mon)
    
    # Create motoneurons (MN)
    n_MN = neuron_pop['MN']
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm: volt
    Isyn = gIa*(E_ex - v): amp
    dgIa/dt = -gIa / tau_e: siemens 
    '''
    
    MN = NeuronGroup(n_MN, mn_eq, 
                   threshold='v > threshold_v', 
                   reset='v = Eleaky', method='euler')
    
    MN.v = initial_state_neurons['MN']['v']
    MN.gIa=initial_state_neurons['MN']['gIa']
    
    spike_mon_MN = SpikeMonitor(MN)
    
    net.add([MN, spike_mon_MN])
    group_map['MN'] = MN
    monitors.append(spike_mon_MN)
    
    # Create synaptic connections
    synapses = {}
    
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
        
        # Extract the base name for the conductance update 
        pre_base = pre_name.split('_')[0]
        
        syn = Synapses(pre, post, 
                      model="w: siemens", 
                      on_pre=f"g{pre_base}_post += w", 
                      method='exact')
        
        syn.connect(p=p)
        noise=0.2
        syn.w = np.clip(weight + noise * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        syn.delay=np.clip(1*ms+0.25*ms*noise*randn(len(syn.delay)), 0*ms, np.inf*ms)
        net.add(syn)
        synapses[key] = syn
    
    # Create state monitor for motoneurons
    mon_MN_state = StateMonitor(MN, ['Isyn', 'v'], n_MN//2)
    monitors.append(mon_MN_state)
    net.add(mon_MN_state)
    
    # Handle EES stimulation if enabled for efferent neurons
    mon_ees_MN = None
    if ees_params is not None and freq > 0 and ees_params['recruitment'].get('MN', 0) > 0:
        eff_recruited = ees_params['recruitment'].get('MN')
        ees_MN = PoissonGroup(N=eff_recruited, rates=freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])
        monitors.append(mon_ees_MN)
    
    # Run simulation
    net.run(T)
    
    # Process results
    result = {}
    
    # Extract spike trains from monitors
    result['Ia'] = Ia_spike_mon.spike_trains()
    result['MN'] = spike_mon_MN.spike_trains()
    
    # Process EES-stimulated motoneuron spikes if applicable
    if mon_ees_MN:
        ees_spikes = mon_ees_MN.spike_trains()
        result["MN"] = process_motoneuron_spikes(
            neuron_pop, result["MN"], ees_spikes, T_refr)
    
    
    # Store state monitors for plotting
    state_monitors = [{
        'IPSP_MN': mon_MN_state.Isyn[0]/nA,
        'potential_MN': mon_MN_state.v[0]/mV
    }]
    #Store final state
    final_state_neurons={
      'MN':{'v' : MN.v[:],
      'gIa':MN.gIa[:]}
    }
                                  
    return [result], final_state_neurons, state_monitors


def run_disynaptic_simulation_with_ib(stretch_input, stretch_velocity_input, stretch_delay_input, normalized_force_input, 
                                     neuron_pop, connections, dt_run, T, spindle_model, seed_run, 
                                     initial_state_neurons, Eleaky, gL, Cm, E_ex, tau_e, E_inh, tau_i, threshold_v, T_refr,
                                     ees_params=None):
    """
    Run a simulation with Ia, II, and Ib pathways (disynaptic pathway with inhibitory component).
    
    Parameters:
    ----------
    stretch_input : list of arrays
        Stretch inputs.
    stretch_velocity_input : list of arrays
        Velocity inputs.
    stretch_delay_input : list of arrays
        Delayed stretch inputs.
    normalized_force_input : list of arrays
        Normalized force inputs for Ib afferents.
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'II','Ib', 'MN', 'exc', 'inhb').
    connections: dict
        Dictionary with the weights and probability of connection all synapse of the network
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    T_refr : time
        Refractory period.
    spindle_model: dict
        Equations that relate afferent firing rate with stretch or joint
    initial_state_neurons : dict
        Initial membrane potentials for neuron groups.
    Eleaky : volt
        Leaky potential.
    gL : siemens
        Leak conductance.
    Cm : farad
        Membrane capacitance.
    E_ex : volt
        Excitatory reversal potential.
    tau_e : time
        Excitatory time constant.
    E_inh : volt
        Inhibitory reversal potential.
    tau_i : time
        Inhibitory time constant.
    threshold_v : volt
        Voltage threshold.
    ees_params : dict, optional
        EES frequency and number of afferent/efferent neurons recruited for each neuron.

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data,
        a dictionary with final membrane potentials and post_synaptic current recorded in motoneurons.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()
    group_map = {}
    
    # Create TimedArray inputs
    stretch_array = TimedArray(stretch_input[0], dt=dt_run)
    stretch_velocity_array = TimedArray(stretch_velocity_input[0], dt=dt_run)
    force_array = TimedArray(normalized_force_input[0], dt=dt_run)
    
    # Extract EES parameters
    Ia_recruited = 0
    II_recruited = 0
    Ib_recruited = 0
    freq = 0
    if ees_params is not None:
        freq = ees_params['frequency']
        Ia_recruited = ees_params['recruitment']['Ia']
        II_recruited = ees_params['recruitment']['II']
        Ib_recruited = ees_params['recruitment']['Ib']
    
    # Create Ia afferent neurons
    n_Ia = neuron_pop['Ia']
    equation_Ia = spindle_model['Ia']

    equation_baseline_Ia = f"""
        stretch = stretch_array(t): 1
        stretch_velocity = stretch_velocity_array(t): 1
        """
    if freq > 0 and Ia_recruited > 0:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline_Ia + f'''
            is_ees = (i < {Ia_recruited}): boolean
            rate = ({equation_Ia})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline_Ia + f'''
            rate = ({equation_Ia})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    Ia_spike_mon = SpikeMonitor(Ia_neurons)
    net.add([Ia_neurons, Ia_spike_mon])
    group_map['Ia'] = Ia_neurons
    
    # Create II afferent neurons
    n_II = neuron_pop['II']
    equation_II = spindle_model['II']
      
    if "Ia_II_delta_delay" in spindle_model:
        stretch_delay_array = TimedArray(stretch_delay_input[0], dt=dt_run)
        equation_baseline_II = """
            stretch_delay=stretch_delay_array(t):1
            """
    else:
        equation_baseline_II = """
            stretch = stretch_array(t): 1
            """

    if freq > 0 and II_recruited > 0:
        II_neurons = NeuronGroup(
            n_II, 
            equation_baseline_II + f'''
            is_ees = (i < {II_recruited}): boolean
            rate = ({equation_II})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        II_neurons = NeuronGroup(
            n_II, 
            equation_baseline_II + f'''
            rate = ({equation_II})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    II_spike_mon = SpikeMonitor(II_neurons)
    net.add([II_neurons, II_spike_mon])
    group_map['II'] = II_neurons
    
    # Create Ib afferent neurons
    n_Ib = neuron_pop['Ib']
    equation_Ib = spindle_model['Ib']
    
    if freq > 0 and Ib_recruited > 0:
        Ib_neurons = NeuronGroup(
            n_Ib, 
            f'''
            force_normalized=force_array(t):1
            is_ees = (i < {Ib_recruited}): boolean
            rate = ({equation_Ib})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        Ib_neurons = NeuronGroup(
            n_Ib, 
            f'''
            force_normalized=force_array(t):1
            rate = ({equation_Ib})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    Ib_spike_mon = SpikeMonitor(Ib_neurons)
    net.add([Ib_neurons, Ib_spike_mon])
    group_map['Ib'] = Ib_neurons

    # Create inhibitory interneurons (for Ib pathway)
    n_inhb = neuron_pop['inhb']
    inhb_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm: volt
    Isyn = gIb*(E_ex - v): amp
    dgIb/dt = -gIb / tau_e: siemens
    '''
    inhb_neurons = NeuronGroup(n_inhb, inhb_eq, 
                        threshold='v > threshold_v', 
                        reset='v = Eleaky', method='euler')

    inhb_neurons.v = initial_state_neurons['inhb']['v']
    inhb_neurons.gIb = initial_state_neurons['inhb']['gIb']

    inhb_spike_mon = SpikeMonitor(inhb_neurons)
    net.add([inhb_neurons, inhb_spike_mon])
    group_map['inhb'] = inhb_neurons
    
    # Create excitatory interneurons (for II pathway)
    n_exc = neuron_pop['exc']
    exc_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm: volt
    Isyn = gII*(E_ex - v): amp
    dgII/dt = -gII / tau_e: siemens
    '''
    
    exc_neurons = NeuronGroup(n_exc, exc_eq, 
                            threshold='v > threshold_v', 
                            reset='v = Eleaky', method='euler')
    
    exc_neurons.v = initial_state_neurons['exc']['v']
    exc_neurons.gII = initial_state_neurons['exc']['gII']
    
    exc_spike_mon = SpikeMonitor(exc_neurons)
    net.add([exc_neurons, exc_spike_mon])
    group_map['exc'] = exc_neurons
    
    # Create motoneurons (MN) - with inhibitory input
    n_MN = neuron_pop['MN']
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm: volt
    Isyn = gIa*(E_ex - v) + gexc*(E_ex-v) + gi*(E_inh-v): amp
    dgIa/dt = -gIa / tau_e: siemens 
    dgexc/dt = -gexc / tau_e: siemens
    dgi/dt = (ginhb-gi)/tau_i : siemens
    dginhb/dt = -ginhb / tau_i : siemens
    '''
    
    MN = NeuronGroup(n_MN, mn_eq, 
                   threshold='v > threshold_v', 
                   reset='v = Eleaky', method='euler')
    
    MN.v = initial_state_neurons['MN']['v']
    MN.gIa = initial_state_neurons['MN']['gIa']
    MN.gexc = initial_state_neurons['MN']['gexc']
    MN.gi = initial_state_neurons['MN']['gi']
    MN.ginhb = initial_state_neurons['MN']['ginhb']
                           
    spike_mon_MN = SpikeMonitor(MN)
    net.add([MN, spike_mon_MN])
    group_map['MN'] = MN
    
    # Create synaptic connections
    synapses = {}
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
        
        pre_base = pre_name.split('_')[0]
        
        syn = Synapses(pre, post, 
                      model="w: siemens", 
                      on_pre=f"g{pre_base}_post += w", 
                      method='exact')
        
        syn.connect(p=p)
        noise = 0.2
        syn.w = np.clip(weight + noise * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        syn.delay = np.clip(1*ms + 0.25*ms*noise*randn(len(syn.delay)), 0*ms, np.inf*ms)
        net.add(syn)
        synapses[key] = syn
    
    # Create state monitor for motoneurons
    mon_MN_state = StateMonitor(MN, ['Isyn', 'v'], n_MN//2)
    net.add(mon_MN_state)
    
    # Handle EES stimulation if enabled for efferent neurons
    mon_ees_MN = None
    if ees_params is not None and freq > 0 and ees_params['recruitment'].get('MN', 0) > 0:
        eff_recruited = ees_params['recruitment'].get('MN')
        ees_MN = PoissonGroup(N=eff_recruited, rates=freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])
    
    # Run simulation
    net.run(T)
    
    # Prepare final state
    final_state_neurons = {
        'MN': {
            'v': MN.v[:],
            'gIa': MN.gIa[:],
            'gexc': MN.gexc[:],
            'ginhb': MN.ginhb[:],
            'gi': MN.gi[:]
        },
        'exc': {
            'v': exc_neurons.v[:],
            'gII': exc_neurons.gII[:]
        },
        'inhb': {
            'v': inhb_neurons.v[:],
            'gIb': inhb_neurons.gIb[:]        
        }
    }
    
    # Process results
    result = {}
    result['Ia'] = Ia_spike_mon.spike_trains()
    result['II'] = II_spike_mon.spike_trains()
    result['Ib'] = Ib_spike_mon.spike_trains()
    result['exc'] = exc_spike_mon.spike_trains()
    result['inhb'] = inhb_spike_mon.spike_trains()
    result['MN'] = spike_mon_MN.spike_trains()
    
    # Process EES-stimulated motoneuron spikes if applicable
    if mon_ees_MN:
        ees_spikes = mon_ees_MN.spike_trains()
        result["MN"] = process_motoneuron_spikes(
            neuron_pop, result["MN"], ees_spikes, T_refr)
    
    
    # Store state monitors for plotting
    state_monitors = [{
        'IPSP_MN': mon_MN_state.Isyn[0]/nA,
        'potential_MN': mon_MN_state.v[0]/mV
    }]
                           
    return [result], final_state_neurons, state_monitors


def run_disynaptic_simulation(stretch_input, stretch_velocity_input, stretch_delay_input, 
                                        neuron_pop, connections, dt_run, T, spindle_model, seed_run, 
                                        initial_state_neurons, Eleaky, gL, Cm, E_ex, tau_e, threshold_v, T_refr,
                                        ees_params=None):
    """
    Run a simulation with Ia and II pathways only (disynaptic pathway without inhibitory component).
    
    Parameters:
    ----------
    stretch_input : list of arrays
        Stretch inputs.
    stretch_velocity_input : list of arrays
        Velocity inputs.
    stretch_delay_input : list of arrays
        Delayed stretch inputs.
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'II', 'MN', 'exc').
    connections: dict
        Dictionary with the weights and probability of connection all synapse of the network
    dt_run : time
        Simulation time step.
    T : time
        Total simulation time.
    T_refr : time
        Refractory period.
    spindle_model: dict
        Equations that relate afferent firing rate with stretch or joint
    initial_state_neurons : dict
        Initial membrane potentials for neuron groups.
    Eleaky : volt
        Leaky potential.
    gL : siemens
        Leak conductance.
    Cm : farad
        Membrane capacitance.
    E_ex : volt
        Excitatory reversal potential.
    tau_e : time
        Excitatory time constant.
    threshold_v : volt
        Voltage threshold.
    ees_params : dict, optional
        EES frequency and number of afferent/efferent neurons recruited for each neuron.

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data,
        a dictionary with final membrane potentials and post_synaptic current recorded in motoneurons.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()
    group_map = {}
    
    # Create TimedArray inputs
    stretch_array = TimedArray(stretch_input[0], dt=dt_run)
    stretch_velocity_array = TimedArray(stretch_velocity_input[0], dt=dt_run)
    
    # Extract EES parameters
    Ia_recruited = 0
    II_recruited = 0
    freq = 0
    if ees_params is not None:
        freq = ees_params['frequency']
        Ia_recruited = ees_params['recruitment']['Ia']
        II_recruited = ees_params['recruitment']['II']
    
    # Create Ia afferent neurons
    n_Ia = neuron_pop['Ia']
    equation_Ia = spindle_model['Ia']

    equation_baseline_Ia = f"""
        stretch = stretch_array(t): 1
        stretch_velocity = stretch_velocity_array(t): 1
        """
    if freq > 0 and Ia_recruited > 0:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline_Ia + f'''
            is_ees = (i < {Ia_recruited}): boolean
            rate = ({equation_Ia})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        Ia_neurons = NeuronGroup(
            n_Ia, 
            equation_baseline_Ia + f'''
            rate = ({equation_Ia})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    Ia_spike_mon = SpikeMonitor(Ia_neurons)
    net.add([Ia_neurons, Ia_spike_mon])
    group_map['Ia'] = Ia_neurons
    
    # Create II afferent neurons
    n_II = neuron_pop['II']
    equation_II = spindle_model['II']
      
    if "Ia_II_delta_delay" in spindle_model:
        stretch_delay_array = TimedArray(stretch_delay_input[0], dt=dt_run)
        equation_baseline_II = """
            stretch_delay=stretch_delay_array(t):1
            """
    else:
        equation_baseline_II = """
            stretch = stretch_array(t): 1
            """

    if freq > 0 and II_recruited > 0:
        II_neurons = NeuronGroup(
            n_II, 
            equation_baseline_II + f'''
            is_ees = (i < {II_recruited}): boolean
            rate = ({equation_II})*hertz + freq * int(is_ees): Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    else:
        II_neurons = NeuronGroup(
            n_II, 
            equation_baseline_II + f'''
            rate = ({equation_II})*hertz: Hz
            ''', 
            threshold='rand() < rate*dt', 
            refractory=T_refr, 
            method='euler'
        )
    
    II_spike_mon = SpikeMonitor(II_neurons)
    net.add([II_neurons, II_spike_mon])
    group_map['II'] = II_neurons
    
    # Create excitatory interneurons (for II pathway)
    n_exc = neuron_pop['exc']
    exc_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm: volt
    Isyn = gII*(E_ex - v): amp
    dgII/dt = -gII / tau_e: siemens
    '''
    
    exc_neurons = NeuronGroup(n_exc, exc_eq, 
                            threshold='v > threshold_v', 
                            reset='v = Eleaky', method='euler')
    
    exc_neurons.v = initial_state_neurons['exc']['v']
    exc_neurons.gII = initial_state_neurons['exc']['gII']
    
    exc_spike_mon = SpikeMonitor(exc_neurons)
    net.add([exc_neurons, exc_spike_mon])
    group_map['exc'] = exc_neurons
    
    # Create motoneurons (MN) - without inhibitory input
    n_MN = neuron_pop['MN']
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm: volt
    Isyn = gIa*(E_ex - v) + gexc*(E_ex-v): amp
    dgIa/dt = -gIa / tau_e: siemens 
    dgexc/dt = -gexc / tau_e: siemens
    '''
    
    MN = NeuronGroup(n_MN, mn_eq, 
                   threshold='v > threshold_v', 
                   reset='v = Eleaky', method='euler')
    
    MN.v = initial_state_neurons['MN']['v']
    MN.gIa = initial_state_neurons['MN']['gIa']
    MN.gexc = initial_state_neurons['MN']['gexc']
                           
    spike_mon_MN = SpikeMonitor(MN)
    net.add([MN, spike_mon_MN])
    group_map['MN'] = MN
    
    # Create synaptic connections
    synapses = {}
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
        
        pre_base = pre_name.split('_')[0]
        
        syn = Synapses(pre, post, 
                      model="w: siemens", 
                      on_pre=f"g{pre_base}_post += w", 
                      method='exact')
        
        syn.connect(p=p)
        noise = 0.2
        syn.w = np.clip(weight + noise * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        syn.delay = np.clip(1*ms + 0.25*ms*noise*randn(len(syn.delay)), 0*ms, np.inf*ms)
        net.add(syn)
        synapses[key] = syn
    
    # Create state monitor for motoneurons
    mon_MN_state = StateMonitor(MN, ['Isyn', 'v'], n_MN//2)
    net.add(mon_MN_state)
    
    # Handle EES stimulation if enabled for efferent neurons
    mon_ees_MN = None
    if ees_params is not None and freq > 0 and ees_params['recruitment'].get('MN', 0) > 0:
        eff_recruited = ees_params['recruitment'].get('MN')
        ees_MN = PoissonGroup(N=eff_recruited, rates=freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])
    
    # Run simulation
    net.run(T)
    
    # Store final state for MNs and interneurons
    final_state_neurons = {
        'MN': {
            'v': MN.v[:],
            'gIa': MN.gIa[:],
            'gexc': MN.gexc[:]
        },
        'exc': {
            'v': exc_neurons.v[:],
            'gII': exc_neurons.gII[:]
        }
    }

    # Collect spike trains into result dictionary
    result = {
        'Ia': Ia_spike_mon.spike_trains(),
        'II': II_spike_mon.spike_trains(),
        'exc': exc_spike_mon.spike_trains(),
        'MN': spike_mon_MN.spike_trains()
    }

    # Optionally process MN spikes if EES was applied
    if mon_ees_MN is not None:
        ees_spikes = mon_ees_MN.spike_trains()
        result['MN'] = process_motoneuron_spikes(
            neuron_pop, result['MN'], ees_spikes, T_refr
        )


    # Prepare state monitor output for plotting
    state_monitors = [{
        'IPSP_MN': mon_MN_state.Isyn[0] / nA,
        'potential_MN': mon_MN_state.v[0] / mV
    }]

    return [result], final_state_neurons, state_monitors
    
def run_flexor_extensor_neuron_simulation(stretch_input, stretch_velocity_input, stretch_delay_input, neuron_pop, connections, dt_run, T,
                                          spindle_model, seed_run, initial_state_neurons, 
                                          Eleaky, gL, Cm, E_ex, E_inh, tau_e, tau_i, threshold_v, T_refr,
                                          ees_params):
    """
    Run a simulation of flexor-extensor neuron dynamics.
    
    Parameters:
    ----------
    stretch_input : list of arrays
        Stretch inputs for flexor [0] and extensor [1].
    stretch_velocity_input : list of arrays
        Velocity inputs for flexor [0] and extensor [1].
    stretch_delay_input: list of arrays
        Stretch(t-delay) inputs for flexor [0] and extensor [1], with delay defined in the spindle model.
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
    ees_params : dict
        EES frequency
        and afferents recruitment for flexor and extensor

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data for flexor and extensor pathways,
        a dictionary with final membrane potentials and post_synapstic current recorded in interesting neurons type.
    """


    #start_build = time.time()

    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()

    # Input arrays
    stretch_flexor_array = TimedArray(stretch_input[0], dt=dt_run)
    velocity_flexor_array = TimedArray(stretch_velocity_input[0], dt=dt_run) 
    stretch_extensor_array = TimedArray(stretch_input[1], dt=dt_run)
    velocity_extensor_array = TimedArray(stretch_velocity_input[1], dt=dt_run)
                                      
      
    # Extract neuron counts from dictionary
    n_Ia_flexor = neuron_pop['Ia_flexor']
    n_II_flexor = neuron_pop['II_flexor']
    n_exc_flexor = neuron_pop['exc_flexor']
    n_inh_flexor = neuron_pop['inh_flexor']
    n_MN_flexor = neuron_pop['MN_flexor']
    n_Ia_extensor = neuron_pop['Ia_extensor']
    n_II_extensor = neuron_pop['II_extensor']
    n_exc_extensor = neuron_pop['exc_extensor']
    n_inh_extensor = neuron_pop['inh_extensor']
    n_MN_extensor = neuron_pop['MN_extensor']  

    #Extract EES_Params:
    if ees_params is not None:
        ees_freq=ees_params['frequency']
        Ia_flexor_recruited=ees_params['recruitment']['Ia_flexor']
        II_flexor_recruited=ees_params['recruitment']['II_flexor']
        MN_flexor_recruited=ees_params['recruitment']['MN_flexor']
        Ia_extensor_recruited=ees_params['recruitment']['Ia_extensor']
        II_extensor_recruited=ees_params['recruitment']['II_extensor']
        MN_extensor_recruited=ees_params['recruitment']['MN_extensor']
    else:
        ees_freq=0*hertz
        Ia_flexor_recruited=0
        II_flexor_recruited=0
        MN_flexor_recruited=0
        Ia_extensor_recruited=0
        II_extensor_recruited=0
        MN_extensor_recruited=0


    # Afferent neuron equations
    equation_Ia = spindle_model['Ia']
    ia_eq = f'''
    is_flexor = (i < n_Ia_flexor) : boolean
    stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    stretch_velocity = velocity_flexor_array(t) * int(is_flexor) + velocity_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < Ia_flexor_recruited) or (not is_flexor and i < n_Ia_flexor + Ia_extensor_recruited)) : boolean
    rate = ({equation_Ia})*hertz + ees_freq * int(is_ees) : Hz
    '''
                                            
    equation_II = spindle_model['II']
    if 'Ia_II_delta_delay' in spindle_model:
        stretch_delay_flexor_array = TimedArray(stretch_delay_input[0], dt=dt_run)
        stretch_delay_extensor_array = TimedArray(stretch_delay_input[1], dt=dt_run)
        ii_eq = f'''
        is_flexor = (i < n_II_flexor) : boolean
        stretch_delay = stretch_delay_flexor_array(t) * int(is_flexor) + stretch_delay_extensor_array(t) * int(not is_flexor) : 1
        is_ees = ((is_flexor and i < II_flexor_recruited) or (not is_flexor and i < n_II_flexor + II_extensor_recruited)) : boolean
        rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
        '''
    else:
        ii_eq = f'''
        is_flexor = (i < n_II_flexor) : boolean
        stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
        is_ees = ((is_flexor and i < II_flexor_recruited) or (not is_flexor and i < n_II_flexor + II_extensor_recruited)) : boolean  
        rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
        '''
    
    
    # Create afferent neurons
    Ia = NeuronGroup(n_Ia_flexor+ n_Ia_extensor, ia_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II = NeuronGroup(n_II_flexor+ n_II_extensor, ii_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
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
    inh = NeuronGroup(n_inh_flexor+ n_inh_extensor, inh_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', method='euler')
  
    exc = NeuronGroup(n_exc_flexor+n_exc_extensor, ex_eq, threshold='v > threshold_v', 
                      reset='v = Eleaky', method='euler')
                                         
    MN = NeuronGroup(n_MN_flexor+n_MN_extensor, mn_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky', method='euler')
                       
    # Initialize membrane potentials
    inh.v = initial_state_neurons['inh']['v']
    exc.v = initial_state_neurons['exc']['v']
    MN.v = initial_state_neurons['MN']['v']
                                            
    #Initialize conductances
    inh.gIa = initial_state_neurons['inh']['gIa']
    inh.gII = initial_state_neurons['inh']['gII']
    inh.gi = initial_state_neurons['inh']['gi']
    inh.ginh = initial_state_neurons['inh']['ginh']
    exc.gII = initial_state_neurons['exc']['gII']
    MN.gIa = initial_state_neurons['MN']['gIa']
    MN.gexc = initial_state_neurons['MN']['gexc']
    MN.gi = initial_state_neurons['MN']['gi']
    MN.ginh = initial_state_neurons['MN']['ginh']

    # Add neuron groups to the network
    net.add([inh, exc, MN])
                                            
    group_map = {
        "Ia_flexor": Ia[:n_Ia_flexor],
        "Ia_extensor": Ia[n_Ia_flexor:],
        "II_flexor": II[:n_II_flexor],
        "II_extensor": II[n_II_flexor:],
        "exc_flexor": exc[:n_exc_flexor],
        "exc_extensor": exc[n_exc_flexor:],
        "inh_flexor": inh[:n_inh_flexor],
        "inh_extensor": inh[n_inh_flexor:],
        "MN_flexor": MN[:n_MN_flexor],
        "MN_extensor": MN[n_MN_flexor:],
    }
    
    # Create synaptic connections
    synapses = {}
    Ia_II_delta_delay=spindle_model["Ia_II_delta_delay"]
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
  
        syn = Synapses(pre, post, model="w : siemens", on_pre=f"g{pre_name.split('_')[0]}_post += w", method='exact')
        syn.connect(p=p)
        noise=0.2
        syn.w = np.clip(weight + noise * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        syn.delay=np.clip(1*ms+0.25*ms*noise*randn(len(syn.delay)), 0*ms, np.inf*ms)
        net.add(syn)
        synapses[key] = syn
          
    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)
    mon_exc = SpikeMonitor(exc)
    mon_inh = SpikeMonitor(inh)
    mon_MN = SpikeMonitor(MN)
    
    mon_inh_flexor = StateMonitor(inh, ['Isyn'], n_inh_flexor/2)
    mon_MN_flexor = StateMonitor(MN, ['Isyn'], n_MN_flexor/2)
    
    mon_inh_extensor = StateMonitor(inh, ['Isyn'], n_inh_flexor+ n_inh_extensor/2)
    mon_MN_extensor = StateMonitor(MN, ['Isyn'], n_MN_flexor+ n_MN_extensor/2)
    
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
    if ees_freq > 0 :
        ees_MN = PoissonGroup(N=MN_flexor_recruited+MN_extensor_recruited, rates=ees_freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])

    #end_build = time.time()

    #start_run=time.time()
    # Run simulation
    net.run(T)
    
    #end_run = time.time()

    #print(f"Network build time: {end_build - start_build:.2f} s")
    #print(f"Simulation run time: {end_run - start_run:.2f} s")
    #start_postprocessing = time.time()

    # Cache spike trains once
    all_mn_spikes = mon_MN.spike_trains()
    all_ees_spikes = mon_ees_MN.spike_trains() if ees_freq > 0 else None
    all_Ia_spikes = mon_Ia.spike_trains()
    all_II_spikes = mon_II.spike_trains()
    all_exc_spikes = mon_exc.spike_trains()
    all_inh_spikes = mon_inh.spike_trains()

    # Extract motoneuron spikes
    MN_flexor_spikes = {i: all_mn_spikes[i] for i in range(n_MN_flexor)}
    MN_extensor_spikes = {i: all_mn_spikes[i + n_MN_flexor] for i in range(n_MN_extensor)}

    # EES correction if applicable
    if ees_freq > 0:
        if MN_flexor_recruited > 0:
            MN_flexor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_flexor_spikes,
                {i: all_ees_spikes[i] for i in range(MN_flexor_recruited)},
                T_refr
            )

            MN_extensor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_extensor_spikes,
                {
                    i % MN_flexor_recruited: all_ees_spikes[i + MN_flexor_recruited]
                    for i in range(MN_extensor_recruited)
                },
                T_refr
            )

        elif MN_extensor_recruited > 0:
            MN_extensor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_extensor_spikes,
                {i: all_ees_spikes[i] for i in range(MN_extensor_recruited)},
                T_refr
            )


    # Save neuron states
    final_state_neurons = {
        'inh': {
            'v': inh.v[:],
            'gIa': inh.gIa[:],
            'gII': inh.gII[:],
            'gi': inh.gi[:],
            'ginh': inh.ginh[:]
        },
        'exc': {
            'v': exc.v[:],
            'gII': exc.gII[:],
        },
        'MN': {
            'v': MN.v[:],
            'gIa': MN.gIa[:],
            'gexc': MN.gexc[:],
            'gi': MN.gi[:],
            'ginh': MN.ginh[:]
        }
    }

    # Store state monitors for plotting
    state_monitors = [
        {
            'IPSP_inh': mon_inh_flexor.Isyn[0] / nA,
            'IPSP_MN': mon_MN_flexor.Isyn[0] / nA
        },
        {
            'IPSP_inh': mon_inh_extensor.Isyn[0] / nA,
            'IPSP_MN': mon_MN_extensor.Isyn[0] / nA,
        }
    ]

    # Build results
    result_flexor = {
        "Ia": {i: all_Ia_spikes[i] for i in range(n_Ia_flexor)},
        "II": {i: all_II_spikes[i] for i in range(n_II_flexor)},
        "exc": {i: all_exc_spikes[i] for i in range(n_exc_flexor)},
        "inh": {i: all_inh_spikes[i] for i in range(n_inh_flexor)},
        "MN": MN_flexor_spikes
    }

    result_extensor = {
        "Ia": {i % n_Ia_flexor: all_Ia_spikes[i] for i in range(n_Ia_flexor, n_Ia_flexor + n_Ia_extensor)},
        "II": {i % n_II_flexor: all_II_spikes[i] for i in range(n_II_flexor, n_II_flexor + n_II_extensor)},
        "exc": {i % n_exc_flexor: all_exc_spikes[i] for i in range(n_exc_flexor, n_exc_flexor + n_exc_extensor)},
        "inh": {i % n_inh_flexor: all_inh_spikes[i] for i in range(n_inh_flexor, n_inh_flexor + n_inh_extensor)},
        "MN": MN_extensor_spikes
    }

    #end_postprocessing = time.time()
    #print(f"PostProcessing run time: {end_postprocessing - start_postprocessing:.2f} s")

   
    return [result_flexor, result_extensor], final_state_neurons, state_monitors



def run_spinal_circuit_with_Ib(stretch_input, stretch_velocity_input,stretch_delay_input, normalized_force_input, neuron_pop, connections, dt_run, T,
                                         spindle_model, seed_run, initial_state_neurons, 
                                         Eleaky, gL, Cm, E_ex, E_inh, tau_e, tau_i, threshold_v, T_refr,
                                         ees_params):
    """
    Run a simulation of complete spinal reflex network with Ia, Ib, II afferents and MN, RC, IA, IB, IN, EX neurons.
    Implements mono-, di-, and trisynaptic pathways as specified in the network architecture.
    
    Parameters:
    ----------
    stretch_input : list of arrays
        Stretch inputs for flexor [0] and extensor [1].
    stretch_velocity_input : list of arrays
        Velocity inputs for flexor [0] and extensor [1].
    normalized_force_input : list of arrays
        Force generated by the flexor [0] and extensor [1] divded by the total force.
    neuron_pop : dict
        Dictionary with counts of different neuron populations ('Ia', 'Ib', 'II', 'MN', 'RC', 'IA', 'IB', 'IN', 'EX').
    connections: dict
        Dictionary with the weights and probability of connection for all synapses of the network
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
    ees_params : dict
        EES frequency and afferents recruitment for flexor and extensor

    Returns:
    -------
    tuple
        Tuple containing a list of dictionaries with spike train data for flexor and extensor pathways,
        a dictionary with final membrane potentials and post_synaptic current recorded in key neurons.
    """
    # Set up random seeds for reproducibility
    np.random.seed(seed_run)
    seed(seed_run)
    defaultclock.dt = dt_run

    net = Network()

    # Input arrays
    stretch_flexor_array = TimedArray(stretch_input[0], dt=dt_run)
    velocity_flexor_array = TimedArray(stretch_velocity_input[0], dt=dt_run)
    force_flexor_array=TimedArray(normalized_force_input[0], dt=dt_run)
    stretch_extensor_array = TimedArray(stretch_input[1], dt=dt_run)
    velocity_extensor_array = TimedArray(stretch_velocity_input[1], dt=dt_run)
    force_extensor_array=TimedArray(normalized_force_input[1], dt=dt_run)
                                           
    # Extract neuron counts from dictionary
    n_Ia_flexor = neuron_pop['Ia_flexor']
    n_Ib_flexor = neuron_pop['Ib_flexor']
    n_II_flexor = neuron_pop['II_flexor']
    n_MN_flexor = neuron_pop['MN_flexor']
    n_inh_flexor = neuron_pop['inh_flexor']
    n_inhb_flexor = neuron_pop['inhb_flexor']
    n_exc_flexor = neuron_pop['exc_flexor']
    
    n_Ia_extensor = neuron_pop['Ia_extensor']
    n_Ib_extensor = neuron_pop['Ib_extensor']
    n_II_extensor = neuron_pop['II_extensor']
    n_MN_extensor = neuron_pop['MN_extensor']
    n_inh_extensor = neuron_pop['inh_extensor']
    n_inhb_extensor = neuron_pop['inhb_extensor']
    n_exc_extensor = neuron_pop['exc_extensor']

    # Extract EES parameters
    if ees_params is not None:
        ees_freq = ees_params['frequency']
        Ia_flexor_recruited = ees_params['recruitment']['Ia_flexor']
        Ib_flexor_recruited = ees_params['recruitment']['Ib_flexor']
        II_flexor_recruited = ees_params['recruitment']['II_flexor']
        MN_flexor_recruited = ees_params['recruitment']['MN_flexor']
        Ia_extensor_recruited = ees_params['recruitment']['Ia_extensor']
        Ib_extensor_recruited = ees_params['recruitment']['Ib_extensor']
        II_extensor_recruited = ees_params['recruitment']['II_extensor']
        MN_extensor_recruited = ees_params['recruitment']['MN_extensor']
    else:
        ees_freq = 0*hertz
        Ia_flexor_recruited = Ib_flexor_recruited = II_flexor_recruited = MN_flexor_recruited = 0
        Ia_extensor_recruited = Ib_extensor_recruited = II_extensor_recruited = MN_extensor_recruited = 0

    # Afferent neuron equations
    equation_Ia = spindle_model['Ia']
    ia_eq = f'''
    is_flexor = (i < n_Ia_flexor) : boolean
    stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    stretch_velocity = velocity_flexor_array(t) * int(is_flexor) + velocity_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < Ia_flexor_recruited) or (not is_flexor and i < n_Ia_flexor + Ia_extensor_recruited)) : boolean
    rate = ({equation_Ia})*hertz + ees_freq * int(is_ees) : Hz
    '''
    
    equation_Ib = spindle_model['Ib']  # Assumes Ib model is provided in spindle_model
    ib_eq = f'''
    is_flexor = (i < n_Ib_flexor) : boolean
    force_normalized = force_flexor_array(t) * int(is_flexor) + force_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < Ib_flexor_recruited) or (not is_flexor and i < n_Ib_flexor + Ib_extensor_recruited)) : boolean
    rate = ({equation_Ib})*hertz + ees_freq * int(is_ees) : Hz
    '''
                                           
    equation_II = spindle_model['II']
    if 'Ia_II_delta_delay' in spindle_model:
        stretch_delay_flexor_array = TimedArray(stretch_delay_input[0], dt=dt_run)
        stretch_delay_extensor_array = TimedArray(stretch_delay_input[1], dt=dt_run)
        ii_eq = f'''
        is_flexor = (i < n_II_flexor) : boolean
        stretch_delay = stretch_delay_flexor_array(t) * int(is_flexor) + stretch_delay_extensor_array(t) * int(not is_flexor) : 1
        is_ees = ((is_flexor and i < II_flexor_recruited) or (not is_flexor and i < n_II_flexor + II_extensor_recruited)) : boolean
        rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
        '''
    else:
        ii_eq = f'''
        is_flexor = (i < n_II_flexor) : boolean
        stretch = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
        is_ees = ((is_flexor and i < II_flexor_recruited) or (not is_flexor and i < n_II_flexor + II_extensor_recruited)) : boolean  
        rate = ({equation_II})*hertz + ees_freq * int(is_ees) : Hz
        '''

    
    # Create afferent neurons
    Ia = NeuronGroup(n_Ia_flexor + n_Ia_extensor, ia_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    Ib = NeuronGroup(n_Ib_flexor + n_Ib_extensor, ib_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II = NeuronGroup(n_II_flexor + n_II_extensor, ii_eq, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    net.add([Ia, Ib, II])

    # LIF neuron equations for different neuron types
    
    # Motoneuron (MN) 
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = gIa*(E_ex - v) + gexc*(E_ex - v)+ gi1*(E_inh - v)  + gi2*(E_inh - v) : amp
    dgIa/dt = -gIa / tau_e : siemens 
    dgexc/dt = -gexc / tau_e : siemens
    dgi1/dt = (ginh-gi1)/tau_i : siemens
    dginh/dt = -ginh / tau_i : siemens 
    dgi2/dt = (ginhb-gi2)/tau_i : siemens
    dginhb/dt = -ginhb / tau_i : siemens
    '''
                                           
    inh_eq = '''
    dv/dt = (gL*(Eleaky - v)+Isyn ) / Cm : volt
    Isyn = gi*(E_inh - v) + gIa*(E_ex-v) + gII*(E_ex - v) :amp
    dgIa/dt = -gIa / tau_e : siemens 
    dgII/dt = -gII / tau_e : siemens
    dgi/dt = (ginh-gi)/tau_i : siemens
    dginh/dt = -ginh/tau_i :siemens                                           
    '''

    # inhb interneuron - inhibitory
    inhb_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = gIa*(E_ex - v) + gIb*(E_ex - v) + gIa*(E_ex - v) : amp
    dgIa/dt = -gIa / tau_e : siemens
    dgIb/dt = -gIb / tau_e : siemens
    '''
    
    # exc interneuron - excitatory
    ex_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn) / Cm : volt
    Isyn = gII*(E_ex - v) : amp
    dgII/dt = -gII / tau_e : siemens
    '''

    # Create neuron groups
    MN = NeuronGroup(n_MN_flexor + n_MN_extensor, mn_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky', method='euler')
    inh = NeuronGroup(n_inh_flexor + n_inh_extensor, inh_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky',  method='euler')
    inhb = NeuronGroup(n_inhb_flexor + n_inhb_extensor, inhb_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky', method='euler')
    exc = NeuronGroup(n_exc_flexor + n_exc_extensor, ex_eq, threshold='v > threshold_v', 
                     reset='v = Eleaky', method='euler')

    # Initialize membrane potentials
    MN.v = initial_state_neurons['MN']['v']
    inh.v = initial_state_neurons['inh']['v']
    inhb.v = initial_state_neurons['inhb']['v']
    exc.v = initial_state_neurons['exc']['v']
    #Initialize conductances
    inh.gIa = initial_state_neurons['inh']['gIa']
    inh.gII = initial_state_neurons['inh']['gII']
    inh.gi = initial_state_neurons['inh']['gi']
    inh.ginh = initial_state_neurons['inh']['ginh']
    inhb.gIa = initial_state_neurons['inhb']['gIa']
    inhb.gIb = initial_state_neurons['inhb']['gIb']
    exc.gII = initial_state_neurons['exc']['gII']
    MN.gIa = initial_state_neurons['MN']['gIa']
    MN.gexc = initial_state_neurons['MN']['gexc']
    MN.gi1 = initial_state_neurons['MN']['gi1']
    MN.ginh = initial_state_neurons['MN']['ginh']
    MN.gi2 = initial_state_neurons['MN']['gi2']
    MN.ginhb = initial_state_neurons['MN']['ginhb']
                                           
    # Add neuron groups to the network
    net.add([MN, inh, inhb, exc])

    # Create group mapping for connections
    group_map = {
        "Ia_flexor": Ia[:n_Ia_flexor],
        "Ia_extensor": Ia[n_Ia_flexor:],
        "Ib_flexor": Ib[:n_Ib_flexor],
        "Ib_extensor": Ib[n_Ib_flexor:],
        "II_flexor": II[:n_II_flexor],
        "II_extensor": II[n_II_flexor:],
        "MN_flexor": MN[:n_MN_flexor],
        "MN_extensor": MN[n_MN_flexor:],
        "inh_flexor": inh[:n_inh_flexor],
        "inh_extensor": inh[n_inh_flexor:],
        "inhb_flexor": inhb[:n_inhb_flexor],
        "inhb_extensor": inhb[n_inhb_flexor:],
        "exc_flexor": exc[:n_exc_flexor],
        "exc_extensor": exc[n_exc_flexor:],
    }
    
    # Create synaptic connections based on the network architecture
    synapses = {}
    Ia_II_delta_delay=spindle_model["Ia_II_delta_delay"]
    for (pre_name, post_name), conn_info in connections.items():
        key = f"{pre_name}_to_{post_name}"
        pre = group_map[pre_name]
        post = group_map[post_name]
        weight = conn_info["w"]
        p = conn_info["p"]
        
        # Determine the appropriate conductance variable based on presynaptic neuron type
        pre_type = pre_name.split('_')[0]
        conductance_var = f"g{pre_type}"
        
        syn = Synapses(pre, post, model="w : siemens", 
                      on_pre=f"{conductance_var}_post += w", method='exact')
        syn.connect(p=p)
        
        # Add noise to weights and delays
        noise = 0.2
        syn.w = np.clip(weight + noise * weight * randn(len(syn.w)), 0*nS, np.inf*nS)
        syn.delay=np.clip(1*ms+0.25*ms*noise*randn(len(syn.delay)), 0*ms, np.inf*ms)
        
        net.add(syn)
        synapses[key] = syn

    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    mon_Ib = SpikeMonitor(Ib)
    mon_II = SpikeMonitor(II)
    mon_MN = SpikeMonitor(MN)
    mon_inh = SpikeMonitor(inh)
    mon_inhb = SpikeMonitor(inhb)
    mon_exc = SpikeMonitor(exc)
    
    # State monitors for key neurons
    mon_MN_flexor = StateMonitor(MN, ['Isyn'], n_MN_flexor//2)
    mon_MN_extensor = StateMonitor(MN, ['Isyn'], n_MN_flexor + n_MN_extensor//2)
    
    # Add all monitors to the network
    monitors = [
        mon_Ia, mon_Ib, mon_II, mon_MN, mon_inh, mon_inhb, mon_exc,
        mon_MN_flexor, mon_MN_extensor
    ]
    net.add(monitors)
    
    # Handle EES stimulation if enabled
    if ees_freq > 0:
        ees_MN = PoissonGroup(N=MN_flexor_recruited + MN_extensor_recruited, rates=ees_freq)
        mon_ees_MN = SpikeMonitor(ees_MN)
        net.add([ees_MN, mon_ees_MN])

    # Run simulation
    net.run(T)
   
    mn_spikes = mon_MN.spike_trains()
    ia_spikes = mon_Ia.spike_trains()
    ib_spikes = mon_Ib.spike_trains()
    ii_spikes = mon_II.spike_trains()
    inh_spikes = mon_inh.spike_trains()
    inhb_spikes = mon_inhb.spike_trains()
    exc_spikes = mon_exc.spike_trains()

    ees_spikes = mon_ees_MN.spike_trains() if ees_freq > 0 else None

    # Extract motoneuron spikes
    MN_flexor_spikes = {i: mn_spikes[i] for i in range(n_MN_flexor)} 
    MN_extensor_spikes = {i: mn_spikes[i + n_MN_flexor] for i in range(n_MN_extensor)} 

    # EES correction
    if ees_freq > 0:
        if MN_flexor_recruited > 0:
            MN_flexor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_flexor_spikes,
                {i: ees_spikes[i] for i in range(MN_flexor_recruited)},
                T_refr
            )
            MN_extensor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_extensor_spikes,
                {
                    i % MN_flexor_recruited: ees_spikes[i + MN_flexor_recruited]
                    for i in range(MN_extensor_recruited)
                },
                T_refr
            )
        elif MN_extensor_recruited > 0:
            MN_extensor_spikes = process_motoneuron_spikes(
                neuron_pop,
                MN_extensor_spikes,
                {i: ees_spikes[i] for i in range(MN_extensor_recruited)},
                T_refr
            )


    # Store the final state to continue the simulation
    final_state_neurons = {
        'inh': {
            'v': inh.v[:],
            'gIa': inh.gIa[:],
            'gII': inh.gII[:],
            'gi': inh.gi[:],
            'ginh': inh.ginh[:]
        },
        'inhb': {
            'v': inhb.v[:],
            'gIa': inhb.gIa[:],
            'gIb': inhb.gIb[:]  
        },
        'exc': {
            'v': exc.v[:],
            'gII': exc.gII[:],
        },
        'MN': {
            'v': MN.v[:],
            'gIa': MN.gIa[:],
            'gexc': MN.gexc[:],
            'gi1': MN.gi1[:],
            'ginh': MN.ginh[:],
            'gi2': MN.gi2[:],
            'ginhb': MN.ginhb[:]
        }
    }

    # Store state monitors for plotting
    state_monitors = [
        {'IPSP_MN': mon_MN_flexor.Isyn[0] / nA},
        {'IPSP_MN': mon_MN_extensor.Isyn[0] / nA}
    ]

    # Organize results
    result_flexor = {
        "Ia": {i: ia_spikes[i] for i in range(n_Ia_flexor)},
        "Ib": {i: ib_spikes[i] for i in range(n_Ib_flexor)},
        "II": {i: ii_spikes[i] for i in range(n_II_flexor)},
        "MN": MN_flexor_spikes,
        "inh": {i: inh_spikes[i] for i in range(n_inh_flexor)},
        "inhb": {i: inhb_spikes[i] for i in range(n_inhb_flexor)},
        "exc": {i: exc_spikes[i] for i in range(n_exc_flexor)}
    }

    result_extensor = {
        "Ia": {i % n_Ia_flexor: ia_spikes[i] for i in range(n_Ia_flexor, n_Ia_flexor + n_Ia_extensor)},
        "Ib": {i % n_Ib_flexor: ib_spikes[i] for i in range(n_Ib_flexor, n_Ib_flexor + n_Ib_extensor)},
        "II": {i % n_II_flexor: ii_spikes[i] for i in range(n_II_flexor, n_II_flexor + n_II_extensor)},
        "MN": MN_extensor_spikes,
        "inh": {i % n_inh_flexor: inh_spikes[i] for i in range(n_inh_flexor, n_inh_flexor + n_inh_extensor)},
        "inhb": {i % n_inhb_flexor: inhb_spikes[i] for i in range(n_inhb_flexor, n_inhb_flexor + n_inhb_extensor)},
        "exc": {i % n_exc_flexor: exc_spikes[i] for i in range(n_exc_flexor, n_exc_flexor + n_exc_extensor)}
    }

    return [result_flexor, result_extensor], final_state_neurons, state_monitors

                                           
                                           
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

