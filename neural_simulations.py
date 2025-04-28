from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional

def run_flexor_extensor_neuron_simulation(stretch, velocity, 
                                          neuron_pop, dt_run, T, w=500*uS, p=0.4, Eleaky=-70*mV,
                                          gL=0.1*mS, Cm=1*uF, E_ex=0*mV, E_inh=-75*mV, 
                                          tau_exc=0.5*ms, tau_inh=3*ms, threshold_v=-55*mV, 
                                          ees_freq=0*hertz, aff_recruited=0, eff_recruited=0, T_refr=10*ms):
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


    n_Ia=neuron_pop['Ia']
    n_II=neuron_pop['II']

    # Afferent neuron equations
    ia_eq = '''
    is_flexor = (i < n_Ia) : boolean
    stretch_array = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    velocity_array = velocity_flexor_array(t) * int(is_flexor) + velocity_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < aff_recruited) or (not is_flexor and i < n_Ia + aff_recruited)) : boolean
    rate = 50*hertz + 2*hertz*stretch_array + 4.3*hertz*sign(velocity_array)*abs(velocity_array)**0.6 + ees_freq * int(is_ees) : Hz
    '''

    ii_eq = '''
    is_flexor = (i < n_II) : boolean
    stretch_array = stretch_flexor_array(t) * int(is_flexor) + stretch_extensor_array(t) * int(not is_flexor) : 1
    is_ees = ((is_flexor and i < aff_recruited) or (not is_flexor and i < n_II + aff_recruited)) : boolean
    rate = 80*hertz + 13.5*hertz*stretch_array + ees_freq * int(is_ees) : Hz
    '''
    
    # Create afferent neurons
    Ia = NeuronGroup(2*n_Ia, ia_eq, threshold='rand() < rate*dt', refractory=T_refr)
    II = NeuronGroup(2*n_II, ii_eq, threshold='rand() < rate*dt', refractory=T_refr)
    net.add([Ia, II])

    # LIF neuron equation
    neuron_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn = gi*(E_inh - v) + (ge1 + ge2 + ge3)*(E_ex - v) : amp
    ge1 : siemens  # First excitatory input
    ge2 : siemens  # Second excitatory input
    ge3 : siemens  # Third excitatory input
    gi : siemens   # Inhibitory input
    '''
    
    #Model all non afferent neuron with on neurongroup
    non_afferent = {k: v for k, v in neuron_pop.items() if k not in ['Ia', 'II']}
    n_total = sum(list(non_afferent.values()))* 2  # Total for flexor and extensor

    # Create single neuron group for all non-afferent neurons
    neurons = NeuronGroup(n_total, neuron_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
    neurons.v = Eleaky
    net.add(neurons)
    # Calculate indices for each neuron type
    indices = {}
    start_idx = 0

    # Generate indices dictionary with a consistent pattern
    for side in ['flexor', 'extensor']:
        for neuron_type, count in non_afferent.items():
            key = f"{neuron_type}_{side}"
            indices[key] = slice(start_idx, start_idx + count)
            start_idx += count

    def get_neurons_by_type(neuron_type):
        neuron_class, side = neuron_type.split('_')
    
        if neuron_class == "Ia":
            return Ia[:neuron_pop['Ia']] if side == "flexor" else Ia[neuron_pop['Ia']:]
        elif neuron_class == "II":
            return II[:neuron_pop['II']] if side == "flexor" else II[neuron_pop['II']:]
        else:
            return neurons[indices[neuron_type]]
    
    synapse_eqs = {
    "exc1": """
        dx/dt = -x / tau_exc : siemens (clock-driven)
        ge1_post = x : siemens (summed)
        w_: siemens
    """,
    "exc2": """
        dx/dt = -x / tau_exc : siemens (clock-driven)
        ge2_post = x : siemens (summed)
        w_: siemens
    """,
    "exc3": """
        dx/dt = -x / tau_exc : siemens (clock-driven)
        ge3_post = x : siemens (summed)
        w_: siemens
    """,
    "inh": """
        dg/dt = (x - g) / tau_inh : siemens (clock-driven)
        dx/dt = -x / tau_inh : siemens (clock-driven)
        gi_post = g : siemens (summed)
        w_: siemens
    """
    }

    # Define connections with targets
    connections = {
    ("Ia_flexor", "motor_flexor"): "exc1",
    ("Ia_flexor", "inh_flexor"): "exc1",
    ("Ia_extensor", "motor_extensor"): "exc1",
    ("Ia_extensor", "inh_extensor"): "exc1",
    
    ("II_flexor", "exc_flexor"): "exc2",
    ("II_flexor", "inh_flexor"): "exc2",
    ("II_extensor", "exc_extensor"): "exc2",
    ("II_extensor", "inh_extensor"): "exc2",
    
    ("exc_flexor", "motor_flexor"): "exc3",
    ("exc_extensor", "motor_extensor"): "exc3",
    
    # All inhibitory connections use the same "inh" type
    ("inh_flexor", "motor_extensor"): "inh",
    ("inh_extensor", "motor_flexor"): "inh",
    ("inh_flexor", "inh_extensor"): "inh",
    ("inh_extensor", "inh_flexor"): "inh"
}
    synapses = {}
    # Then when creating synapses:
    for pre, post in connections:
        pre_neurons = get_neurons_by_type(pre)
        post_neurons = get_neurons_by_type(post)
        key = f"{pre}_to_{post}"
        # Get the correct synapse type
        syn_type = connections[(pre, post)]
        
        # Create synapse
        syn = Synapses(pre_neurons, post_neurons, model=synapse_eqs[syn_type], 
                      on_pre='x += w_', method='exact')
        
        # Connect with probability p (your original method)
        syn.connect(p=p)
        syn.w_ = w
        
        net.add(syn)
        synapses[key] = syn
        net.add(syn)  
      
    # Setup monitors
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)
    M_motoneuron_flexor = SpikeMonitor(neurons[indices[f"motor_flexor"]])
    M_motoneuron_extensor = SpikeMonitor(neurons[indices[f"motor_extensor"]])
      
    monitors = [mon_Ia, mon_II, M_motoneuron_flexor, M_motoneuron_extensor]
    net.add(monitors)
    
    if ees_freq>0 and eff_recruited>0:
        ees_motoneuron=PoissonGroup(N=2*eff_recruited*neuron_pop['motor'], rates= ees_freq)
        mon_ees_motor=SpikeMonitor(ees_motoneuron)
        net.add([ees_motoneuron, mon_ees_motor])

    net.run(T)
    
    # Extract motoneuron spikes
    motor_flexor_spikes = M_motoneuron_flexor.spike_trains()
    motor_extensor_spikes = M_motoneuron_extensor.spike_trains()

    if ees_freq>0 and eff_recruited>0:
        # Process motoneuron spikes by adding EES effect
        motor_flexor_spikes = process_motoneuron_spikes(
        neuron_pop, motor_flexor_spikes,
        [mon_ees_motor.spike_trains()[i] for i in range(int(eff_recruited * neuron_pop['motor']))],
        T_refr)
        motor_extensor_spikes = process_motoneuron_spikes(
        neuron_pop, motor_extensor_spikes,
        [mon_ees_motor.spike_trains()[i] for i in range(int(eff_recruited * neuron_pop['motor']), neuron_pop['motor'])],
        T_refr)


    
    return {'flexor':{"Ia": {i: mon_Ia.spike_trains()[i] for i in range(neuron_pop['Ia'])},
                      "II": {i: mon_II.spike_trains()[i] for i in range(neuron_pop['II'])},
                      "MN": motor_flexor_spikes},
            'extensor':{"Ia": {i: mon_Ia.spike_trains()[i] for i in range(neuron_pop['Ia'], 2*neuron_pop['Ia'])},
                        "II": {i: mon_II.spike_trains()[i] for i in range(neuron_pop['II'],2*neuron_pop['II'] )},
                        "MN": motor_extensor_spikes}}

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
    stretch_array = TimedArray(stretch, dt=dt_run)
    velocity_array = TimedArray(velocity, dt=dt_run)
    
    ia_recruited=aff_recruited*neuron_pop['Ia']
    # Dynamic response (Ia) depends on stretch and velocity
    ia_eq = '''
        is_ees = int(i < ia_recruited): boolean  
        # Rate equation for non-EES neurons
        rate_non_ees = 50 * hertz + 2 * hertz * stretch_array(t) + 4.3 * hertz * sign(velocity_array(t)) * abs(velocity_array(t)) ** 0.6 :hertz
        # Rate equation for EES-activated neurons
        rate_ees = rate_non_ees + ees_freq :Hz
        # Use a conditional to select the appropriate rate for each neuron
        rate = is_ees * rate_ees + (1-is_ees) * rate_non_ees : Hz
    '''
    Ia = NeuronGroup(neuron_pop["Ia"], ia_eq, threshold='rand() < rate*dt',reset='v=Eleaky', refractory=T_refr)

    ii_recruited=aff_recruited*neuron_pop['II']
    # Static response (II) depends primarily on stretch
    ii_eq = '''
        is_ees = int(i < ii_recruited) : boolean 
        # Rate equation for non-EES neurons
        rate_non_ees = 80 * hertz + 13.5 * hertz * stretch_array(t): Hz
        # Rate equation for EES-activated neurons
        rate_ees = rate_non_ees + ees_freq : Hz
        # Use a conditional to select the appropriate rate for each neuron
        rate = is_ees * rate_ees + (1-is_ees) * rate_non_ees : Hz
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

    ees_motoneuron=PoissonGroup(N=eff_recruited*neuron_pop['motor'], rates= ees_freq)
    mon_ees_motor=SpikeMonitor(ees_motoneuron)
    net.add([ees_motoneuron, mon_ees_motor])

    net.run(T)
 
    # Process motoneuron spikes by adding EES effect
    moto_spike_dict = process_motoneuron_spikes(
        neuron_pop, mon_motor.spike_trains(), mon_ees_motor.spike_trains(), T_refr
        )

  
    # Return final results
    return {
        "Ia": mon_Ia.spike_trains(), 
        "II": mon_II.spike_trains(), 
        "MN": moto_spike_dict
    }
        
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



