from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional

def run_flexor_extensior_neuron_simulation():

    # Set simulation parameters
    defaultclock.dt = 0.025 * ms
    cycle_time = 100 * ms
    
    # Constants
    El, gL, Cm = -70 * mV, 0.1 * mS, 1 * uF
    E_ex, E_inh = 0 * mV, -75 * mV
    tau_exc, tau_inh = 0.5 * ms, 3 * ms
    
    # Neuron model (Leaky Integrate-and-Fire)
    eqs = '''
    dv/dt = (gL*(El - v) + Isyn)/ Cm : volt
    Isyn = gi * (E_inh - v) + ge * (E_ex - v) : amp
    ge : siemens
    gi : siemens
    '''
    
    # Neuron populations
    n = {"Ia": 50, "II": 50, "inh": 50, "exc": 50, "motor": 50}
    n_tot = sum(list(n.values()))
    
    neurons = NeuronGroup(n_tot * 2, eqs, threshold="v > -50*mV", reset="v = El", method="exact")
    neurons.v = El  # Set initial voltage
    
    # Dynamic neuron group mapping
    def get_indices(start, sizes):
        indices = {}
        current_index = start
        for k, v in sizes.items():
            indices[k] = slice(current_index, current_index + v)
            current_index += v
        return indices
    
    neurons_type = get_indices(0, {k + "_flexor": v for k, v in n.items()} | {k + "_extensor": v for k, v in n.items()})
    
    # Function to select a fraction of neurons
    def select_fraction(neuron_group, fraction):
        start, stop = neuron_group.start, neuron_group.stop
        return slice(start, start + int(fraction * (stop - start)))
    
    # Select 50% of neurons for Poisson input
    new_sli_II_flexor = select_fraction(neurons_type["II_flexor"], 0.5)
    new_sli_Ia_flexor = select_fraction(neurons_type["Ia_flexor"], 0.5)
    
    # Create Poisson inputs for selected neurons
    poisson_inputs = [
        PoissonInput(neurons[new_sli_II_flexor], "v", N=10, rate=1/(10*ms), weight=100 * mV),
        PoissonInput(neurons[new_sli_Ia_flexor], "v", N=10, rate=1/(10*ms), weight=100 * mV),
        PoissonInput(neurons[neurons_type["II_extensor"]], "v", N=10, rate=1/(10*ms), weight=100 * mV),
        PoissonInput(neurons[neurons_type["Ia_extensor"]], "v", N=10, rate=1/(10*ms), weight=100 * mV)
    ]
    
    # Synapse model templates
    synapse_eqs = {
        "exc": """
            dx/dt = -x / tau_exc : siemens (clock-driven)
            ge_post = x : siemens  (summed)
            w: siemens # Synaptic weight
        """,
        "inh": """
            dg/dt = (x - g) / tau_inh : siemens (clock-driven)
            dx/dt = -x / tau_inh : siemens (clock-driven)
            gi_post = g : siemens  (summed)
            w: siemens # Synaptic weight
        """
    }
    
    # Function to create synapses
    def create_synapse(pre, post, syn_type, p=1.0):
        syn = Synapses(neurons[neurons_type[pre]], neurons[neurons_type[post]], method='exact',
                       model=synapse_eqs[syn_type], on_pre=' x += w')
        syn.connect(p=p)
        syn.w = '10*uS'
        return syn
    
    # Synapse connections
    synapses = {name: create_synapse(*name.split("_to_"), "exc" if "exc" in name or "Ia" in name or "II" in name else "inh", p=0.9)
                for name in [
                    "Ia_extensor_to_motor_extensor", "Ia_extensor_to_inh_extensor", "II_extensor_to_exc_extensor",
                    "II_extensor_to_inh_extensor", "exc_extensor_to_motor_extensor", "inh_extensor_to_motor_flexor",
                    "Ia_flexor_to_motor_flexor", "Ia_flexor_to_inh_flexor", "II_flexor_to_exc_flexor", "II_flexor_to_inh_flexor",
                    "exc_flexor_to_motor_flexor", "inh_flexor_to_motor_extensor", "inh_flexor_to_inh_extensor", "inh_extensor_to_inh_flexor"
                ]}
    
    # Network construction
    net = Network([neurons] + poisson_inputs[0:2] + list(synapses.values()))
    
    # Monitoring
    spikes = {name: SpikeMonitor(neurons[neurons_type[name]]) for name in neurons_type}
    net.add(spikes.values())
    
    # Run the simulation
    net.run(T)

    
    

    


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

