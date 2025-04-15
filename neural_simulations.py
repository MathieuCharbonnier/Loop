from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional




def setup_afferent_neurons(neuron_pop: Dict[str, int], stretch_array: TimedArray, velocity_array: TimedArray) -> Tuple[PoissonGroup, PoissonGroup]:
    """
    Set up afferent neurons (Ia and II) with appropriate firing rates.
    
    Parameters:
    ----------
    neuron_pop : dict
        Dictionary with neuron population sizes
    stretch_array : TimedArray
        TimedArray of stretch values
    velocity_array : TimedArray
        TimedArray of velocity values
        
    Returns:
    -------
    tuple
        Ia and II PoissonGroups
    """
    # Dynamic response (Ia) depends on stretch and velocity
    ia_eq = '50 * hertz + 2 * hertz * stretch_array(t) + 4.3 * hertz * sign(velocity_array(t)) * abs(velocity_array(t)) ** 0.6'
    Ia_without_ees = PoissonGroup(neuron_pop["Ia"], rates=ia_eq)
    
    # Static response (II) depends primarily on stretch
    ii_eq = '80 * hertz + 13.5 * hertz * stretch_array(t)'
    II_without_ees = PoissonGroup(neuron_pop["II"], rates=ii_eq)
    
    return Ia_without_ees, II_without_ees


def setup_ees_stimulation(neuron_pop: Dict[str, int], aff_recruited: float, eff_recruited: float, ees_freq: Quantity) -> Tuple[Dict[str, int], np.ndarray, Optional[PoissonGroup]]:
    """
    Set up epidural electrical stimulation (EES) group.
    
    Parameters:
    ----------
    neuron_pop : dict
        Dictionary with neuron population sizes
    aff_recruited : float
        Proportion of afferent neurons recruited by EES
    eff_recruited : float
        Proportion of efferent neurons recruited by EES
    ees_freq : Quantity
        Frequency of EES
        
    Returns:
    -------
    tuple
        Dictionary of recruited neurons, cumulative sum, and EES group
    """
    # Create dictionary of recruited neurons
    n_poisson = {
        "Ia": int(neuron_pop["Ia"] * aff_recruited), 
        "II": int(neuron_pop["II"] * aff_recruited), 
        "motor": int(neuron_pop["motor"] * eff_recruited)
    }
    
    # Calculate cumulative sum for indexing
    cum_n = np.cumsum(list(n_poisson.values()))
    
    # Create EES stimulation group only if needed
    total_recruited = sum(n_poisson.values())
    if total_recruited > 0 and ees_freq > 0*hertz:
        Gees = PoissonGroup(total_recruited, rates=ees_freq)
    else:
        Gees = None
    
    return n_poisson, cum_n, Gees


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
        return np.array([], dtype=float)
    
    if len(natural_spikes) == 0:
        natural_spikes = np.array([], dtype=float)
    if len(ees_spikes) == 0:
        ees_spikes = np.array([], dtype=float)
    
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
    final_spikes[0] = spikes[0]['time']
    final_count = 1
    last_spike_time = spikes[0]['time']
    
    # Process remaining spikes more efficiently
    for i in range(1, len(spikes)):
        current_time = spikes[i]['time']
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
    return final_spikes[:final_count]


def process_afferent_spikes(neuron_pop: Dict[str, int], afferent_spikes: Dict, ees_spikes: Dict, 
                          n_poisson: Dict[str, int], cum_n: np.ndarray, T_refr: Quantity) -> Dict[str, List]:
    """
    Process afferent (Ia and II) spike trains, combining natural and EES-induced spikes.
    
    Parameters:
    ----------
    neuron_pop : dict
        Dictionary with neuron population sizes
    afferent_spikes : dict
        Dictionary with natural spike trains for Ia and II
    ees_spikes : dict
        Dictionary with EES-induced spikes
    n_poisson : dict
        Dictionary of recruited neurons
    cum_n : array
        Cumulative sum of recruited neurons
    T_refr : Quantity
        Refractory period
        
    Returns:
    -------
    dict
        Dictionary with processed spike indices and times for Ia and II
    """
    result = {
        "Ia_indices": [],
        "Ia_times": [],
        "II_indices": [],
        "II_times": []
    }
    
    # Process Ia spikes
    for i in range(neuron_pop["Ia"]):
        nat_spikes = afferent_spikes["Ia"][i] if i in afferent_spikes["Ia"] else np.array([])
        
        # Get EES-induced spikes for this neuron if it's in the recruited population
        ees_i_spikes = np.array([])
        if i < n_poisson["Ia"] and cum_n[0] > 0:
            ees_idx = i
            ees_i_spikes = ees_spikes[ees_idx] if ees_idx in ees_spikes else np.array([])
        
        # Merge and filter spikes
        final_spikes = merge_and_filter_spikes(nat_spikes, ees_i_spikes, T_refr)
        
        if len(final_spikes) > 0:
            result["Ia_indices"].extend([i] * len(final_spikes))
            result["Ia_times"].extend(final_spikes)
    
    # Process II spikes
    for i in range(neuron_pop["II"]):
        nat_spikes = afferent_spikes["II"][i] if i in afferent_spikes["II"] else np.array([])
        
        # Get EES-induced spikes for this neuron if it's in the recruited population
        ees_i_spikes = np.array([])
        if i < n_poisson["II"] and cum_n[0] > 0:
            ees_idx = i + cum_n[0]
            ees_i_spikes = ees_spikes[ees_idx] if ees_idx in ees_spikes else np.array([])
        
        # Merge and filter spikes
        final_spikes = merge_and_filter_spikes(nat_spikes, ees_i_spikes, T_refr)
        
        if len(final_spikes) > 0:
            result["II_indices"].extend([i] * len(final_spikes))
            result["II_times"].extend(final_spikes)
    
    return result


def create_lif_neuron_model() -> Tuple[str, Dict[str, Quantity]]:
    """
    Create a leaky integrate-and-fire (LIF) neuron model.
    
    Returns:
    -------
    tuple
        Equations string and constants dictionary
    """
    # Constants for the model
    constants = {
        "El": -70 * mV,
        "gL": 0.1 * mS,
        "Cm": 1 * uF,
        "E_ex": 0 * mV,
        "E_inh": -75 * mV,
        "tau_exc": 0.5 * ms,
        "tau_inh": 3 * ms,
        "threshold_v": -55 * mV
    }
    
    # Neuron model (Leaky Integrate-and-Fire)
    eqs = '''
    dv/dt = (gL*(El - v) + Isyn) / Cm : volt
    Isyn = (ge + ge2) * (E_ex - v) : amp
    ge : siemens
    ge2 : siemens
    '''
    
    return eqs, constants


def create_synapse_models(tau_exc: Quantity) -> Dict[str, str]:
    """
    Create equations for different synapse models.
    
    Parameters:
    ----------
    tau_exc : Quantity
        Time constant for excitatory synapses
        
    Returns:
    -------
    dict
        Dictionary with synapse model equations
    """
    synapse_models = {
        "II_Ex": """
        dx/dt = -x / tau_exc : siemens (clock-driven)
        ge_post = x : siemens (summed)
        w: siemens # Synaptic weight
        """,
        
        "Ia_Motoneuron": """
        dy/dt = -y / tau_exc : siemens (clock-driven)
        ge_post = y : siemens (summed)
        w: siemens # Synaptic weight
        """,
        
        "Ex_Motoneuron": """
        dz/dt = -z / tau_exc : siemens (clock-driven)
        ge2_post = z : siemens (summed)
        w: siemens # Synaptic weight
        """
    }
    
    return synapse_models


def process_motoneuron_spikes(neuron_pop: Dict[str, int], motor_spikes: Dict, 
                            ees_spikes: Dict, n_poisson: Dict[str, int], 
                            cum_n: np.ndarray, T_refr: Quantity) -> Dict:
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
    cum_n : array
        Cumulative sum of recruited neurons
    T_refr : Quantity
        Refractory period
        
    Returns:
    -------
    dict
        Dictionary with processed motoneuron spike times
    """
    moto_spike_dict = {}
    
    for i in range(neuron_pop["motor"]):
        nat_spikes = motor_spikes[i] if i in motor_spikes else np.array([])
        
        # Get EES-induced spikes for this neuron if it's in the recruited population
        ees_i_spikes = np.array([])
        if i < n_poisson["motor"] and cum_n[1] > 0:
            ees_idx = i + cum_n[1]
            ees_i_spikes = ees_spikes[ees_idx] if ees_idx in ees_spikes else np.array([])
        
        # Merge and filter spikes
        final_spikes = merge_and_filter_spikes(nat_spikes, ees_i_spikes, T_refr)
        
        if len(final_spikes) > 0:
            moto_spike_dict[i] = final_spikes
    
    return moto_spike_dict


def run_neural_simulations(stretch, velocity, neuron_pop, dt_run, T, w_run=500*uS, p_run=0.4, 
                         ees_freq=0*hertz, aff_recruited=0, eff_recruited=0, T_refr=10*ms):
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
    
    # Set up afferent neurons
    Ia_without_ees, II_without_ees = setup_afferent_neurons(neuron_pop, stretch_array, velocity_array)
    
    # Set up EES stimulation
    n_poisson, cum_n, Gees = setup_ees_stimulation(neuron_pop, aff_recruited, eff_recruited, ees_freq)
    
    # Set up monitoring for initial simulation
    mon_init_Ia = SpikeMonitor(Ia_without_ees)
    mon_init_II = SpikeMonitor(II_without_ees)
    mon_gees = SpikeMonitor(Gees) if Gees is not None else None
    
    # Create and run initial network (to generate baseline spike trains)
    Net_init = Network()
    Net_init.add(Ia_without_ees, II_without_ees)
    if Gees is not None:
        Net_init.add(Gees, mon_gees)
    Net_init.add(mon_init_Ia, mon_init_II)
    Net_init.run(T)
    
    # Get spike trains from initial run
    afferent_spikes = {
        "Ia": mon_init_Ia.spike_trains(),
        "II": mon_init_II.spike_trains()
    }
    
    # Get EES-induced spikes if applicable
    ees_spikes = mon_gees.spike_trains() if mon_gees is not None else {}
    
    # Process afferent spike trains (combine natural and EES-induced spikes)
    spike_data = process_afferent_spikes(neuron_pop, afferent_spikes, ees_spikes, n_poisson, cum_n, T_refr)
    
    # Create spike generator groups for Ia and II
    Ia = SpikeGeneratorGroup(neuron_pop['Ia'], spike_data["Ia_indices"], spike_data["Ia_times"])
    II = SpikeGeneratorGroup(neuron_pop['II'], spike_data["II_indices"], spike_data["II_times"])
    
    # Create neuron model
    eqs, constants = create_lif_neuron_model()
    
    # Creating the interneuron and motoneuron groups
    Excitatory = NeuronGroup(
        neuron_pop["exc"], 
        eqs, 
        threshold=f"v > {constants['threshold_v']}", 
        reset=f"v = {constants['El']}", 
        method="exact"
    )
    Excitatory.v = constants["El"]  # Set initial voltage
    
    Motoneuron = NeuronGroup(
        neuron_pop["motor"], 
        eqs, 
        threshold=f"v > {constants['threshold_v']}", 
        reset=f"v = {constants['El']}", 
        method="exact"
    )
    Motoneuron.v = constants["El"]  # Set initial voltage
    
    # Define synapse models
    synapse_models = create_synapse_models(constants["tau_exc"])
    
    # Create and connect synapses
    II_Ex = Synapses(II, Excitatory, model=synapse_models["II_Ex"], on_pre='x += w', method='exact')
    II_Ex.connect(p=p_run)
    II_Ex.w = w_run
    
    Ia_Motoneuron = Synapses(Ia, Motoneuron, model=synapse_models["Ia_Motoneuron"], on_pre='y += w', method='exact')
    Ia_Motoneuron.connect(p=p_run)
    Ia_Motoneuron.w = w_run
    
    Ex_Motoneuron = Synapses(Excitatory, Motoneuron, model=synapse_models["Ex_Motoneuron"], on_pre='z += w', method='exact')
    Ex_Motoneuron.connect(p=p_run)
    Ex_Motoneuron.w = w_run
    
    # Set up monitoring for main simulation
    mon_motor = SpikeMonitor(Motoneuron)
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)
    
    # Create and run main network
    net = Network()
    net.add([
        Ia, II, Excitatory, Motoneuron,  
        II_Ex, Ia_Motoneuron, Ex_Motoneuron,
        mon_Ia, mon_II, mon_motor
    ])
    net.run(T)
    
    # Process motoneuron spikes
    moto_spike_dict = process_motoneuron_spikes(
        neuron_pop, mon_motor.spike_trains(), ees_spikes, 
        n_poisson, cum_n, T_refr
    )
    
    # Return final results
    return {
        "Ia": mon_Ia.spike_trains(), 
        "II": mon_II.spike_trains(), 
        "MN": moto_spike_dict
    }

