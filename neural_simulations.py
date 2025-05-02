from brian2 import *
import numpy as np
import os
from typing import Dict, List, Union, Tuple, Optional

from brian2 import *
import numpy as np

def run_flexor_extensor_neuron_simulation(stretch, velocity, 
                                          neuron_pop, dt_run, T, initial_potentials=None, Eleaky=-70*mV,
                                          gL=0.1*mS, Cm=1*uF, E_ex=0*mV, E_inh=-75*mV, 
                                          tau_exc=0.5*ms, tau_inh_1=1.5*ms, tau_inh_2=2*ms, threshold_v=-55*mV, 
                                          ees_freq=0*hertz, aff_recruited=0, eff_recruited=0, T_refr=10*ms):
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
    tau_inh_1 : time, optional
        First inhibitory time constant.
    tau_inh_2 : time, optional
        Second inhibitory time constant.
    threshold_v : volt, optional
        Voltage threshold.
    ees_freq : hertz, optional
        EES frequency.
    aff_recruited : int, optional
        Number of afferent neurons recruited.
    eff_recruited : float, optional
        Fraction of efferent neurons recruited.
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
    ia_eq_flexor = '''
    is_ees = (i < aff_recruited) : boolean
    rate = 10*hertz + 0.4*hertz*stretch_flexor_array(t) + 0.86*hertz*sign(velocity_flexor_array(t))*abs(velocity_flexor_array(t))**0.6 + ees_freq * int(is_ees) : Hz
    '''
    ia_eq_extensor = '''
    is_ees = (i < aff_recruited) : boolean
    rate = 10*hertz + 0.4*hertz*stretch_extensor_array(t) + 0.86*hertz*sign(velocity_extensor_array(t))*abs(velocity_extensor_array(t))**0.6 + ees_freq * int(is_ees) : Hz
    '''

    ii_eq_flexor = '''
    is_ees = (i < aff_recruited) : boolean
    rate = 20*hertz + 3.375*hertz*stretch_flexor_array(t) + ees_freq * int(is_ees) : Hz
    '''
                                            
    ii_eq_extensor = '''
    is_ees = (i < aff_recruited) : boolean
    rate = 20*hertz + 3.375*hertz*stretch_extensor_array(t) + ees_freq * int(is_ees) : Hz
    '''
    
    # Create afferent neurons
    Ia_flexor = NeuronGroup(n_Ia, ia_eq_flexor, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    Ia_extensor = NeuronGroup(n_Ia, ia_eq_extensor, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II_flexor = NeuronGroup(n_II, ii_eq_flexor, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    II_extensor = NeuronGroup(n_II, ii_eq_extensor, threshold='rand() < rate*dt', refractory=T_refr, method='euler')
    net.add([Ia_flexor, Ia_extensor, II_flexor, II_extensor])

    # LIF neuron equations
    ex_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn = gII*(E_ex - v) : amp
    dgII/dt = -gII / tau_exc : siemens 
    '''
    
    mn_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn =  (gIa+gex)*(E_ex - v) + gi*(E_inh - v) : amp
    dgIa/dt = -gIa / tau_exc : siemens 
    dgex/dt = -gex / tau_exc : siemens 
    dgi/dt = ((tau_inh_2/tau_inh_1)**(tau_inh_1/(tau_inh_2-tau_inh_1))*x-gi) / tau_inh_1 : siemens 
    dx/dt = -x / tau_inh_2 : siemens 
    '''

    inh_eq = '''
    dv/dt = (gL*(Eleaky - v) + Isyn)/Cm : volt
    Isyn = gi*(E_inh - v) + (gIa+gII)*(E_ex - v) : amp
    dgIa/dt = -gIa / tau_exc : siemens 
    dgII/dt = -gII / tau_exc : siemens 
    dgi/dt = ((tau_inh_2/tau_inh_1)**(tau_inh_1/(tau_inh_2-tau_inh_1))*x-gi) / tau_inh_1 : siemens 
    dx/dt = -x / tau_inh_2 : siemens 
    '''
  
    # Create neuron groups
    inh_flexor = NeuronGroup(n_inh, inh_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
    inh_extensor = NeuronGroup(n_inh, inh_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
    exc_flexor = NeuronGroup(n_exc, ex_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
    exc_extensor = NeuronGroup(n_exc, ex_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')                                        
    moto_extensor = NeuronGroup(n_motor, mn_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
    moto_flexor = NeuronGroup(n_motor, mn_eq, threshold='v > threshold_v', 
                         reset='v = Eleaky', refractory=T_refr, method='exact')
                         
    # Initialize membrane potentials
    if initial_potentials is None: 
        inh_flexor.v = Eleaky
        inh_extensor.v = Eleaky
        exc_flexor.v = Eleaky
        exc_extensor.v = Eleaky
        moto_flexor.v = Eleaky
        moto_extensor.v = Eleaky
    else:
        inh_flexor.v = initial_potentials['inh_flexor']
        inh_extensor.v = initial_potentials['inh_extensor']
        exc_flexor.v = initial_potentials['exc_flexor']
        exc_extensor.v = initial_potentials['exc_extensor']
        moto_flexor.v = initial_potentials['moto_flexor']
        moto_extensor.v = initial_potentials['moto_extensor']
    
    # Add neuron groups to the network
    net.add([inh_flexor, inh_extensor, exc_flexor, exc_extensor, moto_extensor, moto_flexor])
    
    # Define neural connections
    connections = {
        (Ia_flexor, moto_flexor): {"model": "gIa_post += 2.1*nS", "p": 1},
        (Ia_flexor, inh_flexor): {"model": "gIa_post += 3.64*nS", "p": 1},
        (Ia_extensor, moto_extensor): {"model": "gIa_post += 2.1*nS", "p": 1},
        (Ia_extensor, inh_extensor): {"model": "gIa_post += 3.64*nS", "p": 1},
        
        (II_flexor, exc_flexor): {"model": "gII_post += 1.65*nS", "p": 1},
        (II_flexor, inh_flexor): {"model": "gII_post += 2.9*nS", "p": 1},
        (II_extensor, exc_extensor): {"model": "gII_post += 1.65*nS", "p": 1},
        (II_extensor, inh_extensor): {"model": "gII_post += 2.9*nS", "p": 1},
        
        (exc_flexor, moto_flexor): {"model": "gex_post += 0.7*nS" , "p": 1},
        (exc_extensor, moto_extensor): {"model": "gex_post += 0.7*nS", "p": 1},
        
        (inh_flexor, moto_extensor): {"model": "gi_post += 0.2*nS", "p": 1},
        (inh_extensor, moto_flexor): {"model": "gi_post +=  0.2*nS", "p": 1},
        (inh_flexor, inh_extensor): {"model": "gi_post += 0.76*nS", "p": 1},
        (inh_extensor, inh_flexor): {"model": "gi_post += 0.76*nS", "p": 1}
    }
    
    # Create synaptic connections
    synapses = {}
    for (pre_neurons, post_neurons), conn_info in connections.items():
        pre_neurons_name = pre_neurons.__class__.__name__.lower()
        post_neurons_name = post_neurons.__class__.__name__.lower()
        key = f"{pre_neurons_name}_to_{post_neurons_name}"

        model = conn_info["model"]
        p = conn_info["p"]

        syn = Synapses(pre_neurons, post_neurons,
                      on_pre=model, method='exact')
        syn.connect(p=p)
      
        net.add(syn)
        synapses[key] = syn
          
    # Setup monitors
    mon_Ia_flexor = SpikeMonitor(Ia_flexor)
    mon_Ia_extensor = SpikeMonitor(Ia_extensor)
    mon_II_flexor = SpikeMonitor(II_flexor)
    mon_II_extensor = SpikeMonitor(II_extensor)
    mon_exc_flexor = SpikeMonitor(exc_flexor)
    mon_exc_extensor = SpikeMonitor(exc_extensor)
    mon_inh_flexor = SpikeMonitor(inh_flexor)
    mon_inh_extensor = SpikeMonitor(inh_extensor)
    mon_motoneuron_flexor = SpikeMonitor(moto_flexor)
    mon_motoneuron_extensor = SpikeMonitor(moto_extensor)

    monitors = [mon_Ia_flexor, mon_Ia_extensor, mon_II_flexor, mon_II_extensor,
                mon_exc_flexor, mon_exc_extensor, mon_inh_flexor, mon_inh_extensor,
                mon_motoneuron_flexor, mon_motoneuron_extensor]
                                            
    net.add(monitors)
    
    # Variables for EES monitors
    mon_ees_moto_flexor = None
    mon_ees_moto_extensor = None
    
    # Handle EES stimulation if enabled
    if ees_freq > 0 and eff_recruited > 0:
        num_ees_neurons = int(eff_recruited * n_motor)
        ees_motoneuron_extensor = PoissonGroup(N=num_ees_neurons, rates=ees_freq)
        ees_motoneuron_flexor = PoissonGroup(N=num_ees_neurons, rates=ees_freq)
        mon_ees_moto_flexor = SpikeMonitor(ees_motoneuron_flexor)
        mon_ees_moto_extensor = SpikeMonitor(ees_motoneuron_extensor)
      
        net.add([ees_motoneuron_extensor, ees_motoneuron_flexor, 
                mon_ees_moto_flexor, mon_ees_moto_extensor])

    # Run simulation
    net.run(T)
    
    # Extract motoneuron spikes
    moto_flexor_spikes = mon_motoneuron_flexor.spike_trains()
    moto_extensor_spikes = mon_motoneuron_extensor.spike_trains()

    # Final motor neuron spike trains
    motor_flexor_spikes = moto_flexor_spikes
    motor_extensor_spikes = moto_extensor_spikes

    # Process motoneuron spikes by adding EES effect if enabled
    if ees_freq > 0 and eff_recruited > 0:
        motor_flexor_spikes = process_motoneuron_spikes(
            neuron_pop, moto_flexor_spikes, mon_ees_moto_flexor.spike_trains(), T_refr)
        motor_extensor_spikes = process_motoneuron_spikes(
            neuron_pop, moto_extensor_spikes, mon_ees_moto_extensor.spike_trains(), T_refr)
   
    # Final membrane potentials
    final_potentials = {
        'inh_flexor': inh_flexor.v[:],
        'inh_extensor': inh_extensor.v[:],
        'exc_flexor': exc_flexor.v[:],
        'exc_extensor': exc_extensor.v[:],
        'moto_flexor': moto_flexor.v[:],
        'moto_extensor': moto_extensor.v[:]
    } 
    
    # Return results
    return [
        {
            "Ia": mon_Ia_flexor.spike_trains(),
            "II": mon_II_flexor.spike_trains(),
            "exc": mon_exc_flexor.spike_trains(),
            "inh": mon_inh_flexor.spike_trains(),
            "MN": motor_flexor_spikes
        },
        {
            "Ia": mon_Ia_extensor.spike_trains(),
            "II": mon_II_extensor.spike_trains(),
            "exc": mon_exc_extensor.spike_trains(),
            "inh": mon_inh_extensor.spike_trains(),
            "MN": motor_extensor_spikes
        }
    ], final_potentials



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



