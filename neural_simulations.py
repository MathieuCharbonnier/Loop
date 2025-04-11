from brian2 import *
import numpy as np
import os
import warnings


def run_neural_simulations(stretch, velocity, neuron_population, dt, T, w=500*uS, p=0.4):
    """
    Run neural simulations with stretch and velocity inputs.
    
    Parameters:
    ----------
    stretch : array-like
        Stretch signal for muscle spindles
    velocity : array-like
        Velocity signal for muscle spindles
    neuron_population : dict
        Dictionary with neuron population sizes {"Ia": int, "II": int, "exc": int, "motor": int}
    dt : Quantity
        Time step for the simulation
    T : Quantity
        Total simulation time
    w : Quantity, optional
        Synaptic weight (default: 500*uS)
    p : float, optional
        Connection probability (default: 0.4)
        
    Returns:
    -------
    tuple
        Spike trains for Ia, II, and motor neurons
    """
    # Validate inputs
    if not isinstance(dt, Quantity):
        warnings.warn("dt should be a Brian2 Quantity. Converting from seconds.")
        dt = dt * second
    
    if not isinstance(T, Quantity):
        warnings.warn("T should be a Brian2 Quantity. Converting from seconds.")
        T = T * second
    
    # Ensure neuron population is properly formatted
    required_keys = ["Ia", "II", "exc", "motor"]
    for key in required_keys:
        if key not in neuron_population:
            raise ValueError(f"neuron_population missing required key: {key}")

    # Setting the simulation time step
    defaultclock.dt = dt

    # Constants for the model
    El = -70 * mV
    gL = 0.1 * mS
    Cm = 1 * uF
    E_ex = 0 * mV
    E_inh = -75 * mV
    tau_exc = 0.5 * ms
    tau_inh = 3 * ms
    threshold_v = -55 * mV

    # Neuron model (Leaky Integrate-and-Fire)
    eqs = '''
    dv/dt = (gL*(El - v) + Isyn) / Cm : volt
    Isyn = (ge + ge2) * (E_ex - v) : amp
    ge : siemens
    ge2 : siemens
    '''

    # Convert lists to numpy arrays if necessary
    if not isinstance(stretch, np.ndarray):
        stretch = np.array(stretch, dtype=float)
    
    if not isinstance(velocity, np.ndarray):
        velocity = np.array(velocity, dtype=float)

    # Converting stretch and velocity into TimedArrays
    stretch_array = TimedArray(stretch, dt=dt)
    velocity_array = TimedArray(velocity, dt=dt)

    # ---- Ia and II Sensory Receptors ----
    # Dynamic response (Ia) depends on stretch and velocity
    # Fix the syntax error by removing line breaks in the rate equation
    ia_eq = ' 50 * hertz + 2 * hertz * stretch_array(t) + 4.3 * hertz * sign(velocity_array(t)) * abs(velocity_array(t)) ** 0.6 '
    Ia = PoissonGroup(neuron_population["Ia"], rates=ia_eq)
    
    # Static response (II) depends primarily on stretch
    ii_eq = ' 80 * hertz + 13.5 * hertz * stretch_array(t)'
    II = PoissonGroup(neuron_population["II"], rates=ii_eq)

    # Creating the interneuron and motoneuron groups
    Excitatory = NeuronGroup(
        neuron_population["exc"], 
        eqs, 
        threshold="v > threshold_v", 
        reset="v = El", 
        method="exact"
    )
    Excitatory.v = El  # Set initial voltage for Excitatory neurons

    Motoneuron = NeuronGroup(
        neuron_population["motor"], 
        eqs, 
        threshold="v > threshold_v", 
        reset="v = El", 
        method="exact"
    )
    Motoneuron.v = El  # Set initial voltage for Motoneurons

    # Define synapse models with clear variable names
    synapse_eqs_II_Ex = """
    dx/dt = -x / tau_exc : siemens (clock-driven)
    ge_post = x : siemens (summed)
    w: siemens # Synaptic weight
    """
    
    synapse_eqs_Ia_Motoneuron = """
    dy/dt = -y / tau_exc : siemens (clock-driven)
    ge_post = y : siemens (summed)
    w: siemens # Synaptic weight
    """
    
    synapse_eqs_Ex_Motoneuron = """
    dz/dt = -z / tau_exc : siemens (clock-driven)
    ge2_post = z : siemens (summed)
    w: siemens # Synaptic weight
    """

    # Create and connect synapses
    II_Ex = Synapses(II, Excitatory, model=synapse_eqs_II_Ex, on_pre='x += w', method='exact')
    II_Ex.connect(p=p)
    II_Ex.w = w
    
    Ia_Motoneuron = Synapses(Ia, Motoneuron, model=synapse_eqs_Ia_Motoneuron, on_pre='y += w', method='exact')
    Ia_Motoneuron.connect(p=p)
    Ia_Motoneuron.w = w
    
    Ex_Motoneuron = Synapses(Excitatory, Motoneuron, model=synapse_eqs_Ex_Motoneuron, on_pre='z += w', method='exact')
    Ex_Motoneuron.connect(p=p)
    Ex_Motoneuron.w = w

    # Set up monitoring
    mon_motor = SpikeMonitor(Motoneuron)
    mon_Ia = SpikeMonitor(Ia)
    mon_II = SpikeMonitor(II)

    # Creating the network and adding all components
    net = Network()
    net.add([
        Ia, II, Excitatory, Motoneuron,  
        II_Ex, Ia_Motoneuron, Ex_Motoneuron,
        mon_Ia, mon_II, mon_motor
    ])

    # Running the simulation
    net.run(T)

    spikes_Ia=binary_spike_train(dt, T, mon_Ia.spike_trains())
    spikes_II=binary_spike_train(dt, T, mon_II.spike_trains())
    spikes_motor=binary_spike_train(dt, T, mon_motor.spike_trains())

    # Return the spike trains
    return spikes_Ia, spikes_II,spikes_motor

def binary_spike_train(dt,T, spike_times):

  n_neuron=len(spike_times)
  spike_trains = np.zeros((n_neuron, int(T/dt)))  # one row per neuron

  for i in range(n_neuron):
    # Convert spike times to indices
    indices = (np.array([value/dt for value in spike_times[i]])).astype(int)
    # Set 1s in the spike train at the spike indices
    spike_trains[i, indices] = 1
    return spike_trains

