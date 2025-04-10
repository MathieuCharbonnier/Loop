import argparse
from brian2 import *
import numpy as np

def run_neural_simulations(stretch, velocity, neuron_population, dt, T, w=500*uS, p=0.4):
    # Setting the simulation time step
    defaultclock.dt = dt

    # Constants for the model
    El, gL, Cm = -70 * mV, 0.1 * mS, 1 * uF
    E_ex, E_inh = 0 * mV, -75 * mV
    tau_exc, tau_inh = 0.5 * ms, 3 * ms

    # Neuron model (Leaky Integrate-and-Fire)
    eqs = '''
    dv/dt = (gL*(El - v) + Isyn)/ Cm : volt
    Isyn = (ge+ge2) * (E_ex - v) : amp
    ge : siemens
    ge2 : siemens
    '''

    # Converting stretch and velocity into TimedArrays
    stretch_array = TimedArray(stretch, dt=dt)
    velocity_array = TimedArray(velocity, dt=dt)

    # ---- Ia and II Firing Rates ----
    Ia = PoissonGroup(10, rates=""" 50 * hertz + 2 * hertz * stretch_array(t) + 4.3 * hertz * sign(velocity_array(t)) * abs(velocity_array(t)) ** 0.6""")
    II = PoissonGroup(10, rates=""" 80 * hertz + 13.5 * hertz * stretch_array(t)""")

    # Creating the neuron groups
    Excitatory = NeuronGroup(neuron_population["exc"], eqs, threshold="v > -55 * mV", reset="v = El", method="exact")
    Excitatory.v = El  # Set initial voltage for Excitatory neurons

    Motoneuron = NeuronGroup(neuron_population["motor"], eqs, threshold="v > -55 * mV", reset="v = El", method="exact")
    Motoneuron.v = El  # Set initial voltage for Motoneurons

    # Synapse model templates
    synapse_eqs_II_Ex = """
    dx/dt = -x / tau_exc : siemens (clock-driven)
    ge_post = x : siemens  (summed)
    w: siemens # Synaptic weight
    """
    synapse_eqs_Ia_Motoneuron = """
    dy/dt = -y / tau_exc : siemens (clock-driven)
    ge_post = y : siemens  (summed) # Using 'y' instead of 'x' for Ia_Motoneuron
    w: siemens # Synaptic weight
    """
    synapse_eqs_Ex_Motoneuron = """
    dz/dt = -z / tau_exc : siemens (clock-driven)
    ge2_post = z : siemens  (summed) # Using 'z' instead of 'x' for Ex_Motoneuron
    w: siemens # Synaptic weight
    """

    # Synapse connections
    II_Ex = Synapses(II, Excitatory, method='exact', model=synapse_eqs_II_Ex, on_pre='x += w')
    Ia_Motoneuron = Synapses(Ia, Motoneuron, method='exact', model=synapse_eqs_Ia_Motoneuron, on_pre='y += w')
    Ex_Motoneuron = Synapses(Excitatory, Motoneuron, method='exact', model=synapse_eqs_Ex_Motoneuron, on_pre='z += w')

    II_Ex.connect(p=p)
    Ia_Motoneuron.connect(p=p)
    Ex_Motoneuron.connect(p=p)

    II_Ex.w = w
    Ia_Motoneuron.w = w
    Ex_Motoneuron.w = w

    # Spike Monitors for recording the spikes of each population
    mon_motor = SpikeMonitor(Motoneuron, variables='v')
    mon_Ia = SpikeMonitor(Ia, variables='v')
    mon_II = SpikeMonitor(II, variables='v')

    # Creating the network and adding all components
    net = Network()
    net.add(Ia, II, Excitatory, Motoneuron, mon_motor, mon_Ia, mon_II, II_Ex, Ia_Motoneuron, Ex_Motoneuron)

    # Running the simulation
    net.run(T)

    # Returning the spike trains for Ia, II, and Motor neurons
    return mon_Ia.spike_trains(), mon_II.spike_trains(), mon_motor.spike_trains()


