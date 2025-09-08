####
# This should be run as python toy_example.py

import pyNN.spiNNaker as sim
from numpy import arange
import numpy as np
from spynnaker.pyNN.models.neuron.neuron_models.neuron_model_leaky_integrate_and_fire import NeuronModelLeakyIntegrateAndFire
from spynnaker.pyNN.models.neuron.builds.if_curr_exp_base import IFCurrExpBase

from scipy.sparse import csr_array

w=np.array([[-1,0,1]])
test_w = [[] for i in range(w.shape[0]*w.shape[1])]
for i in range(w.shape[0]):
    for j in range(w.shape[1]):
        test_w[i+w.shape[0]*j] = [i,j,w[i,j],0] 
print(test_w)

# === Configure the simulator ================================================

sim.setup(timestep=1) #It appears PyNN likes miliseconds as its default unit. Ew.


# === Build and instrument the network =======================================

#hidden_layer = sim.Population(3, NeuronModelLeakyIntegrateAndFire(v_init=None, cm=1, v_rest=0, tau_m=20, i_offset=0, v_reset=0, tau_refrac=0))
hidden_layer = sim.Population(3, IFCurrExpBase(cm=1, v_rest=-65, tau_m=20, i_offset=5, v_reset=-65, tau_refrac=0, v_thresh=-50))
#            {V: 'mV', V_REST: 'mV', TAU_M: 'ms', CM: 'nF', I_OFFSET: 'nA', V_RESET: 'mV', TAU_REFRAC: 'ms'})
# Another option is
#spynnaker/pyNN/models/neuron/builds/if_curr_alpha.py
#Lots of default parameter here though, not sure which will be more suitable...

#Not sure if this record works
hidden_layer.record(["spikes", "v"])

spike_source = sim.Population(1, sim.SpikeSourceArray(spike_times=arange(20, 51, 1)))

connection = sim.Projection(spike_source, hidden_layer, sim.FromListConnector(test_w, safe=False),
                            sim.StaticSynapse(),receptor_type='inhibitory'),#
#It looks like delay, and excitatory can be removed... that will be necessary, but nice to know the option, leaving here for now

# === Run the simulation =====================================================

sim.run(100.0)


# === Save the results, optionally plot a figure =============================

spikes = hidden_layer.get_data("spikes")
volts = hidden_layer.get_data("v")  # if you recorded 'v'


#we will eventually want to run np.max(volts) to see if we can replicate  


# === Clean up and quit ========================================================

sim.end()

print(spikes.segments[0].spiketrains[0])
print(spikes.segments[0].spiketrains[1])
print(spikes.segments[0].spiketrains[2])
#print(dir(spikes.segments[0]))
#print(volts)

