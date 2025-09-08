####
# This should be run as python SmallNetwork.py
import os
path = os.getcwd().removesuffix('\pynn_models')
os.chdir(path)
import sys
sys.path.insert(1, path)

from algorithm.model import *

import pyNN.spiNNaker as sim
import numpy as np
from algorithm import *
from helpers import *
from spynnaker.pyNN.models.neuron.builds.if_curr_exp_base import IFCurrExpBase
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':

    I_c = np.load('pynn_models\\SmallNetworkParams\\I_c_array.npy')
    tau_mem = np.load('pynn_models\\SmallNetworkParams\\tau_mem_array.npy')*1000 #s to ms conversion
    w1 = np.load('pynn_models\\SmallNetworkParams\\w1_array.npy')
    w2 = np.load('pynn_models\\SmallNetworkParams\\w2_array.npy')
    v = np.load('pynn_models\\SmallNetworkParams\\v_array.npy')

    def sparsify(matrix):
        #need to implement something to check if value is zero
        test_w = [[] for i in range(matrix.shape[0]*matrix.shape[1])]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                test_w[i+matrix.shape[0]*j] = [i,j,matrix[i,j],0] 
        return test_w
    
    w1_sparse = sparsify(w1)
    w2_sparse = sparsify(w2)
    v_sparse = sparsify(v)

    #Get input data
    data_set_name='shd'
    config_path = 'PN_eq_dt2_oi'
    settings = Config(data_set_name,settings_file=config_path)
    trainloader, valloader,testloader, nb_steps = choose_data_params(
            data_set_name, settings, num_workers=4,pre_path='data/shd'
        ) 
    data_module = DataModule(trainloader, valloader, testloader)
    data_module.setup(stage="test")  # make sure test data is prepared
    test_loader = data_module.train_dataloader()
    test_batch = next(iter(test_loader))
    inputs, target = test_batch
    #inputs [time, batch, 1 dimensional, inputs]
    #inputs [586, 256, 1, 70]
    n_in = 70
    n_hi = 128   
    n_out = 20


    #converts the binary array of shape (n_neurons, time_steps) to a list n_neurons long, with each element of the list is another list of indices when the input neurons release a spike.
    spike_times = [[] for i in range(n_in)]
    spikerf_times = [[] for i in range(n_in)]
    for i in range(n_in):
        spike_times[i] = np.nonzero(inputs[:,0,0,i])     #hardcoded first input for now.
    for i in range(n_in):
        for j in range(len(spike_times[i])):
            spikerf_times[i].append(spike_times[i][j][0].item())
    # === Configure the simulator ================================================
    sim.setup(timestep=1) #It appears PyNN likes miliseconds as its default unit.


    #sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100) #May need to change this... found it in docs, don't know what it does.


    # === Build and instrument the network =======================================
    spike_source = sim.Population(n_in, sim.SpikeSourceArray(spike_times=spikerf_times)) #this may not be the correct syntax for setting multiple inputs.


    hidden_layer = sim.Population(n_hi, IFCurrExpBase(cm=1, v_rest=0, tau_m=tau_mem, i_offset=I_c, v_reset=0, tau_refrac=0, v_thresh=1000, tau_syn_E=20.0, tau_syn_I=20.0,))
    #            {V: 'mV', V_REST: 'mV', TAU_M: 'ms', CM: 'nF', I_OFFSET: 'nA', V_RESET: 'mV', TAU_REFRAC: 'ms'})
    hidden_layer.initialize(v=0)
    #Trying an "analog" neuron by having a LIF neuron with a very high threshold so that it never spikes. Still leaky though, uncertain about that.
    output_layer = sim.Population(n_out,IFCurrExpBase(cm=1, v_rest=0, tau_m=40, i_offset=0, v_reset=0, tau_refrac=0, v_thresh=32767, tau_syn_E=40.0, tau_syn_I=40.0,))
    #Threshold set to 32.767Volts. This is due to the 16.15 fixed point limitation. It can only represent up to 32767 sadly, no infinite threshold
    output_layer.initialize(v=0)

    hidden_layer.record(["spikes"])
    output_layer.record(["v"])

    conn_in2hid = sim.Projection(spike_source, hidden_layer, sim.FromListConnector(w1_sparse), #AllToAllConnector does not allow you to set weights for some reason.
                                sim.StaticSynapse()),

    conn_hid2hid = sim.Projection(hidden_layer, hidden_layer, connector = sim.FromListConnector(v_sparse),
                               synapse_type=sim.StaticSynapse()),
    
    conn_hid2out = sim.Projection(hidden_layer, output_layer, connector = sim.FromListConnector(w2_sparse),
                               synapse_type=sim.StaticSynapse()),


    # === Run the simulation =====================================================
    nb_steps=inputs.shape[0]
    sim.run(nb_steps)


    # === Save the results, optionally plot a figure =============================

    spikes = hidden_layer.get_data("spikes")
    volts = output_layer.get_data("v")  # if you recorded 'v'

    # === Clean up and quit ========================================================

    sim.end()

    spikes_handle = open('pynn_models\\spikes.pkl', 'wb') 
    pickle.dump(spikes, spikes_handle)
    spikes_handle.close()
    volts_handle = open('pynn_models\\volts.pkl', 'wb') 
    pickle.dump(volts, volts_handle)
    volts_handle.close()
    
