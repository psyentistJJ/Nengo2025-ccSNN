import pyNN.spiNNaker as sim
import numpy as np
from spynnaker.pyNN.models.neuron.builds.if_curr_exp_base import IFCurrExpBase


class Build_sPyNNaker_Network(object):    
    '''
    Builds the network structure for running on Spinnaker.
    Arguments
        input: A binary Pytorch tensor of the input [time_steps, 1, number_of_neurons] 
        settings: collection of network parameters,
        w1: weights from the input to the hidden layer
        v: feedback weights for the hidden layer
        w2: weights from the hidden layer to the output layer.
    
    '''
    def __init__(self, input, model, alrdy_sprse=False, sep_cell_types=True,w1=None,v=None):
        self.time_step = model.time_step
        if  w1 == None:
            w1=model.w1
        if  v == None:
            v=model.v

        #converts the binary array of shape (n_neurons, time_steps) to a list n_neurons long, with each element of the list is another list of indices when the input neurons release a spike.
        spike_times = [[] for i in range(model.nb_inputs)]
        spikerf_times = [[] for i in range(model.nb_inputs)]
        for i in range(model.nb_inputs):
            spike_times[i] = np.nonzero(input[:,0,i])  
        for i in range(model.nb_inputs):
            for j in range(len(spike_times[i])):
                spikerf_times[i].append(model.time_step*1000*spike_times[i][j][0].item())

        #Convert the weights to a format more efficient for sparse matrices.
        if alrdy_sprse:
            w1_sparse_exc , w1_sparse_inh = sep_exc_inh(w1.detach())
            w2_sparse_exc , w2_sparse_inh = sparsify(model.w2.detach().numpy())
            v_sparse_exc , v_sparse_inh = sep_exc_inh(v.detach())
        else:
            w1_sparse_exc , w1_sparse_inh = sparsify(w1.detach().numpy())
            w2_sparse_exc , w2_sparse_inh = sparsify(model.w2.detach().numpy())
            v_sparse_exc , v_sparse_inh = sparsify(v.detach().numpy())
        
        #get time membrane constants for each neuron
        if sep_cell_types:
            tau_mem_h = np.zeros(model.nb_hidden)
            I_c_h = np.zeros(model.nb_hidden)
            for i in range(model.nb_hidden):
                tau_mem_h[i] = -1000*model.time_step/np.log(model.beta.detach().numpy()[model.cell_types[i]])
                I_c_h[i] = model.I_c.detach().numpy()[model.cell_types[i]]

        else:
            tau_mem_h = -1000*model.time_step/np.log(model.beta.detach().numpy())
            I_c_h = model.I_c.detach().numpy()


        # === Configure the simulator ================================================
        sim.setup(timestep=model.time_step*1000) #It appears PyNN likes miliseconds as its default unit.


        #sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100) #May need to change this... found it in docs, don't know what it does.
        #"The default number of neurons that can be simulated on each core is 256; larger populations are split up into 256-neuron chunks automatically by the software.
        # Note though that the cores are also used for other things, such as input sources, and delay extensions (which are used when any delay is more than 16 timesteps), 
        # reducing the number of cores available for neurons.""

        # === Build and instrument the network =======================================
        self.spike_source = sim.Population(model.nb_inputs, sim.SpikeSourceArray(spike_times=spikerf_times)) #this may not be the correct syntax for setting multiple inputs.


        self.hidden_layer = sim.Population(model.nb_hidden, IFCurrExpBase(cm=tau_mem_h, v_rest=model.rest, tau_m=tau_mem_h, i_offset=I_c_h, v_reset=model.reset.detach().numpy(), tau_refrac=0, v_thresh=model.th*1, tau_syn_E=(-model.time_step/np.log(model.alpha.detach())).numpy()*1000, tau_syn_I=(-model.time_step/np.log(model.alpha.detach())).numpy()*1000,))
        #            {V: 'mV', V_REST: 'mV', TAU_M: 'ms', CM: 'nF', I_OFFSET: 'nA', V_RESET: 'mV', TAU_REFRAC: 'ms'})
        self.hidden_layer.initialize(v=0)
        #Trying an "analog" neuron by having a LIF neuron with a very high threshold so that it never spikes. Still leaky though, uncertain about that.
        self.output_layer = sim.Population(model.nb_outputs,IFCurrExpBase(cm=(-model.time_step/np.log(model.beta_out.detach())).numpy()*1000, v_rest=0, tau_m=(-model.time_step/np.log(model.beta_out.detach())).numpy()*1000, i_offset=0, v_reset=0, tau_refrac=0, v_thresh=32767, tau_syn_E=(-model.time_step/np.log(model.alpha_out.detach())).numpy()*1000, tau_syn_I=(-model.time_step/np.log(model.alpha_out.detach())).numpy()*1000,))
        #Threshold set to 32.767Volts. This is due to the 16.15 fixed point limitation. It can only represent up to 32767 sadly, no infinite threshold
        self.output_layer.initialize(v=0)

        self.hidden_layer.record(["spikes"])
        self.hidden_layer.record(["v"])
        self.output_layer.record(["v"])

        self.conn_in2hid_exc = sim.Projection(self.spike_source, self.hidden_layer, sim.FromListConnector(w1_sparse_exc), #AllToAllConnector does not allow you to set weights for some reason.
                                    sim.StaticSynapse(), receptor_type = 'excitatory')
        self.conn_in2hid_inh = sim.Projection(self.spike_source, self.hidden_layer, sim.FromListConnector(w1_sparse_inh), #AllToAllConnector does not allow you to set weights for some reason.
                                    sim.StaticSynapse(), receptor_type = 'inhibitory')

        self.conn_hid2hid_exc = sim.Projection(self.hidden_layer, self.hidden_layer, connector = sim.FromListConnector(v_sparse_exc),
                                synapse_type=sim.StaticSynapse(), receptor_type = 'excitatory')
        self.conn_hid2hid_inh = sim.Projection(self.hidden_layer, self.hidden_layer, connector = sim.FromListConnector(v_sparse_inh),
                                synapse_type=sim.StaticSynapse(), receptor_type = 'inhibitory')
        
        self.conn_hid2out_exc = sim.Projection(self.hidden_layer, self.output_layer, connector = sim.FromListConnector(w2_sparse_exc),
                                synapse_type=sim.StaticSynapse(), receptor_type = 'excitatory')
        self.conn_hid2out_inh = sim.Projection(self.hidden_layer, self.output_layer, connector = sim.FromListConnector(w2_sparse_inh),
                                synapse_type=sim.StaticSynapse(), receptor_type = 'inhibitory')                        


    def run_the_sim(self, nb_steps):
        # === Run the simulation =====================================================
        sim.run(nb_steps*self.time_step*1000)

        # === Save the results, optionally plot a figure =============================

        spikes = self.hidden_layer.get_data("spikes")
        print(spikes)
        try:
            volts_h = self.hidden_layer.get_data("v")
            print(volts_h)
            volts_o = self.output_layer.get_data("v")  # if you recorded 'v'
            print(volts_v)
        except:
            return spikes, 1,1 #should have done this

        return spikes, volts_h, volts_o
    
    def reset(self):
        # Reset the simulation to t=0
        sim.reset()
    def end(self):
        # === Clean up and quit ========================================================
        sim.end()
        

def sparsify(matrix):
    w_positive = []
    w_negative = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0 :
                w_positive.append([i,j,matrix[i,j],0])
            else:
                w_negative.append([i,j,matrix[i,j],0])
    return w_positive, w_negative

def sep_exc_inh(matrix):
    '''
    Seperate the excitatory and inhibitory weights into two arrays.
    '''
    w_positive = []
    w_negative = []
    #print(matrix.indices().shape[1])
    for i in range(np.array(matrix.indices()).shape[1]):
            if matrix.values()[i] > 0 :
                w_positive.append([matrix.indices()[0,i].item(),matrix.indices()[1,i].item(),matrix.values()[i].item(),0])
            else:
                w_negative.append([matrix.indices()[0,i].item(),matrix.indices()[1,i].item(),matrix.values()[i].item(),0])
    return w_positive, w_negative