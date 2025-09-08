import numpy as np


import torch
import torch.nn as nn

from algorithm.neuron.neurons import *
from algorithm.model.str_helpers import *
from algorithm.model.weight_init import *
from algorithm.model.activation import *


class SNN(nn.Module):

    def __init__(
        self,
        net_size,
        neuron_fct,
        neuron_intrinsic,
        synapse_intrinsic,
        time_step,
        weight_info,
        regularizers=None,
        train_intrinsic=[],
        clip_intrinsic={},
        surr_grad_scale=100.0,
        adapt_intrinsic=False,
        tau_mem_LIF=None,
        training_mode=None,
        params_from_target=None,
        hidden_in_loss=None,
        hidden_loss_scale=1,
        noise=None,
        is_student=False,
        shuffle=False,
        N_train_only_weights=0

    ):
        super().__init__()
        self.training_mode=training_mode

        self.surr_grad_scale = surr_grad_scale

        self.time_step = time_step

        self.batch_size, self.nb_inputs, self.nb_hidden, self.nb_outputs = net_size
        
        self.size_string = get_size_str(self.batch_size,self.nb_inputs,self.nb_outputs)
        self.train_weights,_,self.weight_init_,weight_init_dict,self.weight_scaling_factor,self.recurrent_sparse,self.low_rank_sparse = weight_info
        self.hidden_in_loss, self.hidden_loss_scale,self.mask_all,self.mask_rec,self.mask_unrec,self.sort_rec_unrec = self.set_hidden_in_loss(hidden_in_loss, hidden_loss_scale)

        (
            self.neuron_fct,
            self.neuron_name,
            self.U_L,
            self.tau_mem,
            self.th,
            self.reset,
            self.a,
            self.I_c,
            self.rest,
            self.beta,
            self.beta_out,
        ) = self.set_neuron_intrinsic(
            neuron_fct, neuron_intrinsic, time_step, train_intrinsic
        )
        
        self.activation_fct, self.name, self.is_spiking = self.set_activation_fct()

        # values to save to model irregardless of whether they are trained or not
        

        self.adapt_QIF_params = adapt_intrinsic if "QIF" in self.neuron_name else False
        if self.adapt_QIF_params:
            assert tau_mem_LIF is not None
            self.tau_mem_LIF = tau_mem_LIF
            print(f'adapting QIF params')
        self.clip_intrinsic = clip_intrinsic

        (
            self.synapse_fct,
            self.synapse_name,
            self.tau_syn,
            self.alpha,
            self.alpha_out,
            self.synapse_noise
        ) = self.set_synapse_intrinsic(synapse_intrinsic, time_step)
        self.set_neuron_params(params_from_target)

        shuffle_neuron, shuffle_weights = self.set_shuffle(shuffle,train_intrinsic,self.train_weights)

        self.noise = noise
        self.train_intrinsic = self.set_intrinsic_params_(train_intrinsic, params_from_target,shuffle=shuffle_neuron)

        print(f'\n\ntrain weights: {self.train_weights}, train_intrinsic: {self.train_intrinsic}\n\n')

        self.is_student=is_student
        print(f'self is student: {self.is_student}')
        self.w1, self.v, self.w2,self.Dales_law= self.set_weights(weight_info,shuffle_weights,train_intrinsic)

        self.N_train_only_weights,self.train_weights_later = self.set_split_train_inrt_weights(N_train_only_weights)

        (
            self.loss_fn,
            self.log_softmax_fn,
            self.regularize,
            self.reg_min_activity,
            self.reg_max_activity,
        ) = self.set_training_fcts(regularizers)

        self.batch_padded = False


    def set_activation_fct(self):
        if 'nonsp' in self.neuron_name:
            activation_fct = SaturatedReLU(1.0)
            name='RNN'
            is_spiking=False
        else:
            activation_fct = SurrGradSpike
            name='SNN'
            is_spiking=True
        print(f'model is {name}, is_spiking {is_spiking}')
        return activation_fct, name, is_spiking


    def set_split_train_inrt_weights(self,N_train_only_weights):
        """check if current epoch is in last last N epochs, if so, freeze intrinsic parameters and instead train some of the weights"""
        train_weights_later=[]
        if N_train_only_weights>0:
            for name, value in self.named_parameters():
                print(f'in SNN set_split: {name}')
                if value.requires_grad and (name in ['w1','w2','v']):#,'wsf']):
                    print('grad off')
                    value.requires_grad=False
                    train_weights_later.append(name)
        return N_train_only_weights, train_weights_later


    def set_shuffle(self,shuffle,train_intrinsic,train_weights):
        if shuffle:
            if len(train_intrinsic)==0:
                shuffle_neuron = True
            else:
                shuffle_neuron = False
            if not train_weights:
                shuffle_weights = True
            else:
                shuffle_weights = False
        else:
            shuffle_neuron=False
            shuffle_weights=False
        return shuffle_neuron,shuffle_weights

    def set_neuron_intrinsic(
        self, neuron_fct, neuron_intrinsic, time_step, train_intrinsic
    ):
        """
        Function to set intrinsic parameters for neuron model.

        arguments
            neuron_fct: function
                describes neuron model, choose from {LIF, LIF2, QIF}
            neuron_intrinsic: list of float or list of np.ndarray
                U_L: leak potential
                tau_mem: membrane time constant
                thresh: spiking threshold voltage
                reset: reset voltage
                a: scaling parameter for QIF dv/dy
                I_c: current that determines spike onset for QIF

        returns
            neuron_fct: function
                describes neuron model, choose from {LIF, LIF2, QIF}
            neuron_name: str
                name of neuron model and in QIF case specification for onset bifurcation (HOM/SNIC)
                U_L: leak potential
                tau_mem: membrane time constant
                th: spiking threshold voltage
                reset: reset voltage
                a: scaling parameter for QIF dv/dy
                I_c: current that determines spike onset for QIF
        """

        neuron_fct = neuron_fct
        neuron_name = neuron_fct.__name__

        U_L, tau_mem, tau_mem_out, th, reset, a, I_c = neuron_intrinsic

        print(f'first instance of tau_mem in SNN: {tau_mem.shape}')
        

        if neuron_fct == QIF:
            rest = U_L - np.sqrt(I_c / a)
            crit = U_L  # for QIF: this is V_sn
            if isinstance(reset,torch.Tensor):
                if (reset > crit).all():
                    neuron_name = neuron_name + "_HOM"
                elif (reset < crit).all():
                    neuron_name = neuron_name + "_SNIC"
                elif (reset == crit).all():
                    neuron_name = neuron_name + "_SNL"
                else:
                    neuron_name = neuron_name + "_mixed"
            else:
                if reset > crit:
                    neuron_name = neuron_name + "_HOM"
                elif reset < crit:
                    neuron_name = neuron_name + "_SNIC"
                elif reset == crit:
                    neuron_name = neuron_name + "_SNL"
                else:
                    neuron_name = neuron_name + "_mixed"
        else:
            rest = U_L
        print(
            f"{neuron_name} model with, U_rest {U_L}, rest {rest}, tau mem {tau_mem}, thresh {th}, I_c {I_c}"
        )
        print(f'tau_mem shape before tensor: {tau_mem.shape}')
        if not isinstance(tau_mem, torch.Tensor):
            tau_mem = torch.tensor(tau_mem)
        if not isinstance(tau_mem_out, torch.Tensor):
            tau_mem_out = torch.tensor(tau_mem_out)

        beta = None
        if (train_intrinsic is not None and "beta" in train_intrinsic) or ("LIF" in neuron_name  or "BLK_nonsp" in neuron_name):
            beta = torch.exp(
                -time_step / tau_mem
            )  # forward for QIF is fater with tau instead of indirect time constant beta
            print(f'tau_mem in SNN after tensor: {tau_mem.shape}')
            print(f'beta in SNN: {beta.shape}')
        # output layer is always LIF
        beta_out = torch.exp(-time_step / tau_mem_out)



        return (
            neuron_fct,
            neuron_name,
            U_L,
            tau_mem,
            th,
            reset,
            a,
            I_c,
            rest,
            beta,
            beta_out,
        )

    def set_synapse_intrinsic(self, synapse_intrinsic, time_step):
        if len(synapse_intrinsic)==3:
            synapse_noise=0.0 #old models cannot be loaded otherwise
            synapse_intrinsic.append(synapse_noise)
        synapse_fct, tau_syn, tau_syn_out, synapse_noise = synapse_intrinsic
        synapse_name = synapse_fct.__name__
        if not isinstance(tau_syn, torch.Tensor):
            tau_syn = torch.tensor(tau_syn)
        if not isinstance(tau_syn_out, torch.Tensor):
            tau_syn_out = torch.tensor(tau_syn_out)
        alpha = torch.exp(-time_step / tau_syn)
        alpha_out = torch.exp(-time_step / tau_syn_out)

        return synapse_fct, synapse_name, tau_syn, alpha, alpha_out,synapse_noise

    

    def set_weights(self, weight_info,shuffle=False,train_intrinsic=[]):
        """
        Function to set intial weights for synapses. Weights can be parameters for optimization or constants.

        arguments
            params: bool
                whether or not to optimize weights during training (True) or keep them constant (False)
            weight_init: fct or dict
                if function is provided, use that for initialization, otherwise, a dict with weight names and values should be provided
            Dales_law: None or Float
                decide if recurrent weight matrix should follow Dale's law
        returns
            weights: list of torch.Tensor
                w1: weights for synapases between input and hidden layer
                v: None or weights for recurrent synapses of hidden layer
                w2: weights for synapses between hidden and output layer
        """
        train_weights,Dales_law,weight_init_,weight_init_dict,weight_scaling_factor,recurrent_sparse,low_rank_sparse = weight_info

        print(f'model is student: {self.is_student}, low_rank is {low_rank_sparse}')
        print(f'init w1')
        if ('BLK' in self.training_mode) or (self.is_student):
            assert weight_init_dict is not None, "please provide target weights"
            w1 = weight_init_dict['w1'].clone()
            w1=nn.Parameter(w1,requires_grad=False)
        else:
            w1 = torch.empty((self.nb_inputs, self.nb_hidden), requires_grad=train_weights)
            weight_init_(w1,scale=weight_scaling_factor)
            w1=nn.Parameter(w1,requires_grad=train_weights)
            if 'w1' in train_intrinsic:
                #overwrite train_weights if weight name is specifically mentioned
                w1=nn.Parameter(w1,requires_grad=True)
            else:
                w1=nn.Parameter(w1,requires_grad=train_weights)

        print(f'init w2')
        if ('BLK' in self.training_mode) or (self.is_student and (not train_weights)):
            w2 = weight_init_dict['w2'].clone()
            w2=nn.Parameter(w2,requires_grad=False)
        else:
            w2 = torch.empty((self.nb_hidden, self.nb_outputs), requires_grad=train_weights)
            weight_init_(w2,scale=weight_scaling_factor)
            if 'w2' in train_intrinsic:
                #overwrite train_weights if weight name is specifically mentioned
                w2=nn.Parameter(w2,requires_grad=True)
            else:
                w2=nn.Parameter(w2,requires_grad=train_weights)

        print(f'init v')
        if self.synapse_name == "recurrent_synapse":
            if (self.is_student and (not train_weights)):
                assert weight_init_dict is not None, "please provide target weights"
                v = weight_init_dict['v'].clone()
                if shuffle:
                    idx = torch.randperm(v.nelement())
                    v = v.view(-1)[idx].view(v.size())
                v=nn.Parameter(v,requires_grad=train_weights)
            else:   
                assert (recurrent_sparse==1.0) or low_rank_sparse is None, 'Please choose between sparsity percentage or low-rank constraint for recurrent weight matrix v'
                if low_rank_sparse is not None:
                    print(f'low rank sparse dim is {low_rank_sparse}')
                    #overwrite v - instead construct from low-rank matrices and train those
                    #low_rank_sparse is dimensionality of each low-rank matrix
                    if self.is_student and (not train_weights):
                        self.subv1 = weight_init_dict['subv1'].clone()
                        self.subv2 = weight_init_dict['subv2'].clone()
                        self.subv1 = nn.Parameter(self.subv1,requires_grad=train_weights)
                        self.subv2 = nn.Parameter(self.subv2,requires_grad=train_weights)
                    else:
                        self.subv1 = nn.Parameter(torch.empty((self.nb_hidden, low_rank_sparse)), requires_grad=train_weights)
                        self.subv2 = nn.Parameter(torch.empty((self.nb_hidden, low_rank_sparse)), requires_grad=train_weights)
                        # for 1D tensors, manually provide fan_in of recurrent weight matrix
                        if weight_init_ == almost_xavier_normal_:
                            #half the variance for each 1D gaussian
                            weight_init_(self.subv1,scale=weight_scaling_factor,fan_in=self.nb_hidden, sq=True) 
                            weight_init_(self.subv2,scale=weight_scaling_factor,fan_in=self.nb_hidden, sq=True)
                        elif weight_init_ == almost_xavier_uniform_:
                            #initialize beta distributions to get uniform??? - TODO
                            weight_init_(self.subv1,scale=weight_scaling_factor,fan_in=self.nb_hidden, sq=True)
                            weight_init_(self.subv2,scale=weight_scaling_factor,fan_in=self.nb_hidden, sq=True)
                        else:
                            raise NotImplementedError
                    v = self.subv1@self.subv2.T
                    v = nn.Parameter(v,requires_grad=False)
                    print(f'low rank v shape is {v.shape}')
                    print(f'v requires grad: {v.requires_grad}')
                else:
                    v = torch.empty((self.nb_hidden, self.nb_hidden))
                    weight_init_(v,scale=weight_scaling_factor)
                    print(f'initializing new v')
                    if recurrent_sparse <1.0:
                        v,self.sparse_mask = set_recurrent_sparse(v,recurrent_sparse)
                    
                    #Dales law only when not low-rank
                    if Dales_law is not None:
                        v,Dales_law=set_recurrent_Dales_law(v,Dales_law,self.nb_hidden)
                    else:
                        print('no Dales law')
                    
                    if 'v' in train_intrinsic:
                        #overwrite train_weights if weight name is specifically mentioned
                        v = nn.Parameter(v,requires_grad=True)
                    else:
                        v = nn.Parameter(v,requires_grad=train_weights)

            if not (train_weights or self.is_student):
                
                if 'wsf' in train_intrinsic:
                    #overwrite train_weights if weight name is specifically mentioned
                    self.weight_scaling_factor= nn.Parameter(torch.tensor(float(weight_scaling_factor)), requires_grad=True)
                else:
                    self.weight_scaling_factor= nn.Parameter(torch.tensor(float(weight_scaling_factor)), requires_grad=False)
                self.v_scaled = v*self.weight_scaling_factor
        else:
            v = None

        weights = [w1, v, w2,Dales_law]

        return weights

    def set_neuron_params(self, params_from_target):
        """helper function to use I_c and beta from target model"""
        if params_from_target is not None:
            for name in params_from_target.keys():
                self.__setattr__(name, params_from_target[name].clone())


    def set_distr_value_(self, name, param=False, noise=None,shuffled_hidden=None):
        """
        Function to distribute values for one intrinsic parameter for all neurons in the hidden layer.

        arguments
            name: str
                name of intrinsic neuron parameter (and attribute of the SNN class)
            param: bool
                determine if the intrinsic parameter should be optimized during training (True) or contant (False)
        """
        value = self.__getattribute__(name)
        if not isinstance(value, (np.ndarray, list, tuple, torch.Tensor)):
            value = torch.full((self.nb_hidden,), value)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if shuffled_hidden is not None:
            value=value[shuffled_hidden]

        if noise is not None:
            add_noise = noise* torch.randn_like(value)
            value = value+add_noise

        if param:
            value = nn.Parameter(value)
            # self.neuron_name += f'{name}' instead check hyperparams
            # print(name)
        else:
            value = nn.Parameter(value, requires_grad=False)
        self.__setattr__(name, value)

    def set_intrinsic_params_(self, train_intrinsic, params_from_target,shuffle=False):
        """
        Function to distribute values for all intrinsic parameters that are optimized during training for all neurons in the hidden layer.
        This way, each neuron in the hidden layer can have different intrinsic parameters.
        #TODO: add distribution for values who are not optimized, add different kinds of distributions (normal, uniform,...)

        arguments
            train_intrinsic: list of str
                list of neuron_intrsinsic attributes that should be trained

        returns
            train_intrinsic: list of str
                    list of neuron_intrsinsic attributes that should be trained
        """
        print(f"train_intrinsic: {train_intrinsic}")
        print(len(train_intrinsic))
        if shuffle: #same shuffling index for all parameters - whole neurons are shuffled
            shuffled_hidden=np.arange(self.nb_hidden)
            np.random.shuffle(shuffled_hidden)
        else:
            shuffled_hidden=None
        if len(train_intrinsic) > 0:
            for name in train_intrinsic:
                if name not in ['w1','w2','v','wsf']:#avoid weight-related parameters in this function
                    if (params_from_target is not None) and (name in params_from_target.keys()) and self.noise is not None:
                        self.set_distr_value_(name, param=True, noise=self.noise,shuffled_hidden=shuffled_hidden)
                    else:
                        self.set_distr_value_(name, param=True, noise=None,shuffled_hidden=shuffled_hidden)
        if 'reset' not in train_intrinsic:
            self.set_distr_value_('reset', param=False,shuffled_hidden=shuffled_hidden)
        if 'I_c' not in train_intrinsic:
            self.set_distr_value_('I_c', param=False,shuffled_hidden=shuffled_hidden)
        if 'beta' not in train_intrinsic and self.beta is not None:
            self.set_distr_value_('beta', param=False,shuffled_hidden=shuffled_hidden)



        return train_intrinsic.copy()

    def set_training_fcts(self, regularizers):
        """
        Function to prepare other functions for training and evaluation.
        arguments
            mode: in {'classification','trace_learning'}

        returns
            mode:
            loss_fn: loss function
            log_softmax_fn: function to evaluate output
            regularize
        """
        for n, p in self.named_parameters():
            print(f"Parameter name: {n}, requires_grad: {p.requires_grad}")

        # if loss == 'MNIST':
        #    self.loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        # else:
        if self.training_mode == 'classification' or self.training_mode == 'classification_BLK':
            loss_fn = nn.NLLLoss()
            log_softmax_fn = nn.LogSoftmax(dim=1)
            print('\n\nusing classification loss\n\n')
        elif self.training_mode=='trace_learning' or self.training_mode=='trace_learning_BLK':
            loss_fn = nn.MSELoss() #mean over all time points and all neurons
            log_softmax_fn = None
            print('\n\nusing trace_learning loss\n\n')
        if regularizers is None:
            reg_min_activity = None
            reg_max_activity = None
            regularize = False
        else:
            reg_min_activity, reg_max_activity = regularizers
            regularize = True

        return loss_fn, log_softmax_fn, regularize, reg_min_activity, reg_max_activity

    def set_hidden_in_loss(self, hidden_in_loss, hidden_loss_scale):
        if hidden_in_loss is not None:
            print(f'hidden in loss is {hidden_in_loss}')
            print(f'hidden in loss is {type(hidden_in_loss)}')
            if isinstance(hidden_in_loss, (list, np.ndarray)): #choose specific hidden units
                pass
            elif isinstance(hidden_in_loss, float): #randomly choose hidden units
                assert hidden_in_loss>=0.0 and hidden_in_loss<=1.0, 'please provide a fraction 0<=p<=1 of hidden units'
                N_hidden_provided = int(self.nb_hidden * hidden_in_loss)
                print(f'N hidden provided: {N_hidden_provided}')
                hidden_in_loss = np.random.choice(np.arange(self.nb_hidden),N_hidden_provided,replace=False)
                print(f'len hidden in loss: {len(hidden_in_loss)}')
            else:
                raise ValueError('please provide a fraction 0<=p<=1 for random hidden units or an array/list of specific hidden units')
            
            mask_all = np.ones(self.nb_hidden, dtype=bool)
            mask_rec = np.zeros(self.nb_hidden, dtype=bool)
            mask_rec[hidden_in_loss] = True
            
            mask_unrec = np.ones(self.nb_hidden, dtype=bool)
            mask_unrec[hidden_in_loss] = False

        else:
            mask_all = np.ones(self.nb_hidden, dtype=bool)
            mask_rec = np.zeros(self.nb_hidden, dtype=bool)
            mask_unrec = np.ones(self.nb_hidden, dtype=bool)
        print(f'N rec = {np.sum(mask_rec)}')
        print(f'N unrec = {np.sum(mask_unrec)}')

        print(f'hidden in loss in SNN are {hidden_in_loss}')

        all_hidden=np.ones(self.nb_hidden)
        all_hidden[hidden_in_loss]=0 #mark recorded neurons
        sort_rec_unrec=np.argsort(all_hidden)

        return hidden_in_loss, hidden_loss_scale, mask_all,mask_rec,mask_unrec,sort_rec_unrec

    def permute_batch_dim(self, inputs):
        permute_order = list(np.arange(len(inputs.size())))
        permute_order[:2] = [1, 0]

        inputs = inputs.permute(permute_order)

        return inputs

    def pad_batch_(self, inputs, a):
        # changes attribute batch_padded
        #print(f'a in pad bacth is {a}')
        #print(f'inputs in pad bacth is {inputs.shape}')
        print(self.batch_size)
        if a == self.batch_size:
            self.batch_padded = False
            pass
        elif a < self.batch_size:
            self.batch_padded = True

            # pad first dim to self.batch_size
            p1d = [0, 1]
            p1d = [0] * 2 * len(inputs.size())
            p1d[-1] = self.batch_size - a
            # p1d = (0, 1) # pad last dim by 1 on each side
            inputs = torch.nn.functional.pad(inputs, p1d, "constant", 0)
        else:
            raise NotImplementedError

        return inputs

    def to_input_device_(self, list_attr, inputs):
        for name in list_attr:
            if not name in self.train_intrinsic:
                value = self.__getattribute__(name)
                if isinstance(value, torch.Tensor):
                    value = value.to(inputs)
                    self.__setattr__(name, value)

    def _prep_forward(self,inputs):
        if self.low_rank_sparse is not None:
            print('creating low-rank v')
            #self.v.data = self.subv1@ self.subv2.T

            v = self.subv1@ self.subv2.T
            self.v.data = v #self.v itself cannot be used -> breaks gradients

            #self.v = self.subv1@ self.subv2.T
            #self.v = self.v.to(inputs)
            v = v.to(inputs)
        elif not (self.train_weights or self.is_student):
            print(f'device v: {self.v.device}, device wsf: {self.weight_scaling_factor.device}')
            self.v = self.v.to(inputs)
            self.v_scaled =self.v*self.weight_scaling_factor
            v = self.v_scaled
        else:
            v = self.v

        if len(inputs.size()) > 3:
            inputs = self.permute_batch_dim(
                inputs
            )  # since time dim was padded before -> now set batch first
            sz = inputs.size()
            print(inputs.size())
            a, nb_steps = sz[0], sz[1]
            inputs = inputs.reshape((a, nb_steps, -1))
            print(inputs.size())

        else:
            sz = inputs.size()
            a, nb_steps = sz[0], sz[1]

        inputs = self.pad_batch_(inputs, a)

        if self.adapt_QIF_params:
            QIF = match_QIF_params(
                tau_mem_LIF=self.tau_mem_LIF, V_reset_QIF=self.reset  # V_sn = 0.5
            )
            self.tau_mem = QIF.tau_mem
            self.th = QIF.V_peak
            print(f'adapting QIF params in forward')

        self.to_input_device_(
            #["U_L", "tau_mem", "th", "reset", "a", "I_c", "rest", "alpha", "beta"],
            ["U_L", "tau_mem", "th", "a", "rest", "alpha",],
            inputs,
        )

        syn = torch.zeros((self.batch_size, self.nb_hidden)).to(inputs)

        mem = (
            torch.ones((self.batch_size, self.nb_hidden)).to(inputs)
            # * self.rest
            * self.reset
        )

        mem_rec = []
        spk_rec = []

        return inputs,v,mem,syn,mem_rec,spk_rec,nb_steps

    

    def forward(self, inputs, target):
        """
        -> adapted from Zenke notebooks (https://github.com/fzenke/spytorch/tree/main/notebooks)
        Function to generate network output for a specific input.

        arguments
            inputs: torch.tensor
                input values, shape should be (batch_size x time_steps x input_neurons)

        returns
            out_rec: torch.Tensor
                voltages of output neurons
            other_recs: list of torch.Tensor
                mem_rec: voltages of hidden neurons
                spk_rec: spike trains of hidden neurons (binary)
        """
        # TODO: split into 2 layers??

        inputs,v,mem,syn,mem_rec,spk_rec,nb_steps = self._prep_forward(inputs)


        #input -> hidden
        h1 = torch.einsum("abc,cd->abd", (inputs, self.w1))
        self.last_h1 = h1

        #train time constant indirectly through beta
        if self.beta is not None:
            self.tau_mem = -self.time_step / torch.log(self.beta)

        # simulate hidden layer over time
        print(f"nb_steps in forward is {nb_steps}")
        for t in range(nb_steps):
            #rst, out, mthr = self.call_activation_fct(mem,inputs)
            if self.is_spiking:
                mthr = mem - self.th
                out = SurrGradSpike.apply(mthr, self.surr_grad_scale).to(inputs) #apparently, SurrGradSpike cannot be called in helper fct - gradients break?
                rst = out.detach()  # We do not want to backprop through the reset
            else:
                out=self.activation_fct(mem).to(inputs)
                rst=out
                
            new_syn = self.synapse_fct(self.alpha, syn, h1, t, rst, v,self.synapse_noise) #TODO: add I_c here instead of in neuron_fct???
            new_mem = self.neuron_fct(
                mem=mem,
                syn=syn,
                rst=rst,
                U_L=self.U_L,
                tau_mem=self.tau_mem,
                reset=self.reset,
                a=self.a,
                I_c=self.I_c,
                beta=self.beta,
                time_step=self.time_step,
            )

            mem_rec.append(mem)
            spk_rec.append(out)

            mem = new_mem
            syn = new_syn

        mem_rec = torch.stack(mem_rec, dim=1).to(inputs)
        spk_rec = torch.stack(spk_rec, dim=1).to(inputs)
        other_recs = [mem_rec, spk_rec]

        # hidden->readout
        h2 = torch.einsum(
            "abc,cd->abd", (spk_rec, self.w2)
        )
        flt = torch.zeros((self.batch_size, self.nb_outputs)).to(inputs)
        out = torch.zeros((self.batch_size, self.nb_outputs)).to(inputs)
        out_rec = [out]

        #simulate readout layer over time
        for t in range(nb_steps):
            new_flt = self.alpha_out * flt + h2[:, t]
            new_out = self.beta_out * out + (1 - self.beta_out) * flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)

        return out_rec, other_recs


    def clip_params(self, device):
        if len(self.train_intrinsic) > 0:

            taus_cotrained = (
                True
                if ("alpha" in self.train_intrinsic) or ("beta" in self.train_intrinsic)
                else False
            )

            for name, value in self.named_parameters():
                
                if (name in self.train_intrinsic) and (name not in ['w1','w2','v','wsf']):
                    clamp_min, clamp_max = self.clip_intrinsic[name]
                    if (clamp_min is not None) or (
                        clamp_max is not None
                    ):  # only clip if one value is not None
                        value.data.clamp_(clamp_min, clamp_max)

                    if name == "reset":
                        value.data.clamp_(
                            torch.tensor([-3.0], device=device),
                            torch.tensor(self.th - self.time_step, device=device), #new
                        )  # 1mV below threshold
                '''
                elif (
                    name in ["w1", "v", "w2"] and taus_cotrained
                ):  # clamp weights tp avoid explofing gradients? #TODO -> still not working for QIF
                    if value.requires_grad:
                        print(f"\nclamping {name}\n")
                        value.data.clamp_(-1.0, 1.0)'''

        # when weights trained with exc vs inh: obey Dale's law
        if self.recurrent_sparse <1.0:
            for name, value in self.named_parameters():
                if name == 'v' and value.requires_grad:
                    with torch.no_grad():
                        print(f'sparse mask shape: {self.sparse_mask.shape}')
                        value[self.sparse_mask] = torch.clamp(value.data[self.sparse_mask], min=0, max=0)
                        print(f'clamped to sparsity {self.recurrent_sparse}')
                        print(f'clamped shape: {value.shape}')


        if (self.Dales_law is not None):
            assert self.low_rank_sparse is None, 'low rank and Dales law do not go together'
            for name, value in self.named_parameters():
                if name == 'v' and value.requires_grad:
                    with torch.no_grad():
                        print(f'self.Dales_law in clamp: {self.Dales_law},{type(self.Dales_law)} ')
                        value[:self.Dales_law,:] = torch.clamp(value.data[:self.Dales_law,:], min=0, max=None)
                        value[self.Dales_law:,:] =torch.clamp(value.data[self.Dales_law:,:], min=None, max=0)


