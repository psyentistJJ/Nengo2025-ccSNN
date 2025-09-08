import torch
from algorithm.model.Regularizers import LowerBoundL2, UpperBoundL1
from algorithm.model.weight_init import *
from algorithm.model.LightningSNN import Lightning_SNN
from algorithm.synapses import forward_synapse, recurrent_synapse

# Perez Nieves et al (2021) settings

# neuron and synapse
factor = 4
time_step = 0.5 * 1e-3 * factor
tau_mem = 20e-3 * factor
tau_mem_out = 20e-3 * factor
tau_syn = 10e-3 * factor
tau_syn_out = 10e-3 * factor
synapse_noise = 0.0

neuron_name = "LIF"  # LIF, QIF: [SNIC, HOM, mixed, other]
V_reset = 0.0
V_thresh = 1.0
V_rest = 0.0  # only for LIF
# ignore below for QIF
a = None  # only for QIF
V_sn = 0.5  # only for QIF
div_factor = 0.1  # only for QIF
other_reset = None  # only for QIF

# network parameters
batch_size = 256  # unknown
optim_class = torch.optim.Adam

optim_params = (0.9, 0.999)  # betas
scheduler = None
learning_rate = 0.001
weight_init_ = almost_xavier_normal_
target_weights = "target_model"  # None or 'target_model'
Dales_law = 0.7
N_train_only_weights = 0

# surrogate gradient scale
surr_grad_scale = 100.0

# regularization
reg = False

# baseline current abs value
set_I_c_range = True
set_tau_mem_range = True
I_c_abs = 1.0

train_weights = False
train_out = True
train_hidden = False
weight_scaling_factor = 1.0
recurrent_sparse = 1.0
low_rank_sparse = None
train_intrinsic = ["I_c", "beta"]
synapse = recurrent_synapse
training_mode = "classification"
hidden_mode = None
target_Lightning = None
target_model = None
params_from_target = None
hidden_in_loss = None
hidden_loss_scale = None
noise = None  # if parameter is initialized from target model and trained as well, add this factor *gaussian noise to the parameter init

specifics = dict(
    rms=dict(  # random manifolds -> not used in paper
        nb_inputs=3,
        nb_hidden=100,  # from Zenke & Vogels (2021)
        nb_outputs=2,
        N_epochs=40,
        reg=True,  # needed, otherwise 0 spikes
        LowerBound=LowerBoundL2,
        UpperBound=UpperBoundL1,
        lower_thresh=10 ** (-3),
        lower_strength=100.0,
        upper_thresh=100.0,  # 0-1000
        upper_strength=1.0,  # or 100
    ),
    rm=dict(  # random manifolds -> not used in paper
        nb_inputs=20,
        nb_hidden=100,  # from Zenke & Vogels (2021)
        nb_outputs=10,
        N_epochs=50,
        reg=True,  # needed, otherwise 0 spikes
        LowerBound=LowerBoundL2,
        UpperBound=UpperBoundL1,
        lower_thresh=10 ** (-3),
        lower_strength=100.0,
        upper_thresh=100.0,  # 0-1000
        upper_strength=1.0,  # or 100
    ),
    rml=dict(  # random manifolds -> not used in paper
        nb_inputs=20,
        nb_hidden=100,  # from Zenke & Vogels (2021)
        nb_outputs=10,
        N_epochs=40,
        reg=True,  # needed, otherwise 0 spikes
        LowerBound=LowerBoundL2,
        UpperBound=UpperBoundL1,
        lower_thresh=10 ** (-3),
        lower_strength=100.0,
        upper_thresh=100.0,  # 0-1000
        upper_strength=1.0,  # or 100
    ),
    mnist=dict(  # N-MNIST
        nb_inputs=2312,  # probably
        nb_hidden=128,
        nb_outputs=10,
        N_epochs=50,  # less than in paper
    ),
    shd=dict(  # SHD
        nb_inputs=70,  # probably 700 in original (not specified?), here, 70 is used because Cramer et al. (2020) said it's better, and it's faster
        nb_hidden=128,
        nb_outputs=20,
        N_epochs=150,  # double compared to paper
    ),
    other=dict(  # shd settings
        nb_hidden=128,
        N_epochs=50,
    ),
)
