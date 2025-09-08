import torch.nn.init as init
import math
import torch
import numpy as np


def almost_xavier_uniform_(x, scale=1.0,fan_in=None,sq=False):
    if fan_in is None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(x)
    bound = scale / math.sqrt(fan_in)
    if sq:
        bound = math.sqrt(bound)
    print(f'scale: {scale}, bound is {bound}')
    init.uniform_(
        x, -bound, bound
    )  # similar to xavier but instead of sqrt(6/(fan_in + fan_out)), we use sqrt(1/fan_in)? from pereznieves (&zenke&vogels?)


def almost_xavier_normal_(x, scale=5.0,fan_in=None,sq=False):  # scale?
    if fan_in is None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(x)
    std = scale / math.sqrt(fan_in)
    if sq:
        std = math.sqrt(std)
    print(f'scale: {scale}, std is {std}')
    init.normal_(x, mean=0.0, std=std)


def set_recurrent_Dales_law(v,Dales_law,nb_hidden):
    print(f'Dales law is {Dales_law}')
    assert Dales_law<=1.0 and Dales_law>=0.0 , 'please provide fraction of excitatory neurons in hidden layer'
    Dales_law = int(nb_hidden * Dales_law) #convert Dales_law from fraction to absolute value for #exc hidden neurons
    print(f'Dales law is {Dales_law}')
    
    v[:Dales_law,:] = torch.abs(v[:Dales_law,:])
    v[Dales_law:,:] = -torch.abs(v[Dales_law:,:])
    return v,Dales_law

def set_recurrent_sparse(v,non_zero_sparse):
    """non_zero_sparse is percentage of non-zero elements in recurrent weights"""
    N_elements = torch.numel(v)
    ones=int(non_zero_sparse*N_elements)
    mask=np.zeros(N_elements)
    mask[:ones]=1
    np.random.shuffle(mask)
    sparse_mask=torch.tensor(mask.reshape(v.shape)).bool()
    v_sparse = torch.where(sparse_mask,v,0)
    return v_sparse,sparse_mask
