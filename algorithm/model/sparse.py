import math
import torch
import numpy as np

#-> function from SWAT (https://github.com/AamirRaihan/SWAT/blob/bbe271afa6e5ead3c02b20409594e17b6265b958/SWAT-code/cifar10-100-code/topk/drop.py#L5)

def matrix_drop_th(tensor,select_percentage):
        tensor_shape    =   tensor.shape
        k               =   int(math.ceil(select_percentage*(tensor_shape[0]*tensor_shape[1])))
        topk            =   tensor.view(-1).abs().topk(k)
        threshold       =   topk[0][-1]
        index           =   tensor.abs()>=(threshold)
        index           =   index.type_as(tensor)
        tensor          =   (tensor*index)
        return tensor,threshold
