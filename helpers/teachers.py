import os
import glob
import numpy as np
import torch
from algorithm import *


def find_teacher(data_set_name, teacher_index, teacher_params, teacher_low_rank_sparse, teacher_sparsity, teacher_weight_init,teacher_neuron_model,size_string, synapse,pre_results='results'):
    sparse='sparse'
    if teacher_low_rank_sparse is None:
            sparse+=f'{int(10*teacher_sparsity)}'
    else:
        #low rank is only ran for small subset of other parameter combinations
        if teacher_sparsity!=0.5:
            #only consider a single setting for low-rank runs, random sparsity setting picked - overwritten further down anyway
            return None
        if teacher_params=='oi':
            #no low-rank teacher when weights are not trained
            return None
        sparse='lowrank'

    
    model_inst=get_inst(teacher_index,padding=3)

    if teacher_params=='oi':
        ckpt='epoch=199*.ckpt'
    else:
        ckpt='epoch=99*.ckpt'
    
    if pre_results !='':
        pre_results+='/'
    pre_teacher=f'{pre_results}{data_set_name}/teacher'

    print(f'{pre_teacher}/{teacher_neuron_model}_{synapse}')

    all_teachers = glob.glob(f'{pre_teacher}/{teacher_neuron_model}_{synapse}/{size_string}_*_{teacher_params}_{teacher_weight_init}-{sparse}/*/checkpoints/{ckpt}')
    
    all_accs = np.zeros(len(all_teachers))
    print(f'all teachers: {all_teachers}')
    print(f'all accs init: {all_accs}')
    all_str = []



    for i,name in enumerate(all_teachers):
        print(f'name in find teachers: {name}')
        print(i)
        print(f'acc string: {(name.split('val_acc=')[1]).split('.ckpt')[0]}')
        all_accs[i] = float((name.split('val_acc=')[1]).split('.ckpt')[0])
        all_str.append(name)
    print(all_accs)

    #use negative to sort descending
    idx_desc=np.argsort(-all_accs)
    print(f'idx desc: {idx_desc}')
    print(f'teacher_index: {teacher_index}')
    print(f'teacher_index: {idx_desc}, {idx_desc[teacher_index]}')
    try:
        i_best_teacher = all_str[idx_desc[teacher_index]]
    except:
        raise ValueError(f'Teacher not found: {pre_teacher}/{teacher_neuron_model}_{synapse}/{size_string}_*_{teacher_params}_{teacher_weight_init}-{sparse}/{model_inst}/checkpoints/{ckpt}')

    target_Lightning = Lightning_SNN.load_from_checkpoint(i_best_teacher, strict=False).eval().model

    return target_Lightning
