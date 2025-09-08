import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  ModelCheckpoint #,DeviceStatsMonitor, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler as Profiler  # PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from algorithm.neuron.neurons import *
import os

from algorithm.model.LightningSNN import Lightning_SNN

#from config:
self.V_reset = 0.0
self.V_thresh = 1.0
self.V_rest = 0.0
self.V_sn = 0.5
def get_neuron_intrinsic():
    
    return (neuron_fcts,
        U_L,
        tau_mems,
        tau_mems_out,
        resets,
        th,
        a,
        I_c,
        train_intrinsic,)

def get_QIF(
    settings,
    name,
    V_sn=0.5,
    div_factor=0.1,
    V_rest=0.0,
    V_thresh=1.0,
    V_reset=0.0,
    other_reset=None,
    perc_SNIC=0.5,
    match_QIF=True
):
    # V_sn = 0.5 #0.5 to match V_t = V_thresh
    # div_factor = 0.1 #!!! otherwise, spiking thresh < unstable FP
    div_SNIC = -div_factor
    div_HOM = div_factor

    QIF_options=['SNIC','HOM','mixed','other']

    if match_QIF:
        if name=='SNIC':
            QIF = match_QIF_params(
                settings.tau_mem,
                V_sn=V_sn,
                div=div_SNIC,  # -!!
                V_rest_LIF=V_rest,
                V_reset_LIF=V_reset,
                V_thresh_LIF=V_thresh,
            )
        elif name=='HOM':
            QIF = match_QIF_params(
                settings.tau_mem,
                V_sn=V_sn,
                div=div_HOM,  # +!!
                V_rest_LIF=V_rest,
                V_reset_LIF=V_reset,
                V_thresh_LIF=V_thresh,
            )

        elif name=='other':
            if other_reset is None:
                other_reset = 0.5
            QIF = match_QIF_params(
                settings.tau_mem,
                V_sn=V_sn,
                V_reset_QIF=other_reset,  # +!!
                V_rest_LIF=V_rest,
                V_reset_LIF=V_reset,
                V_thresh_LIF=V_thresh,
            )
        elif name=='mixed':
            QIF_SNIC = match_QIF_params(
                settings.tau_mem,
                V_sn=V_sn,
                div=div_SNIC,  # -!!
                V_rest_LIF=V_rest,
                V_reset_LIF=V_reset,
                V_thresh_LIF=V_thresh,
            )
            QIF_HOM = match_QIF_params(
                settings.tau_mem,
                V_sn=V_sn,
                div=div_HOM,  # +!!
                V_rest_LIF=V_rest,
                V_reset_LIF=V_reset,
                V_thresh_LIF=V_thresh,
            )
            QIF = get_mixed_QIF(QIF_SNIC, QIF_HOM, settings, perc=perc_SNIC)
    elif name in QIF_options:
        if V_reset_QIF is None:
        V_reset_QIF = V_sn + div
        I_c = 1.0 + I_c_LIF # add I_c of LIF for accurate natching??
        a = I_c / V_sn**2
        QIF = QIF_cont(
            V_sn=V_sn, V_reset=V_reset_QIF, tau_mem=tau_mem_QIF, V_peak=V_peak, a=a, I_c=I_c
        )
    elif name=='LIF':
        QIF=None
    else:
        raise NotImplementedError

    return QIF
