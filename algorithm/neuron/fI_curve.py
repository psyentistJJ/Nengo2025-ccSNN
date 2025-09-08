import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import snntorch.functional as SF

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler as Profiler  # PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import date, datetime


def dv_LIF(dt, v, tau_m, V_rest, I, I_c):
    """
    Function to calculate euler step for LIF.

    arguments:
        dt: float
            time step for euler step
        v: float
            voltage before euler step
        tau_m: float
            membrane time constant
        V_rest: float
            leak potential
        I: float
            input current

    returns
        dvdt*dt: float
            euler step value
    """
    dvdt = (-v + V_rest + (I-I_c)) / tau_m
    return dvdt * dt


def dv_QIF(dt, v, tau_m, V_sn, I, I_c, a):
    """
    Function to calculate euler step for QIF.

    arguments:
        dt: float
            time step for euler step
        tau_m: float
            membrane time constant
        V_sn: float
            leak potential
        I: float
            input current
        I_c: float
            current that determines spike onset
        a: float
            scaling factor for dv/dt parabola

    returns
        dvdt*dt: float
            euler step value
    """
    dvdt = (a * (v - V_sn) ** 2 + (I - I_c)) / tau_m
    return dvdt * dt


def int_LIF(dt, v0, tau_m, V_rest, I, I_c, V_thresh, V_reset):
    """
    Function to use euler integration to determine the membrane voltage over time for LIF.

    arguments:
        dt: float
            time step for euler step
        v0: float
            initial membrane voltage
        tau_m: float
            membrane time constant
        V_rest: float
            leak potential
        I: float
            input current
        V_thresh: float
            threshold voltage for spiking
        V_reset: float
            reset voltage after spike

    returns
        vs: list of float
            membrane voltage over time
    """
    v = v0
    vs = [v]
    for step in I:
        v += dv_LIF(dt, v, tau_m, V_rest, step, I_c)
        if v >= V_thresh:
            v = V_reset
        vs.append(v)
    return vs


def int_QIF(dt, v0, tau_m, V_sn, I, I_sn, V_peak, V_reset, a):
    """
    Function to use euler integration to determine the membrane voltage over time for QIF.

    arguments:
        dt: float
            time step for euler step
        v0: float
            initial membrane voltage
        tau_m: float
            membrane time constant
        V_sn: float
            leak potential
        I: float
            input current
        I_sn: float
            current that determines spike onset
        V_peak: float
            threshold voltage for spiking
        V_reset: float
            reset voltage after spike
        a: float
            scaling factor for dv/dt parabola

    returns
        vs: list of float
            membrane voltage over time
    """
    v = v0
    vs = [v]
    for step in I:
        v += dv_QIF(dt, v, tau_m, V_sn, step, I_sn, a)
        if v >= V_peak:
            v = V_reset
        vs.append(v)
    return vs


def get_freq(dt, voltage, cut):
    """
    Function to determine spiking frequency for a given voltage.

    arguments:
        voltage: np.ndarray
            membrane voltage over time
        cut: float
            threshold/peak voltage for neuron

    returns:
        freq: float
            spiking frequency for voltage

    """
    spike_idxs = np.where(np.isclose(voltage, cut))
    if len(spike_idxs[-1]) > 2:
        ISI = (spike_idxs[-1][-1] - spike_idxs[-1][-2]) * dt
        freq = 1 / ISI
    else:
        freq = 0
    return freq
