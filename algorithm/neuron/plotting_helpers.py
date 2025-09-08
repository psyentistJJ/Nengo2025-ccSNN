import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from algorithm.neuron.fI_curve import *


def spike_fn(x):
    """
    -> from Zenke notebook
    Function to rewrite input as binary spike train with spike=1 vs non-spike=0.
    The input is the membrane voltage with the spiking threshold subtracted, s.t. all values > 0 represent a spike.

    arguments
        x: torch.Tensor
            membrane voltage with spiking threshold subtracted

    returns
        out: torch.Tensor
            spike train

    """
    out = torch.zeros_like(x)
    out[x > 0] = 1.0
    return out


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5, name=""):
    """
    -> from Zenke notebook
    Function to plot voltage traces .

    arguments
        mem: torch.Tensor
            membrane voltage
        spk:
        dim: Tuple
            dimensionality of plot grid
        spike_height:
        name:
    """
    gs = GridSpec(*dim)
    if spk is not None:
        dat = 1.0 * mem
        dat[spk > 0.0] = spike_height
        dat = dat.detach().cpu().numpy()
    else:
        dat = mem.detach().cpu().numpy()
    print(f"dat shape is: {dat.shape}")
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
            # a0.set_ylim(0, 1)
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i])
        # ax.axis("off")
    plt.suptitle(name)
    return ax


def plot_example_input(
    neuron_fct,
    U_L,
    tau_mem,
    V_reset,
    V_th,
    a,
    I_c,
    beta,
    time_step,
    nb_hidden,
    batch_size,
    nb_steps,
    I=1.001,
):

    syn_shape = torch.ones((batch_size, nb_hidden))
    mem = torch.ones((batch_size, nb_hidden)) * V_reset

    # Here we define two lists which we use to record the membrane potentials and output spikes
    mem_rec = []
    spk_rec = []

    # Here we loop over time
    for t in range(nb_steps):
        mthr = mem - V_th
        out = spike_fn(mthr)  # just calculating the spiking output of the hidden layer
        # the next step is probably to do with the auto diff, getting the spike train out of the differentiation process
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = syn_shape * (I)
        syn = new_syn

        if not "LIF" in neuron_fct.__name__:
            beta = None

        new_mem = neuron_fct(
            mem=mem,
            syn=syn,
            rst=rst,
            U_L=U_L,
            tau_mem=tau_mem,
            reset=V_reset,
            a=a,
            I_c=I_c,
            beta=beta,
            time_step=time_step,
        )  # every neuron that spikes rst=1, gets reset to zero, all others are just multiplied by one, has to be changed to -reset_value*rst to allow for variable rsts
        new_mem[rst == 1] = V_reset

        mem_rec.append(mem)
        spk_rec.append(out)

        mem = new_mem
        syn = new_syn

    # Now we merge the recorded membrane potentials into a single tensor
    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    fig = plt.figure(dpi=100)
    plot_voltage_traces(
        mem_rec,
        spk_rec,
        dim=(1, 1),
        name=f"{neuron_fct.__name__} for input current {I}",
    )


def plot_fI_curves(
    dt,
    nb_steps,
    V_reset,
    tau_mem,
    V_rest,
    V_thresh,
    V_reset_SNIC,
    tau_mem_SNIC,
    V_peak_SNIC,
    V_reset_HOM,
    tau_mem_HOM,
    V_peak_HOM,
    V_sn,
    I_c,
    a,
    v0,
):
    N_currents = 500
    currents = np.linspace(-1, 3, N_currents)
    freqs_LIF1 = np.zeros((N_currents))
    freqs_SNIC = np.zeros((N_currents))
    freqs_HOM = np.zeros((N_currents))

    for i, current in enumerate(currents):
        I = np.ones(nb_steps) * current

        # v_LIF1_fixed = int_LIF(dt,V_rest, tau_mem, V_rest, I, V_thresh, V_reset)
        v_LIF1_fixed = int_LIF(dt, v0, tau_mem, V_rest, I, V_thresh, V_reset)
        freqs_LIF1[i] = get_freq(dt, v_LIF1_fixed, V_reset)

        # v_SNIC_free  = int_QIF(dt,V_rest, tau_m_SNIC, V_sn_SNIC, I, I_c_SNIC, V_peak_SNIC, V_reset_SNIC, a_SNIC)
        v_SNIC_free = int_QIF(
            dt, v0, tau_mem_SNIC, V_sn, I, I_c, V_peak_SNIC, V_reset_SNIC, a
        )
        freqs_SNIC[i] = get_freq(dt, v_SNIC_free, V_reset_SNIC)

        # v_HOM_free = int_QIF(dt,V_rest, tau_m_HOM, V_sn_HOM, I, I_c_HOM, V_peak_HOM, V_reset_HOM, a_HOM)
        v_HOM_free = int_QIF(
            dt, v0, tau_mem_HOM, V_sn, I, I_c, V_peak_HOM, V_reset_HOM, a
        )
        freqs_HOM[i] = get_freq(dt, v_HOM_free, V_reset_HOM)

    plt.plot(currents, freqs_LIF1, label="LIF")
    plt.plot(currents, freqs_HOM, label="HOM")
    plt.plot(currents, freqs_SNIC, label="SNIC")

    plt.legend()

    plt.xlabel(f"I")
    plt.ylabel("frequency")
    plt.legend()
    plt.show()
