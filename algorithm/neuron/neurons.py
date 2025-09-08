import torch


def BLK_nonsp(mem, syn, rst, U_L, tau_mem, reset, a, I_c, beta, time_step):
    """Non-spiking neuron model for U_L=0"""
    new_mem = (beta * mem + (1 - beta) * (syn-I_c))  #added: -I_c
    return new_mem

# neuron functions for membrane potential
def LIF(mem, syn, rst, U_L, tau_mem, reset, a, I_c, beta, time_step):
    """
    -> adapted from Zenke notebook https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb
    Leaky integrate and fire neuron with direct integration for U_L=0.

    arguments
        mem: torch.Tensor
            membrane voltage
        syn: torch.Tensor
            synapse current from last timestep
        rst: torch.Tensor
            binary spike input information from last timestep
        U_L: float or torch.Tensor
            leak potential
        tau_mem: float or torch.Tensor
            membrane time constant
        reset: float or torch.Tensor
            reset voltage
        a: float or torch.Tensor
            scaling parameter
        I_c: float or torch.Tensor
            current that determines spike onset

    returns
        new_mem: torch.Tensor
            membrane potential for next timestep
    """
    # beta = torch.exp(-time_step / tau_mem)
    new_mem = (beta * mem + (1 - beta) * (syn-I_c)) #* (1.0 - rst) #added: -I_c
    new_mem = torch.where(rst == 1, reset, new_mem)

    return new_mem


def LIF2(mem, syn, rst, U_L, tau_mem, reset, a, I_c, beta, time_step):
    """
    Leaky integrate and fire neuron with euler integration for V_rest=0.

    arguments
        mem: torch.Tensor
            membrane voltage
        syn: torch.Tensor
            synapse current from last timestep
        rst: torch.Tensor
            binary spike input information from last timestep
        U_L: float or torch.Tensor
            leak potential
        tau_mem: float or torch.Tensor
            membrane time constant
        reset: float or torch.Tensor
            reset voltage
        a: float or torch.Tensor
            scaling parameter
        I_c: float or torch.Tensor
            current that determines spike onset

    returns
        new_mem: torch.Tensor
            membrane potential for next timestep
    """
    new_mem = (mem + time_step / tau_mem * (-(mem - U_L) + syn-I_c)) * (1.0 - rst) #added: -I_c
    new_mem = torch.where(rst == 1, reset, new_mem)

    return new_mem


def QIF(mem, syn, rst, U_L, tau_mem, reset, a, I_c, beta, time_step):
    """
    ->from Zenke notebook
    Quadratic integrate and fire neuron with euler integration.

    arguments
        mem: torch.Tensor
            membrane voltage
        syn: torch.Tensor
            synapse current from last timestep
        rst: torch.Tensor
            binary spike input information from last timestep
        U_L: float or torch.Tensor
            V_sn = 0.5 * (V_rest + V_th_Q), vertex of QIF parabola
        tau_mem: float or torch.Tensor
            membrane time constant
        reset: float or torch.Tensor
            reset voltage
        a: float or torch.Tensor
            scaling parameter
        I_c: float or torch.Tensor
            current that determines spike onset

    returns
        new_mem: torch.Tensor
            membrane potential for next timestep
    """

    if isinstance(U_L, (torch.Tensor)):
        U_L = torch.squeeze(U_L)
    if isinstance(I_c, (torch.Tensor)):
        I_c = torch.squeeze(I_c)
    if isinstance(a, (torch.Tensor)):
        a = torch.squeeze(a)

    new_mem = (mem + time_step / tau_mem * (a * (mem - U_L) ** 2 + syn - I_c)) * (
        1.0 - rst
    )
    new_mem = torch.where(rst == 1, reset, new_mem)

    return new_mem
