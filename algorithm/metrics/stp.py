import torch


def get_coinc(spikes_pred, spikes_ref, Δ):
    """
    -> adapted from NeuronPy: neuronpy.util.spiketrain.get_sync_masks (https://pythonhosted.org/neuronpy/spiketrain.html?highlight=get_sync_masks#neuronpy.util.spiketrain.get_sync_masks)
    returns number of coincidences of spikes between two spike trains with precision Δ

    arguments:
        spiketr_pred: spike train from prediction by model
        spiketr_ref: spike train reference/target data
        Δ: precision of coincidence in seconds
    returns:
        Ncoinc: number of coincidences of the two spike trains
    """
    idx_a = 0
    idx_b = 0

    mask_a = torch.zeros_like(spikes_pred)

    len_a = len(spikes_pred)
    len_b = len(spikes_ref)

    while idx_a < len_a and idx_b < len_b:
        val_a = spikes_pred[idx_a]
        val_b = spikes_ref[idx_b]

        diff = abs(val_a - val_b)
        if diff <= Δ:
            mask_a[idx_a] = 1

        if val_a == val_b:
            idx_a += 1
            idx_b += 1
        else:
            if val_a < val_b:
                idx_a += 1
            else:
                idx_b += 1
    Ncoinc = torch.sum(mask_a)

    return Ncoinc


def spike_time_precision(
    spk_pred, spk_ref, Δ=None, duration=None, dt=None
):
    """
    returns spike time precision according to Naud et al. (2009) for a deterministic model (N=1 repetition only)
    with gamma coincidence factor from Jolivet et al.(2006)

    arguments:
        volt_pred: voltage trace that was predicted by model
        volt_ref: reference/target voltage trace
        crossing0: value of 0 mv scaled according to model output
        Δ: precision of coincidence in seconds
        duration: duration of spike train in seconds
        dt: duration of a time steps in the data in seconds
    returns:
        A: gamma coincidence factor scaled by intrinsic reliability

    TODO: reliability??
    """
    R = 1  # 0.4  # 0.4 - 0.75????

    # use peaks instead
    # TODO: use 0mV as threshold for peaks?

    #Ncoinc = get_coinc(spk_pred, spk_ref, Δ)
    Ncoinc = torch.sum(torch.where(spk_pred == 1, torch.where(spk_pred==spk_ref,1,0), 0))

    # number of spikes per spike train
    Npred = torch.sum(spk_pred)
    Nref = torch.sum(spk_ref)

    v = Npred / duration
    exp_Ncoinc = 2 * v * Δ * Nref
    N = 1 - 2 * v * Δ

    Γ = (1 / N) * (Ncoinc - exp_Ncoinc) / (0.5 * (Nref + Npred))
    A = Γ / R

    return A