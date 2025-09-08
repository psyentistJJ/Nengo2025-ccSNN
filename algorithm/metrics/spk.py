import pyspike as spk
import numpy as np
from quantities import s

# metrics from PySpike.pyspike (spk_)

def get_spike_train_spk(spk_t,time_step):
    """transform spike train from 0-1 to event-based format"""
    timeseries=spk_t.flatten().numpy(force=True)
    spike_idxs=np.where(timeseries)[0]

    return spk.SpikeTrain(spike_idxs*time_step,[0.0, len(timeseries)*time_step],is_sorted=True)


def sync_metrics(spk1,spk2,time_step):
    spk1_=get_spike_train_spk(spk1,time_step)
    spk2_=get_spike_train_spk(spk2,time_step)

    _,_,Nhidden=spk1.shape

    spike_sync = spk.spike_sync(spk1_, spk2_)#, interval=ival)


    return spike_sync



    


