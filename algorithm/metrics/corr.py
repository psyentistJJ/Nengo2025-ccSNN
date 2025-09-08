import numpy as np

def get_corr(arr1,arr2):
    _,t1,N_neurons=arr1.shape
    _,t2,_=arr2.shape

    #make sure target and prediction have same number of time steps
    if t1<t2:
        arr2=arr2[:,:t1,:]
    elif t2<t1:
        arr1=arr1[:,:t2,:]

    arr1_=arr1.flatten(start_dim=0,end_dim=1).numpy(force=True)
    arr2_=arr2.flatten(start_dim=0,end_dim=1).numpy(force=True)

    #print(f'spk1 mean shape: {arr1_.mean(axis=0).shape}')
    x=np.corrcoef(arr1_-arr1_.mean(axis=0),arr2_-arr2_.mean(axis=0),rowvar=False) #correlate between neurons (flattened over time and batch size)
    #print(f'corrcoeff matrix shape: {x.shape}')
    #print(f'N hidden: {N_neurons}')
    corr = np.nanmean(x[N_neurons+np.arange(N_neurons),np.arange(N_neurons)]) # index only pairs of corresponding neurons (e.g. correlate neuron 100 in prediction with neuron 100 in target)average over all hidden neurons
    return corr