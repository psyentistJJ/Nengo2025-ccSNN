import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torchvision

import os
import glob
import shutil
import h5py

from algorithm.data.CustomSpikeDataset import CustomSpikeDataset


def bin_spike_events(events, num_neurons, bin_size_ms=1.0, t_max_ms=None):
    # ->chatgpt
    """
    Convert event-based spikes to a dense matrix [num_neurons, num_bins]

    Args:
        events: Tensor of shape [N_spikes, 2], where each row is [timestamp (s), neuron_idx]
        num_neurons: Total number of neurons
        bin_size_ms: Width of each time bin in milliseconds
        t_max_ms: Max time to bin (if None, use max timestamp)

    Returns:
        Tensor of shape [num_neurons, num_bins], binary spike counts
    """
    if events.shape[0] == 0:
        return torch.zeros((num_neurons, 1), dtype=torch.float32)

    timestamps_ms = events[:, 0] * 1000  # convert to ms
    neuron_ids = events[:, 1].long()

    if t_max_ms is None:
        t_max_ms = timestamps_ms.max().item()

    num_bins = int(t_max_ms // bin_size_ms) + 1
    spike_matrix = torch.zeros((num_neurons, num_bins), dtype=torch.float32)

    bin_indices = (timestamps_ms // bin_size_ms).long()

    for t, n in zip(bin_indices, neuron_ids):
        if 0 <= n < num_neurons and 0 <= t < num_bins:
            # spike_matrix[inputs_to_single_neurons[int(n)]['inputlayer_idxs'], t] += 1.0  # or just = 1.0 for binary
            # instead : sapmple sperately for each neuron!!!
            spike_matrix[n, t] += 1.0

    return spike_matrix.T.unsqueeze(1)


class HDF5SpikeDataset(Dataset):
    # ->chatgpt
    def __init__(self, file_paths, num_neurons, bin_size_ms=1.0, t_max_ms=500.0):
        self.file_paths = file_paths
        self.num_neurons = num_neurons
        self.bin_size_ms = bin_size_ms
        self.t_max_ms = t_max_ms

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with h5py.File(file_path, "r") as f:
            events = f["data"][:]  # [N, 2]: timestamp (s), neuron_id
            target = f["target"][()]
            duration = torch.tensor(f["duration"][()], dtype=torch.float32)
        events = torch.tensor(events, dtype=torch.float32)
        binned = bin_spike_events(
            events, self.num_neurons, self.bin_size_ms, self.t_max_ms
        )
        return binned, target, duration
