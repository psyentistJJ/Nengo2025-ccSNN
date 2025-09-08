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
from algorithm.data.HDF5SpikeDataset import HDF5SpikeDataset
from algorithm.data.customPadTensors import customPadTensors


def get_data_loaders(
    x_data,
    y_data,
    x_data_val,
    y_data_val,
    x_data_test,
    y_data_test,
    plot_ex=False,
    num_workers=0,
    batch_size=256,
):
    """
    Function to create data loaders for train and test set given input and target tensors, e.g. for random manifolds dataset.

    arguments
        x_data: torch.Tensor
            training set input data
        y_data: torch.Tensor
            training set target data
        x_data_test: torch.Tensor
            test set input data
        y_data_test: torch.Tensor
            test set target data

    returns
        train_dataloader: DataLoader
            data loader for training data
        test_dataloader: DataLoader
            data loader for test data
    """

    training_data = CustomSpikeDataset(x_data, y_data)
    val_data = CustomSpikeDataset(x_data_val, y_data_val)
    test_data = CustomSpikeDataset(x_data_test, y_data_test)

    print(f"len {training_data.__len__()}")

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # show example of laoded data
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    if plot_ex:
        train_features, train_labels = next(iter(train_dataloader))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img.cpu().t(), cmap=plt.cm.gray_r, aspect="auto")
        plt.xlabel("Time (ms)")
        plt.ylabel("Unit")
        sns.despine()
        print(f"Label: {label}")

    return train_dataloader, val_dataloader, test_dataloader


def get_transform(data_set_name, sensor_size, time_step, encoding_dim=100):
    if data_set_name == "mnist":
        # -> from tonic docs
        # Denoise removes isolated, one-off events
        # time_window
        transform = transforms.Compose(
            [
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(
                    sensor_size=sensor_size,
                    # time_window=1000)
                    time_window=time_step * 1e6,
                ),  # s to us
            ]
        )
    elif data_set_name == "shd":

        print(f"encoding dim: {encoding_dim}, sensor size: {sensor_size}")

        # -> Gregor Lenz (https://lenzgregor.com/posts/train-snns-fast/)

        transform = tonic.transforms.Compose(
            [
                transforms.Downsample(
                    spatial_factor=encoding_dim / np.prod(sensor_size)
                ),  # 700), #downsample to encoding_dim
                # transforms.CropTime(max=1e6),
                transforms.ToFrame(
                    # sensor_size=sensor_size, time_window=time_step * 1e6, include_incomplete=True
                    sensor_size=(encoding_dim, 1, 1),
                    time_window=time_step * 1e6,
                    include_incomplete=True,
                ),
            ]
        )
    else:
        raise NotImplementedError

    return transform


def get_tonic_dataset(
    data_set_name, time_step, mode=None, transform=False, encoding_dim=100, pre_path=""
):
    """
    Download tonic dataset.

    arguments
        data_set_name: str
            data set, choose from {mnist, shd}
        mdoe: str
            whether to generate 'train' or 'test' data
        transform: bool
            whether to rotate dataset, only recommended if train=True and for image data?

    returns
        cached_dataset: DiskCachedDataset
            tonic dataset for data_set_name
    """
    train = True if mode == "train" else False

    if data_set_name == "mnist":
        Data = tonic.datasets.NMNIST
    elif data_set_name == "shd":
        Data = tonic.datasets.hsd.SHD
    else:
        raise NotImplementedError

    sensor_size = Data.sensor_size
    init_transform = get_transform(
        data_set_name,
        sensor_size=sensor_size,
        time_step=time_step,
        encoding_dim=encoding_dim,
    )

    dataset = Data(
        save_to=f"{pre_path}data/{data_set_name}/pre_cache_{time_step}",
        transform=init_transform,
        train=train,
    )
    print(f"data already downloaded: {dataset._check_exists()}")

    dataloader = DataLoader(dataset, num_workers=2)
    print(next(iter(dataloader)))

    # also apply random rotations (for image data in training set)
    rotate = tonic.transforms.Compose(
        [torch.from_numpy, torchvision.transforms.RandomRotation([-10, 10])]
    )
    if transform:
        cached_dataset = DiskCachedDataset(
            dataset,
            transform=rotate,
            cache_path=f"{pre_path}data/{data_set_name}/cache_{time_step}/{mode}",
        )
    else:
        cached_dataset = DiskCachedDataset(
            dataset,
            cache_path=f"{pre_path}data/{data_set_name}/cache_{time_step}/{mode}",
        )

    return cached_dataset


def train_val_split(dataset, batch_size):
    """
    split tonic 'training' set into actual training and validation sets
    """
    dataset_size = len(dataset)

    split_ratio = 0.9
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    # split deterministically
    train_indices = list(range(train_size))  # first 90% of training set
    val_indices = list(
        range(train_size, dataset_size)
    )  # remaining data points for validation set

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
    )

    return trainloader, valloader


def choose_data_params(data_set_name, settings, num_workers=4, pre_path=""):
    """
    Function to create data loaders for a data set.
    WARNING: data_set_name has to be chosen from {'synth', 'rm', 'mnist'}.

    arguments
        data_set_name: str
            short name of data set that models should be trained on, choose from {'synth', 'rm', 'mnist'}
            synth: synthetic random data set, 1 batch, no generalization
            rm: random manifolds data set, several batches, generalization possible
            mnist: N-MNIST (spiking digits), several batches, generalization possible
        settings: Config
            should contain all relevant training parameters for each data set

    returns
        trainloader: DataLoader
            data loader for training data
        testloader: DataLoader
            data loader for test data
        net_size: list of int
            batch_size: batch size
            nb_inputs: number of input neurons
            nb_hidden: number of hidden neurons
            nb_outputs: number of output neurons
    """

    if data_set_name == "rms":
        assert settings.nb_inputs == 3  # 3
        assert settings.nb_outputs == 2  # 2
        samples = 12800  # 2560

        nb_steps = 100
        # or classes_5 for more complex data set
        x_data_all = torch.tensor(
            np.load(
                f"data/randman/data_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_4_classes_{settings.nb_outputs}.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )
        y_data_all = torch.tensor(
            np.load(
                f"data/randman/labels_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_4_classes_{settings.nb_outputs}.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )

        cut_test_train = int(len(x_data_all) * 0.9)
        cut_val_train = int(len(x_data_all) * 0.8)

        x_data = x_data_all[:cut_val_train]
        y_data = y_data_all[:cut_val_train]

        x_data_val = x_data_all[cut_val_train:cut_test_train]
        y_data_val = y_data_all[cut_val_train:cut_test_train]

        x_data_test = x_data_all[cut_test_train:]
        y_data_test = y_data_all[cut_test_train:]

        trainloader, valloader, testloader = get_data_loaders(
            x_data,
            y_data,
            x_data_val,
            y_data_val,
            x_data_test,
            y_data_test,
            plot_ex=False,
            num_workers=num_workers,
            batch_size=settings.batch_size,
        )

    if data_set_name == "rm":
        assert settings.nb_inputs == 20  # 3
        assert settings.nb_outputs == 10  # 2
        samples = 12800  # 2560

        nb_steps = 100
        # or classes_5 for more complex data set
        x_data_all = torch.tensor(
            np.load(
                f"data/randman/data_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_2_classes_{settings.nb_outputs}.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )
        y_data_all = torch.tensor(
            np.load(
                f"data/randman/labels_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_2_classes_{settings.nb_outputs}.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )

        cut_test_train = int(len(x_data_all) * 0.9)
        cut_val_train = int(len(x_data_all) * 0.8)

        x_data = x_data_all[:cut_val_train]
        y_data = y_data_all[:cut_val_train]

        x_data_val = x_data_all[cut_val_train:cut_test_train]
        y_data_val = y_data_all[cut_val_train:cut_test_train]

        x_data_test = x_data_all[cut_test_train:]
        y_data_test = y_data_all[cut_test_train:]

        trainloader, valloader, testloader = get_data_loaders(
            x_data,
            y_data,
            x_data_val,
            y_data_val,
            x_data_test,
            y_data_test,
            plot_ex=False,
            num_workers=num_workers,
            batch_size=settings.batch_size,
        )

    if data_set_name == "rml":
        assert settings.nb_inputs == 20  # 3
        assert settings.nb_outputs == 10  # 2
        samples = 12800  # 2560

        nb_steps = 100
        # or classes_5 for more complex data set
        x_data_all = torch.tensor(
            np.load(
                f"data/randman/data_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_2_classes_{settings.nb_outputs}_mfdim_2.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )
        y_data_all = torch.tensor(
            np.load(
                f"data/randman/labels_samples_{samples}_steps_{nb_steps}_units_{settings.nb_inputs}_alpha_2_classes_{settings.nb_outputs}_mfdim_2.npy"
            ),
            device=settings.device,
            dtype=settings.dtype,
            requires_grad=False,
        )

        cut_test_train = int(len(x_data_all) * 0.9)
        cut_val_train = int(len(x_data_all) * 0.8)

        x_data = x_data_all[:cut_val_train]
        y_data = y_data_all[:cut_val_train]

        x_data_val = x_data_all[cut_val_train:cut_test_train]
        y_data_val = y_data_all[cut_val_train:cut_test_train]

        x_data_test = x_data_all[cut_test_train:]
        y_data_test = y_data_all[cut_test_train:]

        trainloader, valloader, testloader = get_data_loaders(
            x_data,
            y_data,
            x_data_val,
            y_data_val,
            x_data_test,
            y_data_test,
            plot_ex=False,
            num_workers=num_workers,
            batch_size=settings.batch_size,
        )

    elif data_set_name == "mnist":
        assert settings.nb_inputs == 2312
        assert settings.nb_outputs == 10

        cached_trainset = get_tonic_dataset(
            data_set_name,
            settings.time_step,
            mode="train",
            transform=True,
            pre_path=pre_path,
        )

        # no augmentations for the testset
        cached_testset = get_tonic_dataset(
            data_set_name,
            settings.time_step,
            mode="test",
            transform=False,
            pre_path=pre_path,
        )

        print(f"len {cached_trainset.__len__()}")

        # pad time dimension for test and training set -> batch_first=False
        testloader = DataLoader(
            cached_testset,
            batch_size=settings.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
            shuffle=False,
        )

        trainloader, valloader = train_val_split(cached_trainset, settings.batch_size)
        nb_steps = None

    elif data_set_name == "shd":
        assert settings.nb_inputs <= 700
        assert settings.nb_outputs == 20

        # rotations don't make sense for audio file?
        cached_trainset = get_tonic_dataset(
            data_set_name,
            settings.time_step,
            mode="train",
            transform=False,  # no image data -> no rotations
            encoding_dim=settings.nb_inputs,
            pre_path=pre_path,
        )
        cached_testset = get_tonic_dataset(
            data_set_name,
            settings.time_step,
            mode="test",
            transform=False,
            encoding_dim=settings.nb_inputs,
            pre_path=pre_path,
        )

        # pad time dimension for test and training set -> batch_first=False
        testloader = DataLoader(
            cached_testset,
            batch_size=settings.batch_size,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
        )

        trainloader, valloader = train_val_split(cached_trainset, settings.batch_size)

        # nb_steps = next(iter(trainloader))[0].size()[0]  # 250
        nb_steps = None
    elif data_set_name == "olfactory":
        assert settings.nb_inputs == 2264
        assert settings.nb_hidden == 7616
        assert settings.nb_outputs == 284

        # pad time dimension for test and training set -> batch_first=False

        val_split = 0.2
        test_split = 0.1
        num_workers = 0
        t_max_ms = 500

        all_files = sorted(glob.glob(os.path.join(pre_path, "*.hdf5")))
        dataset = HDF5SpikeDataset(
            all_files,
            num_neurons=settings.nb_inputs,
            bin_size_ms=settings.time_step * 1000,
            t_max_ms=t_max_ms,
        )

        total = len(dataset)
        test_size = max(1, int(test_split * total)) if total >= 3 else 0
        val_size = max(1, int(val_split * total)) if total >= 3 else 0
        train_size = total - val_size - test_size

        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )
        print(f"settings.batch_size: {settings.batch_size}")
        trainloader = DataLoader(
            train_set,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=customPadTensors(
                batch_first=False
            ),  # tonic.collation.PadTensors(batch_first=False),
        )

        valloader = DataLoader(
            val_set,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=customPadTensors(
                batch_first=False
            ),  # tonic.collation.PadTensors(batch_first=False),
        )

        testloader = DataLoader(
            test_set,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=customPadTensors(
                batch_first=False
            ),  # tonic.collation.PadTensors(batch_first=False),
        )

        # nb_steps = next(iter(trainloader))[0].size()[0]  # 250
        nb_steps = None

    return trainloader, valloader, testloader, nb_steps
