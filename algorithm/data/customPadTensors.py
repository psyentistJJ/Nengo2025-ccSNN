import torch


class customPadTensors:
    # -> modified from tonic https://github.com/neuromorphs/tonic/blob/96f683e4e11d68f8a40b039dbf9725ae79199f45/tonic/collation.py#L7 to allow returning metadata (i.e. duration of stim)
    """This is a custom collate function for a pytorch dataloader to load multiple event recordings
    at once. It's intended to be used in combination with sparse tensors. All tensor sizes are
    extended to the largest one in the batch, i.e. the longest recording.

    Example:
        >>> dataloader = torch.utils.data.DataLoader(dataset,
        >>>                                          batch_size=10,
        >>>                                          collate_fn=tonic.collation.PadTensors(),
        >>>                                          shuffle=True)
    """

    def __init__(self, batch_first: bool = True):
        self.batch_first = batch_first

    def __call__(self, batch):
        samples_output = []
        targets_output = []
        durations_output = []

        max_length = max([sample.shape[0] for sample, target, duration in batch])
        for sample, target, duration in batch:
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            if not isinstance(duration, torch.Tensor):
                duration = torch.tensor(duration)
            if sample.is_sparse:
                sample.sparse_resize_(
                    (max_length, *sample.shape[1:]),
                    sample.sparse_dim(),
                    sample.dense_dim(),
                )
            else:
                sample = torch.cat(
                    (
                        sample,
                        torch.zeros(
                            max_length - sample.shape[0],
                            *sample.shape[1:],
                            device=sample.device,
                        ),
                    )
                )
            samples_output.append(sample)
            targets_output.append(target)
            durations_output.append(duration)

        samples_output = torch.stack(samples_output, 0 if self.batch_first else 1)
        if len(targets_output[0].shape) > 1:
            targets_output = torch.stack(targets_output, 0 if self.batch_first else -1)
        else:
            targets_output = torch.tensor(targets_output, device=target.device)
        if len(durations_output[0].shape) > 1:
            durations_output = torch.stack(
                durations_output, 0 if self.batch_first else -1
            )
        else:
            durations_output = torch.tensor(durations_output, device=target.device)
        return (samples_output, targets_output, durations_output)
