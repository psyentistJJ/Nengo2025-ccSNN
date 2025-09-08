import torch


class SurrGradSpike(torch.autograd.Function):
    """
    -> from Zenke notebooks (https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb)
    -> adapted for scale as input
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    @staticmethod
    def forward(ctx, input, scale):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.scale = scale
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0  # threshold at zero...
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            # grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
            grad_input
            / (ctx.scale * torch.abs(input) + 1.0) ** 2
        )  # calculate once, should be dNL/dInp
        return grad, None
