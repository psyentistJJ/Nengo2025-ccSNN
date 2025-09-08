import torch.nn as nn
import torch

class SaturatedReLU(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, 0, self.threshold)