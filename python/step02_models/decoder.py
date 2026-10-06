"""Reconstruct paired time/STFT inputs from task and background features."""
from torch import nn
from torch.nn import functional as F


class PairedDecoder(nn.Module):
    def __init__(self, dim, bins):
        super().__init__()
        self.bins = bins
        self.time_seed = nn.Linear(dim, 32 * 16)
        self.time_output = nn.Conv1d(32, 810, 1)
        self.stft_seed = nn.Linear(dim, 32 * 8 * 8)
        self.stft_output = nn.Conv2d(32, 810, 1)

    def forward(self, latent, time_width, stft_width):
        n = len(latent)
        t = F.gelu(self.time_seed(latent)).reshape(n, 32, 16)
        t = self.time_output(F.interpolate(t, size=time_width, mode="linear", align_corners=False))
        f = F.gelu(self.stft_seed(latent)).reshape(n, 32, 8, 8)
        f = self.stft_output(F.interpolate(f, size=(self.bins, stft_width // self.bins),
                                         mode="bilinear", align_corners=False))
        return t.reshape(n, 3, 270, time_width), f.transpose(-1, -2).reshape(n, 3, 270, stft_width)
