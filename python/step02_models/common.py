"""Shared lightweight encoders; feature rows are channels, not image pixels."""
import torch
from torch import nn
from .dual_cnn import RestoreSTFT


class Residual(nn.Module):
    def __init__(self, channels, dimensions=1, dilation=1):
        super().__init__()
        conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
        self.layers = nn.Sequential(
            conv(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.GELU(),
            conv(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


def time_stem():
    # Compress the 3 x 270 feature channels independently at each time sample.
    return nn.Sequential(nn.Flatten(1, 2), nn.Conv1d(810, 64, 1),
                         nn.GroupNorm(8, 64), nn.GELU(),
                         nn.Conv1d(64, 64, 7, stride=2, padding=3),
                         nn.GroupNorm(8, 64), nn.GELU())


def stft_encoder(bins):
    return nn.Sequential(
        nn.Conv2d(3, 32, (270, 1)), RestoreSTFT(bins),
        nn.GroupNorm(8, 32), nn.GELU(), Residual(32, dimensions=2),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.GroupNorm(8, 64), nn.GELU(), Residual(64, dimensions=2),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())


class PairedClassifier(nn.Module):
    def __init__(self, classes, embedding_dim, stft_frequency_bins):
        super().__init__()
        if classes < 2 or embedding_dim < 1:
            raise ValueError("Require at least two classes and a positive embedding dimension")
        if not isinstance(stft_frequency_bins, int) or stft_frequency_bins < 3:
            raise ValueError("STFT requires at least 3 frequency bins")
        self.bins = stft_frequency_bins
        self.stft_encoder = stft_encoder(stft_frequency_bins)
        self.fusion = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, classes))

    def encode(self, time, stft):
        for x in (time, stft):
            if x.ndim != 4 or x.shape[1:3] != (3, 270):
                raise ValueError("Expected paired [B,3,270,T] inputs")
        if time.shape[0] != stft.shape[0]:
            raise ValueError("Time and STFT batch sizes must match")
        if time.shape[-1] < 1:
            raise ValueError("Time input must be nonempty")
        if stft.shape[-1] % self.bins or stft.shape[-1] // self.bins < 3:
            raise ValueError("STFT width must contain whole frames and at least 3 frames")
        z = torch.cat((self.time_encoder(time), self.stft_encoder(stft)), dim=1)
        return self.embedding(self.fusion(z))

    def forward(self, time, stft, return_embedding=False):
        z = self.encode(time, stft)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits
