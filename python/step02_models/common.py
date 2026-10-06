"""Shared lightweight encoders; feature rows are channels, not image pixels."""
import torch
from torch import nn


def conv_block(inputs, outputs, kernel, stride=1, padding=0, *,
               dimensions=2, dilation=1):
    """Convolution followed by GroupNorm and GELU."""
    if dimensions not in (1, 2):
        raise ValueError("Convolution dimensions must be 1 or 2")
    conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
    return nn.Sequential(
        conv(inputs, outputs, kernel, stride=stride, padding=padding, dilation=dilation),
        nn.GroupNorm(8, outputs),
        nn.GELU())


class RestoreSTFT(nn.Module):
    """Undo MATLAB frequency-first flattening after feature fusion."""
    def __init__(self, frequency_bins):
        super().__init__()
        self.frequency_bins = frequency_bins

    def forward(self, x):
        batch, channels, _, width = x.shape
        if width % self.frequency_bins or width // self.frequency_bins < 3:
            raise ValueError("STFT width must contain whole frames and at least 3 frames")
        return x.reshape(batch, channels, -1, self.frequency_bins).transpose(-2, -1)


class Residual(nn.Module):
    def __init__(self, channels, dimensions=1, dilation=1):
        super().__init__()
        conv = nn.Conv1d if dimensions == 1 else nn.Conv2d
        self.layers = nn.Sequential(
            *conv_block(channels, channels, 3, padding=dilation, dilation=dilation, dimensions=dimensions).children(),
            conv(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class PairedModel(nn.Module):
    """Shared paired-input contract and forward flow; subclasses own every layer."""
    min_time_length = 1

    @classmethod
    def validate_time_length(cls, length):
        if length < cls.min_time_length:
            raise ValueError(f"Time input is too short; requires at least {cls.min_time_length} samples")

    def __init__(self, classes, embedding_dim, stft_frequency_bins):
        super().__init__()
        if classes < 2 or embedding_dim < 1:
            raise ValueError("Require at least two classes and a positive embedding dimension")
        if not isinstance(stft_frequency_bins, int) or stft_frequency_bins < 3:
            raise ValueError("STFT requires at least 3 frequency bins")
        self.bins = stft_frequency_bins

    def validate_inputs(self, time, stft):
        for x in (time, stft):
            if x.ndim != 4 or x.shape[1:3] != (3, 270):
                raise ValueError("Expected paired [B,3,270,T] inputs")
        if time.shape[0] != stft.shape[0]:
            raise ValueError("Time and STFT batch sizes must match")
        self.validate_time_length(time.shape[-1])
        if stft.shape[-1] % self.bins or stft.shape[-1] // self.bins < 3:
            raise ValueError("STFT width must contain whole frames and at least 3 frames")

    def encode(self, time, stft):
        self.validate_inputs(time, stft)
        z = torch.cat((self.time_encoder(time), self.stft_encoder(stft)), dim=1)
        return self.embedding(self.fusion(z))

    def forward(self, time, stft, return_embedding=False):
        z = self.encode(time, stft)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits
