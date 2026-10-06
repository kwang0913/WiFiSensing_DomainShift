"""Temporal CNN/Transformer and channel-fused time-frequency patch Transformer."""
import math
import torch
from torch import nn
from .common import PairedModel, conv_block, RestoreSTFT


def time_stem():
    # Feature channels are mixed independently at each time sample.
    return nn.Sequential(nn.Flatten(1, 2),
                         *conv_block(810, 64, 1, dimensions=1).children(),
                         *conv_block(64, 64, 7, stride=2, padding=3, dimensions=1).children())


def stft_fusion(bins):
    return nn.Sequential(nn.Conv2d(3, 32, (270, 1)),
                         RestoreSTFT(bins),
                         nn.GroupNorm(8, 32),
                         nn.GELU())


def transformer_encoder(width=64, heads=4, feedforward=128, layers=2, dropout=0.1):
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(width,
                                   heads,
                                   dim_feedforward=feedforward,
                                   dropout=dropout,
                                   activation="gelu",
                                   batch_first=True),
        num_layers=layers, enable_nested_tensor=False)


def sinusoidal_positions(length, width, device, dtype):
    if width % 2:
        raise ValueError("Sinusoidal position width must be even")
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                      * (-math.log(10000.0) / width))
    angles = torch.arange(length, device=device, dtype=dtype)[:, None] * scale
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = time_stem()
        # Bound attention cost for long segments; retain order in <=64 time tokens.
        self.transformer = transformer_encoder()

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool1d(x, min(64, x.shape[-1])).transpose(1, 2)
        pe = sinusoidal_positions(x.shape[1], 64, x.device, x.dtype)
        x = self.transformer(x + pe).transpose(1, 2)
        return nn.functional.adaptive_avg_pool1d(x, 4).flatten(1)


class SpectralPatchAttention(nn.Module):
    """Fuse CSI channels, then tokenize 8-frequency by 9-frame patches."""
    def __init__(self, frequency_bins):
        super().__init__()
        self.fusion = stft_fusion(frequency_bins)
        self.patch = nn.Conv2d(32, 64, kernel_size=(8, 9), stride=(8, 9))
        self.transformer = transformer_encoder()

    @staticmethod
    def position_encoding(height, width, device, dtype):
        # Separate 32-dimensional sinusoidal coordinates for frequency and time.
        def axis(length):
            return sinusoidal_positions(length, 32, device, dtype)
        frequency = axis(height)[:, None, :].expand(height, width, 32)
        time = axis(width)[None, :, :].expand(height, width, 32)
        return torch.cat((frequency, time), dim=-1).reshape(height * width, 64)

    def forward(self, x):
        x = self.fusion(x)
        # Pad only high-frequency/right-time edges after normalization. Every
        # patch contains real data; partial boundary patches retain zero padding.
        x = nn.functional.pad(x, (0, (-x.shape[-1]) % 9, 0, (-x.shape[-2]) % 8))
        x = self.patch(x)
        height, width = x.shape[-2:]
        tokens = x.flatten(2).transpose(1, 2)
        positions = self.position_encoding(height, width, x.device, x.dtype)
        x = self.transformer(tokens + positions).transpose(1, 2)
        x = x.reshape(x.shape[0], 64, height, width)
        return nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1)


class CNNTransformer(PairedModel):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = TemporalAttention()
        self.stft_encoder = SpectralPatchAttention(stft_frequency_bins)
        self.fusion = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
