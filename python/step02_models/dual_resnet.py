"""Gradual temporal downsampling and residual time-frequency processing."""
from torch import nn
from .common import PairedClassifier, Residual, time_stem


class DualResNet(PairedClassifier):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = nn.Sequential(
            time_stem(), Residual(64),
            nn.Conv1d(64, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.GELU(), Residual(64),
            nn.AdaptiveAvgPool1d(4), nn.Flatten())
