"""Noncausal dilated temporal residual network paired with an STFT CNN.

Symmetric padding uses the complete observed segment for classification.
"""
from torch import nn
from .common import PairedClassifier, Residual, time_stem


class TCNSTFT(PairedClassifier):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = nn.Sequential(
            time_stem(), *(Residual(64, dilation=d) for d in (1, 2, 4, 8)),
            nn.AdaptiveAvgPool1d(4), nn.Flatten())
