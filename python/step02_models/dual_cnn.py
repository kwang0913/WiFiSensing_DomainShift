"""Dual CNN with temporal and explicit time-frequency valid convolutions.

Inputs are paired [B,3,270,T] tensors. Time standardization remains in the
Dataset; normalization follows each convolution, with no input normalization.
"""
from torch import nn

from .common import conv_block, RestoreSTFT, PairedModel


class DualCNN(PairedModel):
    min_time_length = 96

    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = nn.Sequential(
            # Collapse all 270 feature rows; subsequent convolutions slide in time.
            conv_block(3, 32, (270, 24), (1, 24)),
            conv_block(32, 64, (1, 4), (1, 2)),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
        )
        self.stft_encoder = nn.Sequential(
            # Fuse features independently at each time-frequency point.
            conv_block(3, 32, (270, 1), (1, 1)),
            RestoreSTFT(stft_frequency_bins),
            conv_block(32, 64, (3, 3), (1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(nn.Linear(512, 128), nn.GELU())
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, classes)
