"""Dual CNN with temporal and explicit time-frequency valid convolutions.

Inputs are paired [B,3,270,T] tensors. Time standardization remains in the
Dataset; this model retains ordinary input BatchNorm on both branches.
"""
import torch
from torch import nn

# Keras momentum=0.99 weights the old running statistic; PyTorch weights the new.
_BN = dict(eps=1e-3, momentum=0.01)


def conv_block(inputs, outputs, kernel, stride, dropout=False):
    layers = [nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=0),
            #   nn.LeakyReLU(0.1),
              nn.ReLU(),
              nn.BatchNorm2d(outputs, **_BN)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


def dense_block(inputs, outputs, dropout=False):
    layers = [nn.Linear(inputs, outputs),
            #   nn.LeakyReLU(0.1),
              nn.ReLU(),
              nn.BatchNorm1d(outputs, **_BN)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


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


class DualCNN(nn.Module):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__()
        if classes < 2 or embedding_dim < 1:
            raise ValueError("Require at least two classes and a positive embedding dimension")
        if not isinstance(stft_frequency_bins, int) or stft_frequency_bins < 3:
            raise ValueError("STFT requires at least 3 frequency bins")
        self.time_encoder = nn.Sequential(
            nn.BatchNorm2d(3, **_BN),
            # Collapse all 270 feature rows; subsequent convolutions slide in time.
            conv_block(3, 128, (270, 24), (1, 24)),
            conv_block(128, 256, (1, 4), (1, 2)),
            nn.AdaptiveAvgPool2d((1, 4)),
            nn.Flatten(),
            dense_block(1024, 1024, dropout=False),
        )
        self.stft_encoder = nn.Sequential(
            nn.BatchNorm2d(3, **_BN),
            # Fuse features independently at each time-frequency point.
            conv_block(3, 128, (270, 1), (1, 1)),
            RestoreSTFT(stft_frequency_bins),
            conv_block(128, 256, (3, 3), (1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            dense_block(1024, 1024, dropout=False),
        )
        self.fusion = nn.Sequential(
            dense_block(2048, 1024),
            dense_block(1024, 512),
            dense_block(512, 128),
        )
        # Export the embedding block output: after BatchNorm, with Dropout disabled in eval mode.
        self.embedding = nn.Sequential(
            # nn.Linear(128, embedding_dim),
            # nn.LeakyReLU(0.1)
            dense_block(128, embedding_dim, dropout=True),
        )
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(embedding_dim, **_BN),
            # nn.Dropout(0.5),
            nn.Linear(embedding_dim, classes),
        )
        self.apply(self._initialize)
        nn.init.xavier_uniform_(self.classifier[-1].weight)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.uniform_(layer.weight, -0.05, 0.05)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, time, stft):
        for x in (time, stft):
            if x.ndim != 4 or x.shape[1:3] != (3, 270):
                raise ValueError("Expected paired [B,3,270,T] inputs")
        if time.shape[0] != stft.shape[0]:
            raise ValueError("Time and STFT batch sizes must match")
        if time.shape[-1] < 96:
            raise ValueError("Time input is too short for the convolution kernels")
        if self.training and time.shape[0] < 2:
            raise ValueError("BatchNorm1d requires training batches of at least 2 samples")
        x = torch.cat((self.time_encoder(time), self.stft_encoder(stft)), dim=1)
        return self.embedding(self.fusion(x))

    def forward(self, time, stft, return_embedding=False):
        z = self.encode(time, stft)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits
