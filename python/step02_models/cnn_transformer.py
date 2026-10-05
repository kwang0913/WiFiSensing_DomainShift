"""Local temporal CNN followed by a small position-aware Transformer."""
import math
import torch
from torch import nn
from .common import PairedClassifier, time_stem


class TemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = time_stem()
        # Bound attention cost for long segments; retain order in <=64 time tokens.
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, dim_feedforward=128,
                                       dropout=0.1, activation="gelu", batch_first=True),
            num_layers=2, enable_nested_tensor=False)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool1d(x, min(64, x.shape[-1])).transpose(1, 2)
        position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
        scale = torch.exp(torch.arange(0, 64, 2, device=x.device, dtype=x.dtype)
                          * (-math.log(10000.0) / 64))
        pe = torch.zeros(x.shape[1], 64, device=x.device, dtype=x.dtype)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        x = self.transformer(x + pe).transpose(1, 2)
        return nn.functional.adaptive_avg_pool1d(x, 4).flatten(1)


class CNNTransformer(PairedClassifier):
    def __init__(self, classes, embedding_dim=32, stft_frequency_bins=32):
        super().__init__(classes, embedding_dim, stft_frequency_bins)
        self.time_encoder = TemporalAttention()
