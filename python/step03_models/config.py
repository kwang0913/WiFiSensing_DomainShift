"""Parameters for the dual CNN architecture."""
from dataclasses import dataclass, field


@dataclass
class DualCNNConfig:
    embedding_dim: int = 32
    stft_frequency_bins: int = 32
    kind: str = field(default="dual_cnn", init=False)
