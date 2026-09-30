"""Build a seeded PyTorch classifier from notebook NPY shapes (H,W,C)."""
import torch
from .dual_cnn import DualCNN


def build_dualcnn(config, time_shape, stft_shape, classes, seed=42):
    for shape in (time_shape, stft_shape):
        if len(shape) != 3 or shape[0] != 270 or shape[2] != 3:
            raise ValueError("Expected notebook shapes [270,T,3]")
    if time_shape[1] < 96:
        raise ValueError("Time input is too short for the convolution kernels")
    bins = config.stft_frequency_bins
    if not isinstance(bins, int) or bins < 3:
        raise ValueError("STFT requires at least 3 frequency bins")
    if stft_shape[1] % bins or stft_shape[1] // bins < 3:
        raise ValueError("STFT width must contain whole frames and at least 3 frames")
    torch.manual_seed(seed)
    return DualCNN(classes, config.embedding_dim, bins)
