"""Build a seeded PyTorch classifier from notebook NPY shapes (H,W,C)."""
import torch
from .dual_cnn import DualCNN
from .adversarial import DomainClassifier


def build_dualcnn(config, time_shape, stft_shape, classes, seed=42,
                  domain_classes=0, domain_hidden_dim=64, domain_conditional=False):
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
    model = DualCNN(classes, config.embedding_dim, bins)
    if domain_classes:
        if domain_classes < 2:
            raise ValueError("Domain-adversarial training requires at least two training domains")
        model.domain_classifier = DomainClassifier(config.embedding_dim, domain_classes, domain_hidden_dim,
                                                   task_classes=classes if domain_conditional else 0)
    return model
