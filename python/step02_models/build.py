"""Build a seeded PyTorch classifier from notebook NPY shapes (H,W,C)."""
from dataclasses import asdict
import torch
from .dual_cnn import DualCNN
from .cnn_transformer import CNNTransformer
from .dual_link_graph import DualLinkGraph
from .adversarial import AdversarialClassifier


def build_model(config, time_shape, stft_shape, classes, seed=42,
                domain_classes=0, domain_hidden_dim=64, domain_conditional=False,
                encoder_background=None, background_domains=0, decoder=None):
    architectures = {"dual_cnn": DualCNN, "cnn_transformer": CNNTransformer,
                     "dual_link_graph": DualLinkGraph}
    if config.kind not in architectures:
        raise ValueError(f"Unknown model kind: {config.kind}")
    architecture = architectures[config.kind]
    for shape in (time_shape, stft_shape):
        if len(shape) != 3 or shape[0] != 270 or shape[2] != 3:
            raise ValueError("Expected notebook shapes [270,T,3]")
    architecture.validate_time_length(time_shape[1])
    bins = config.stft_frequency_bins
    if not isinstance(bins, int) or bins < 3:
        raise ValueError("STFT requires at least 3 frequency bins")
    if stft_shape[1] % bins or stft_shape[1] // bins < 3:
        raise ValueError("STFT width must contain whole frames and at least 3 frames")
    torch.manual_seed(seed)
    model = architecture(classes, config.embedding_dim, bins)
    from step00_config.schema import EncoderBackgroundConfig, DecoderConfig
    from .encoder_background import TaskBackgroundClassifier
    background = encoder_background or EncoderBackgroundConfig()
    decoder = decoder or DecoderConfig()
    if background.enabled:
        if background_domains < 2:
            raise ValueError("Background encoder requires at least two source domains")
        background_model = architecture(background_domains, config.embedding_dim, bins)
        model = TaskBackgroundClassifier(model, background_model, background, classes,
                                       config.embedding_dim, bins, background_domains, decoder)
    model.architecture_config = asdict(config)
    if domain_classes:
        if domain_classes < 2:
            raise ValueError("Domain-adversarial training requires at least two training domains")
        model.domain_classifier = AdversarialClassifier(
            config.embedding_dim, domain_classes, domain_hidden_dim,
            condition_classes=classes if domain_conditional else 0)
    return model
