"""Matched independent task/background encoders; prediction uses task features only."""
from dataclasses import asdict
from torch import nn
from .adversarial import AdversarialClassifier
from .decoder import PairedDecoder


class TaskBackgroundClassifier(nn.Module):
    def __init__(self, task_model, background_model, config, classes, embedding_dim, bins, background_domains, decoder_config):
        super().__init__()
        self.task_model = task_model
        self.encoder_background_config = asdict(config)
        # Same encoder architecture, separate initialization and parameter storage.
        self.background_encoder = background_model
        if config.domain["enabled"]:
            self.background_domain_classifier = background_model.classifier
        # Keep only the encoder here; do not register an unused or duplicated head.
        background_model.classifier = nn.Identity()
        if config.adversarial["enabled"]:
            self.background_activity_classifier = AdversarialClassifier(
                embedding_dim, classes, config.adversarial["hidden_dim"],
                condition_classes=background_domains if config.adversarial["conditional"] else 0)
        if decoder_config.enabled:
            self.decoder = PairedDecoder(2 * embedding_dim, bins)

    def encode(self, time, stft):
        return self.task_model.encode(time, stft)

    def encode_background(self, time, stft):
        return self.background_encoder.encode(time, stft)

    def forward(self, time, stft, return_embedding=False):
        return self.task_model(time, stft, return_embedding=return_embedding)
