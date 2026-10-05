"""Configuration types for experiment components."""
from dataclasses import dataclass, field
import math


@dataclass
class SplitConfig:
    validation_fraction: float = 0.10
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    mode: str = "domain_holdout"
    domain_key: str = "activity"
    test_domains: list = field(default_factory=lambda: ["walk"])

    @property
    def train_fraction(self):
        test_fraction = self.test_fraction if self.mode == "random" else 0
        return 1 - self.validation_fraction - self.calibration_fraction - test_fraction


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 8
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    patience: int | None = 8
    min_delta: float = 0.0
    weight_decay: float = 1e-4
    optimizer_kwargs: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    name: str = "none"
    kwargs: dict = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    enabled: bool = False
    time_noise: dict = field(default_factory=lambda: {
        "enabled": True, "kind": "gaussian", "relative_std": 0.02, "probability": 0.5,
    })
    time_mask: dict = field(default_factory=lambda: {
        "enabled": False, "max_fraction": 0.05, "probability": 0.5,
    })
    stft_mask: dict = field(default_factory=lambda: {
        "enabled": True, "frequency_bins": 32, "time_max_fraction": 0.05,
        "frequency_max_fraction": 0.05, "probability": 0.5,
    })


@dataclass
class SoftmaxConfig:
    kind: str = field(default="softmax", init=False)


@dataclass
class KDEConfig:
    kernel: str = "gaussian"
    bandwidth: float = 1.0
    kind: str = field(default="kde", init=False)


@dataclass
class SVMConfig:
    C: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    kind: str = field(default="svm", init=False)


@dataclass
class HBGBConfig:
    max_iter: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int = 31
    kind: str = field(default="hbgb", init=False)


@dataclass
class CalibrationConfig:
    alpha: float = 0.1
    class_conditional: bool = True
    weighted_cp: bool = False


@dataclass
class WeightingConfig:
    reference_domains: list = field(default_factory=list)
    C: float = 1.0
    max_iter: int = 1000
    clip_min: float | None = None
    clip_max: float | None = None

    def __post_init__(self):
        if not isinstance(self.reference_domains, list) or any(not isinstance(v, str) for v in self.reference_domains):
            raise ValueError("weighting.reference_domains must be a list of domain strings")
        if len(set(self.reference_domains)) != len(self.reference_domains):
            raise ValueError("weighting.reference_domains must not contain duplicates")
        if not math.isfinite(self.C) or self.C <= 0:
            raise ValueError("weighting.C must be finite and positive")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("weighting.max_iter must be a positive integer")
        for value in (self.clip_min, self.clip_max):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Weight clipping bounds must be finite and positive")
        if self.clip_min is not None and self.clip_max is not None and self.clip_min > self.clip_max:
            raise ValueError("clip_min must not exceed clip_max")


@dataclass
class DualCNNConfig:
    embedding_dim: int = 32
    stft_frequency_bins: int = 32
    kind: str = "dual_cnn"

    def __post_init__(self):
        if self.kind not in ("dual_cnn", "dual_resnet", "tcn_stft", "cnn_transformer", "dual_link_graph"):
            raise ValueError(f"Unknown model kind: {self.kind}")


@dataclass
class AdversarialConfig:
    enabled: bool = False
    conditional: bool = False
    domain_key: str = "activity"
    max_weight: float = 0.1
    warmup_epochs: int = 10
    hidden_dim: int = 64
    steps_per_batch: int = 2
    optimizer: str = "sgd"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer_kwargs: dict = field(default_factory=lambda: {"momentum": 0.9, "nesterov": True})
    scheduler: dict = field(default_factory=lambda: {"name": "none", "kwargs": {}})

    def __post_init__(self):
        if not isinstance(self.steps_per_batch, int) or self.steps_per_batch < 1:
            raise ValueError("Adversarial steps_per_batch must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Adversarial learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Adversarial weight_decay must be finite and nonnegative")
        if not math.isfinite(self.max_weight) or self.max_weight < 0:
            raise ValueError("Adversarial max_weight must be finite and nonnegative")
        if not isinstance(self.warmup_epochs, int) or self.warmup_epochs < 0:
            raise ValueError("Adversarial warmup_epochs must be a nonnegative integer")
        if not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise ValueError("Adversarial hidden_dim must be a positive integer")


@dataclass
class ContrastiveConfig:
    enabled: bool = False
    weight: float = 0.1
    temperature: float = 0.1

    def __post_init__(self):
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("Contrastive weight must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Contrastive temperature must be finite and positive")


@dataclass
class SamplingConfig:
    enabled: bool = False
    classes_per_batch: int = 6
    domains_per_class: int = 3
    samples_per_domain: int = 4

    def __post_init__(self):
        for name in ("classes_per_batch", "domains_per_class", "samples_per_domain"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Sampling {name} must be a positive integer")
        if self.enabled and (self.classes_per_batch < 2 or self.domains_per_class < 2):
            raise ValueError("Balanced batches require at least two classes and two domain slots per class")
