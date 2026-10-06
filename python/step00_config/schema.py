"""Configuration types for experiment components."""
from dataclasses import dataclass, field, asdict
import math


@dataclass
class SplitConfig:
    validation_fraction: float = 0.10
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    mode: str = "domain_holdout"
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

    scheduler: dict = field(default_factory=lambda: {"name": "none", "kwargs": {}})

    def __post_init__(self):
        if not isinstance(self.scheduler, dict) or self.scheduler.keys() - {"name", "kwargs"}:
            raise ValueError("Invalid training.scheduler")
        self.scheduler = {"name": "none", "kwargs": {}, **self.scheduler}
        if self.scheduler["name"] not in ("none", "cosine", "plateau") or not isinstance(self.scheduler["kwargs"], dict):
            raise ValueError("Invalid training.scheduler")


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
    weighted_cp: bool = False

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("calibration.alpha must be between 0 and 1")
        if not isinstance(self.weighted_cp, bool):
            raise ValueError("calibration.weighted_cp must be boolean")


@dataclass
class WeightingConfig:
    C: float = 1.0
    max_iter: int = 1000
    clip_min: float | None = None
    clip_max: float | None = None

    features: str = "task"
    method: str = "density_ratio"

    def __post_init__(self):
        if self.features not in ("task", "background"):
            raise ValueError("Invalid weighting.features")
        if self.method not in ("density_ratio", "domain_mixture"):
            raise ValueError("Invalid weighting.method")
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
class ModelConfig:
    embedding_dim: int = 32
    stft_frequency_bins: int = 32
    kind: str = "dual_cnn"

    def __post_init__(self):
        if self.kind not in ("dual_cnn", "cnn_transformer", "dual_link_graph"):
            raise ValueError(f"Unknown model kind: {self.kind}")


@dataclass
class AdversarialConfig:
    enabled: bool = False
    conditional: bool = False
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
        for key in ("enabled", "conditional"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
        for key in ("steps_per_batch", "hidden_dim", "warmup_epochs"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if self.optimizer not in ("sgd", "adam", "adamw", "rmsprop"):
            raise ValueError("Invalid adversarial optimizer")
        if not isinstance(self.optimizer_kwargs, dict):
            raise ValueError("Adversarial optimizer_kwargs must be a mapping")
        if not isinstance(self.scheduler, dict) or self.scheduler.keys() - {"name", "kwargs"}:
            raise ValueError("Invalid adversarial scheduler")
        self.scheduler = {"name": "none", "kwargs": {}, **self.scheduler}
        if self.scheduler["name"] not in ("none", "cosine", "plateau") or not isinstance(self.scheduler["kwargs"], dict):
            raise ValueError("Invalid adversarial scheduler")
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


@dataclass
class EncoderBackgroundConfig:
    enabled: bool = False
    domain: dict = field(default_factory=lambda: {"enabled": True, "weight": 0.2})
    contrastive: dict = field(default_factory=lambda: {"enabled": True, "weight": 0.1, "temperature": 0.1})
    adversarial: dict = field(default_factory=lambda: asdict(AdversarialConfig(enabled=True)))
    difference: dict = field(default_factory=lambda: {"enabled": True, "weight": 0.01})

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("encoder_background.enabled must be boolean")
        self.adversarial = asdict(AdversarialConfig(**{
            **asdict(AdversarialConfig(enabled=True)), **self.adversarial}))
        defaults = {f.name: f.default_factory() for f in self.__dataclass_fields__.values()
                    if f.name in ("domain", "contrastive", "difference")}
        for key, default in defaults.items():
            value = getattr(self, key)
            if not isinstance(value, dict) or value.keys() - default.keys():
                raise ValueError(f"Invalid encoder_background.{key}")
            value = {**default, **value}
            if not isinstance(value["enabled"], bool):
                raise ValueError(f"{key}.enabled must be boolean")
            for name, x in value.items():
                if name == "enabled":
                    continue
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                    raise ValueError(f"Invalid {key}.{name}")
                if name == "temperature" and x <= 0:
                    raise ValueError(f"{key}.{name} must be positive")
            setattr(self, key, value)


@dataclass
class DecoderConfig:
    enabled: bool = False
    weight: float = 0.1

    def __post_init__(self):
        validate_difference({"enabled": self.enabled, "weight": self.weight}, "decoder")


def validate_difference(settings, name):
    if not isinstance(settings, dict) or settings.keys() - {"enabled", "weight"}:
        raise ValueError(f"Invalid {name}")
    result = {"enabled": True, "weight": 0.01, **settings}
    if not isinstance(result["enabled"], bool):
        raise ValueError(f"{name}.enabled must be boolean")
    value = result["weight"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}.weight must be finite and nonnegative")
    return result


@dataclass
class EncoderTaskConfig:
    task: dict = field(default_factory=lambda: {"enabled": True, "weight": 1.0})
    contrastive: dict = field(default_factory=lambda: asdict(ContrastiveConfig()))
    adversarial: dict = field(default_factory=lambda: asdict(AdversarialConfig()))

    difference: dict = field(default_factory=lambda: {"enabled": True, "weight": 0.01})

    def __post_init__(self):
        if not isinstance(self.task, dict) or self.task.keys() - {"enabled", "weight"}:
            raise ValueError("Invalid encoder_task.task")
        self.task = {"enabled": True, "weight": 1.0, **self.task}
        if not isinstance(self.task["enabled"], bool):
            raise ValueError("encoder_task.task.enabled must be boolean")
        weight = self.task["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("encoder_task.task.weight must be finite and nonnegative")
        self.difference = validate_difference(self.difference, "encoder_task.difference")
        self.contrastive = asdict(ContrastiveConfig(**self.contrastive))
        self.adversarial = asdict(AdversarialConfig(**self.adversarial))



@dataclass
class InferenceConfig:
    batch_size: int = 4

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("inference.batch_size must be a positive integer")
