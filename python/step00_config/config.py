"""Default experiment parameters; override them in run.ipynb."""
from dataclasses import dataclass, field


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
