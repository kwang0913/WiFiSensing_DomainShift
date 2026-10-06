"""Load one YAML experiment, using baseline.yaml for omitted settings."""
from dataclasses import asdict
from pathlib import Path

import yaml

from .schema import KDEConfig, SVMConfig, HBGBConfig, SoftmaxConfig, WeightingConfig, ModelConfig, EncoderBackgroundConfig, EncoderTaskConfig, DecoderConfig, TrainingConfig, InferenceConfig

BASELINE = Path(__file__).resolve().parents[1] / "experiments/baseline.yaml"
SCORE_CONFIGS = {"kde": KDEConfig, "svm": SVMConfig, "hbgb": HBGBConfig, "softmax": SoftmaxConfig}


def _merge(base, overrides, section="experiment"):
    """Merge structured settings; optimizer/scheduler kwargs are replaced as a unit."""
    unknown = overrides.keys() - base.keys()
    if unknown:
        raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    # Apply this at every nesting level, including adversarial.scheduler.
    for selector, parameters in (("optimizer", "optimizer_kwargs"), ("name", "kwargs")):
        if selector in base and parameters in base and overrides.get(selector, base[selector]) != base[selector]:
            base[parameters] = {}
    for key, value in overrides.items():
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{section}.{key} must be a mapping")
            if key in ("kwargs", "optimizer_kwargs"):
                base[key] = value
            else:
                _merge(base[key], value, f"{section}.{key}")
        else:
            base[key] = value


def load_config(path=None):
    config = yaml.safe_load(BASELINE.read_text())
    overrides = yaml.safe_load(Path(path).expanduser().read_text()) if path else {}
    if not isinstance(overrides, dict):
        raise ValueError("Experiment YAML must contain a mapping of settings")
    # Normalize older YAML layouts before strict merging; the returned config is canonical.
    for old, new in (("disentanglement", "encoder_background"),):
        if old in overrides:
            if new in overrides:
                raise ValueError(f"Use only {new}, not both {old} and {new}")
            overrides[new] = overrides.pop(old)
    for key in ("contrastive", "adversarial"):
        if key in overrides:
            group = overrides.setdefault("encoder_task", {})
            if not isinstance(group, dict) or key in group:
                raise ValueError(f"Conflicting legacy and encoder_task.{key} settings")
            group[key] = overrides.pop(key)
    # Older experiment files used one seed for all randomness.
    if "seed" in overrides:
        old_seed = overrides.pop("seed")
        overrides.setdefault("split_seed", old_seed)
        overrides.setdefault("training_seed", old_seed)
    score = overrides.get("score", {})
    if isinstance(score, dict) and score.get("kind", config["score"]["kind"]) not in SCORE_CONFIGS:
        raise ValueError(f"Unknown score.kind: {score.get('kind')}")
    if isinstance(score, dict) and score.get("kind", config["score"]["kind"]) != config["score"]["kind"]:
        config["score"] = asdict(SCORE_CONFIGS[score["kind"]]())
    _merge(config, overrides)
    score_settings = config["score"].copy()
    score_kind = score_settings.pop("kind")
    if score_kind not in SCORE_CONFIGS:
        raise ValueError(f"Unknown score.kind: {score_kind}")
    try:
        SCORE_CONFIGS[score_kind](**score_settings)
    except TypeError as error:
        raise ValueError(f"Invalid settings for score.kind={score_kind}: {error}") from error
    datasets = config["data"]["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a nonempty list of cache directories")
    InferenceConfig(**config["inference"])
    TrainingConfig(**config["training"])
    ModelConfig(**config["model"])
    EncoderTaskConfig(**config["encoder_task"])
    background = EncoderBackgroundConfig(**config["encoder_background"])
    DecoderConfig(**config["decoder"])
    if not isinstance(config["data"]["domain_key"], str) or not config["data"]["domain_key"]:
        raise ValueError("data.domain_key must be a nonempty string")
    weighting = WeightingConfig(**config["weighting"])
    enabled = config["calibration"]["weighted_cp"]
    if not isinstance(enabled, bool):
        raise ValueError("calibration.weighted_cp must be true or false")
    if enabled and weighting.features == "background" and not background.enabled:
        raise ValueError("Background weighting requires encoder_background.enabled")
    return config
