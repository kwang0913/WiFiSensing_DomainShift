"""Load one YAML experiment, using baseline.yaml for omitted settings."""
from dataclasses import asdict
from pathlib import Path

import yaml

from .schema import (ModelConfig, EncoderBackgroundConfig, EncoderTaskConfig,
                     DecoderConfig, TrainingConfig, CalibrationConfig, SplitConfig, SamplingConfig)
from .evaluation import SCORE_CONFIGS, parse_evaluation_settings

BASELINE = Path(__file__).resolve().parents[1] / "experiments/baseline.yaml"


def _merge(base, overrides, section="experiment"):
    """Merge structured settings; optimizer/scheduler kwargs are replaced as a unit."""
    unknown = overrides.keys() - base.keys()
    if unknown:
        raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    # Optimizer and scheduler options are replaced when switching algorithms.
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
    # Schema-default calibration fields must also be accepted in override YAML.
    config["calibration"] = asdict(CalibrationConfig(**config.get("calibration", {})))
    overrides = yaml.safe_load(Path(path).expanduser().read_text()) if path else {}
    if not isinstance(overrides, dict):
        raise ValueError("Experiment YAML must contain a mapping of settings")
    score = overrides.get("score", {})
    if isinstance(score, dict) and score.get("kind", config["score"]["kind"]) not in SCORE_CONFIGS:
        raise ValueError(f"Unknown score.kind: {score.get('kind')}")
    if isinstance(score, dict) and score.get("kind", config["score"]["kind"]) != config["score"]["kind"]:
        config["score"] = asdict(SCORE_CONFIGS[score["kind"]]())
    _merge(config, overrides)
    evaluation = parse_evaluation_settings(config)
    datasets = config["data"]["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a nonempty list of cache directories")
    SplitConfig(**config["split"])
    TrainingConfig(**config["training"])
    SamplingConfig(**config["sampling"])
    ModelConfig(**config["model"])
    EncoderTaskConfig(**config["encoder_task"])
    background = EncoderBackgroundConfig(**config["encoder_background"])
    DecoderConfig(**config["decoder"])
    if not isinstance(config["data"]["domain_key"], str) or not config["data"]["domain_key"]:
        raise ValueError("data.domain_key must be a nonempty string")
    weighting = evaluation.weighting
    enabled = evaluation.calibration.weighted_cp
    if enabled and weighting.features == "background" and not background.enabled:
        raise ValueError("Background weighting requires encoder_background.enabled")
    return config
