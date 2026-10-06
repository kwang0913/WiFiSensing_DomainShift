"""One parser for the evaluation settings used by training and saved-run evaluation."""
from dataclasses import dataclass
from .schema import (KDEConfig, SVMConfig, HBGBConfig, SoftmaxConfig,
                     CalibrationConfig, WeightingConfig, InferenceConfig)

SCORE_CONFIGS = {"kde": KDEConfig, "svm": SVMConfig, "hbgb": HBGBConfig, "softmax": SoftmaxConfig}


@dataclass
class EvaluationSettings:
    score: object
    calibration: CalibrationConfig
    weighting: WeightingConfig
    inference: InferenceConfig
    alpha_grid: list


def normalize_alpha_grid(levels, alpha):
    if not isinstance(levels, (list, tuple)) or not levels:
        raise ValueError("alpha_grid must be a nonempty list")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not 0 < value < 1 for value in levels):
        raise ValueError("Every alpha must be between 0 and 1")
    return sorted(set([*levels, alpha]))


def parse_evaluation_settings(config):
    score = config["score"].copy()
    kind = score.pop("kind")
    if kind not in SCORE_CONFIGS:
        raise ValueError(f"Unknown score.kind: {kind}")
    try:
        score = SCORE_CONFIGS[kind](**score)
    except TypeError as error:
        raise ValueError(f"Invalid settings for score.kind={kind}: {error}") from error
    calibration = CalibrationConfig(**config["calibration"])
    weighting = WeightingConfig(**config["weighting"])
    inference = InferenceConfig(**config["inference"])
    levels = normalize_alpha_grid(config["alpha_grid"], calibration.alpha)
    return EvaluationSettings(score, calibration, weighting, inference, levels)
