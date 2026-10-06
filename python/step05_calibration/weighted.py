"""Fit target weights and a weighted rank calibrator from frozen representations."""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .calibrate import WeightedRankCalibrator
from .weighting import (DensityRatioEstimator, DomainMixtureEstimator,
                        source_matching_weights, weight_summary)


@dataclass
class WeightedCalibration:
    calibrator: WeightedRankCalibrator
    ratio_estimator: object
    calibration_weights: np.ndarray
    test_weights: np.ndarray
    source_fit_weights: np.ndarray
    diagnostics: dict


def fit_weighted_calibration(scores, cal, train, target, experiment, config):
    key = "background" if config.features == "background" else "embeddings"
    domain_key = experiment["data"]["domain_key"]
    domains = np.asarray([r["metadata"][domain_key] for r in experiment["recordings"]])
    source_files = np.asarray(experiment["splits"]["train"])[:, 0]
    cal_files = np.asarray(experiment["splits"]["calibration"])[:, 0]
    source_fit_weights = source_matching_weights(
        list(zip(domains[source_files], train["labels"])),
        list(zip(domains[cal_files], cal["labels"])))
    estimator = (DomainMixtureEstimator if config.method == "domain_mixture"
                 else DensityRatioEstimator)
    ratio_estimator = estimator(config, seed=experiment["split_seed"])
    args = [train[key], target[key], source_fit_weights]
    if config.method == "domain_mixture":
        args.append(domains[source_files])
    # Only target features enter weight estimation, never target labels.
    ratio_estimator.fit(*args)
    calibration_weights = ratio_estimator.weights(cal[key])
    test_weights = ratio_estimator.weights(target[key])
    calibrator = WeightedRankCalibrator().fit(
        scores, cal["labels"], calibration_weights)
    weight_diagnostics = {
        "fit": ratio_estimator.fit_summary,
        "calibration": weight_summary(calibration_weights),
        "test": weight_summary(test_weights),
        "calibration_per_class": {name: weight_summary(calibration_weights[cal["labels"] == i])
                                  for i, name in enumerate(experiment["class_names"]) if (cal["labels"] == i).any()},
        "target_domain_segments": dict(Counter(domains[np.asarray(experiment["splits"]["test"])[:, 0]].tolist())),
        "features": config.features, "method": config.method,
        "weight_estimation_split": "test",
        "clipping": {"min": config.clip_min, "max": config.clip_max},
        "interpretation": "Unlabeled target inputs reused for estimated weights and evaluation; empirical transductive adaptation, no exact coverage guarantee.",
    }
    return WeightedCalibration(calibrator, ratio_estimator, calibration_weights,
                               test_weights, source_fit_weights, weight_diagnostics)
