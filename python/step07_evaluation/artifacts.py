"""Persist evaluation artifacts; no fitting or calibration happens here."""
from dataclasses import asdict
import json
import joblib
import numpy as np
from .diagnostics import background_diagnostics


def write_configuration(evaluation):
    counts = np.bincount(evaluation.exported["calibration"]["labels"], minlength=len(evaluation.class_names))
    evaluation.cp_config = {
        "inference": {"batch_size": evaluation.batch_size}, "checkpoint_sha256": evaluation.checkpoint_sha256,
        "settings_source": evaluation.settings_source, "device": str(next(evaluation.model.parameters()).device),
        "split": evaluation.experiment["split"], "weighting": asdict(evaluation.weighting_config),
        "score": asdict(evaluation.score_config), "calibration": asdict(evaluation.calibration_config),
        "alpha_grid": evaluation.alpha_grid, "class_names": evaluation.class_names,
        "split_seed": evaluation.experiment["split_seed"], "training_seed": evaluation.experiment["training_seed"],
        "checkpoint": str(evaluation.export_checkpoint_path), "checkpoint_epoch": evaluation.checkpoint["epoch"],
        "score_validation_accuracy": evaluation.score_validation_accuracy,
        "calibration_counts": counts.tolist(), "evaluation_unit": "segment",
        "split_manifest": str((evaluation.run_dir / "experiment.json").resolve()),
        "aps_seed": evaluation.aps_seed,
    }
    (evaluation.cp_dir / "config.json").write_text(json.dumps(evaluation.cp_config, indent=2))


def write_calibrators(evaluation):
    joblib.dump({"scorer": evaluation.scorer, "calibrator": evaluation.calibrator, "config": evaluation.cp_config,
                 "ratio_estimator": evaluation.ratio_estimator, "aps_calibrator": evaluation.aps_calibrator,
                 "aps_weighted_calibrator": evaluation.aps_weighted_calibrator}, evaluation.cp_dir / "scorer_calibrator.joblib")


def write_embeddings(evaluation):
    for split, arrays in evaluation.exported.items():
        dataset = evaluation.datasets[split]
        recording = np.asarray([str(dataset.time_files[i]) for i, j in dataset.samples])
        segment = np.asarray([j for i, j in dataset.samples])
        if len(recording) != len(arrays["labels"]):
            raise ValueError("Exported rows do not match recording/segment identifiers")
        np.savez_compressed(evaluation.cp_dir / f"{split}_embeddings.npz", **arrays,
                            recording=recording, segment_index=segment)


def write_predictions(evaluation):
    labels = evaluation.exported["test"]["labels"]
    alpha = evaluation.calibration_config.alpha
    np.savez_compressed(evaluation.cp_dir / "predictions.npz", labels=labels, scores=evaluation.test_scores,
                        p_values=evaluation.test_p_values, sets=evaluation.test_sets, point_predictions=evaluation.point_predictions,
                        cnn_probabilities=evaluation.exported["test"]["probabilities"], class_names=evaluation.class_names, alpha=alpha)
    np.savez_compressed(evaluation.cp_dir / "aps_predictions.npz", calibration_scores=evaluation.aps_cal_scores,
                        test_scores=evaluation.aps_test_scores, calibration_uniform=evaluation.aps_cal_uniform,
                        test_uniform=evaluation.aps_test_uniform,
                        unweighted_p_values=evaluation.method_p_values["APS unweighted"],
                        **({"weighted_p_values": evaluation.method_p_values["APS weighted"]} if evaluation.test_weights is not None else {}))


def write_weights(evaluation):
    if evaluation.ratio_estimator is not None:
        np.savez_compressed(evaluation.cp_dir / "calibration_weights.npz", weights=evaluation.calibration_weights,
                            source_fit_weights=evaluation.source_fit_weights)
        np.savez_compressed(evaluation.cp_dir / "test_weights.npz", weights=evaluation.test_weights)
        key = "background" if evaluation.weighting_config.features == "background" else "embeddings"
        pairs = evaluation.experiment["splits"]["test"]
        records = evaluation.experiment["recordings"]
        domain_key = evaluation.experiment["data"]["domain_key"]
        np.savez_compressed(evaluation.cp_dir / "target_features.npz", embeddings=evaluation.exported["test"][key],
                            recording=np.asarray([str(evaluation.datasets["test"].time_files[i]) for i, _ in pairs]),
                            segment_index=np.asarray([j for _, j in pairs]),
                            domains=np.asarray([records[i]["metadata"][domain_key] for i, _ in pairs]))
        (evaluation.cp_dir / "weighting.json").write_text(json.dumps(evaluation.weight_diagnostics, indent=2, allow_nan=False))


def write_diagnostics(evaluation):
    if hasattr(evaluation.model, "encode_background"):
        report = {}
        domain_map = {name: i for i, name in enumerate(evaluation.experiment["domain_names"])}
        for split, arrays in evaluation.exported.items():
            domain_ids = np.asarray([domain_map.get(evaluation.experiment["recordings"][i]["metadata"][evaluation.experiment["data"]["domain_key"]], -1)
                                     for i, _ in evaluation.experiment["splits"][split]])
            report[split] = background_diagnostics(evaluation.model, arrays, domain_ids)
        (evaluation.cp_dir / "background_diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False))

