"""Shared frozen-model evaluation stages for run.ipynb and evaluate.ipynb."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from step00_config.schema import CalibrationConfig, WeightingConfig
from step00_config.evaluation import parse_evaluation_settings, normalize_alpha_grid
from step04_scores.scorers import build_scorer
from step04_scores.aps import randomized_aps_scores
from step05_calibration.calibrate import RankCalibrator, WeightedRankCalibrator
from step05_calibration.weighted import fit_weighted_calibration
from step06_prediction.representations import export_representations
from step06_prediction.predict import prediction_sets
from step07_evaluation.metrics import summarize_predictions
from step07_evaluation.artifacts import (write_configuration, write_calibrators, write_embeddings,
                                         write_predictions, write_weights, write_diagnostics)


class ConformalEvaluation:
    """One selected checkpoint, one ordered export, and one evaluation configuration.

    Recreate the session to change settings. Rerunning an upstream stage invalidates
    its downstream results; failed stages cannot expose a completed old result.
    No network optimizer or training operation is created here.
    """
    def __init__(self, model, datasets, experiment, config, checkpoint_path,
                 *, settings_source="run configuration", aps_seed=42):
        self.model = model
        self.datasets = datasets
        self.experiment = deepcopy(experiment)
        self.checkpoint_path = Path(checkpoint_path)
        self.run_dir = self.checkpoint_path.parent
        self.settings_source = settings_source
        self.aps_seed = aps_seed
        self.class_names = self.experiment["class_names"]
        settings = parse_evaluation_settings(config)
        self.batch_size = settings.inference.batch_size
        self.score_config = settings.score
        self.calibration_config = settings.calibration
        self.weighting_config = settings.weighting
        self.alpha_grid = settings.alpha_grid
        if (self.calibration_config.weighted_cp and self.weighting_config.features == "background"
                and not hasattr(model, "encode_background")):
            raise ValueError("Background weighting requires a trained background encoder")
        self.export_complete = self.score_fit_complete = False
        self.calibration_complete = self.prediction_complete = False
        self.saved_complete = False

    def _check_model(self):
        state = self.model.state_dict()
        saved = self.checkpoint["model_state_dict"]
        if state.keys() != saved.keys() or not all(
                torch.equal(value.detach().cpu(), saved[name]) for name, value in state.items()):
            raise ValueError("Model weights changed after export; rerun export and score fitting")

    def export(self):
        self.saved_complete = False
        self.export_complete = self.score_fit_complete = False
        self.calibration_complete = self.prediction_complete = False
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.export_checkpoint_path = self.checkpoint_path.resolve()
        with self.checkpoint_path.open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model.load_state_dict(self.checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.exported = {}
        for split in ("train", "validation", "calibration", "test"):
            loader = DataLoader(self.datasets[split], batch_size=self.batch_size,
                                shuffle=False, num_workers=0, drop_last=False)
            self.exported[split] = export_representations(
                self.model, loader, inference_batch_size=self.batch_size, include_background=True)
            arrays = self.exported[split]
            print(f"{split}: {len(arrays['labels'])} embeddings, shape {arrays['embeddings'].shape}")
        self.export_complete = True

    def fit_score(self):
        self.saved_complete = False
        self.score_fit_complete = self.calibration_complete = self.prediction_complete = False
        if not self.export_complete:
            raise ValueError("Representation export incomplete; rerun export successfully")
        self._check_model()
        train, validation = self.exported["train"], self.exported["validation"]
        if not np.array_equal(np.unique(train["labels"]), np.arange(len(self.class_names))):
            raise ValueError("The scoring training split must contain every class")
        self.scorer = build_scorer(self.score_config, len(self.class_names), seed=self.experiment["split_seed"])
        if self.scorer.input_key == "embeddings":
            self.scorer.fit(train["embeddings"], train["labels"])
        scores = self.scorer.score(validation[self.scorer.input_key])
        self.score_validation_accuracy = float((scores.argmin(1) == validation["labels"]).mean())
        print("Score settings:", asdict(self.score_config))
        print(f"Scorer validation accuracy: {self.score_validation_accuracy:.4f}")
        self.fitted_score_config = asdict(self.score_config)
        self.score_fit_complete = True

    def calibrate(self):
        self.saved_complete = False
        self.calibration_complete = self.prediction_complete = False
        if not self.score_fit_complete:
            raise ValueError("Score fitting incomplete; rerun export and score fitting before calibration")
        self._check_model()
        if asdict(self.score_config) != self.fitted_score_config:
            raise ValueError("Score settings changed; refit the score before calibration")
        CalibrationConfig(**asdict(self.calibration_config))
        WeightingConfig(**asdict(self.weighting_config))
        self.alpha_grid = normalize_alpha_grid(self.alpha_grid, self.calibration_config.alpha)
        self.frozen_calibration_config = asdict(self.calibration_config)
        self.frozen_weighting_config = asdict(self.weighting_config)
        self.frozen_alpha_grid = list(self.alpha_grid)
        self.frozen_aps_seed = self.aps_seed
        cal, train, target = (self.exported[k] for k in ("calibration", "train", "test"))
        self.calibration_scores = self.scorer.score(cal[self.scorer.input_key])
        self.ratio_estimator = None
        self.calibration_weights = self.test_weights = None
        self.weight_diagnostics = {}
        if self.calibration_config.weighted_cp:
            weighted = fit_weighted_calibration(self.calibration_scores, cal, train, target,
                                                self.experiment, self.weighting_config)
            self.calibrator, self.ratio_estimator = weighted.calibrator, weighted.ratio_estimator
            self.calibration_weights, self.test_weights = weighted.calibration_weights, weighted.test_weights
            self.source_fit_weights, self.weight_diagnostics = weighted.source_fit_weights, weighted.diagnostics

        else:
            self.calibrator = RankCalibrator().fit(self.calibration_scores, cal["labels"])
        # Independent calibration/test RNG streams; fixed draws across the alpha sweep.
        streams = np.random.SeedSequence(self.aps_seed).spawn(2)
        self.aps_cal_scores, self.aps_cal_uniform = randomized_aps_scores(cal["probabilities"], np.random.default_rng(streams[0]))
        self.aps_test_scores, self.aps_test_uniform = randomized_aps_scores(target["probabilities"], np.random.default_rng(streams[1]))
        self.aps_calibrator = RankCalibrator().fit(self.aps_cal_scores, cal["labels"])
        self.aps_weighted_calibrator = None
        if self.calibration_weights is not None:
            self.aps_weighted_calibrator = WeightedRankCalibrator().fit(
                self.aps_cal_scores, cal["labels"], self.calibration_weights)
        self.calibrated_scorer = self.scorer
        self.calibration_complete = True

    def _check_calibration(self):
        if not self.calibration_complete:
            raise ValueError("Calibration incomplete; rerun calibration successfully")
        self._check_model()
        if self.scorer is not self.calibrated_scorer:
            raise ValueError("Score model was replaced after calibration; rerun calibration")
        if asdict(self.score_config) != self.fitted_score_config or asdict(self.calibration_config) != self.frozen_calibration_config:
            raise ValueError("Settings changed after calibration; refit or recalibrate")
        if asdict(self.weighting_config) != self.frozen_weighting_config:
            raise ValueError("Weighting settings changed after calibration; rerun calibration")
        if self.alpha_grid != self.frozen_alpha_grid:
            raise ValueError("Alpha grid changed after calibration; rerun calibration")
        if self.aps_seed != self.frozen_aps_seed:
            raise ValueError("APS seed changed after calibration; rerun calibration")

    def predict(self):
        self.saved_complete = self.prediction_complete = False
        self._check_calibration()
        target = self.exported["test"]
        self.test_scores = self.scorer.score(target[self.scorer.input_key])
        self.test_p_values = (self.calibrator.p_values(self.test_scores, self.test_weights)
                              if self.test_weights is not None else self.calibrator.p_values(self.test_scores))
        self.test_sets = prediction_sets(self.test_p_values, self.calibration_config.alpha)
        self.point_predictions = self.test_scores.argmin(1)
        self.nn_predictions = target["probabilities"].argmax(1)
        name = self.score_config.kind + (" weighted CP" if self.calibration_config.weighted_cp else " CP")
        self.primary_method = name
        self.method_p_values = {name: self.test_p_values,
                               "APS unweighted": self.aps_calibrator.p_values(self.aps_test_scores)}
        if self.aps_weighted_calibrator is not None:
            self.method_p_values["APS weighted"] = self.aps_weighted_calibrator.p_values(self.aps_test_scores, self.test_weights)
        self.prediction_complete = True

    def save_results(self):
        self.saved_complete = False
        if not self.prediction_complete:
            raise ValueError("Prediction incomplete; rerun prediction successfully")
        self._check_calibration()
        summary = summarize_predictions(
            self.exported["test"]["labels"], self.test_sets, self.point_predictions,
            self.nn_predictions, self.method_p_values, self.alpha_grid,
            self.calibration_config.alpha, self.class_names, self.experiment,
            primary_method=self.primary_method)
        self.nn_accuracy = summary["nn_accuracy"]
        self.metrics, self.per_class = summary["metrics"], summary["per_class"]
        self.method_results, self.alpha_results = summary["method_results"], summary["alpha_results"]
        self.cp_dir = self.run_dir / ("conformal-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        self.cp_dir.mkdir(exist_ok=False)
        write_configuration(self)
        write_calibrators(self)
        write_embeddings(self)
        write_predictions(self)
        write_weights(self)
        write_diagnostics(self)
        # Write completion marker last, after all requested artifacts succeed.
        results = {"per_domain": summary["per_domain"], "weighted_cp": self.calibration_config.weighted_cp,
                   "overall": self.metrics, "per_class": self.per_class, "alpha_sweep": self.alpha_results,
                   "methods": self.method_results, "aps_seed": self.aps_seed}
        (self.cp_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))
        for name, rows in self.method_results.items():
            row = next(row for row in rows if row["alpha"] == self.calibration_config.alpha)
            print(f"{name}: coverage={row['coverage']:.3f}, mean size={row['mean_set_size']:.3f}, empty={row['empty_fraction']:.3f}")
        print("Results:", self.cp_dir)
        self.saved_complete = True
        return self.cp_dir

    def require_saved_results(self):
        """Prevent plotting/logging stale results after a failed or repeated stage."""
        if not self.saved_complete:
            raise ValueError("Saved evaluation incomplete; rerun prediction and save_results")
        self._check_calibration()
