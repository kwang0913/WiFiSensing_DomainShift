# Python experiments

Train paired time-domain/STFT models and evaluate conformal prediction under domain shift. Experiments use YAML configuration, memory-mapped NPY data, and optional W&B logging.

## Setup and run

From the repository root, in your Python environment:

```bash
python -m pip install -r python/requirements.txt
python -m ipykernel install --user --name wifi-cp --display-name "Python (wifi-cp)"
python python/run_experiment.py --config python/experiments/baseline.yaml
```

| Entry point | Purpose |
|---|---|
| [run.ipynb](run.ipynb) | Prepare data, train, select a checkpoint, calibrate, and evaluate |
| [evaluate.ipynb](evaluate.ipynb) | Compare selected runs; optionally reevaluate frozen weights without training |
| [run_experiment.py](run_experiment.py) | Execute the complete run notebook through Papermill |

For interactive training, select the kernel, set `config_path` in the first cell, and run in order. For comparison, edit `selected_runs` in `evaluate.ipynb`: each selected run adds CP/APS curves and one t-SNE row (task, domain, and split); colors/markers distinguish setups (Plain, DA, CL, DA + CL), while line styles distinguish CP, APS, and NN references. Saved results are reused by default. Set `recompute=True` to reevaluate checkpoints sequentially; `evaluation_config_path` optionally overrides evaluation settings. Restart the kernel after updating Python modules.

The CLI accepts `--kernel NAME` and `--output-dir NEW_PATH`. Use `--skip-plots` to skip plots, projections, and evaluation image logging while retaining numeric results and embeddings. Cluster instructions are in the [Slurm guide](slurm/README.md).

## Configuration and data

[baseline.yaml](experiments/baseline.yaml) defines defaults and documents the switches. Partial experiment YAML files inherit omitted values; changing an optimizer or scheduler clears inherited kwargs for the previous algorithm.

- **Data:** `data.datasets` selects caches under `python/dataset/` or absolute paths. Enable `prepare_npy_cache` to convert MAT features. Use 32 STFT bins for setting_dataset/self_time and 64 for crossroom; model and augmentation settings must match the cache.
- **Splits:** `data.domain_key` defines the shared domain. `domain_holdout` reserves `test_domains` for test. Optional `split.validation_domains` isolates entire validation users; otherwise validation is sampled from source users. Remaining source segments are split into train/calibration (and validation when not isolated), using the same fractions within each domain/class. `random` splits whole recordings. Source segment splits may share recordings, never segments.
- **Models:** `model.kind` selects `dual_cnn`, `cnn_transformer`, or `dual_link_graph`. `encoder_task` controls task objectives; `encoder_background` enables a separate encoder, with optional reconstruction through `decoder`.
- **Training:** `training`, `augmentation`, and `sampling` control optimization and batches. All enabled encoders, classifiers and the decoder share one optimizer/scheduler and one update per batch; GRL reverses only encoder gradients. Balanced sampling overrides `training.batch_size`. `inference.batch_size` separately controls export microbatches. Validation CE and accuracy weight users equally. Early stopping compares task CE only after active DA warmup and requires patience plus stability of all active DA/CL metrics; `training.early_stopping` groups `patience`, `min_delta`, and the auxiliary-window settings. Set its `patience` to `null` to disable stopping. DA/CL monitors use training metrics, including when validation users are unseen.
- **Calibration:** `score.kind` selects softmax, KDE, SVM, or HBGB. Randomized APS is evaluated alongside it. `calibration.weighted_cp` enables target-input weighting and weighted APS; background weighting requires a trained background encoder.

`best.pt` selects the lowest domain-mean validation CE strictly after the longest active DA warmup (warmup 10 means selection starts at epoch 11). Disabled or zero-weight adversaries do not delay selection. Total epochs must exceed this warmup. `last.pt` retains every last completed epoch; an interruption during warmup leaves no best checkpoint and null best metrics in `history.json`. To evaluate the latter, set `checkpoint_name="last.pt"` and `recompute=True` in evaluate. Calibration uses held-out segments from training domains, excluding isolated validation domains. Target labels are used for evaluation, not fitting. Estimated weighting under domain shift does not guarantee nominal coverage. Dataset layouts and feature formats are in the [data guide](../data/README.md).

## Code layout

| Module | Responsibility |
|---|---|
| `experiments/` | Experiment YAML files |
| `step00_config/` | Configuration types, loading, and validation |
| `step01_preprocessing/` | Split indices, memory-mapped datasets, and normalization |
| `step02_models/` | Encoders, classifiers, decoder, and shared blocks |
| `step03_training/` | Objectives, validation, stopping, checkpoints, sampling, augmentation, scheduling, and monitoring |
| `step04_scores/` | Nonconformity scores, including APS |
| `step05_calibration/` | Rank calibration and weight estimation |
| `step06_prediction/` | Representation export and prediction sets |
| `step07_evaluation/` | Metrics, diagnostics, saved results, and projection data |
| [evaluation_pipeline.py](evaluation_pipeline.py) | Shared export → score fitting → calibration → prediction → saving workflow |

Saved-result loading and comparison checks live in `step07_evaluation/comparison.py`; frozen checkpoint/data restoration lives in `restoration.py`, and projection data preparation in `projections.py`.

The training loop stays in `run.ipynb`; plotting stays in both notebooks for interactive adjustment. Rerun loading after changing the comparison selection; plot styling can be edited without recomputing t-SNE.

## Saved results

Each experiment writes to `python/runs/<timestamp>/` unless another output directory is selected.

| Artifact | Contents |
|---|---|
| `config.yaml`, `experiment.json` | Resolved configuration, recording metadata, and exact split indices |
| `best.pt`, `last.pt`, `history.json`, `training.png` | Best/final weights and training history |
| `run.ipynb`, `execution.json` | Executed notebook and status for CLI runs |
| `conformal-*/` | Evaluation settings, scorer/calibrators, exported embeddings, CP/APS predictions, metrics, and plots |

Weighted runs also save weights and estimation diagnostics. Reevaluation creates a new conformal result directory; comparison plots and selection provenance are saved separately under `python/comparisons/`. Checkpoints require matching model code and support inference, not optimizer-state resume. Data caches and generated results are excluded from Git.
