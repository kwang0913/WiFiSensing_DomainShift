# Python experiments

[run.ipynb](run.ipynb) contains data loading, training, and evaluation. YAML configures experiments; model and utility modules keep the notebook readable.

## Setup and run

From the repository root, in your Python environment:

```bash
python -m pip install -r python/requirements.txt
python -m ipykernel install --user --name wifi-cp --display-name "Python (wifi-cp)"
python python/run_experiment.py --config python/configs/baseline.yaml
```

For interactive use, open the notebook with that kernel, set `config_path` in the first cell, and run in order. Restart the kernel for a fresh experiment after changing model code.

The CLI executes a notebook copy through Papermill. `stage: train` stops after training plots; `stage: full` starts a new training run and also evaluates conformal prediction. This setting only limits CLI execution. Optional arguments: `--kernel NAME`, `--output-dir NEW_PATH`.

## Configuration

Edit [baseline.yaml](configs/baseline.yaml) or supply a partial override. [sgd_cosine.yaml](configs/sgd_cosine.yaml) shows SGD, scheduling, and augmentation.

| Setting | Purpose |
|---|---|
| `data.datasets` | Cache paths relative to `python/dataset/`, or absolute paths |
| `prepare_npy_cache` | Convert new/changed MAT features to NPY in the notebook |
| `split_seed` / `training_seed` | Data splits/scorer folds versus initialization, shuffle, and augmentation |
| `training` | Epochs, batch size, LR, weight decay, optimizer; `patience: null` disables early stopping |
| `training.optimizer_kwargs` | Extra arguments for `adamw`, `adam`, `sgd`, or `rmsprop` |
| `scheduler` | `none`, `cosine`, or validation-loss `plateau` |
| `augmentation` | Master switch plus independent `time_noise`, `time_mask`, and `stft_mask` switches |
| `score.kind` | `kde`, `svm`, or `hbgb` |
| `calibration.alpha` | Target miscoverage level |

Omitted fields inherit the baseline. Optimizer/scheduler kwargs replace inherited dictionaries; changing algorithms clears inherited algorithm-specific kwargs. Scheduler and augmentation default to off.

## Data and splitting

The default combines setting_dataset and self_time. Other caches use `crossroom/generated_features` or `deep_l/generated_features`. Combined inputs must have matching shapes; see the [data guide](../data/README.md) for labels and formats. Loading uses mmap. Time features are standardized independently per segment/group over all feature rows and time samples, reducing overall offset/scale differences without fitting statistics across splits. STFT magnitudes retain their stored scale; the two branches therefore preserve different amplitude information.

The experiment measures conformal prediction under domain shift: target domains are unseen during model fitting, selection, and calibration. For user recognition, an activity can be the domain; for activity recognition, a user can be the domain. This is a known-class problem: the domain field must differ from the task, and test labels must exist in training.

Default configuration:

```yaml
data:
  task: user
split:
  mode: domain_holdout
  domain_key: activity
  test_domains: [walk]       # Multiple values are supported
  validation_fraction: 0.10
  calibration_fraction: 0.20
```

1. Reserve every segment of the selected target domains for test.
2. Expand source files into `(recording_index, segment_index)` pairs. Within each task label, shuffle and allocate validation/calibration fractions; round down with at least one segment each. The remainder goes to training.
3. Check that target labels exist in training. Source subsets may share recordings, but never the same segment; no target-domain segment enters them.

`test_fraction` is unused in domain-holdout mode. For activity recognition across users, set `task: activity`, `domain_key: user`, and list held-out users. `mode: random` uses a four-way **recording** split. Only indices are split; full signal arrays stay on disk.

**Why split source segments?** With limited recordings, this retains samples for every task class in training, validation, and calibration. It deliberately permits shared source recordings; distinct segment indices do not imply independent observations. The target domains remain fully held out. This is the current data-availability tradeoff, not a claim of recording-independent validation or calibration.

Source validation selects the CNN checkpoint and scorer settings; it need not simulate another unseen domain for this CP experiment. Freeze those choices before calibration/test. Calibration fits score ranks on separate source segments, and test measures target-domain coverage and set size empirically; `alpha` alone does not guarantee coverage under domain shift. SVM's internal probability calibration uses five training folds and requires at least five training segments per class.

The default dataset combination is not a complete user-by-activity grid: `walk` tests 6 of the 10 training user labels. Dataset/label associations remain part of this experiment; see the [label inventory](../data/README.md#datasets-and-labels) when choosing a different subset.

## Model and kernel choices

[dual_cnn.py](step03_models/dual_cnn.py) accepts paired `[B,3,270,T]` tensors. The 270 rows combine subcarriers and antenna links/pairs, rather than one homogeneous spatial axis. Full-height first kernels fuse these rows without sliding across their concatenation boundaries.

| Branch | First convolution: channels; kernel; stride | Second convolution: channels; kernel; stride | Adaptive pooling |
|---|---|---|---|
| Time | 3 → 128; `(270,24)`; `(1,24)` | 128 → 256; `(1,4)`; `(1,2)` | `(1,4)` |
| STFT | 3 → 128; `(270,1)`; `(1,1)` | 128 → 256; `(3,3)`; `(1,1)` on frequency/time | `(2,2)` |

- **Time:** the first layer mixes all feature rows over 24 samples (24 ms at 1000 Hz), reducing the sequence length early. The second learns local temporal patterns. This convolution is equivalent to a temporal Conv1d over 810 input channels after reshaping; using Conv2d keeps the input layout explicit. Kernel widths and strides are baseline choices, not established optima; early downsampling trades temporal detail for computation.
- **STFT:** `(270,1)` mixes features independently at each time-frequency point. The stored frequency-first sequence is then reshaped to `[B,128,frames,bins]` and transposed to `[B,128,bins,frames]`. A `(3,3)` kernel learns local spectral/temporal patterns without treating a frame's last frequency and the next frame's first frequency as neighbors.
- **Pooling and fusion:** time pooling preserves four coarse temporal positions; STFT pooling preserves a coarse `2×2` frequency/time grid. Each produces 1024 values followed by a 1024 → 1024 dense block. Concatenation feeds `2048 → 1024 → 512 → 128 → embedding → class logits`; the default embedding has 32 dimensions for downstream scoring.

All convolutions use valid padding to operate on observed input regions. Hidden blocks use LeakyReLU and BatchNorm; both inputs also have BatchNorm. Only the embedding block uses Dropout(0.5), disabled during evaluation/export. BatchNorm uses source-training running statistics at evaluation; the architecture does not enforce domain invariance.

Set `model.stft_frequency_bins` to 32 for setting_dataset/self_time or 64 for crossroom/deep_l; enabled STFT masking must match. Time and STFT windows share a detection peak but cover different durations, so dual-branch gains can also reflect additional context.

## Augmentation choices

Augmentation operates on stored features because source CSI is unavailable for some datasets. It regularizes feature use rather than simulating a specified wireless SNR or channel. No crops or shifts are applied.

`augmentation.enabled` is the master switch. Each component also has its own `enabled` flag, so any combination is supported. The baseline master switch is off; enabling it with the default component flags activates time noise and STFT masking.

| Component | Operation and purpose |
|---|---|
| `time_noise` | Gaussian or uniform noise on standardized time features; discourages reliance on exact feature values. `relative_std` scales noise standard deviation by each sample/group's RMS. |
| `time_mask` | Zeros one contiguous time interval across all rows/groups; encourages use of the remaining temporal evidence. Runs after noise so masked values remain zero. |
| `stft_mask` | Restores the time/frequency axes and independently masks a time band and a frequency band, shared across rows/groups; encourages use of distributed spectral and temporal information. |

`probability` is a per-sample selection probability, independently applied to each STFT axis. Mask width is sampled uniformly from zero through `floor(axis_length × max_fraction)`, then placed at a random valid position. A fraction is a maximum width, not an average or an application probability; selected masks can have zero width. For example, 21 frames with `time_max_fraction: 0.05` permits only zero or one masked frame.

Fresh randomness is drawn on each training batch from an RNG seeded once with `training_seed`; patterns are not fixed across epochs. Time and STFT masks are independent because their observation windows differ; noise does not trigger STFT recomputation. These are feature perturbations, not guaranteed physically consistent paired signals. Validation, calibration, test, and all embedding exports use clean inputs.

## Monitoring and outputs

W&B defaults to offline. Use `monitoring.enabled: false` to disable it, or run `wandb login` and set `mode: online` for live charts. Upload existing logs with `wandb sync <offline-run-directory>`. Initialization failures stop training; later logging failures preserve training and local saves.

Each experiment writes to `python/runs/<timestamp>/`:

- CLI: merged `config.yaml`, executed `run.ipynb`, and `execution.json` status.
- Training: `experiment.json`, `best.pt`, `history.json`, and `training.png`.
- Full evaluation: a `conformal-*` directory with embeddings, scorer/calibrator, predictions, metrics, and plots.

The final notebook cell jointly projects saved embeddings with t-SNE, colored by task label, domain, and split. Set `tsne_dir` to inspect an existing result without retraining; `tsne_max_per_split` limits plotting cost. It saves `embedding_tsne.png` and coordinates/labels/sample IDs in `embedding_tsne.npz`. Interpret local neighborhoods rather than treating projected gaps as original-space distances.

The manifest's `recordings` table and `splits` pairs preserve exact subset membership and order; `split_index_columns` defines the pair fields. Summaries report file/segment counts. `best.pt` supports inference with matching model code, not optimizer-state resume. Caches and outputs are excluded from Git.
