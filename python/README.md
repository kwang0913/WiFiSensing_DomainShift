# Python experiments

[run.ipynb](run.ipynb) contains data loading, training, and evaluation. YAML configures experiments; model and utility modules keep the notebook readable.

## Setup and run

From the repository root, in your Python environment:

```bash
python -m pip install -r python/requirements.txt
python -m ipykernel install --user --name wifi-cp --display-name "Python (wifi-cp)"
python python/run_experiment.py --config python/experiments/baseline.yaml
```

For interactive use, open the notebook with that kernel, set `config_path` in the first cell, and run in order. Restart the kernel for a fresh experiment after changing model code.

The CLI executes the complete notebook through Papermill: training, embedding export, scoring, conformal evaluation, and plots. To run only selected parts, execute cells interactively. Optional arguments: `--kernel NAME`, `--output-dir NEW_PATH`.

For Amarel job arrays and environment setup, see the [Slurm guide](slurm/README.md).

## Code layout

| Directory | Responsibility |
|---|---|
| `experiments/` | Experiment YAML files |
| `step00_config/` | Configuration types (`schema.py`) and loading/merging (`loader.py`) |
| `step01_preprocessing/` | Feature normalization |
| `step02_models/` | CNN architecture and model construction |
| `step03_training/` | Augmentation, balanced sampling, contrastive loss, scheduling, and monitoring |
| `step04_scores/` | KDE and classifier-based nonconformity scores |
| `step05_calibration/` | Class-conditional or pooled rank calibration |
| `step06_prediction/` | Prediction-set construction |
| `step07_evaluation/` | Coverage, set size, and classification metrics |

The training loop stays in `run.ipynb`; `run_experiment.py` executes it from YAML. Edit YAML for experiments; `step00_config/schema.py` defines the Python configuration objects used by components.

## Configuration

Edit [baseline.yaml](experiments/baseline.yaml) or supply a partial override. Inline examples show optimizer and scheduler options.

| Setting | Purpose |
|---|---|
| `data.datasets` | Cache paths relative to `python/dataset/`, or absolute paths |
| `prepare_npy_cache` | Convert new/changed MAT features to NPY in the notebook |
| `split_seed` / `training_seed` | Data splits/scorer folds versus initialization, shuffle, and augmentation |
| `training` | Epochs, batch size, LR, weight decay, optimizer; `patience: null` disables early stopping |
| `training.optimizer_kwargs` | Extra arguments for `adamw`, `adam`, `sgd`, or `rmsprop` |
| `scheduler` | `none`, `cosine`, or validation-loss `plateau` |
| `augmentation` | Master switch plus independent `time_noise`, `time_mask`, and `stft_mask` switches |
| `adversarial` | Optional conditional domain classifier, reversed gradient weight, and warmup |
| `contrastive` | Cross-domain same-class positives; independent switch, weight, and temperature |
| `sampling` | Training class/domain-balanced batches; overrides training batch size when enabled |
| `score.kind` | `kde`, `svm`, or `hbgb` |
| `calibration.alpha` | Target miscoverage level |

Omitted fields inherit the baseline. Optimizer/scheduler kwargs replace inherited dictionaries. Changing algorithms in an override YAML clears inherited algorithm-specific kwargs; when editing baseline directly, replace the matching kwargs yourself. The YAML is the source of truth for active settings.

## Data and splitting

Select setting_dataset, self_time, or both through `data.datasets`. The default uses `crossroom/generated_features`. Combined inputs must have matching shapes; see the [data guide](../data/README.md) for labels and formats. Loading uses mmap. Time features are standardized independently per segment/group over all feature rows and time samples, reducing overall offset/scale differences without fitting statistics across splits. STFT magnitudes retain their stored scale; the two branches therefore preserve different amplitude information.

The experiment measures conformal prediction under domain shift: target domains are unseen during model fitting, selection, and calibration. For user recognition, an activity can be the domain; for activity recognition, a user can be the domain. This is a known-class problem: the domain field must differ from the task, and test labels must exist in training.

Default configuration:

```yaml
data:
  datasets: [crossroom/generated_features]
  task: activity
split:
  mode: domain_holdout
  domain_key: user
  test_domains: [cong]       # Multiple values are supported
  validation_fraction: 0.10
  calibration_fraction: 0.20
```

1. Reserve every segment of the selected target domains for test.
2. Expand source files into `(recording_index, segment_index)` pairs. Within each task label, shuffle and allocate validation/calibration fractions; round down with at least one segment each. The remainder goes to training.
3. Check that target labels exist in training. Source subsets may share recordings, but never the same segment; no target-domain segment enters them.

`test_fraction` is unused in domain-holdout mode. For user recognition across activities, set `task: user`, `domain_key: activity`, and list held-out activities. `mode: random` uses a four-way **recording** split. Only indices are split; full signal arrays stay on disk.

**Why split source segments?** With limited recordings, this retains samples for every task class in training, validation, and calibration. It deliberately permits shared source recordings; distinct segment indices do not imply independent observations. The target domains remain fully held out. This is the current data-availability tradeoff, not a claim of recording-independent validation or calibration.

Source validation selects the CNN checkpoint and scorer settings; it need not simulate another unseen domain for this CP experiment. Freeze those choices before calibration/test. Calibration fits score ranks on separate source segments, and test measures target-domain coverage and set size empirically; `alpha` alone does not guarantee coverage under domain shift. SVM's internal probability calibration uses five training folds and requires at least five training segments per class.

Combining setting_dataset and self_time does not produce a complete user-by-activity grid: `walk` tests 6 of the 10 training user labels. Dataset/label associations remain part of this experiment; see the [label inventory](../data/README.md#datasets-and-labels) when choosing a different subset.

## Model and kernel choices

[dual_cnn.py](step02_models/dual_cnn.py) accepts paired `[B,3,270,T]` tensors. The 270 rows combine subcarriers and antenna links/pairs, rather than one homogeneous spatial axis. Full-height first kernels fuse these rows without sliding across their concatenation boundaries.

| Branch | First convolution: channels; kernel; stride | Second convolution: channels; kernel; stride | Adaptive pooling |
|---|---|---|---|
| Time | 3 → 128; `(270,24)`; `(1,24)` | 128 → 256; `(1,4)`; `(1,2)` | `(1,4)` |
| STFT | 3 → 128; `(270,1)`; `(1,1)` | 128 → 256; `(3,3)`; `(1,1)` on frequency/time | `(2,2)` |

- **Time:** the first layer mixes all feature rows over 24 samples (24 ms at 1000 Hz), reducing the sequence length early. The second learns local temporal patterns. This convolution is equivalent to a temporal Conv1d over 810 input channels after reshaping; using Conv2d keeps the input layout explicit. Kernel widths and strides are baseline choices, not established optima; early downsampling trades temporal detail for computation.
- **STFT:** `(270,1)` mixes features independently at each time-frequency point. The stored frequency-first sequence is then reshaped to `[B,128,frames,bins]` and transposed to `[B,128,bins,frames]`. A `(3,3)` kernel learns local spectral/temporal patterns without treating a frame's last frequency and the next frame's first frequency as neighbors.
- **Pooling and fusion:** time pooling preserves four coarse temporal positions; STFT pooling preserves a coarse `2×2` frequency/time grid. Each produces 1024 values followed by a 1024 → 1024 dense block. Concatenation feeds `2048 → 1024 → 512 → 128 → embedding → class logits`; the default embedding has 32 dimensions for downstream scoring.

All convolutions use valid padding to operate on observed input regions. Hidden blocks use ReLU and BatchNorm; both inputs also have BatchNorm. Only the embedding block uses Dropout(0.5), disabled during evaluation/export. BatchNorm uses source-training running statistics at evaluation; the architecture does not enforce domain invariance.

Set `model.stft_frequency_bins` to 32 for setting_dataset/self_time or 64 for crossroom; enabled STFT masking must match. Time and STFT windows share a detection peak but cover different durations, so dual-branch gains can also reflect additional context.

## Augmentation choices

Augmentation operates on stored features because source CSI is unavailable for some datasets. It regularizes feature use rather than simulating a specified wireless SNR or channel. No crops or shifts are applied.

`augmentation.enabled` is the master switch. Each component also has its own `enabled` flag, so any combination is supported.

| Component | Operation and purpose |
|---|---|
| `time_noise` | Gaussian or uniform noise on standardized time features; discourages reliance on exact feature values. `relative_std` scales noise standard deviation by each sample/group's RMS. |
| `time_mask` | Zeros one contiguous time interval across all rows/groups; encourages use of the remaining temporal evidence. Runs after noise so masked values remain zero. |
| `stft_mask` | Restores the time/frequency axes and independently masks a time band and a frequency band, shared across rows/groups; encourages use of distributed spectral and temporal information. |

`probability` is a per-sample selection probability, independently applied to each STFT axis. Mask width is sampled uniformly from zero through `floor(axis_length × max_fraction)`, then placed at a random valid position. A fraction is a maximum width, not an average or an application probability; selected masks can have zero width. For example, 21 frames with `time_max_fraction: 0.05` permits only zero or one masked frame.

Fresh randomness is drawn on each training batch from an RNG seeded once with `training_seed`; patterns are not fixed across epochs. Time and STFT masks are independent because their observation windows differ; noise does not trigger STFT recomputation. These are feature perturbations, not guaranteed physically consistent paired signals. Validation, calibration, test, and all embedding exports use clean inputs.

## Domain-adversarial training

Set `adversarial.enabled: true` in baseline or an override YAML; `false` uses the
ordinary task classifier unless contrastive learning is enabled.

A small domain classifier branches off the shared embedding. It learns the chosen
`domain_key` (activity for user recognition), while gradient reversal makes the encoder
suppress that information and retain task discrimination. Only training segments supply
optimization targets; source validation labels provide diagnostics. Held-out domains,
calibration and test never enter this objective.
At least two training domains are required, distinct from the prediction task.
For activity recognition across users, also change `adversarial.domain_key` to `user`.

`max_weight` scales the reversed encoder gradient, not the domain classifier's own
cross-entropy gradient. It ramps from zero over `warmup_epochs` (0 means immediate);
`hidden_dim` sets the domain classifier width. Start with 0.1 and 10 epochs as an
experimental setting, not a tuned optimum. Strong suppression can remove useful identity
cues. Low domain accuracy alone is not proof of invariance or unseen-domain performance.

The domain head has independent `optimizer`, `learning_rate`, `weight_decay`,
`optimizer_kwargs`, and `scheduler` settings. Each batch performs `steps_per_batch`
head updates on the same detached embedding, followed by one encoder/task update with
the head fixed. The encoder runs once, so extra head steps do not repeat BatchNorm updates.
The baseline uses five head steps and a fixed domain LR; these are experimental choices.

History and W&B record train/clean-validation domain loss and accuracy, domain LR, and
mean reversal weight. Training domain metrics are measured after the head updates on
the augmented batch. Validation uses eval mode without augmentation. Domains absent
from training are excluded from domain metrics; `validation_domain_samples` reports the
count. Domain `plateau` scheduling monitors this validation loss and skips epochs with
no known domains. Main-model scheduling and checkpoint selection still use task loss.

The checkpoint includes the domain head, vocabulary and configuration. Rebuild with
`domain_classes=len(domain_names)`, `domain_hidden_dim=adversarial.hidden_dim`, and
`domain_conditional=adversarial.conditional` to load it; inference and KDE/CP still use the usual task logits and embeddings.

## Cross-domain representation learning

`contrastive.enabled`, `adversarial.enabled`, and `sampling.enabled` are independent.
All three use `adversarial.domain_key`, even when the adversarial loss is disabled.
For user recognition this is activity; for activity recognition set it to user.

With `adversarial.conditional: true`, the domain head receives the embedding and the
true task-label one-hot vector. Only the embedding gradient is reversed. This aims to
suppress domain information within each task class; task labels never enter the encoder
or task classifier. The head-only diagnostic also supplies task labels and skips when
validation contains no known domains. Its reference accuracy depends on domain frequencies
within each class, not necessarily uniform chance.

The contrastive loss uses L2-normalized embeddings: positives share the task label but
have different domains; negatives have different task labels. Same-class/same-domain
pairs are excluded. `temperature` scales cosine similarities, and `weight` scales the
loss added to task CE. There is no projection head; training retains the existing
embedding Dropout, and export uses eval mode with Dropout disabled. The encoder receives
task CE + weighted contrastive loss - weighted domain CE; the domain head minimizes its
own CE. GRL applies the domain weight once. Checkpoint selection still uses task validation CE.

Balanced sampling selects task classes, domains within each class, and segments within
each group. It prefers distinct recordings where available. The batch size is
`min(classes_per_batch, available_classes) * domains_per_class * samples_per_domain`
(72 in baseline); `training.batch_size` still controls evaluation. Missing domains use
repeated slots from available domains. Small groups repeat segments only after exhausting
the group, and classes with a single domain remain in training but have no contrastive
positives. Training fails if no class has cross-domain positives at all.

One epoch draws enough full batches to cover the original training size numerically;
it does not guarantee visiting each segment. A seeded RNG advances between epochs.
History reports actual draws, contrastive loss averaged over valid anchors, and
`train_contrastive_anchor_fraction`. Calibration/test are never resampled. Embedding
export visits each original segment exactly once in manifest order, including training.
The manifest, checkpoint, and W&B record both new configurations and the effective
training batch size. Disable all three switches for ordinary classification.

## Monitoring and outputs

Select `monitoring.mode: offline` for local logging. Use `monitoring.enabled: false` to disable it, or run `wandb login` and set `mode: online` for live charts. Upload existing logs with `wandb sync <offline-run-directory>`. Initialization failures stop training; later logging failures preserve training and local saves.

Each experiment writes to `python/runs/<timestamp>/`:

- CLI: merged `config.yaml`, executed `run.ipynb`, and `execution.json` status.
- Training: `experiment.json`, `best.pt`, `history.json`, and `training.png`.
- Full evaluation: a `conformal-*` directory with embeddings, scorer/calibrator, predictions, metrics, and plots.

The embedding-comparison cell jointly projects saved embeddings with t-SNE, colored by task label, domain, and split. Set `viz_dir` to inspect an existing result without retraining; `viz_max_per_split` limits plotting cost. It saves `embedding_tsne.png` and coordinates/labels/sample IDs in `embedding_tsne.npz`. In the same cell, Isomap, Spectral Embedding and metric MDS reuse the same samples/colors and save `embedding_isomap`, `embedding_spectral`, and `embedding_mds` PNG/NPZ files. Adjust `map_neighbors` for the graph methods; heed disconnected-graph warnings. Interpret each method according to its distance/neighborhood objective, not as proof of domain invariance.

The manifest's `recordings` table and `splits` pairs preserve exact subset membership and order; `split_index_columns` defines the pair fields. Summaries report file/segment counts. `best.pt` supports inference with matching model code, not optimizer-state resume. Caches and outputs are excluded from Git.
