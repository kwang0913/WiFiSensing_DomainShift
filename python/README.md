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
| `step05_calibration/` | Pooled rank calibration |
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
| `training.scheduler` | `none`, `cosine`, or validation-loss `plateau` |
| `augmentation` | Master switch plus independent `time_noise`, `time_mask`, and `stft_mask` switches |
| `encoder_task.adversarial` | Optional conditional domain classifier, reversed gradient weight, and warmup |
| `encoder_task.contrastive` | Cross-domain same-class positives; independent switch, weight, and temperature |
| `sampling` | Training class/domain-balanced batches; overrides training batch size when enabled |
| `score.kind` | `softmax` (direct 1-p), `kde`, `svm`, or `hbgb` |
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
  domain_key: user
split:
  mode: domain_holdout
  test_domains: [cong]       # Multiple values are supported
  validation_fraction: 0.10
  calibration_fraction: 0.20
```

1. Reserve every segment of the selected target domains for test.
2. Expand source files into `(recording_index, segment_index)` pairs. Within each task label, shuffle and allocate validation/calibration fractions; round down with at least one segment each. The remainder goes to training.
3. Check that target labels exist in training. Source subsets may share recordings, but never the same segment; no target-domain segment enters them.

`test_fraction` is unused in domain-holdout mode. For user recognition across activities, set `data.task: user`, `data.domain_key: activity`, and list held-out activities. `mode: random` uses a four-way **recording** split. Only indices are split; full signal arrays stay on disk.

**Why split source segments?** With limited recordings, this retains samples for every task class in training, validation, and calibration. It deliberately permits shared source recordings; distinct segment indices do not imply independent observations. The target domains remain fully held out. This is the current data-availability tradeoff, not a claim of recording-independent validation or calibration.

Source validation selects the CNN checkpoint and scorer settings; it need not simulate another unseen domain for this CP experiment. Freeze those choices before calibration/test. Calibration fits score ranks on separate source segments, and test measures target-domain coverage and set size empirically; `alpha` alone does not guarantee coverage under domain shift. SVM's internal probability calibration uses five training folds and requires at least five training segments per class.

Combining setting_dataset and self_time does not produce a complete user-by-activity grid: `walk` tests 6 of the 10 training user labels. Dataset/label associations remain part of this experiment; see the [label inventory](../data/README.md#datasets-and-labels) when choosing a different subset.

## Model and kernel choices

[dual_cnn.py](step02_models/dual_cnn.py) accepts paired `[B,3,270,T]` tensors. The 270 rows combine subcarriers and antenna links/pairs, rather than one homogeneous spatial axis. Full-height first kernels fuse these rows without sliding across their concatenation boundaries.

| Branch | First convolution: channels; kernel; stride | Second convolution: channels; kernel; stride | Adaptive pooling |
|---|---|---|---|
| Time | 3 → 32; `(270,24)`; `(1,24)` | 32 → 64; `(1,4)`; `(1,2)` | `(1,4)` |
| STFT | 3 → 32; `(270,1)`; `(1,1)` | 32 → 64; `(3,3)`; `(1,1)` on frequency/time | `(2,2)` |

- **Time:** the first layer mixes all feature rows over 24 samples (24 ms at 1000 Hz), reducing the sequence length early. The second learns local temporal patterns. This convolution is equivalent to a temporal Conv1d over 810 input channels after reshaping; using Conv2d keeps the input layout explicit. Kernel widths and strides are baseline choices, not established optima; early downsampling trades temporal detail for computation.
- **STFT:** `(270,1)` mixes features independently at each time-frequency point. The stored frequency-first sequence is then reshaped to `[B,32,frames,bins]` and transposed to `[B,32,bins,frames]`. A `(3,3)` kernel learns local spectral/temporal patterns without treating a frame's last frequency and the next frame's first frequency as neighbors.
- **Pooling and fusion:** time pooling preserves four coarse temporal positions; STFT pooling preserves a coarse `2×2` frequency/time grid. Each produces 256 values directly, with no branch dense block. Concatenation feeds `512 → 128 → embedding → class logits`; the default embedding has 32 dimensions for downstream scoring.

DualCNN convolutions use valid padding. Its branch hidden blocks use GroupNorm and GELU; neither branch normalizes the raw input. The compact shared embedding is linear, without BatchNorm or Dropout. GroupNorm uses per-sample statistics in both training and evaluation; it does not guarantee domain invariance.

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

Set `encoder_task.adversarial.enabled: true` in baseline or an override YAML; `false` uses the
task classifier without task-domain reversal; other encoder objectives remain independent.

A small domain classifier branches off the shared embedding. It learns the chosen
`domain_key` (activity for user recognition), while gradient reversal makes the encoder
suppress that information and retain task discrimination. Only training segments supply
optimization targets; source validation labels provide diagnostics. Held-out domains,
calibration and test never enter this objective.
At least two training domains are required, distinct from the prediction task.
For activity recognition across users, also change `data.domain_key` to `user`.

`max_weight` scales the reversed encoder gradient, not the domain classifier's own
cross-entropy gradient. It ramps from zero over `warmup_epochs` (0 means immediate);
`hidden_dim` sets the domain classifier width. Start with 0.1 and 10 epochs as an
experimental setting, not a tuned optimum. Strong suppression can remove useful identity
cues. Low domain accuracy alone is not proof of invariance or unseen-domain performance.

The domain head has independent `optimizer`, `learning_rate`, `weight_decay`,
`optimizer_kwargs`, and `scheduler` settings. Each batch performs `steps_per_batch`
head updates on the same detached embedding, followed by one encoder/task update with
the head fixed. The encoder runs once; extra head steps reuse detached embeddings.
The baseline uses five head steps and a cosine domain scheduler; these are experimental choices.

History and W&B record train/clean-validation domain loss and accuracy, domain LR, and
mean reversal weight. Training domain metrics are measured after the head updates on
the augmented batch. Validation uses eval mode without augmentation. Domains absent
from training are excluded from domain metrics; `validation_domain_samples` reports the
count. Domain `plateau` scheduling monitors this validation loss and skips epochs with
no known domains. Main-model scheduling and checkpoint selection still use task loss.

The checkpoint includes the domain head, vocabulary and configuration. Rebuild with
`domain_classes=len(domain_names)`, `domain_hidden_dim=encoder_task.adversarial.hidden_dim`, and
`domain_conditional=encoder_task.adversarial.conditional` to load it; inference and KDE/CP still use the usual task logits and embeddings.

## Cross-domain representation learning

`encoder_task.contrastive.enabled`, `encoder_task.adversarial.enabled`, and `sampling.enabled` are independent.
All three use `data.domain_key`, even when the adversarial loss is disabled.
For user recognition this is activity; for activity recognition set it to user.

With `encoder_task.adversarial.conditional: true`, the domain head receives the embedding and the
true task-label one-hot vector. Only the embedding gradient is reversed. This aims to
suppress domain information within each task class; task labels never enter the encoder
or task classifier. Domain-head reference accuracy depends on domain frequencies
within each class, not necessarily uniform chance.

The contrastive loss uses L2-normalized embeddings: positives share the task label but
have different domains; negatives have different task labels. Same-class/same-domain
pairs are excluded. `temperature` scales cosine similarities, and `weight` scales the
loss added to task CE. There is no projection head; training retains the existing
embedding Dropout, and export uses eval mode with Dropout disabled. The encoder receives
weighted task CE + weighted contrastive loss - weighted domain CE; the domain head minimizes its
own CE. GRL applies the domain weight once. Checkpoint selection still uses task validation CE.

Balanced sampling selects task classes, domains within each class, and segments within
each group. It prefers distinct recordings where available. The batch size is
`min(classes_per_batch, available_classes) * domains_per_class * samples_per_domain`
(10 in baseline); `training.batch_size` still controls evaluation. Missing domains use
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

The embedding computation cell runs t-SNE and metric MDS by default, colored by task, domain and split. Set `viz_dir` to inspect an existing result without retraining; `viz_max_per_split` limits plotting cost. Each method saves its PNG and coordinates/labels/sample IDs NPZ. Isomap and Spectral Embedding are optional entries in that cell; graph-neighbor and method settings are directly editable there. Projection structure alone does not prove domain invariance.

The manifest's `recordings` table and `splits` pairs preserve exact subset membership and order; `split_index_columns` defines the pair fields. Summaries report file/segment counts. `best.pt` supports inference with matching model code, not optimizer-state resume. Caches and outputs are excluded from Git.

## Weighted CP using unlabeled target inputs

Enable `calibration.weighted_cp` in baseline to estimate target/source density ratios
on frozen clean task or background features. Weight estimation always uses the current
test inputs, without their task labels. There is no separate domain reservation or
reference selector. Turning weighting on/off does not change the four data splits.

`weighting.features` selects `task` or `background`; the latter requires a trained
background branch. `weighting.method` selects `density_ratio` (source/target logistic
classifier) or `domain_mixture` (target mixture proportions over source domains).
Mixture mode needs at least two supported source domains.

Source training sample weights match calibration domain/class proportions. This uses
source/calibration labels and counts, not target labels or calibration features to fit
the ratio estimator. Every calibration stratum must occur in training. This matching
does not correct within-stratum distribution shift. Logistic ratios include the
source/target sample-prior correction. Optional clipping changes the estimated ratio.

With nonconformity score s (larger is worse), candidate-label p-values are
`(w_test + sum(w_i * (s_i >= s_test))) / (w_test + sum(w_i))`.
Sums use all calibration samples. Ties are included, sets use `p > alpha`, and unit weights recover
ordinary CP. Reusing target inputs for estimated weights and prediction is empirical
transductive adaptation; it does not establish exact target coverage.

Weighted runs save `target_features.npz` without task labels, `calibration_weights.npz`,
`test_weights.npz`, and `weighting.json` with ESS, clipping, convergence and target
counts. The CP bundle includes scorer, calibrator and ratio estimator. Failed notebook
reruns invalidate downstream results before any work begins.

Evaluation requires current-format saved configuration and the exact four-way split
manifest. Missing run settings, obsolete weighting selectors and extra legacy splits
are rejected rather than migrated. Existing result directories are not rewritten.


## Architecture and direct softmax CP switches

Set `model.kind` to `dual_cnn`, `cnn_transformer`, or `dual_link_graph`. Each new model lives in its matching `step02_models/*.py`;
`common.py` supplies shared convolution/residual blocks and paired-input forward flow.
It also owns `RestoreSTFT`, `conv_block` (Conv→GN→GELU), and residual blocks. Transformer construction and sinusoidal position helpers
live in `cnn_transformer.py`, alongside its temporal stem and STFT channel fusion.
`PairedModel` supplies only input validation and paired forward flow (including
DualCNN). Each model explicitly declares both encoders, fusion, embedding and
classifier; no parent creates layers that a subclass later replaces.
`build_model` constructs all architectures from `ModelConfig`. Minimum time lengths
are declared by the model and checked through `PairedModel` during construction
and forward calls. `AdversarialClassifier.condition_classes` describes the optional
conditioning labels for either adversarial head. `decoder.py` owns `PairedDecoder`;
`encoder_background.py` assembles the independent task/background models.
Single-layer classifiers are direct `Linear` modules: checkpoints now use
`classifier.weight`/`classifier.bias` (including the corresponding task/background
prefixes), without the former `.0` component. Old checkpoint keys are not migrated.
Convolution blocks use GN followed by GELU. DualCNN uses 32/64 convolution
channels without input normalization or branch dense layers. Each branch pools
to 256 values; the shared head is Linear(512→128)/GELU, Linear(128→embedding),
and the class head. Singleton training batches are retained. Earlier DualCNN
checkpoints require the previous architecture; retrain for this compact version.
All return the same task logits and embeddings, support the existing DA/CL losses,
and save the model kind in the checkpoint/manifest. `evaluate.ipynb` restores the
saved architecture; old manifests without a kind default to `dual_cnn`.
Changing architecture requires retraining; an old checkpoint cannot be reused.

| Model | Temporal inductive bias |
|---|---|
| `dual_cnn` | Two valid convolutions with early temporal downsampling |
| `cnn_transformer` | Local CNN, at most 64 ordered tokens, sinusoidal positions, two 4-head attention layers |
| `dual_link_graph` | Link-local convolutions and Tx/Rx node-edge message passing |

`cnn_transformer` restores the frequency-first STFT storage order and fuses
810 feature channels to 32 (GroupNorm/GELU), projects nonoverlapping 8×9
frequency/time patches to width 64, adds fixed two-axis sinusoidal positions,
and applies two 4-head Transformer layers (FFN 128, dropout 0.1). Right/bottom
zero padding after fusion makes partial patches complete: 64×65 becomes 64×72,
giving an 8×8 grid of 64 tokens. Other cache sizes use the same patch size and
therefore may produce different token counts. Its STFT encoder has 240,480
parameters; the temporal branch is unchanged. Background encoders use the same
architecture when enabled. Pre-patch `cnn_transformer` checkpoints require the
previous code; they cannot load into this new architecture.
Both branches retain four pooled positions (STFT: 2×2),
then use 512→128→embedding→class logits (CNNTransformer). DualLinkGraph instead
outputs 128 values per branch and uses a 256→128 fusion. All three models use
GroupNorm/GELU convolution blocks; Transformer and graph message-passing blocks
also use LayerNorm. Both adversarial heads use Linear→GELU→Linear without
normalization. Final embedding and classifier layers are linear. None of these
architectures guarantees domain invariance. Changing ReLU to GELU does not change
checkpoint tensor keys, but loading pre-change adversarial weights under GELU
changes their computation; use the matching code for historical reproduction.

A minimal override YAML for a new training run:

```yaml
model:
  kind: cnn_transformer      # or dual_cnn / dual_link_graph
score:
  kind: softmax
calibration:
  weighted_cp: false
```

Run `python run_experiment.py --config /path/to/override.yaml` from this directory.
The loader supplies other settings from baseline.yaml. When editing baseline.yaml
itself, replace the entire active score mapping with `score: {kind: softmax}`;
remove the HBGB/SVM/KDE-only parameters.

`softmax` uses `s(x,y)=1-softmax(logits(x))[y]` from the selected frozen task
classifier. There is no intermediate fitting step. The same score is applied to
validation, calibration, and test probabilities; ordinary finite-sample ranks
and the existing strict p-value threshold are unchanged. `weighted_cp` remains independently selectable. Saved scorers expose `input_key`:
pass probabilities for softmax and embeddings for the other scorers.

For an existing run, use the score/calibration portion of this YAML as
`evaluation_config_path` in `evaluate.ipynb`; no NN retraining is needed to change
only the score. Calibration/test labels never fit the score. Better representations
may reduce score shift, but alone do not prove exchangeability or target coverage.

## Further invariant-representation experiments (not implemented)

- Source-only class-conditional mean/covariance alignment: penalize statistical
  differences between training users within the same activity. This is a proposed
  adaptation of [CORAL](https://arxiv.org/abs/1607.01719), whose original formulation
  uses source and target data. Do not import target samples into this DG experiment.
  With small per-class/domain batches, covariance estimates need pooling/shrinkage.
- [DICA](https://arxiv.org/abs/1301.2115): kernel-based domain-invariant component
  analysis preserves input-output relationships while reducing domain differences;
  a statistical alternative to adversarial representation learning.
- [MixStyle](https://arxiv.org/abs/2104.02008): mix training-instance feature statistics
  to augment source-domain styles. Applying it to CSI is a hypothesis: statistics
  can also encode activity, so first test same-activity, different-user mixing.

Compare each separately on fixed splits against CE and the existing DA/CL setup.
Fit transforms and statistics on training data only, select with source validation,
and report unseen-user classification plus CP coverage/set size separately.


### Dual Link-Graph model

Select `model.kind: dual_link_graph` in your experiment YAML. Existing real-valued
`[B,3,270,T]` time and flattened STFT inputs are unchanged. Use the correct
`stft_frequency_bins` for your files (64 for the current crossroom profile).
No feature regeneration or change to the softmax CP scorer is required.

The MATLAB row order must be Tx, then Rx/pair, then 30 subcarriers. Amplitude
provides nine Tx–Rx nodes; relative phase and conjugate-product magnitude provide
nine edges, ordered 1->2, 1->3, 2->3 within each Tx. These are different physical
objects and are encoded separately. Time link encoders use shared local residual
convolutions. STFT encoders share small time-frequency convolutions across
subcarriers, then convolve along the subcarrier axis. Both retain eight temporal
positions and 64 channels per node/edge.

Each branch applies two edge-aware message-passing layers independently within
each Tx, with different source/target message functions. Ordered node/edge
concatenation maps 384->128 per Tx; ordered Tx concatenation maps 384->128.
A temporal residual block and pooling produce 128 values per branch. Concatenated
branches map 256->128->embedding, followed by the existing classifier. Time/STFT
positions are pooled independently, not assumed to be temporally aligned.

This model retains spatial structure longer than the early global compression
in the other architectures. It does not guarantee domain invariance. Per-link
encoding also uses more activation memory; lower training batch size if needed.

The time branch receives signed relative phase. Its magnitude STFT does not
preserve phase sign; ordered graph endpoints preserve pair identity, not recover
phase discarded by preprocessing. GroupNorm operates within each encoded link
(and within each subcarrier at the first STFT stage), so absolute link-energy
information may be attenuated. This is an architectural tradeoff to validate.


## Optional task/background decomposition

Configure all switches in `experiments/baseline.yaml`; `run.ipynb` uses it when
`config_path = None`. The `encoder_task` and optional `encoder_background` groups follow `model`,
and CP weighting follows `calibration`, keeping settings in pipeline order.
Set `encoder_background.enabled: true` to train the new architecture.
The selected `model.kind` remains the task backbone. `encoder_background.enabled: false`
builds only the task model and disables the paired decoder and both difference losses.
The new architecture must be trained before background evaluation is available.

```text
paired input -> task backbone -> z -> activity classifier
                              -> activity CL / reversed domain head
paired input -> background encoder -> b -> domain classifier / domain CL
                                       -> reversed activity head
[z, b] -> decoder -> standardized time and signed-log STFT reconstruction
z, b -> centered cross-correlation penalty
```

All source users share one background encoder. Task and background encoders use the
same `model.kind`, `model.embedding_dim`, and input processing, with separately
initialized parameters. There is no independent background dimension or model selector.
The background domain head mirrors the task classifier with a domain-sized output;
the reversed activity head uses its own `encoder_background.adversarial.hidden_dim`.
Both adversary widths default to 128 in baseline.yaml and can be changed independently.
The joint decoder receives `2 * model.embedding_dim` features.
Use a new training run for the current configuration schema.

The task objectives retain the
`encoder_task.contrastive` and `encoder_task.adversarial` YAML controls. Under `encoder_background`,
`domain`, `contrastive`, `adversarial`, and `difference` each have
an independent `enabled` switch. The background contrastive positives are same-domain,
different-activity pairs. The background activity head has detached fitting steps and
its own optimizer, learning rate, weight decay, optimizer kwargs and scheduler under
`encoder_background.adversarial`, using the same schema as the task adversary. It is frozen during
the reversed encoder update. The task encoder, background encoder, decoder, and
background domain head use the main optimizer/scheduler. Model selection remains
clean validation task CE. Losses and available anchor fractions are saved in history/W&B.

The decoder reconstructs the actual augmented model input, not raw CSI: standardized
time and signed-log raw STFT magnitude. Frequency-first MATLAB packing is preserved.
Low-resolution decoder seeds are upsampled to the original input shapes. Difference
loss discourages linear correlation; neither it nor adversarial accuracy proves independence.

`evaluate.ipynb` restores architecture settings from the saved manifest/checkpoint,
not from today's YAML, and never updates network parameters. The former head-training
diagnostic has been removed from evaluation. Its existing downstream scorer fitting
and conformal calibration remain available. Set `evaluation_config_path` to
`experiments/baseline.yaml` (or its absolute path) to apply its current evaluation
settings. For background target weighting, set `calibration.weighted_cp: true`,
`weighting.features: background`; weights always use unlabeled target inputs.
Changing baseline for evaluation cannot retrofit a background branch onto an old model.

`weighting` supports:
- `features: task | background` (background requires the saved branch).
- `method: density_ratio | domain_mixture` (logistic source/target ratio, or fitted
  mixture proportions over source domains using a source-domain classifier).

Target mode uses unlabeled test inputs to estimate weights; the same inputs are then
predicted, so this is empirical transductive adaptation, not an exact finite-sample
coverage claim. Task labels are only used for evaluation on target. Calibration labels
supply true-label scores and source/calibration stratum matching, not encoder training.
Estimated ratios, mixture assumptions, feature shift and clipping can affect validity.
Source weights match calibration domain/class proportions before ratio fitting.
Mixture mode records estimated target proportions; it cannot model backgrounds outside
the source component family. Calibration pools all true-label scores.

For controlled comparisons, restore the same checkpoint and change only evaluation
YAML: ordinary CP (`weighted_cp: false`), task weighting, background weighting, then
background mixture weighting. Exports include `background` arrays for both branches'
analysis, weight ESS/clipping diagnostics, and the unchanged NN accuracy line on plots.

Notebook reruns invalidate downstream scorer/calibration/prediction state before
starting work, so failed reruns cannot reuse an earlier CP result. Task and
background arrays are exported together in one ordered pass, with bounded
inference batches. Interactive runs also save `config.yaml`; existing experiment
manifests/checkpoints are protected against accidental overwrite. The old automatic
head-only scheduler experiment is no longer part of `run.ipynb`.

## Encoder objective groups

`encoder_task` groups `task`, `contrastive`, `adversarial`, and `difference`; `encoder_background`
groups `domain`, `contrastive`, `adversarial`, and `difference`. The separate `decoder`
group controls reconstruction and is effective only when background is enabled.
The common `model` section still selects both encoder architectures and dimensions.
The background master switch is independent of task CE.

`encoder_task.task.enabled` defaults to true and `weight` to 1. Disabling it or using
weight 0 removes CE from the training objective without removing the prediction head.
Other enabled objectives can still update the task encoder. Disabled CE supplies no
classifier gradients, so the optimizer does not apply momentum/decay to that head.
If no objective supplies gradients, the main optimizer skips the batch update.
`train_loss` and `validation_loss` remain raw task CE; `train_objective_loss` reports
the combined weighted training objective. Early stopping and checkpoint selection
continue to use clean, unweighted validation CE.

Manifests/checkpoints save `encoder_task`, `encoder_background`, and `decoder`;
the manifest also records the shared `data.domain_key`. Old top-level `seed`,
`contrastive`, `adversarial`, and `disentanglement` YAML keys are rejected. Use
`split_seed`/`training_seed` and the encoder groups. Evaluation requires the
current saved groups and never trains.

`encoder_background.adversarial.conditional` independently controls whether the
background activity adversary receives the true training-domain one-hot alongside b.
Both adversaries use `conditional: true` in baseline.yaml, independently configurable.
Gradient reversal affects b only. Both detached head fitting and reversed encoder
updates use the domain condition. Enabling it changes the adversarial head shape and
requires a new training run. Task prediction and density-ratio estimation still need
only input features. Read-only conditional activity diagnostics skip unseen domains
instead of inventing a condition, and report the evaluated sample count; accuracy
may remain high from domain/activity correlations alone.


The shared `data.domain_key` supplies training, contrastive, sampling, diagnostics and
weighting and holdout domain labels. `split.test_domains` contains values of this
shared key; there is no separate domain selector under `split`.

Each encoder has its own `difference.enabled` and `difference.weight`. The task loss
uses detached background features; the background loss uses detached task features.
Equal weights reproduce the previous joint penalty's gradient on each encoder;
the reported sum of the two loss values is twice the former scalar penalty.
Both losses are inactive without a background encoder.

`decoder.enabled` and `decoder.weight` control paired reconstruction, jointly updating
both encoders and the decoder through the main optimizer/scheduler. CL has no separate
projection head or optimizer; its encoder gradients also use the main optimizer.

When background is enabled, active background domain/activity heads report clean
validation loss, accuracy and sample count every epoch. Unknown domain labels are
excluded from domain-head and conditional activity-head validation. Unconditional
activity validation can use all validation samples. Background `plateau` scheduling
uses `validation_background_activity_loss`, skipping epochs without eligible samples;
cosine scheduling runs once per epoch. Both adversarial optimizer configurations are
independent and initially equal in baseline.yaml. Best checkpoint selection remains
based only on task validation CE.


After training, `run.ipynb` clears gradients and releases the three optimizers,
their schedulers, and retained training tensors before restoring the selected model.
Checkpoints are loaded on CPU; only model weights remain on the GPU for inference.
The same cleanup runs before export, including when the restoration cell was skipped.
Configuration, history, loaders and model architecture remain available. Recreate
training setup to train again; this cleanup does not provide optimizer-state resume.


`inference.batch_size` in baseline.yaml controls run/evaluate export microbatches
(default 4), including unlabeled target features for weighting. It does not change
training batches or exported sample order. Evaluation uses the current baseline's
inference budget, with optional `evaluation_config_path` overrides, rather than the
historical run's memory budget. Each new CP result records `inference.batch_size`.


## Shared notebook evaluation and APS

Both notebooks use `step01_preprocessing/dataset.py` for paired memory-mapped data.
`evaluation_pipeline.py` owns a `ConformalEvaluation` session with explicit
`export`, `fit_score`, `calibrate`, `predict`, and `save_results` stages. The notebook
keeps these stages visible. Recreate the session after changing its configuration;
rerunning an upstream stage invalidates downstream completion flags, including saved
results. Saving, plotting and logging reject changed post-calibration settings rather
than mixing old predictions with new metadata. NN training
remains in run.ipynb; evaluate.ipynb restores frozen weights only.

Each split is exported once per session export with `inference.batch_size`, including
the final partial batch. Weight estimation and prediction reuse cached target features.
Only source/calibration labels enter calibration or ratio fitting; target labels enter
metrics only. Exports occupy CPU RAM. The model smoke check also bounds its device
batch by `inference.batch_size`.

Randomized APS now runs in the regular evaluation, alongside the configured score.
Unweighted APS always runs; weighted APS reuses exactly the ordinary CP weights when
enabled. `aps_seed` defaults to 42 and is saved in config.json. Evaluation restores the saved
seed; an evaluation YAML can explicitly override it. Changing the seed requires
recalibration.
Independent calibration/test random streams provide one uniform draw per sample,
shared across candidate classes and held fixed across all alphas. No forced nonempty
sets are added. `metrics.json` includes the per-method alpha sweeps; `aps_predictions.npz`
saves scores, uniforms and p-values. `coverage_set_size.png` compares these methods
with nominal coverage and a horizontal NN accuracy line, using the former APS plot
style. This remains empirical evaluation under domain shift.

Plotting code lives directly in both notebooks for interactive styling. Embedding visualization
uses three cells: load/sample with the shared `projections.load_projection_data` helper,
compute/save coordinates, and plot/save images. Rerunning the plot cell reuses coordinates
without fitting t-SNE/MDS again. Optional W&B logging uses a shared evaluation helper. Logging initialization
or upload errors warn without discarding local results. Active background heads print
validation loss, accuracy and sample count after each training epoch.


Module boundaries: `evaluation_pipeline.py` coordinates the post-training stages;
`step07_evaluation/metrics.py` summarizes arrays, `artifacts.py` writes result files,
and `diagnostics.py` evaluates saved background heads. Weight fitting and weighted
calibration live in `step05_calibration/weighted.py`. Training calls
`step03_training/adversarial.py` explicitly before computing losses in `objectives.py`;
`validation.py` contains read-only background validation. Evaluation settings share
one parser; `alpha_grid` must be nonempty and contain values strictly between 0 and 1.
