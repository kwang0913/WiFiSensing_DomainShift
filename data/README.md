# Data

Raw CSI recordings and extracted MAT features. Data and generated NPY caches are excluded from Git.

## Datasets and labels

The inventory covers feature files used by the notebook, with NPY labels verified on 2026-09-29. Counts are files, not segments. User identifiers below are anonymized for presentation; YAML domain values must match the actual, case-sensitive filename metadata.

| Dataset | Files | Users (anonymized) | `activity` labels | `position` labels |
|---|---:|---|---|---|
| setting_dataset | 72 | `user_1` ~ `user_6` | `baking`, `cooking`, `raisingarm`, `walk` (4) | `p1` |
| self_time | 150 | `user_1` ~ `user_5` | `doc`, `squat` (2) | `c1`, `c2`, `c3`, `c4`, `c5` |
| crossroom | 125 | `user_1` ~ `user_10` | `doc`, `sit`, `squat`, `walk`, `wipe` (5) | `103`, `501`, `503` |

Each listed user has all activities listed for that dataset, but recording counts and position coverage vary.

## File organization

| Dataset | Recordings and features | Collection metadata |
|---|---|---|
| setting_dataset | `environment1/`, `environment2/`, `environment3/`: 24 active feature files each | Environment is the parent directory; `p1` is an unspecified-position placeholder |
| self_time | `0310_morning/`, `0310_afternoon/`, `0310_night/`: 50 feature files each | Collection period is the parent directory; `c1`–`c5` are condition identifiers |
| crossroom | `recordings/<user>/`, `generated_features/<user>/` | Numeric position/room codes in filenames |

Source DAT recordings are available only for crossroom. Files use `<user>_<activity>_<position>[_rNN]`: `.dat` for recordings, `.mat` for CSI caches, and `_raw.mat` for extracted features. `_rNN` identifies repeated recordings. Physical room dimensions and antenna placements are not documented.

Two active environment1 feature files contain merged recordings; `merge_source_index` and `merge_info` retain their provenance. Canonical activity aliases include `document → doc`, `walking/walkingtraj2 → walk`, `sitting → sit`, and `wiping → wipe`.

## Feature format

The `_raw.mat` suffix means time-domain features, not raw complex CSI.

| Dataset | `segment_data` | `segment_data_stft` |
|---|---|---|
| self_time, setting_dataset | `[N,270,1200,3]` | `[N,270,672,3]` |
| crossroom | `[N,270,4000,3]` | `[N,270,4160,3]` |

- `N`: number of segments in one file.
- `270`: 3 Tx × 3 Rx × 30 subcarriers for amplitude; 3 Tx × 3 Rx pairs × 30 subcarriers for relative features. Rx pairs are `(1,2)`, `(1,3)`, `(2,3)`.
- Last dimension: dB amplitude, time-unwrapped relative phase, conjugate-product magnitude.
- Time width: 1200 or 4000 samples at 1000 Hz. STFT width: 32 bins × 21 frames or 64 bins × 65 frames, flattened with frequency varying fastest.

Time and STFT windows share a detection peak but have different temporal coverage. Filtered time features can be negative. Generated features include `extraction_info` with configuration and source/segment metadata; self_time/setting_dataset files generally lack it.

```matlab
s = load('path/to/feature_raw.mat');
% crossroom; use 32, 21 for self_time/setting_dataset.
S = reshape(s.segment_data_stft(1, 1, :, 1), 64, 65);
```

## Use in experiments

The notebook converts MAT features to float32 NPY pairs and loads segments through mmap. Datasets with different shapes cannot be directly batched together.

Domain holdout reserves complete target domains for test and optionally isolates `validation_domains` for validation. Remaining source **segments** are split into train/calibration, plus validation when no domains are isolated, using the same fractions within each domain/class. Source subsets may share files but never the same segment. The splitter uses filename fields, not parent-directory environment or collection-period names. Multiple held-out values are supported, such as `test_domains: [baking, walk]` for a user task with activity domains.

See the [Python guide](../python/README.md) for split design, its recording-dependence tradeoff, and saved indices, or the [MATLAB guide](../matlab/README.md) to extract features.
