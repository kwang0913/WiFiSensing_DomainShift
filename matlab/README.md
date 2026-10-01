# MATLAB feature extraction

Decode Intel CSI recordings, align packets, and extract paired time-domain/STFT segments. Source DAT files are available for crossroom; dataset labels and array formats are in the [data guide](../data/README.md).

## Setup and run

Requires MATLAB and Signal Processing Toolbox. Included MEX decoders were verified on Linux x86-64 with MATLAB R2024b; other platforms need compatible binaries.

From the repository root:

```matlab
addpath(fullfile(pwd, 'matlab'));
step00_main_raw_seg_extraction_new  % crossroom
```

| Entry point | Input | Output |
|---|---|---|
| `step00_main_raw_seg_extraction_new.m` | `data/crossroom/recordings/` | `data/crossroom/generated_features/` |
| `step00_main_raw_seg_extraction.m` | `data/<dataset>/recordings/<subset>/` | `data/<dataset>/generated_features/<subset>/` |

For crossroom, `name = ''` processes all participant directories. The general entry point supports self_time, setting_dataset, and appleman_0713 but requires their source recordings, which are not included.

Keep `read_from_file = true` for timestamp alignment. Existing outputs are skipped by default; set `config.overwrite = true` after loading the profile to regenerate them. Configuration mismatches warn instead of silently replacing files. Distinct recordings need unique names or `_rNN` suffixes.

## Parameters

Defaults are defined in [step00_processing_profile.m](step00_processing_profile.m).

| Parameter | self_time / setting_dataset / appleman_0713 | crossroom |
|---|---:|---:|
| Aligned sample rate | 1000 Hz | 1000 Hz |
| Tx × Rx | 3 × 3 | 3 × 3 |
| Feature bandpass | 0.8–200 Hz | 0.8–200 Hz |
| Total time-domain segment length | 1.2 s / 1200 samples | 4 s / 4000 samples |
| STFT window length | 250 samples | 125 samples |
| STFT overlap | 125 samples | 63 samples |
| NFFT | 1000 | 1000 |
| Retained frequencies | First 32 bins, 0–31 Hz | First 64 bins, 0–63 Hz |
| Frames on each side of the peak | 10 | 32 |
| Total STFT frames | 21 | 65 |
| Normalized peak height | 0.1 | 0.4 |
| Minimum peak distance | 2.8 s | 3 s |
| Minimum peak prominence | 0.01 | 0.01 |
| First retained sample index / samples trimmed from end | 4000 / 8000 | 100 / 4000 |

Gaps longer than 10 ms split recordings into continuous blocks; segments never cross those boundaries. Four-second segments with a three-second minimum peak distance may overlap.

## Processing

1. Decode and scale all nine links using a common valid-packet mask. Handle timestamp wraparound, duplicates, resets, and long gaps.
2. Construct amplitude, relative phase, and conjugate-product magnitude before interpolation onto the 1000 Hz grid; apply feature filtering.
3. Detect peaks using the first 90 amplitude rows and extract paired time/STFT windows. Save normalized filenames and source metadata.

Relative features use Rx pairs `(1,2)`, `(1,3)`, `(2,3)` within each Tx. Detection uses a separate 2.5–100 Hz bandpass and a normalized STFT-energy curve, so `peak_height` is relative to that curve's maximum.

## Outputs and diagnostics

Each `*_raw.mat` contains `segment_data`, `segment_data_stft`, and `extraction_info`. The last records configuration, labels, source, packet alignment, and segment positions. Empty extractions are not saved. Interpret segment positions together with block IDs and timestamp-reset epochs.

STFT magnitudes are cropped from the continuous spectrogram and flattened with MATLAB `piece(:)`, frequency first. The 21/65-frame windows span approximately 2.75/4.093 seconds, respectively. See the [data guide](../data/README.md) for shapes and reshaping.

Malformed or truncated DAT tails retain complete preceding packets; inspect warnings and decoding metadata. Blocks without usable signal produce no segments. `Data Analysis/` contains optional scripts with their own paths and dependencies.
