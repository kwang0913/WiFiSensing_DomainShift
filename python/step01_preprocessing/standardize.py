"""Standardize each time-domain feature group within each segment."""
import numpy as np


# STFT preprocessing candidates for future validation experiments:
# - Baseline: keep the stored magnitude, preserving its linear amplitude scale.
# - Per-segment standardization or RMS normalization, separately for each feature
#   group: emphasizes spectral patterns but removes offset/overall scale cues.
# - Training-set mean/std scaling per feature group: retains between-segment
#   differences. Fit on training only and freeze for validation/calibration/test.
# - Dynamic-range compression: log1p(S / S_ref) or 20*log10(max(S, eps) / S_ref)
#   for amplitude magnitude. Choose a meaningful reference and floor separately
#   for each feature group; raw log1p(S) implicitly assumes a reference scale.
#   Compression changes amplitude ratios and emphasizes weak components; it is
#   optional, not inherently better. Power spectra use 10*log10 instead of 20.
# - Robust scaling or clipping if outliers dominate: fit thresholds on training
#   only; clipping discards peak information and should be justified by results.
# Keep the three physical feature groups separate. For frequency-wise processing,
# first undo the MATLAB frequency-first STFT flattening; the flattened axis is
# not a pure frequency axis. Select preprocessing using validation, then freeze
# it before calibration and final test evaluation. None of these STFT transforms
# is currently applied, and the NPY cache retains the original magnitudes.


def standardize_time(x):
    """Input [B,270,T,3]; statistics use axes (1,2), independently for each group.

    Each group uses 270*T values. The standard-deviation floor protects nearly
    constant groups. No statistics are fitted across recordings or splits.
    STFT magnitude is not transformed here.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError("Expected [batch,channels,time,features]")
    # Accumulate statistics in float64; keep model inputs in float32.
    mean = x.mean(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x.std(axis=(1, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    floor = np.float32(1.0 / np.sqrt(x.shape[1] * x.shape[2]))
    return (x - mean) / np.maximum(std, floor)
