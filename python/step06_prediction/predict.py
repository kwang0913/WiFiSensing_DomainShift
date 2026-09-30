"""Convert calibrated p-values into prediction sets."""
import numpy as np

def prediction_sets(p_values, alpha=0.1):
    values = np.asarray(p_values)
    if not 0 < alpha < 1 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected valid alpha and a finite [N,K] p-value matrix")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("P-values must be between zero and one")
    return values > alpha
