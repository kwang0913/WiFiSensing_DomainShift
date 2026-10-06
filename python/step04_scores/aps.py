"""Randomized adaptive prediction-set scores from frozen NN probabilities."""
import numpy as np


def randomized_aps_scores(probabilities, rng):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] < 2
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=1e-5)):
        raise ValueError("APS requires finite, nonnegative class probabilities summing to one")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked = np.take_along_axis(probabilities, order, axis=1)
    uniform = rng.random((len(probabilities), 1))
    cumulative_before = np.cumsum(ranked, axis=1) - ranked
    ranked_scores = cumulative_before + uniform * ranked
    scores = np.empty_like(ranked_scores)
    np.put_along_axis(scores, order, ranked_scores, axis=1)
    return scores, uniform[:, 0]
