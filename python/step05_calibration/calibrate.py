"""Finite-sample rank calibration, separate from fitting score models."""
import numpy as np

class RankCalibrator:
    def fit(self, scores, labels):
        scores, labels = np.asarray(scores), np.asarray(labels)
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Calibration labels must be a one-dimensional integer array")
        if scores.ndim != 2 or scores.shape[1] < 1 or len(scores) != len(labels) or not len(labels):
            raise ValueError("Expected nonempty calibration scores [N,K] and labels [N]")
        if np.isnan(scores).any() or (labels < 0).any() or (labels >= scores.shape[1]).any():
            raise ValueError("Invalid calibration scores or labels")
        self.classes = scores.shape[1]
        truth = scores[np.arange(len(labels)), labels]
        self.reference = np.sort(truth)
        return self

    def p_values(self, scores):
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != self.classes or np.isnan(scores).any():
            raise ValueError("Invalid candidate score matrix")
        # Include ties; higher nonconformity means a smaller p-value.
        greater_equal = len(self.reference) - np.searchsorted(self.reference, scores, side="left")
        return (1 + greater_equal) / (len(self.reference) + 1)


class WeightedRankCalibrator(RankCalibrator):
    """Weighted ranks with the candidate's own mass, including score ties.

    Estimated weights fitted on reused target inputs do not in general
    establish an exact target coverage guarantee.
    """
    @staticmethod
    def _weights(weights, count):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (count,) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Expected one finite positive weight per sample")
        if not np.isfinite(weights.sum()):
            raise ValueError("Weight sum overflow")
        return weights

    def fit(self, scores, labels, weights):
        super().fit(scores, labels)
        scores, labels = np.asarray(scores), np.asarray(labels)
        weights = self._weights(weights, len(labels))
        truth = scores[np.arange(len(labels)), labels]
        order = np.argsort(truth, kind="stable")
        sorted_weights = weights[order]
        self.tail_weights = np.r_[np.cumsum(sorted_weights[::-1])[::-1], 0.0]
        return self

    def p_values(self, scores, weights):
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != self.classes or np.isnan(scores).any():
            raise ValueError("Invalid candidate score matrix")
        weights = self._weights(weights, len(scores))
        index = np.searchsorted(self.reference, scores, side="left")
        tail = self.tail_weights
        weights = weights[:, None]
        # Rescale before adding to avoid overflow for large candidate weights.
        scale = np.maximum(weights, tail[0])
        return (weights / scale + tail[index] / scale) / (weights / scale + tail[0] / scale)
