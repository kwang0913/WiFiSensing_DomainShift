"""Finite-sample rank calibration, separate from fitting score models."""
import numpy as np

class RankCalibrator:
    def __init__(self, class_conditional=False):
        self.class_conditional = class_conditional

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
        self.references = []
        for label in range(self.classes):
            values = truth[labels == label] if self.class_conditional else truth
            if not len(values):
                raise ValueError("Each class needs calibration examples for class-conditional calibration")
            self.references.append(np.sort(values))
        return self

    def p_values(self, scores):
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != self.classes or np.isnan(scores).any():
            raise ValueError("Invalid candidate score matrix")
        result = np.empty_like(scores, dtype=float)
        for label, reference in enumerate(self.references):
            # Include ties; higher nonconformity means a smaller p-value.
            greater_equal = len(reference) - np.searchsorted(reference, scores[:, label], side="left")
            result[:, label] = (1 + greater_equal) / (len(reference) + 1)
        return result
