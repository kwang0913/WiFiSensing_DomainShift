"""Report coverage over ALL test samples, including empty prediction sets."""
import numpy as np


def evaluate(labels, sets, point_predictions):
    labels = np.asarray(labels)
    sets = np.asarray(sets, dtype=bool)
    point_predictions = np.asarray(point_predictions)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Evaluation labels must be a one-dimensional integer array")
    if not len(labels) or sets.ndim != 2 or len(sets) != len(labels) or point_predictions.shape != labels.shape:
        raise ValueError("Incompatible evaluation arrays")
    if ((labels < 0) | (labels >= sets.shape[1])).any():
        raise ValueError("Evaluation labels are outside the class range")
    if not np.issubdtype(point_predictions.dtype, np.integer) or ((point_predictions < 0) | (point_predictions >= sets.shape[1])).any():
        raise ValueError("Point predictions must be integer class indices")
    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    return {
        "samples": int(len(labels)),
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "empty_fraction": float((sizes == 0).mean()),
        "singleton_fraction": float((sizes == 1).mean()),
        "multiple_fraction": float((sizes > 1).mean()),
        "classification_accuracy": float((point_predictions == labels).mean()),
        "coverage_given_nonempty": float(covered[sizes > 0].mean()) if (sizes > 0).any() else None,
    }
