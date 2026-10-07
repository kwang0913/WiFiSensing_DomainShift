"""Read completed evaluations and check that comparisons use the same target rows."""
from pathlib import Path
import json
import numpy as np


def resolve_result(path, runs_root):
    """Accept a run or an explicit conformal directory; prefer latest completed result."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(runs_root) / path
    path = path.resolve()
    if (path / "experiment.json").is_file():
        completed = sorted(p.parent for p in path.glob("conformal-*/metrics.json")
                           if (p.parent / "config.json").is_file())
        return path, completed[-1] if completed else None
    if (path / "metrics.json").is_file() and (path.parent / "experiment.json").is_file():
        return path.parent, path
    raise FileNotFoundError(f"Expected a run or completed conformal directory: {path}")


def load_result(directory):
    directory = Path(directory)
    metrics = json.loads((directory / "metrics.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    primary = config["score"]["kind"] + (" weighted CP" if config["calibration"]["weighted_cp"] else " CP")
    with np.load(directory / "predictions.npz", allow_pickle=False) as arrays:
        method_p_values = {primary: arrays["p_values"].copy()}
    with np.load(directory / "aps_predictions.npz", allow_pickle=False) as arrays:
        method_p_values["APS unweighted"] = arrays["unweighted_p_values"].copy()
        if config["calibration"]["weighted_cp"]:
            method_p_values["APS weighted"] = arrays["weighted_p_values"].copy()
    if set(method_p_values) != set(metrics["methods"]):
        raise ValueError(f"Saved methods and p-values differ: {directory}")
    with np.load(directory / "test_embeddings.npz", allow_pickle=False) as arrays:
        labels = arrays["labels"].copy()
        ids = list(zip(arrays["recording"].tolist(), arrays["segment_index"].tolist()))
        nn_accuracy = float(np.mean(arrays["probabilities"].argmax(1) == labels))
    classes = config["class_names"]
    if not len(labels) or len(ids) != len(labels):
        raise ValueError(f"Missing or misaligned target rows: {directory}")
    for method, p_values in method_p_values.items():
        if (p_values.shape != (len(labels), len(classes)) or not np.isfinite(p_values).all()
                or np.any((p_values < 0) | (p_values > 1))):
            raise ValueError(f"Invalid saved p-values for {method}: {directory}")
    return dict(directory=directory, config=config, method_results=metrics["methods"],
                history=json.loads((directory.parent / "history.json").read_text()),
                labels=labels, ids=ids, class_names=classes, method_p_values=method_p_values, nn_accuracy=nn_accuracy)


def validate_comparison(results):
    """Require identical target identities, order and labels, without hiding missing runs."""
    if not results:
        raise ValueError("Select at least one run")
    first = next(iter(results.values()))
    for name, result in results.items():
        if result["class_names"] != first["class_names"]:
            raise ValueError(f"{name}: class names/order differ")
        if result["ids"] != first["ids"] or not np.array_equal(result["labels"], first["labels"]):
            raise ValueError(f"{name}: target samples/order/labels differ; use matching splits")
