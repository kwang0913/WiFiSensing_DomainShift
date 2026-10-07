"""Read and sample saved embeddings with aligned labels and sample identifiers."""
from pathlib import Path
import json
import numpy as np


def load_projection_data(viz_dir, viz_seed=42, viz_max_per_split=1000):
    viz_dir = Path(viz_dir)
    viz_manifest = json.loads((viz_dir.parent / "experiment.json").read_text())
    viz_domain_key = viz_manifest["data"]["domain_key"]
    held_out_domains = (viz_manifest["split"]["test_domains"]
                        if viz_manifest["split"]["mode"] == "domain_holdout" else [])
    viz_rng = np.random.default_rng(viz_seed)
    viz_embeddings, viz_labels, viz_domains, viz_splits = [], [], [], []
    viz_recordings, viz_segments = [], []
    for split in ("train", "validation", "calibration", "test"):
        with np.load(viz_dir / f"{split}_embeddings.npz") as arrays:
            selected = np.arange(len(arrays["labels"]))
            if viz_max_per_split is not None and len(selected) > viz_max_per_split:
                selected = np.sort(viz_rng.choice(selected, viz_max_per_split, replace=False))
            # Saved split pairs have the same row order as the embedding exports.
            pairs = np.asarray(viz_manifest["splits"][split])[selected]
            viz_embeddings.append(arrays["embeddings"][selected])
            viz_labels.extend(viz_manifest["class_names"][label] for label in arrays["labels"][selected])
            viz_domains.extend(viz_manifest["recordings"][i]["metadata"][viz_domain_key]
                                for i, j in pairs)
            viz_splits.extend([split] * len(selected))
            viz_recordings.extend(arrays["recording"][selected])
            viz_segments.extend(arrays["segment_index"][selected])

    viz_embeddings = np.concatenate(viz_embeddings)
    viz_labels, viz_domains, viz_splits = map(np.asarray, (viz_labels, viz_domains, viz_splits))

    return dict(directory=viz_dir, embeddings=viz_embeddings, labels=viz_labels,
                domains=viz_domains, splits=viz_splits, recording=np.asarray(viz_recordings),
                segment_index=np.asarray(viz_segments), task=viz_manifest["task"],
                domain_key=viz_domain_key, held_out_domains=held_out_domains,
                seed=viz_seed, max_per_split=viz_max_per_split)


def load_comparison_projections(directories, viz_seed=42, viz_max_per_split=1000):
    """Load identical sampled rows and metadata for independently fitted embeddings."""
    if not directories:
        raise ValueError("Select at least one run")
    data = {name: load_projection_data(directory, viz_seed, viz_max_per_split)
            for name, directory in directories.items()}
    first = next(iter(data.values()))
    for name, item in data.items():
        for key in ("recording", "segment_index", "labels", "domains", "splits",
                    "task", "domain_key", "held_out_domains"):
            if not np.array_equal(item[key], first[key]):
                raise ValueError(f"{name}: projection {key} differ; use matching split manifests")
    return data
