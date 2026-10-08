"""Four-way splits of lightweight indices; signal arrays remain on disk."""
import numpy as np
from sklearn.model_selection import train_test_split
from step00_config.schema import SplitConfig


def conditional_partition(samples, labels, domains, validation_fraction, calibration_fraction, seed):
    """Allocate each domain/class stratum independently, retaining at least one train sample."""
    rng = np.random.default_rng(seed)
    groups = {}
    for row, (recording, _) in enumerate(samples):
        groups.setdefault((domains[recording], labels[recording]), []).append(row)
    partitions = {key: [] for key in ("train", "validation", "calibration")}
    for key in sorted(groups):
        indices = rng.permutation(groups[key])
        n_val = max(1, int(len(indices) * validation_fraction)) if validation_fraction else 0
        n_cal = max(1, int(len(indices) * calibration_fraction))
        if n_val + n_cal >= len(indices):
            raise ValueError(f"Source domain/class {key!r} has too few segments for the requested splits")
        partitions["validation"].extend(indices[:n_val])
        partitions["calibration"].extend(indices[n_val:n_val + n_cal])
        partitions["train"].extend(indices[n_val + n_cal:])
    return {name: samples[np.asarray(rows, dtype=int)] for name, rows in partitions.items()}


def split_samples(labels, counts, metadata, data_config, config, seed):
    """Isolate specified domains, or retain the existing recording-level random split."""
    config = SplitConfig(**config)
    indices = np.arange(len(labels))
    if config.mode == "random":
        n_classes = len(set(labels))
        n_val, n_cal, n_test = [max(n_classes, int(len(indices) * f)) for f in
                               (config.validation_fraction, config.calibration_fraction, config.test_fraction)]
        train, remaining = train_test_split(indices, test_size=n_val + n_cal + n_test,
                                            stratify=labels, random_state=seed)
        validation, remaining = train_test_split(remaining, train_size=n_val,
                                                 stratify=labels[remaining], random_state=seed)
        calibration, test = train_test_split(remaining, test_size=n_test,
                                             stratify=labels[remaining], random_state=seed)
        return {name: np.asarray([(i, j) for i in records for j in range(counts[i])], dtype=int)
                for name, records in dict(train=train, validation=validation, calibration=calibration, test=test).items()}
    key = data_config["domain_key"]
    if key == data_config["task"]:
        raise ValueError("Domain must differ from the prediction task; held-out classes are not supported")
    if not all(key in record for record in metadata):
        raise ValueError(f"Filename metadata does not contain domain field: {key}")
    domains = np.asarray([record[key] for record in metadata])
    for name, selected in (("Test", config.test_domains), ("Validation", config.validation_domains)):
        missing = set(selected) - set(domains)
        if missing:
            raise ValueError(f"{name} domains not found: {sorted(missing)}")
    test_mask = np.isin(domains, config.test_domains)
    validation_mask = np.isin(domains, config.validation_domains)
    source_mask = ~(test_mask | validation_mask)
    if not source_mask.any():
        raise ValueError("The held-out domains leave no source recordings")
    samples = np.asarray([(i, j) for i, count in enumerate(counts) for j in range(count)], dtype=int)
    result = conditional_partition(samples[source_mask[samples[:, 0]]], labels, domains,
                                   0 if config.validation_domains else config.validation_fraction,
                                   config.calibration_fraction, seed)
    if config.validation_domains:
        result["validation"] = samples[validation_mask[samples[:, 0]]]
    result["test"] = samples[test_mask[samples[:, 0]]]
    return result
