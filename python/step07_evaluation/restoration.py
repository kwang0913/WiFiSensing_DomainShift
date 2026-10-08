"""Restore frozen runs for sequential reevaluation; never train or resplit."""
from pathlib import Path
from dataclasses import asdict
import gc
import json
import numpy as np
import torch
from step00_config.schema import (ModelConfig, AdversarialConfig,
                                 EncoderBackgroundConfig, DecoderConfig)
from step01_preprocessing.dataset import NpySegments
from step02_models.build import build_model
from .settings import restore_evaluation_settings


def restore_datasets(experiment, npy_root):
    time_shape, stft_shape = experiment["time_shape"], experiment["stft_shape"]
    npy_root = Path(npy_root)
    recordings = experiment["recordings"]
    files = [npy_root / r["time"] for r in recordings]
    stft_files = [npy_root / r["stft"] for r in recordings]
    targets = np.asarray([r["label"] for r in recordings], dtype=np.int64)
    # Restore exactly the saved four-way split.
    if set(experiment["splits"]) != {"train", "validation", "calibration", "test"}:
        raise ValueError("Expected current four-way split manifest")
    split_samples = {}
    seen = set()
    for name in ("train", "validation", "calibration", "test"):
        pairs = np.asarray(experiment["splits"][name])
        if not pairs.size:
            raise ValueError(f"Empty saved split: {name}")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError(f"Expected integer recording/segment pairs: {name}")
        if (pairs < 0).any() or (pairs[:, 0] >= len(recordings)).any():
            raise ValueError(f"Invalid saved indices: {name}")
        identities = set(map(tuple, pairs))
        if len(identities) != len(pairs) or seen & identities:
            raise ValueError(f"Duplicate or overlapping saved segments: {name}")
        seen.update(identities)
        split_samples[name] = pairs

    checked_recordings = set()
    for name, pairs in split_samples.items():
        for i in np.unique(pairs[:, 0]):
            if i not in checked_recordings:
                for path, shape in ((files[i], time_shape), (stft_files[i], stft_shape)):
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if list(array.shape) != [recordings[i]["segments"], *shape] or array.dtype != np.float32:
                        raise ValueError(f"Cache shape/dtype differs from saved run: {path}")
                    del array
                checked_recordings.add(i)
            if pairs[pairs[:, 0] == i, 1].max() >= recordings[i]["segments"]:
                raise ValueError(f"Segment outside saved recording: {name}, {i}")

    return {name: NpySegments(files, stft_files, targets, pairs) for name, pairs in split_samples.items()}


def architecture_adversarial(settings):
    """Frozen restoration needs head architecture, not historical optimizer settings."""
    return {key: settings[key] for key in ("enabled", "conditional", "hidden_dim") if key in settings}


def restore_model(experiment, checkpoint_path, device):
    model_config = ModelConfig(**experiment["model"])
    adversarial_config = AdversarialConfig(**architecture_adversarial(experiment["encoder_task"].get("adversarial", {})))
    background = experiment["encoder_background"]
    encoder_background_config = EncoderBackgroundConfig(**{
        **background, "adversarial": architecture_adversarial(background.get("adversarial", {}))})
    decoder_config = DecoderConfig(**experiment["decoder"])
    domain_names = experiment["domain_names"]
    time_shape, stft_shape = experiment["time_shape"], experiment["stft_shape"]
    class_names, training_seed = experiment["class_names"], experiment["training_seed"]
    selected_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_model = selected_checkpoint["model_config"]
    if asdict(ModelConfig(**checkpoint_model)) != asdict(model_config):
        raise ValueError("Checkpoint architecture differs from experiment manifest")
    checkpoint_decoder = DecoderConfig(**selected_checkpoint["decoder"])
    if selected_checkpoint["encoder_background"] != background:
        raise ValueError("Checkpoint background configuration differs from experiment manifest")
    if asdict(checkpoint_decoder) != asdict(decoder_config):
        raise ValueError("Checkpoint decoder differs from experiment manifest")
    model = build_model(model_config, time_shape, stft_shape, len(class_names), training_seed,
                        domain_classes=len(domain_names) if adversarial_config.enabled else 0,
                        domain_hidden_dim=adversarial_config.hidden_dim,
                        domain_conditional=adversarial_config.conditional,
                        encoder_background=encoder_background_config,
                        background_domains=len(domain_names), decoder=decoder_config)
    model.load_state_dict(selected_checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return model


def reevaluate_run(run_dir, npy_root, *, saved_dir=None, override_path=None,
                   checkpoint_name="best.pt", device_name="auto", log_to_wandb=False):
    """Save a new evaluation and release GPU model references before the next run."""
    from evaluation_pipeline import ConformalEvaluation
    from .monitoring import log_evaluation

    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("device_name must be auto, cpu or cuda")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if device_name == "auto" else device_name)
    run_dir = Path(run_dir)
    experiment = json.loads((run_dir / "experiment.json").read_text())
    config, source = restore_evaluation_settings(run_dir, saved_dir, override_path)
    checkpoint_path = run_dir / checkpoint_name
    model = evaluation = datasets = None
    try:
        datasets = restore_datasets(experiment, npy_root)
        model = restore_model(experiment, checkpoint_path, device)
        evaluation = ConformalEvaluation(model, datasets, experiment, config, checkpoint_path,
                                         settings_source=source, aps_seed=config["aps_seed"])
        evaluation.export()
        evaluation.fit_score()
        evaluation.calibrate()
        evaluation.predict()
        directory = evaluation.save_results()
        log_evaluation(evaluation, {**experiment.get("monitoring", {}), "enabled": log_to_wandb})
        return directory
    finally:
        # Also move parameters off GPU on exceptions, whose tracebacks may retain references.
        if model is not None:
            model.cpu()
        del evaluation, model, datasets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
