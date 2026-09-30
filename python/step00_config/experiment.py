"""Load one YAML experiment, using baseline.yaml for omitted settings."""
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import KDEConfig, SVMConfig, HBGBConfig

BASELINE = Path(__file__).resolve().parents[1] / "configs/baseline.yaml"
SCORE_CONFIGS = {"kde": KDEConfig, "svm": SVMConfig, "hbgb": HBGBConfig}


def _merge(base, overrides, section="experiment"):
    """Merge structured settings; optimizer/scheduler kwargs are replaced as a unit."""
    unknown = overrides.keys() - base.keys()
    if unknown:
        raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    for key, value in overrides.items():
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{section}.{key} must be a mapping")
            if key in ("kwargs", "optimizer_kwargs"):
                base[key] = value
            else:
                _merge(base[key], value, f"{section}.{key}")
        else:
            base[key] = value


def load_config(path=None):
    config = yaml.safe_load(BASELINE.read_text())
    overrides = yaml.safe_load(Path(path).expanduser().read_text()) if path else {}
    if not isinstance(overrides, dict):
        raise ValueError("Experiment YAML must contain a mapping of settings")
    # Older experiment files used one seed for all randomness.
    if "seed" in overrides:
        old_seed = overrides.pop("seed")
        overrides.setdefault("split_seed", old_seed)
        overrides.setdefault("training_seed", old_seed)
    score = overrides.get("score", {})
    if isinstance(score, dict) and score.get("kind", config["score"]["kind"]) != config["score"]["kind"]:
        config["score"] = asdict(SCORE_CONFIGS[score["kind"]]())
    # Changing algorithms must not inherit parameters belonging to another one.
    for section, selector, parameters in (("training", "optimizer", "optimizer_kwargs"),
                                          ("scheduler", "name", "kwargs")):
        update = overrides.get(section, {})
        if isinstance(update, dict) and update.get(selector, config[section][selector]) != config[section][selector]:
            config[section][parameters] = {}
    _merge(config, overrides)
    if config["stage"] not in ("train", "full"):
        raise ValueError("stage must be train or full")
    datasets = config["data"]["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a nonempty list of cache directories")
    return config
