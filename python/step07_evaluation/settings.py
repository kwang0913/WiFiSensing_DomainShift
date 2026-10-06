"""Restore current-format evaluation settings without changing training metadata."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import yaml
from step00_config.loader import BASELINE
from step00_config.evaluation import SCORE_CONFIGS, parse_evaluation_settings

SECTIONS = ("score", "calibration", "weighting", "alpha_grid", "inference")


def restore_evaluation_settings(run_dir, saved_dir=None, override_path=None):
    """Saved CP settings take priority over the saved run YAML; no legacy migration."""
    path = Path(saved_dir) / "config.json" if saved_dir is not None else Path(run_dir) / "config.yaml"
    saved = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    config = {key: deepcopy(saved[key]) for key in SECTIONS}
    config["aps_seed"] = saved.get("aps_seed", 42)
    # Memory budget is deliberately independent of historical training settings.
    config["inference"] = yaml.safe_load(BASELINE.read_text())["inference"]
    source = str(path)
    if override_path is not None:
        patch = yaml.safe_load(Path(override_path).expanduser().read_text())
        if not isinstance(patch, dict):
            raise ValueError("Evaluation YAML must contain a mapping")
        for key in (*SECTIONS, "aps_seed"):
            if key not in patch:
                continue
            value = deepcopy(patch[key])
            if key in ("alpha_grid", "aps_seed"):
                config[key] = value
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            if key == "score" and value.get("kind", config[key]["kind"]) != config[key]["kind"]:
                if value["kind"] not in SCORE_CONFIGS:
                    raise ValueError(f"Unknown score.kind: {value['kind']}")
                config[key] = asdict(SCORE_CONFIGS[value["kind"]]())
            config[key].update(value)
        source += f"; explicit evaluation overrides: {override_path}"
    parse_evaluation_settings(config)
    if isinstance(config["aps_seed"], bool) or not isinstance(config["aps_seed"], int) or config["aps_seed"] < 0:
        raise ValueError("aps_seed must be a nonnegative integer")
    return config, source
