"""Execute one configured experiment using the training code in run.ipynb."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nbformat
import papermill
import yaml

from step00_config.loader import BASELINE, load_config

PYTHON_ROOT = Path(__file__).resolve().parent


def run_experiment(config_path=BASELINE, output_dir=None, kernel="wifi-cp"):
    config = load_config(config_path)
    notebook = nbformat.read(PYTHON_ROOT / "run.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir).expanduser().resolve() if output_dir else PYTHON_ROOT / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    saved_config = run_dir / "config.yaml"
    saved_config.write_text(yaml.safe_dump(config, sort_keys=False))
    # Execute a copy: the source notebook is never overwritten.
    notebook_path = run_dir / "run.ipynb"
    nbformat.write(notebook, notebook_path)
    status = {"status": "running", "kernel": kernel,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "source_config": str(Path(config_path).expanduser().resolve())}
    status_path = run_dir / "execution.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(f"Experiment: {run_dir}", flush=True)
    try:
        papermill.execute_notebook(
            str(notebook_path), str(notebook_path),
            parameters={"config_path": str(saved_config), "experiment_dir": str(run_dir),
                        "project_root": str(PYTHON_ROOT.parent)},
            kernel_name=kernel, cwd=str(PYTHON_ROOT), progress_bar=False,
            stdout_file=sys.stdout, stderr_file=sys.stderr,
        )
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error))
        raise
    else:
        status["status"] = "completed"
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))
    print(f"Completed: {notebook_path}", flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASELINE, help="Experiment YAML file")
    parser.add_argument("--output-dir", type=Path, help="New experiment directory; must not exist")
    parser.add_argument("--kernel", default="wifi-cp", help="Installed Jupyter kernel name")
    args = parser.parse_args()
    run_experiment(args.config, args.output_dir, args.kernel)
