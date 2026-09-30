# WiFi Conformal Prediction

Study conformal prediction under domain shift using WiFi CSI segments.

## Workflow

1. **Prepare data.** Extract aligned, filtered CSI segments and STFT features with MATLAB, then convert MAT features to memory-mapped NPY files in the notebook.
2. **Train and select.** Hold out an entire domain for test, then randomly split source segments into training, validation, and calibration sets. Train the CNN and choose model/scorer settings using validation. YAML controls experiments; W&B records metrics and gradients.
3. **Calibrate and evaluate.** Fit the scorer on training embeddings, calibrate on the separate calibration set, and report test accuracy, prediction-set coverage, and set size.

## Run an experiment

Open [python/run.ipynb](python/run.ipynb) and run the cells in order, or execute a configured run from the project root:

```bash
python python/run_experiment.py --config python/configs/baseline.yaml
```

See the [Python guide](python/README.md) for installation, split design, kernel and augmentation choices, and saved results.

## Project contents

| Directory | Contents |
|---|---|
| [data/](data/README.md) | Dataset organization, recordings, and feature formats |
| [matlab/](matlab/README.md) | CSI preprocessing and feature extraction |
| [python/](python/README.md) | Notebook, models, experiment configuration, and evaluation |

Datasets and generated experiment files are excluded from Git. Prepare the data locally before running.
