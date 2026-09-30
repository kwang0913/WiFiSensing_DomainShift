"""Optional W&B logging and sampled gradient diagnostics, independent of the model.

Parameter and activation gradients share one training-step sampling schedule.
Hooks keep only scalar statistics, never activation tensors or computation graphs.
No W&B import or login occurs when monitoring is disabled.
"""
from fnmatch import fnmatchcase
from contextlib import contextmanager
from pathlib import Path
import warnings
import torch
from torch import nn


class TrainingMonitor:
    def __init__(self, model, config, *, run_dir, experiment=None):
        defaults = dict(enabled=True, project="wifi-conformal", mode="offline",
                        entity=None, name=None, gradient_interval=100,
                        parameter_histograms=True, activation_gradients=True,
                        layers=None)
        unknown = set(config) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown monitor settings: {sorted(unknown)}")
        self.config = defaults | config
        interval = self.config["gradient_interval"]
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            raise ValueError("gradient_interval must be a positive integer")
        if self.config["mode"] not in ("online", "offline"):
            raise ValueError("mode must be online or offline; use enabled=False to disable")
        self.model, self.run_dir = model, Path(run_dir)
        self.experiment = experiment or {}
        self.run = self._wandb = None
        self._hooks, self._tensor_hooks = [], []
        self._pending = {}
        self._collect = False
        self._step = 0
        self._entered = False
        self._warned = False
        self.layer_names = []

    def _layers(self):
        modules = dict(self.model.named_modules())
        patterns = self.config["layers"]
        if patterns is None:
            return [(name, layer) for name, layer in modules.items() if name
                    and isinstance(layer, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d))]
        if isinstance(patterns, str):
            raise ValueError("layers must be a list of module-name patterns or None")
        for pattern in patterns:
            if not any(name and fnmatchcase(name, pattern) for name in modules):
                raise ValueError(f"No model layer matches {pattern!r}")
        return [(name, layer) for name, layer in modules.items() if name
                and any(fnmatchcase(name, pattern) for pattern in patterns)]

    def __enter__(self):
        if self._entered:
            raise RuntimeError("This monitor is already active")
        self._entered = True
        self._warned = False
        if not self.config["enabled"]:
            return self
        try:
            import wandb
            self._wandb = wandb
            if wandb.run is not None:
                raise RuntimeError("An existing W&B run is active; finish it before starting this monitor")
            layers = self._layers() if self.config["activation_gradients"] else []
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                project=self.config["project"], entity=self.config["entity"],
                name=self.config["name"] or self.run_dir.name,
                group=self.run_dir.name, job_type="training",
                mode=self.config["mode"], dir=str(self.run_dir),
                settings={"quiet": True},  # Hide W&B's end-of-run history/summary tables.
                config={**self.experiment, "monitoring": self.config},
            )
            self.run.define_metric("train/step")
            for pattern in ("train/*", "gradients/*", "parameters/*"):
                self.run.define_metric(pattern, step_metric="train/step")
            self.run.define_metric("epoch")
            self.run.define_metric("epoch/*", step_metric="epoch")
            for name, layer in layers:
                self._hooks.append(layer.register_forward_hook(self._forward_hook(name)))
            self.layer_names = [name for name, _ in layers]
            self.run.summary["monitored_activation_layers"] = self.layer_names
            return self
        except BaseException:
            self.close(exit_code=1)
            raise

    def _warn(self, error):
        if not self._warned:
            self._warned = True
            warnings.warn(f"W&B monitoring stopped; local training/results are unaffected: {error}",
                          RuntimeWarning, stacklevel=3)

    @contextmanager
    def _logging(self):
        # Catch only diagnostic SDK calls, never forward/backward or file saves.
        try:
            yield
        except Exception as error:
            try:
                self._warn(error)
            finally:
                self.close(exit_code=1)

    @staticmethod
    def _stats(gradient):
        g = gradient.detach().float()
        finite = torch.isfinite(g)
        stats = {"nonfinite_fraction": (~finite).float().mean().item()}
        if not finite.all():
            g = g[finite]
        if g.numel():
            maximum = g.abs().max().item()
            # Scale before squaring to avoid overflow for large finite gradients.
            rms = (g / maximum).square().mean().sqrt().item() * maximum if maximum else 0.0
            stats.update(rms=rms, max_abs=maximum,
                         zero_fraction=(g == 0).float().mean().item())
        return stats

    def _forward_hook(self, name):
        def forward_hook(module, inputs, output):
            if not self._collect or not module.training or not torch.is_grad_enabled():
                return
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                return
            def backward_hook(gradient):
                self._pending.update({f"gradients/activations/{name}/{key}": value
                                      for key, value in self._stats(gradient).items()})
                # Returning None leaves the original gradient untouched.
            self._tensor_hooks.append(output.register_hook(backward_hook))
        return forward_hook

    def _remove_tensor_hooks(self):
        for handle in self._tensor_hooks:
            handle.remove()
        self._tensor_hooks.clear()

    def begin_step(self, global_step):
        self._remove_tensor_hooks()
        self._pending.clear()
        self._step = global_step
        # Sample the first batch as well, so short debugging runs have diagnostics.
        self._collect = self.run is not None and (
            global_step == 1 or global_step % self.config["gradient_interval"] == 0)

    def after_backward(self, model):
        if not self._collect:
            return
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            self._pending.update({f"gradients/parameters/{name}/{key}": value
                                  for key, value in self._stats(parameter.grad).items()})
            if self.config["parameter_histograms"]:
                # Explicit histogram logging shares the activation schedule and avoids
                # watch() forward counters including validation batches.
                for prefix, value in (("parameters", parameter),
                                      ("gradients/parameters", parameter.grad)):
                    flat = value.detach().float().reshape(-1)
                    flat = flat[torch.isfinite(flat)]
                    if flat.numel():
                        values = flat.cpu().numpy()
                        with self._logging():
                            self._pending[f"{prefix}/{name}/histogram"] = self._wandb.Histogram(
                                values, num_bins=64)
                        if self.run is None:
                            return
        self._remove_tensor_hooks()

    def end_step(self, metrics):
        if self.run is not None:
            with self._logging():
                self.run.log({"train/step": self._step, **metrics, **self._pending})
        self._pending.clear()
        self._collect = False
        self._remove_tensor_hooks()

    def log_epoch(self, epoch, metrics):
        if self.run is not None:
            with self._logging():
                self.run.log({"epoch": epoch, **{f"epoch/{k}": v for k, v in metrics.items()}})

    def log_result(self, result):
        if self.run is not None:
            with self._logging():
                self.run.summary.update({k: v for k, v in result.items() if k != "history"})

    def close(self, exit_code=0):
        self._collect = False
        self._remove_tensor_hooks()
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._pending.clear()
        self._entered = False
        run, self.run = self.run, None
        if run is not None:
            try:
                run.finish(exit_code=exit_code)
            except Exception as error:
                self._warn(error)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(exit_code=0 if exc_type is None else 1)
        return False
