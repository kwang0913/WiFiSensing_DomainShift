"""Task patience gated by post-warmup stability of active DA/CL objectives."""
from collections import deque
from dataclasses import asdict
import math


class JointEarlyStopping:
    def __init__(self, training, adversarial, contrastive, background):
        self.settings = dict(training.early_stopping)
        self.patience = self.settings["patience"]
        self.min_delta = self.settings["min_delta"]
        self.warmup = 0
        self.monitors = []
        self.anchor_metrics = []
        branches = [("", asdict(adversarial), asdict(contrastive))]
        if background.enabled:
            branches.append(("background_", background.adversarial, background.contrastive))
        for prefix, da, cl in branches:
            if da["enabled"] and da["max_weight"] > 0:
                head = "activity" if prefix else "domain"
                self.monitors.extend([f"train_{prefix}{head}_loss", f"train_{prefix}{head}_accuracy"])
                self.warmup = max(self.warmup, da["warmup_epochs"])
            if cl["enabled"] and cl["weight"] > 0:
                self.monitors.append(f"train_{prefix}contrastive_loss")
                self.anchor_metrics.append("train_background_anchor_fraction" if prefix
                                           else "train_contrastive_anchor_fraction")
        self.values = {key: deque(maxlen=2 * self.settings["window"]) for key in self.monitors}
        self.best = float("inf")
        self.wait = self.stable_epochs = 0

    def update(self, row):
        epoch, loss = row["epoch"], row["validation_loss"]
        if epoch <= self.warmup:
            self.wait = self.stable_epochs = 0
            return False
        if loss < self.best - self.min_delta:
            self.best, self.wait = loss, 0
        else:
            self.wait += 1
        window = self.settings["window"]
        anchors = [row.get(key) for key in self.anchor_metrics]
        anchors_valid = all(value is not None and math.isfinite(value) and value > 0 for value in anchors)
        stable = anchors_valid
        for key, values in self.values.items():
            value = row.get(key)
            if value is None or not math.isfinite(value) or not anchors_valid:
                values.clear()
                stable = False
                continue
            values.append(value)
            if len(values) < 2 * window:
                stable = False
                continue
            previous, recent = list(values)[:window], list(values)[window:]
            a, b = sum(previous) / window, sum(recent) / window
            tolerance = self.settings["absolute_delta"] + self.settings["relative_delta"] * max(abs(a), abs(b))
            stable = stable and abs(a - b) <= tolerance
        self.stable_epochs = self.stable_epochs + 1 if stable else 0
        return (self.patience is not None and epoch >= self.warmup + self.patience
                and self.wait >= self.patience and self.stable_epochs >= self.patience)
