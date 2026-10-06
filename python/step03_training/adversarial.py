"""Explicit optimizer updates for adversarial heads, separate from encoder losses."""
import torch
from torch.nn import functional as F


def update_background_adversary(model, background, labels, domains, config, optimizer):
    if not config.adversarial["enabled"]:
        return
    if optimizer is None:
        raise ValueError("Background activity adversary requires its independent optimizer")
    head = model.background_activity_classifier
    for _ in range(config.adversarial["steps_per_batch"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(background.detach(), labels=domains), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite background activity loss")
        loss.backward()
        optimizer.step()
