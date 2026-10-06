"""Background and joint objectives. Target and calibration batches never enter this training helper."""
import torch
from torch.nn import functional as F
from .contrastive import cross_domain_contrastive


def joint_objective(model, z, time, stft, labels, domains, config,
                         step=1, steps_per_epoch=1, *, background=None, task_difference=None, decoder=None):
    task_difference = task_difference or {"enabled": False, "weight": 0.0}
    task_diff_on = task_difference["enabled"] and task_difference["weight"] > 0
    decoder_on = decoder is not None and decoder.enabled and decoder.weight > 0
    weighted_terms = (config.domain, config.contrastive, config.difference)
    if not config.adversarial["enabled"] and not task_diff_on and not decoder_on and not any(
            term["enabled"] and term["weight"] > 0 for term in weighted_terms):
        return time.new_zeros(()), {}
    b = background if background is not None else model.encode_background(time, stft)
    loss = b.new_zeros(())
    metrics = {}
    if config.domain["enabled"]:
        logits = model.background_domain_classifier(b)
        ce = F.cross_entropy(logits, domains)
        if config.domain["weight"]:
            loss = loss + config.domain["weight"] * ce
        metrics.update(background_domain_loss=ce.item(),
                       background_domain_accuracy=(logits.argmax(1) == domains).float().mean().item())
    if config.contrastive["enabled"]:
        # Same domain, different activity: swap the labels of task CL.
        cl, anchors = cross_domain_contrastive(b, domains, labels, config.contrastive["temperature"])
        if config.contrastive["weight"]:
            loss = loss + config.contrastive["weight"] * cl
        metrics.update(background_contrastive_loss=cl.item(), background_anchor_fraction=anchors / len(labels))
    if config.adversarial["enabled"]:
        head = model.background_activity_classifier
        ramp = config.adversarial["warmup_epochs"] * steps_per_epoch
        weight = config.adversarial["max_weight"] * (min(1., (step - 1) / ramp) if ramp else 1.)
        head.requires_grad_(False)
        try:
            logits = head(b, weight, labels=domains)
        finally:
            head.requires_grad_(True)
        ce = F.cross_entropy(logits, labels)
        if weight:
            loss = loss + ce
        metrics.update(background_activity_loss=ce.item(), background_adversarial_weight=weight,
                       background_activity_accuracy=(logits.argmax(1) == labels).float().mean().item())
    if config.difference["enabled"] or task_diff_on:
        # Detach the opposite branch so each switch controls only its own gradient.
        if task_diff_on:
            difference = difference_loss(z, b.detach())
            loss = loss + task_difference["weight"] * difference
            metrics["task_difference_loss"] = difference.item()
        if config.difference["enabled"] and config.difference["weight"]:
            difference = difference_loss(z.detach(), b)
            loss = loss + config.difference["weight"] * difference
            metrics["background_difference_loss"] = difference.item()
    if decoder_on:
        t_hat, f_hat = model.decoder(torch.cat((z, b), 1), time.shape[-1], stft.shape[-1])
        # Reconstruct the actual model inputs; signed log STFT retains amplitude ordering.
        f_target = torch.sign(stft) * torch.log1p(stft.abs())
        def relative_mse(pred, target):
            scale = target.detach().square().mean((1, 2, 3)).clamp_min(1e-6)
            return ((pred - target).square().mean((1, 2, 3)) / scale).mean()
        rt, rf = relative_mse(t_hat, time), relative_mse(f_hat, f_target)
        loss = loss + decoder.weight * (rt + rf) / 2
        metrics.update(reconstruction_time_loss=rt.item(), reconstruction_stft_loss=rf.item())
    return loss, metrics


def difference_loss(z, b):
    """Mean squared cross-correlation, with gradient routing chosen by the caller."""
    zc = F.normalize(z - z.mean(0, keepdim=True), dim=0, eps=1e-6)
    bc = F.normalize(b - b.mean(0, keepdim=True), dim=0, eps=1e-6)
    return (zc.T @ bc).square().mean()


