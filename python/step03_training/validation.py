"""Domain-balanced task metrics and read-only background-head validation."""
import torch
from torch.nn import functional as F


class DomainMetrics:
    """Accumulate within-domain means across batches, then weight domains equally."""

    def __init__(self):
        self.totals = {}

    @torch.no_grad()
    def update(self, logits, labels, groups, criterion):
        groups = groups.to(labels.device)
        for group in groups.unique():
            mask = groups == group
            loss = criterion(logits[mask], labels[mask])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            count = int(mask.sum())
            totals = self.totals.setdefault(int(group), [0., 0, 0])
            totals[0] += loss.item() * count
            totals[1] += int((logits[mask].argmax(1) == labels[mask]).sum())
            totals[2] += count

    def compute(self):
        if not self.totals:
            raise ValueError("Validation must be nonempty")
        values = list(self.totals.values())
        return {
            "validation_loss": sum(loss / count for loss, _, count in values) / len(values),
            "validation_accuracy": sum(correct / count for _, correct, count in values) / len(values),
            "validation_samples": sum(count for _, _, count in values),
            "validation_domains": len(values),
        }


@torch.no_grad()
def background_validation(model, time, stft, labels, domains):
    """Evaluate saved background heads on a clean validation batch in eval mode."""
    b = model.encode_background(time, stft)
    result = {}
    known = domains >= 0
    if hasattr(model, "background_domain_classifier") and known.any():
        logits = model.background_domain_classifier(b[known])
        result["domain"] = (F.cross_entropy(logits, domains[known]).item(),
                            (logits.argmax(1) == domains[known]).sum().item(), int(known.sum()))
    if hasattr(model, "background_activity_classifier"):
        head = model.background_activity_classifier
        mask = known if head.condition_classes else torch.ones_like(known)
        if mask.any():
            logits = head(b[mask], labels=domains[mask])
            result["activity"] = (F.cross_entropy(logits, labels[mask]).item(),
                                  (logits.argmax(1) == labels[mask]).sum().item(), int(mask.sum()))
    if any(not torch.isfinite(torch.tensor(values[0])) for values in result.values()):
        raise ValueError("Nonfinite background validation loss")
    return result
