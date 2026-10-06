"""Read-only validation of background heads during training."""
import torch
from torch.nn import functional as F


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
