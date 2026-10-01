"""Cross-domain positives on the embedding used for downstream scoring."""
import torch
from torch.nn import functional as F


def cross_domain_contrastive(embedding, labels, domains, temperature=0.1):
    """Return mean loss and valid-anchor count; exclude same-class/same-domain pairs."""
    same_class = labels[:, None] == labels[None, :]
    different_domain = domains[:, None] != domains[None, :]
    known = (domains[:, None] >= 0) & (domains[None, :] >= 0)
    positives = same_class & different_domain & known
    candidates = positives | (~same_class & known)
    valid = positives.any(dim=1)
    count = int(valid.sum().item())
    if not count:
        return embedding.sum() * 0, 0
    z = F.normalize(embedding, dim=1)
    scores = (z @ z.T)[valid] / temperature
    positive = positives[valid]
    log_denominator = scores.masked_fill(~candidates[valid], -torch.inf).logsumexp(dim=1)
    positive_mean = scores.masked_fill(~positive, 0).sum(dim=1) / positive.sum(dim=1)
    return (log_denominator - positive_mean).mean(), count
