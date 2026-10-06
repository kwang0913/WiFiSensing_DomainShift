"""Read-only background exports, preserving loader order and module modes."""
import numpy as np
import torch


def feature_representations(model, batches, microbatch_size=4, features="background"):
    if features not in ("task", "background"):
        raise ValueError("Expected task or background features")
    if features == "background" and not hasattr(model, "encode_background"):
        raise ValueError("Background weighting requires a saved disentangled model")
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("microbatch_size must be a positive integer")
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    result = []
    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                time, stft = batch[:2]  # Labels deliberately not inspected.
                for start in range(0, len(time), microbatch_size):
                    encode = model.encode_background if features == "background" else model.encode
                    b = encode(time[start:start+microbatch_size].to(device),
                                                stft[start:start+microbatch_size].to(device))
                    if not torch.isfinite(b).all():
                        raise ValueError("Nonfinite weighting representation")
                    result.append(b.cpu().numpy())
    finally:
        for module, training in modes:
            module.training = training
    if not result:
        raise ValueError("Empty weighting feature export")
    return np.concatenate(result)


def background_representations(model, batches, microbatch_size=4):
    return feature_representations(model, batches, microbatch_size, features="background")


def background_diagnostics(model, exported, domain_targets=None):
    """Read-only diagnostics from saved heads, not a newly trained leakage probe."""
    z, b = np.asarray(exported["embeddings"]), np.asarray(exported["background"])
    zc, bc = z - z.mean(0), b - b.mean(0)
    zc = zc / np.maximum(np.linalg.norm(zc, axis=0), 1e-8)
    bc = bc / np.maximum(np.linalg.norm(bc, axis=0), 1e-8)
    result = {"task_background_mean_squared_correlation": float(np.square(zc.T @ bc).mean()),
              "background_mean_std": float(b.std(0).mean()),
              "background_mean_norm": float(np.linalg.norm(b, axis=1).mean()),
              "interpretation": "Saved-head diagnostics do not establish independence."}
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        features = torch.as_tensor(b, device=next(model.parameters()).device)
        with torch.inference_mode():
            if hasattr(model, "background_activity_classifier"):
                head = model.background_activity_classifier
                labels = np.asarray(exported["labels"])
                conditional = bool(head.condition_classes)
                result["background_activity_conditioning"] = "domain" if conditional else "none"
                known = np.ones(len(b), dtype=bool)
                if conditional:
                    domains = np.asarray(domain_targets) if domain_targets is not None else np.full(len(b), -1)
                    known = (domains >= 0) & (domains < head.condition_classes)
                result["background_activity_samples"] = int(known.sum())
                if known.any():
                    condition = (torch.as_tensor(domains[known], dtype=torch.long, device=features.device)
                                 if conditional else None)
                    logits = head(features[known], labels=condition)
                    result["background_activity_accuracy"] = float((logits.argmax(1).cpu().numpy() == labels[known]).mean())
                else:
                    result["background_activity_accuracy"] = None
                    result["background_activity_skipped"] = "No known training-domain labels for conditional head"
            if hasattr(model, "background_domain_classifier") and domain_targets is not None:
                domains = np.asarray(domain_targets)
                known = domains >= 0
                result["known_domain_samples"] = int(known.sum())
                if known.any():
                    pred = model.background_domain_classifier(features).argmax(1).cpu().numpy()
                    result["background_domain_accuracy"] = float((pred[known] == domains[known]).mean())
    finally:
        for module, training in modes:
            module.training = training
    return result
