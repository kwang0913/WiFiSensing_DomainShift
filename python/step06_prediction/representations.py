"""Ordered task/background exports in a single pass over input data."""
import numpy as np
import torch


def export_representations(model, batches, device=None, inference_batch_size=4,
                           include_background=True):
    if not isinstance(inference_batch_size, int) or inference_batch_size < 1:
        raise ValueError("inference_batch_size must be a positive integer")
    device = device or next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    arrays = {key: [] for key in ("probabilities", "embeddings", "labels")}
    if include_background and hasattr(model, "encode_background"):
        arrays["background"] = []
    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                time, stft, labels = batch[:3]
                for start in range(0, len(labels), inference_batch_size):
                    stop = start + inference_batch_size
                    time_part, stft_part = time[start:stop].to(device), stft[start:stop].to(device)
                    logits, embedding = model(time_part, stft_part, return_embedding=True)
                    values = {"probabilities": logits.softmax(1), "embeddings": embedding,
                              "labels": labels[start:stop]}
                    if "background" in arrays:
                        values["background"] = model.encode_background(time_part, stft_part)
                    for key, value in values.items():
                        if not torch.isfinite(value).all():
                            raise ValueError(f"Nonfinite representation: {key}")
                        arrays[key].append(value.cpu().numpy())
    finally:
        for module, training in modes:
            module.training = training
    if not arrays["labels"]:
        raise ValueError("Representation DataLoader is empty")
    return {key: np.concatenate(value) for key, value in arrays.items()}
