"""Create epoch-based schedulers; the notebook controls when they step."""
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def build_scheduler(optimizer, config, epochs):
    if config.name == "none":
        return None
    kwargs = config.kwargs.copy()
    if config.name == "cosine":
        kwargs.setdefault("T_max", epochs)
        return CosineAnnealingLR(optimizer, **kwargs)
    if config.name == "plateau":
        # The notebook always supplies validation loss, not accuracy.
        if kwargs.pop("mode", "min") != "min":
            raise ValueError("Plateau monitors validation loss and requires mode='min'")
        return ReduceLROnPlateau(optimizer, mode="min", **kwargs)
    raise ValueError(f"Unknown scheduler: {config.name}")
