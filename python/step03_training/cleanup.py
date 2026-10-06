"""Release notebook training state while retaining the model for inference."""
import gc
import torch


TRAINING_OBJECTS = (
    "optimizer", "scheduler", "domain_optimizer", "domain_scheduler",
    "background_activity_optimizer", "background_activity_scheduler",
    "main_parameters", "augmenter", "criterion", "monitor",
    "checkpoint", "selected_checkpoint",
)
TRAINING_TENSORS = (
    "time", "stft", "labels", "logits", "embedding", "task_loss", "loss",
    "domain_labels", "fit_logits", "fit_loss", "domain_logits", "domain_loss",
    "background_loss", "contrastive_loss", "known",
)


def release_training_state(namespace, model):
    """Drop known notebook references; keep config, history, loaders and model weights."""
    model.zero_grad(set_to_none=True)
    for name in TRAINING_OBJECTS:
        namespace.pop(name, None)
    for name in TRAINING_TENSORS:
        if isinstance(namespace.get(name), torch.Tensor):
            namespace.pop(name)
    # A custom loader may have supplied a batch containing device tensors.
    namespace.pop("batch", None)
    gc.collect()
    if next(model.parameters()).is_cuda:
        with torch.cuda.device(next(model.parameters()).device):
            torch.cuda.empty_cache()
