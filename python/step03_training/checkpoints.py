"""Atomic inference checkpoints shared by best and last epoch saves."""
import torch


def save_checkpoint(path, model, epoch, validation_loss, metadata):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({**metadata, "model_state_dict": model.state_dict(), "epoch": epoch,
                "validation_loss": validation_loss, "validation_metric": "domain_mean_ce"}, temporary)
    temporary.replace(path)
