from pathlib import Path

import torch
from torch import nn


CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")

    path = Path(path)

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint file not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("invalid checkpoint")

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")

    if "step" not in checkpoint:
        raise ValueError("checkpoint is missing step")

    if "model" not in checkpoint:
        raise ValueError("checkpoint is missing model state")

    if "optimizer" not in checkpoint:
        raise ValueError(
            "checkpoint is missing optimizer state"
        )

    step = checkpoint["step"]

    if not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint contains invalid step")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return step