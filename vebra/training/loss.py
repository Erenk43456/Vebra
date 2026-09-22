import torch
from torch.nn import functional as F


def causal_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")

    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must have matching batch and sequence dimensions"
        )

    if logits.shape[1] < 2:
        raise ValueError(
            "sequence length must be at least 2 for causal language modeling"
        )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
    )