import torch
from torch import nn

from vebra.training.loss import causal_language_model_loss


def evaluate_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    resolved_device = torch.device(device)

    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available"
            )

    total_loss = 0.0
    batches = 0

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(resolved_device)
                labels = labels.to(resolved_device)

                logits, _ = model(input_ids)

                loss = causal_language_model_loss(
                    logits=logits,
                    labels=labels,
                )

                total_loss += loss.item()
                batches += 1
    finally:
        model.train(was_training)

    if batches == 0:
        raise ValueError("dataloader must contain at least one batch")

    return total_loss / batches

def evaluate_perplexity(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> float:
    loss = evaluate_loss(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    return torch.exp(torch.tensor(loss)).item()