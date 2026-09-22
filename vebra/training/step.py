import torch
from torch import nn

from vebra.training.loss import causal_language_model_loss


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    precision: str = "fp32",
) -> float:
    if precision not in {
        "fp32",
        "bf16",
        "fp16",
    }:
        raise ValueError(
            "precision must be one of: fp32, bf16, fp16"
        )

    device = torch.device(device)

    use_autocast = (
        device.type == "cuda"
        and precision != "fp32"
    )

    if precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        logits, _ = model(input_ids)

        loss = causal_language_model_loss(
            logits=logits,
            labels=labels,
        )

    loss.backward()
    optimizer.step()

    return loss.item()