import torch

from vebra.model.model import VebraModel


def generate(
    model: VebraModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    if input_ids.dtype != torch.long:
        raise TypeError("input_ids must have dtype torch.long")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    model_max_length = model.config.max_sequence_length

    if input_ids.shape[1] > model_max_length:
        raise ValueError(
            "input sequence length exceeds configured maximum"
        )

    model_was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            generated = input_ids

            for _ in range(max_new_tokens):
                if generated.shape[1] >= model_max_length:
                    break

                logits, _ = model(generated)

                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)

        return generated

    finally:
        model.train(model_was_training)