import torch
from torch import nn
from torch.nn import functional as F

from vebra.model.block import TransformerBlock
from vebra.model.config import VebraConfig
from vebra.model.embedding import TokenEmbedding
from vebra.model.norm import RMSNorm


class VebraModel(nn.Module):
    def __init__(self, config: VebraConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    intermediate_size=config.intermediate_size,
                    max_sequence_length=config.max_sequence_length,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tie input embedding and output projection weights.
        self.lm_head.weight = self.token_embedding.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")

        sequence_length = input_ids.shape[1]

        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                "sequence length exceeds configured maximum"
            )

        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None

        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    "labels must have the same shape as input_ids"
                )

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return logits, loss