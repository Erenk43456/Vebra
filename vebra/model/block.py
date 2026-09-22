import torch
from torch import nn

from vebra.model.attention import CausalSelfAttention
from vebra.model.mlp import SwiGLU
from vebra.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(hidden_size)

        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
        )

        self.mlp_norm = RMSNorm(hidden_size)

        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(
            self.attention(
                self.attention_norm(x)
            )
        )

        x = x + self.dropout(
            self.mlp(
                self.mlp_norm(x)
            )
        )

        return x