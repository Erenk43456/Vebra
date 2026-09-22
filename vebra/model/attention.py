import math

import torch
from torch import nn

from vebra.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_sequence_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        if max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the range [0, 1)"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, hidden_size)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_sequence_length=max_sequence_length,
        )

        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(
            torch.ones(
                max_sequence_length,
                max_sequence_length,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        if sequence_length > self.causal_mask.shape[0]:
            raise ValueError(
                "sequence length exceeds configured maximum"
            )

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        query, key = self.rope(query, key)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        attention_scores = attention_scores.masked_fill(
            ~mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = torch.matmul(
            attention_weights,
            value,
        )

        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

        return self.output(output)

    def _split_heads(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)