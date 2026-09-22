import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_sequence_length: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        if base <= 0:
            raise ValueError("base must be positive")

        inverse_frequency = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )

        positions = torch.arange(
            max_sequence_length,
            dtype=torch.float32,
        )

        frequencies = torch.outer(positions, inverse_frequency)

        self.register_buffer(
            "cos",
            frequencies.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin",
            frequencies.sin(),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]

        if sequence_length > self.cos.shape[0]:
            raise ValueError(
                "sequence length exceeds configured maximum"
            )

        cos = self.cos[:sequence_length]
        sin = self.sin[:sequence_length]

        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)

        return query, key

    @staticmethod
    def _rotate(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        ).flatten(-2)