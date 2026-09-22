import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")

        return self.embedding(input_ids)