import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive"
            )

        self.gate_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.up_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        self.down_projection = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(
            self.gate_projection(x)
        )

        up = self.up_projection(x)

        return self.down_projection(gate * up)