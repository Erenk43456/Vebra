from dataclasses import dataclass


@dataclass(frozen=True)
class VebraConfig:
    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    intermediate_size: int = 2_048
    max_sequence_length: int = 1_024

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")