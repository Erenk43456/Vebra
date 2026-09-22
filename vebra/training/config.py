from dataclasses import dataclass
from pathlib import Path

import yaml

from vebra.model.config import VebraConfig


@dataclass(frozen=True)
class TokenizerConfig:
    path: str
    vocab_size: int
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError(
                "tokenizer path must not be empty"
            )

        if self.vocab_size < 256:
            raise ValueError(
                "tokenizer vocab_size must be at least 256"
            )

        if self.min_frequency <= 0:
            raise ValueError(
                "tokenizer min_frequency must be positive"
            )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    steps: int
    device: str
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.steps <= 0:
            raise ValueError(
                "steps must be positive"
            )

        if not self.device:
            raise ValueError(
                "device must not be empty"
            )

        if self.precision not in {
            "fp32",
            "bf16",
            "fp16",
        }:
            raise ValueError(
                "precision must be one of: fp32, bf16, fp16"
            )


@dataclass(frozen=True)
class VebraTrainingConfig:
    model: VebraConfig
    tokenizer: TokenizerConfig
    training: TrainingConfig


def load_training_config(
    path: str | Path,
) -> VebraTrainingConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"config file not found: {path}"
        )

    data = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")

    model_data = data.get("model")
    tokenizer_data = data.get("tokenizer")
    training_data = data.get("training")

    if not isinstance(model_data, dict):
        raise ValueError(
            "config is missing model section"
        )

    if not isinstance(tokenizer_data, dict):
        raise ValueError(
            "config is missing tokenizer section"
        )

    if not isinstance(training_data, dict):
        raise ValueError(
            "config is missing training section"
        )

    model_config = VebraConfig(
        **model_data,
    )

    tokenizer_config = TokenizerConfig(
        **tokenizer_data,
    )

    training_config = TrainingConfig(
        **training_data,
    )

    if model_config.vocab_size != tokenizer_config.vocab_size:
        raise ValueError(
            "model vocab_size must match tokenizer vocab_size"
        )

    return VebraTrainingConfig(
        model=model_config,
        tokenizer=tokenizer_config,
        training=training_config,
    )