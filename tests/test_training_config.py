from logging import config

import pytest

from vebra.training.config import (
    TokenizerConfig,
    TrainingConfig,
    VebraTrainingConfig,
    load_training_config,
)


def test_load_training_config(tmp_path):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
model:
  vocab_size: 512
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

tokenizer:
  path: tokenizer.json
  vocab_size: 512
  min_frequency: 2

training:
  batch_size: 4
  learning_rate: 0.001
  steps: 10
  device: cpu
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert isinstance(
        config,
        VebraTrainingConfig,
    )

    assert config.model.vocab_size == 512
    assert config.model.hidden_size == 64
    assert config.model.num_layers == 2
    assert config.training.batch_size == 4
    assert config.training.learning_rate == 0.001
    assert config.training.steps == 10
    assert config.training.device == "cpu"

    assert config.tokenizer.path == "tokenizer.json"
    assert config.tokenizer.vocab_size == 512
    assert config.tokenizer.min_frequency == 2


def test_tokenizer_config_rejects_empty_path():
    with pytest.raises(
        ValueError,
        match="path",
    ):
        TokenizerConfig(
            path="",
            vocab_size=512,
            min_frequency=2,
        )


def test_tokenizer_config_rejects_small_vocab_size():
    with pytest.raises(
        ValueError,
        match="vocab_size",
    ):
        TokenizerConfig(
            path="tokenizer.json",
            vocab_size=255,
            min_frequency=2,
        )


def test_tokenizer_config_rejects_invalid_min_frequency():
    with pytest.raises(
        ValueError,
        match="min_frequency",
    ):
        TokenizerConfig(
            path="tokenizer.json",
            vocab_size=512,
            min_frequency=0,
        )


def test_training_config_rejects_invalid_batch_size():
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        TrainingConfig(
            batch_size=0,
            learning_rate=1e-3,
            steps=10,
            device="cpu",
        )


def test_training_config_rejects_invalid_learning_rate():
    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        TrainingConfig(
            batch_size=4,
            learning_rate=0,
            steps=10,
            device="cpu",
        )


def test_training_config_rejects_invalid_steps():
    with pytest.raises(
        ValueError,
        match="steps",
    ):
        TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            steps=0,
            device="cpu",
        )


def test_training_config_rejects_empty_device():
    with pytest.raises(
        ValueError,
        match="device",
    ):
        TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            steps=10,
            device="",
        )


def test_load_training_config_rejects_missing_file(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_training_config(
            tmp_path / "missing.yaml"
        )


def test_load_training_config_rejects_missing_model(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
tokenizer:
  path: tokenizer.json
  vocab_size: 512
  min_frequency: 2

training:
  batch_size: 4
  learning_rate: 0.001
  steps: 10
  device: cpu
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="model",
    ):
        load_training_config(config_path)


def test_load_training_config_rejects_missing_tokenizer(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
model:
  vocab_size: 512
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

training:
  batch_size: 4
  learning_rate: 0.001
  steps: 10
  device: cpu
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="tokenizer",
    ):
        load_training_config(config_path)


def test_load_training_config_rejects_missing_training(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
model:
  vocab_size: 512
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

tokenizer:
  path: tokenizer.json
  vocab_size: 512
  min_frequency: 2
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="training",
    ):
        load_training_config(config_path)


def test_load_training_config_rejects_vocab_size_mismatch(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
model:
  vocab_size: 512
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

tokenizer:
  path: tokenizer.json
  vocab_size: 1024
  min_frequency: 2

training:
  batch_size: 4
  learning_rate: 0.001
  steps: 10
  device: cpu
  precision: bf16
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="match",
    ):
        load_training_config(config_path)
        assert config.training.precision == "bf16"

def test_training_config_rejects_invalid_precision():
    with pytest.raises(
        ValueError,
        match="precision must be one of: fp32, bf16, fp16",
    ):
        TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            steps=10,
            device="cpu",
            precision="int8",
        )