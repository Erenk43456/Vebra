import pytest

from vebra.model.config import VebraConfig
from vebra.training.config import load_training_config


def test_default_config():
    config = VebraConfig()

    assert config.vocab_size == 32_000
    assert config.hidden_size == 512
    assert config.num_layers == 8
    assert config.num_heads == 8
    assert config.intermediate_size == 2_048
    assert config.max_sequence_length == 1_024


def test_hidden_size_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError):
        VebraConfig(hidden_size=513, num_heads=8)


def test_invalid_layer_count():
    with pytest.raises(ValueError):
        VebraConfig(num_layers=0)

def test_load_h200_config():
    config = load_training_config("configs/h200.yaml")

    assert config.training.device == "cuda"
    assert config.training.precision == "bf16"