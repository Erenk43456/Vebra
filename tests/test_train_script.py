import torch

from vebra.tokenizer.train import train_tokenizer
from vebra.training.config import load_training_config
from vebra.training.train import train_from_corpus


def test_train_from_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    config_path = tmp_path / "config.yaml"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint.pt"

    corpus_path.write_text(
        (
            "Vebra is an open source language model. "
            "This corpus is used for training tests. "
        )
        * 8,
        encoding="utf-8",
    )

    train_tokenizer(
        corpus_path=corpus_path,
        output_path=tokenizer_path,
        vocab_size=256,
        min_frequency=2,
    )

    config_path.write_text(
        f"""
model:
  vocab_size: 256
  hidden_size: 64
  num_layers: 2
  num_heads: 8
  intermediate_size: 256
  max_sequence_length: 16

tokenizer:
  path: {tokenizer_path}
  vocab_size: 256
  min_frequency: 2

training:
  batch_size: 2
  learning_rate: 0.001
  steps: 2
  device: cpu
""",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    mean_loss = train_from_corpus(
        corpus_path=corpus_path,
        checkpoint_path=checkpoint_path,
        config=config,
    )

    assert mean_loss > 0.0
    assert torch.isfinite(
        torch.tensor(mean_loss)
    )
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0