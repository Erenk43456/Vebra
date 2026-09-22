import torch

from vebra.data.loader import create_dataloader
from vebra.data.text_dataset import TextDataset
from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel
from vebra.tokenizer.byte import ByteTokenizer
from vebra.training.step import training_step


def test_raw_text_training_pipeline():
    tokenizer = ByteTokenizer()

    text = (
        "Vebra is an open source language model. "
        "This is a tiny training corpus for integration testing. "
    ) * 8

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=16,
    )

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    config = VebraConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=16,
    )

    model = VebraModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids, labels = next(iter(loader))

    loss_before = training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    loss_after = training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    assert isinstance(loss_before, float)
    assert isinstance(loss_after, float)
    assert loss_before > 0.0
    assert loss_after > 0.0


def test_raw_text_pipeline_produces_valid_logits():
    tokenizer = ByteTokenizer()

    text = "Vebra language model training data. " * 8

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=16,
    )

    loader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    config = VebraConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=16,
    )

    model = VebraModel(config)

    input_ids, labels = next(iter(loader))

    logits, loss = model(
        input_ids,
        labels=labels,
    )

    assert logits.shape == (
        2,
        16,
        tokenizer.vocab_size,
    )

    assert loss is not None
    assert torch.isfinite(loss)