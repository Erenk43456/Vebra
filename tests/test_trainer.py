import pytest
import torch

from unittest.mock import patch

from vebra.data.loader import create_dataloader
from vebra.data.text_dataset import TextDataset
from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel
from vebra.tokenizer.byte import ByteTokenizer
from vebra.training.trainer import Trainer, TrainingMetrics


def create_trainer(
    device: str = "cpu",
    precision: str = "fp32",
) -> Trainer:
    tokenizer = ByteTokenizer()

    text = (
        "Vebra is an open source language model. "
        "Training a tiny transformer on a small corpus. "
    ) * 8

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=16,
    )

    dataloader = create_dataloader(
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

    return Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        precision=precision,
    )


def test_trainer_returns_metrics():
    trainer = create_trainer()

    metrics = trainer.train(steps=3)

    assert isinstance(metrics, TrainingMetrics)
    assert metrics.steps == 3
    assert metrics.mean_loss > 0.0
    assert torch.isfinite(
        torch.tensor(metrics.mean_loss)
    )


def test_trainer_updates_model():
    trainer = create_trainer()

    parameter_before = next(
        trainer.model.parameters()
    ).detach().clone()

    trainer.train(steps=1)

    parameter_after = next(
        trainer.model.parameters()
    ).detach()

    assert not torch.equal(
        parameter_before,
        parameter_after,
    )


def test_trainer_restarts_dataloader_when_exhausted():
    trainer = create_trainer()

    dataset_length = len(trainer.dataloader)

    metrics = trainer.train(
        steps=dataset_length + 2,
    )

    assert metrics.steps == dataset_length + 2
    assert metrics.mean_loss > 0.0


def test_trainer_rejects_invalid_steps():
    trainer = create_trainer()

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        trainer.train(steps=0)


def test_trainer_uses_requested_device():
    trainer = create_trainer(
        device="cpu",
    )

    assert trainer.device == torch.device("cpu")
    assert next(
        trainer.model.parameters()
    ).device == torch.device("cpu")

def test_trainer_rejects_unavailable_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")

    with pytest.raises(
        RuntimeError,
        match="CUDA is not available",
    ):
        create_trainer(
            device="cuda",
        )

def test_trainer_uses_requested_precision():
    trainer = create_trainer(device="cpu")

    assert trainer.precision == "fp32"

def test_trainer_preserves_data_progress_between_train_calls():
    trainer = create_trainer()

    trainer.train(steps=1)
    first_step = trainer.global_step

    trainer.train(steps=1)

    assert first_step == 1
    assert trainer.global_step == 2

def test_trainer_rejects_unsupported_cuda_bf16():
    with patch(
        "torch.cuda.is_available",
        return_value=True,
    ), patch(
        "torch.cuda.is_bf16_supported",
        return_value=False,
    ):
        with pytest.raises(
            RuntimeError,
            match="does not support BF16",
        ):
            create_trainer(
                device="cuda",
                precision="bf16",
            )