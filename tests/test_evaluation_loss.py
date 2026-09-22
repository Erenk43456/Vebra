import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vebra.evaluation.loss import (
    evaluate_loss,
    evaluate_perplexity,
)

from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel


def create_model() -> VebraModel:
    config = VebraConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=16,
    )

    return VebraModel(config)


def test_evaluate_loss_returns_finite_value():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (4, 16),
    )

    labels = torch.randint(
        0,
        128,
        (4, 16),
    )

    dataloader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
    )

    loss = evaluate_loss(
        model=model,
        dataloader=dataloader,
        device="cpu",
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_evaluate_loss_does_not_update_parameters():
    model = create_model()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    input_ids = torch.randint(
        0,
        128,
        (4, 16),
    )

    labels = torch.randint(
        0,
        128,
        (4, 16),
    )

    dataloader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
    )

    evaluate_loss(
        model=model,
        dataloader=dataloader,
        device="cpu",
    )

    for before, after in zip(
        parameters_before,
        model.parameters(),
    ):
        assert torch.equal(
            before,
            after,
        )

def test_evaluate_loss_restores_training_mode():
    model = create_model()
    model.train()

    input_ids = torch.randint(
        0,
        128,
        (4, 16),
    )

    labels = torch.randint(
        0,
        128,
        (4, 16),
    )

    dataloader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
    )

    evaluate_loss(
        model=model,
        dataloader=dataloader,
        device="cpu",
    )

    assert model.training

def test_evaluate_loss_rejects_empty_dataloader():
    model = create_model()

    dataloader = DataLoader(
        TensorDataset(
            torch.empty((0, 16), dtype=torch.long),
            torch.empty((0, 16), dtype=torch.long),
        ),
        batch_size=2,
    )

    with pytest.raises(
        ValueError,
        match="at least one batch",
    ):
        evaluate_loss(
            model=model,
            dataloader=dataloader,
            device="cpu",
        )


def test_evaluate_loss_rejects_unavailable_cuda():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")

    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (4, 16),
    )

    labels = torch.randint(
        0,
        128,
        (4, 16),
    )

    dataloader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA is not available",
    ):
        evaluate_loss(
            model=model,
            dataloader=dataloader,
            device="cuda",
        )

def test_evaluate_perplexity_matches_loss():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (4, 16),
    )

    labels = torch.randint(
        0,
        128,
        (4, 16),
    )

    dataloader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=2,
    )

    loss = evaluate_loss(
        model=model,
        dataloader=dataloader,
        device="cpu",
    )

    perplexity = evaluate_perplexity(
        model=model,
        dataloader=dataloader,
        device="cpu",
    )

    assert torch.isclose(
        torch.tensor(perplexity),
        torch.exp(torch.tensor(loss)),
    )
    assert perplexity >= 1.0