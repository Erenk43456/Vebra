import pytest
import torch

from unittest.mock import MagicMock, patch

from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel
from vebra.training.step import training_step


def create_model() -> VebraModel:
    config = VebraConfig(
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=32,
    )
    return VebraModel(config)


def test_training_step_returns_finite_loss():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    loss = training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_training_step_updates_parameters():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    changed = False

    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter.detach()):
            changed = True
            break

    assert changed


def test_training_step_clears_previous_gradients():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    first_gradients = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
    )

    second_gradients = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(first_gradients) == len(second_gradients)

def test_training_step_accepts_explicit_fp32():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    loss = training_step(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        labels=labels,
        device="cpu",
        precision="fp32",
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))


def test_training_step_rejects_invalid_precision():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    with pytest.raises(
        ValueError,
        match="precision must be one of: fp32, bf16, fp16",
    ):
        training_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            device="cpu",
            precision="int8",
        )

def test_training_step_uses_bf16_autocast_on_cuda():
    model = create_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))

    autocast_context = MagicMock()

    with patch(
        "vebra.training.step.torch.autocast",
        return_value=autocast_context,
    ) as autocast:
        training_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            labels=labels,
            device="cuda",
            precision="bf16",
        )

    autocast.assert_called_once_with(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=True,
    )