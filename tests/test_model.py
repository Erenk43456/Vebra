import torch

from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel


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


def test_model_output_shape():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (2, 16, 128)
    assert loss is None


def test_model_supports_loss():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    labels = torch.randint(
        0,
        128,
        (2, 16),
    )

    logits, loss = model(
        input_ids,
        labels=labels,
    )

    assert logits.shape == (2, 16, 128)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_supports_backward():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    labels = torch.randint(
        0,
        128,
        (2, 16),
    )

    _, loss = model(
        input_ids,
        labels=labels,
    )

    assert loss is not None

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_model_uses_weight_tying():
    model = create_model()

    assert (
        model.lm_head.weight
        is model.token_embedding.embedding.weight
    )


def test_model_rejects_excessive_sequence_length():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 33),
    )

    try:
        model(input_ids)
    except ValueError as error:
        assert "maximum" in str(error)
    else:
        raise AssertionError(
            "expected ValueError for excessive sequence length"
        )


def test_model_rejects_mismatched_labels():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 16),
    )

    labels = torch.randint(
        0,
        128,
        (2, 15),
    )

    try:
        model(input_ids, labels=labels)
    except ValueError as error:
        assert "same shape" in str(error)
    else:
        raise AssertionError(
            "expected ValueError for mismatched labels"
        )