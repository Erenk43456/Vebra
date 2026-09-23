import torch

from vebra.generation import generate
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


def test_generation_adds_requested_tokens():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 8),
    )

    generated = generate(
        model,
        input_ids,
        max_new_tokens=5,
    )

    assert generated.shape == (2, 13)
    assert torch.equal(generated[:, :8], input_ids)


def test_generation_uses_greedy_decoding():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    generated = generate(
        model,
        input_ids,
        max_new_tokens=1,
    )

    logits, _ = model(input_ids)

    expected_token = logits[:, -1, :].argmax(
        dim=-1,
        keepdim=True,
    )

    assert torch.equal(
        generated[:, -1:],
        expected_token,
    )


def test_generation_with_zero_tokens_returns_input():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (2, 8),
    )

    generated = generate(
        model,
        input_ids,
        max_new_tokens=0,
    )

    assert torch.equal(generated, input_ids)


def test_generation_respects_max_sequence_length():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 30),
    )

    generated = generate(
        model,
        input_ids,
        max_new_tokens=10,
    )

    assert generated.shape == (1, 32)


def test_generation_restores_training_mode():
    model = create_model()
    model.train()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    generate(
        model,
        input_ids,
        max_new_tokens=2,
    )

    assert model.training


def test_generation_preserves_eval_mode():
    model = create_model()
    model.eval()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    generate(
        model,
        input_ids,
        max_new_tokens=2,
    )

    assert not model.training


def test_generation_rejects_non_long_input():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    ).float()

    try:
        generate(
            model,
            input_ids,
            max_new_tokens=1,
        )
    except TypeError as error:
        assert "torch.long" in str(error)
    else:
        raise AssertionError(
            "expected TypeError for non-long input"
        )


def test_generation_rejects_non_2d_input():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (8,),
    )

    try:
        generate(
            model,
            input_ids,
            max_new_tokens=1,
        )
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError(
            "expected ValueError for non-2D input"
        )


def test_generation_rejects_negative_max_new_tokens():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    try:
        generate(
            model,
            input_ids,
            max_new_tokens=-1,
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError(
            "expected ValueError for negative max_new_tokens"
        )


def test_generation_rejects_excessive_input_length():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 33),
    )

    try:
        generate(
            model,
            input_ids,
            max_new_tokens=1,
        )
    except ValueError as error:
        assert "maximum" in str(error)
    else:
        raise AssertionError(
            "expected ValueError for excessive input length"
        )


def test_generation_returns_long_tensor():
    model = create_model()

    input_ids = torch.randint(
        0,
        128,
        (1, 8),
    )

    generated = generate(
        model,
        input_ids,
        max_new_tokens=2,
    )

    assert generated.dtype == torch.long
