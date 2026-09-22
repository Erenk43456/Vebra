import pytest
import torch

from vebra.model.mlp import SwiGLU


def test_swiglu_output_shape():
    mlp = SwiGLU(
        hidden_size=64,
        intermediate_size=256,
    )

    x = torch.randn(2, 16, 64)
    output = mlp(x)

    assert output.shape == x.shape


def test_swiglu_has_expected_parameters():
    mlp = SwiGLU(
        hidden_size=64,
        intermediate_size=256,
    )

    assert mlp.gate_projection.weight.shape == (256, 64)
    assert mlp.up_projection.weight.shape == (256, 64)
    assert mlp.down_projection.weight.shape == (64, 256)


def test_swiglu_uses_no_bias():
    mlp = SwiGLU(
        hidden_size=64,
        intermediate_size=256,
    )

    assert mlp.gate_projection.bias is None
    assert mlp.up_projection.bias is None
    assert mlp.down_projection.bias is None


def test_swiglu_supports_backward():
    mlp = SwiGLU(
        hidden_size=64,
        intermediate_size=256,
    )

    x = torch.randn(
        2,
        16,
        64,
        requires_grad=True,
    )

    output = mlp(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in mlp.parameters():
        assert parameter.grad is not None


def test_swiglu_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        SwiGLU(
            hidden_size=0,
            intermediate_size=256,
        )

    with pytest.raises(ValueError):
        SwiGLU(
            hidden_size=64,
            intermediate_size=0,
        )