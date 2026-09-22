import pytest
import torch

from vebra.model.attention import CausalSelfAttention


def test_attention_output_shape():
    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        max_sequence_length=32,
    )

    x = torch.randn(2, 16, 64)
    output = attention(x)

    assert output.shape == x.shape


def test_attention_rejects_invalid_head_configuration():
    with pytest.raises(ValueError):
        CausalSelfAttention(
            hidden_size=65,
            num_heads=8,
            max_sequence_length=32,
        )


def test_attention_rejects_excessive_sequence_length():
    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        max_sequence_length=16,
    )

    x = torch.randn(2, 17, 64)

    with pytest.raises(ValueError):
        attention(x)


def test_attention_supports_backward():
    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        max_sequence_length=32,
    )

    x = torch.randn(
        2,
        16,
        64,
        requires_grad=True,
    )

    output = attention(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in attention.parameters():
        assert parameter.grad is not None


def test_attention_does_not_use_future_tokens():
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=8,
        max_sequence_length=32,
    )

    attention.eval()

    prefix = torch.randn(1, 4, 64)

    first_output = attention(prefix)

    modified = prefix.clone()
    modified[:, 1:, :] = torch.randn(1, 3, 64)

    second_output = attention(modified)

    assert torch.allclose(
        first_output[:, :1, :],
        second_output[:, :1, :],
        atol=1e-5,
        rtol=1e-5,
    )