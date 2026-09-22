import pytest
import torch

from vebra.model.rope import RotaryEmbedding


def test_rope_preserves_shape():
    rope = RotaryEmbedding(
        head_dim=64,
        max_sequence_length=128,
    )

    query = torch.randn(2, 8, 16, 64)
    key = torch.randn(2, 8, 16, 64)

    rotated_query, rotated_key = rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_rope_registers_frequency_buffers():
    rope = RotaryEmbedding(
        head_dim=64,
        max_sequence_length=128,
    )

    assert rope.cos.shape == (128, 32)
    assert rope.sin.shape == (128, 32)


def test_rope_rejects_odd_head_dimension():
    with pytest.raises(ValueError):
        RotaryEmbedding(
            head_dim=63,
            max_sequence_length=128,
        )


def test_rope_rejects_excessive_sequence_length():
    rope = RotaryEmbedding(
        head_dim=64,
        max_sequence_length=16,
    )

    query = torch.randn(1, 8, 17, 64)
    key = torch.randn(1, 8, 17, 64)

    with pytest.raises(ValueError):
        rope(query, key)


def test_rope_supports_backward():
    rope = RotaryEmbedding(
        head_dim=64,
        max_sequence_length=128,
    )

    query = torch.randn(
        2, 8, 16, 64,
        requires_grad=True,
    )
    key = torch.randn(
        2, 8, 16, 64,
        requires_grad=True,
    )

    rotated_query, rotated_key = rope(query, key)

    loss = rotated_query.mean() + rotated_key.mean()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None