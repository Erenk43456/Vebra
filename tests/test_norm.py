import torch

from vebra.model.norm import RMSNorm


def test_rms_norm_preserves_shape():
    norm = RMSNorm(16)

    x = torch.randn(2, 8, 16)
    y = norm(x)

    assert y.shape == x.shape


def test_rms_norm_has_trainable_weight():
    norm = RMSNorm(16)

    assert norm.weight.shape == (16,)
    assert norm.weight.requires_grad


def test_rms_norm_supports_backward():
    norm = RMSNorm(16)

    x = torch.randn(2, 8, 16, requires_grad=True)
    y = norm(x)

    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None