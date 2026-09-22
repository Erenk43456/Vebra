import torch
import pytest

from vebra.training.loss import causal_language_model_loss


def test_causal_language_model_loss_returns_scalar():
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8))

    loss = causal_language_model_loss(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_causal_language_model_loss_supports_backward():
    logits = torch.randn(
        2,
        8,
        32,
        requires_grad=True,
    )
    labels = torch.randint(0, 32, (2, 8))

    loss = causal_language_model_loss(logits, labels)
    loss.backward()

    assert logits.grad is not None


def test_causal_language_model_loss_rejects_short_sequence():
    logits = torch.randn(2, 1, 32)
    labels = torch.randint(0, 32, (2, 1))

    with pytest.raises(ValueError, match="at least 2"):
        causal_language_model_loss(logits, labels)


def test_causal_language_model_loss_rejects_wrong_logits_rank():
    logits = torch.randn(2, 8)
    labels = torch.randint(0, 32, (2, 8))

    with pytest.raises(ValueError, match="\\[batch, sequence, vocab\\]"):
        causal_language_model_loss(logits, labels)


def test_causal_language_model_loss_rejects_wrong_labels_rank():
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 8, 1))

    with pytest.raises(ValueError, match="\\[batch, sequence\\]"):
        causal_language_model_loss(logits, labels)


def test_causal_language_model_loss_rejects_mismatched_dimensions():
    logits = torch.randn(2, 8, 32)
    labels = torch.randint(0, 32, (2, 7))

    with pytest.raises(ValueError, match="matching"):
        causal_language_model_loss(logits, labels)