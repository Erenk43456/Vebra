import pytest
import torch

from vebra.model.embedding import TokenEmbedding


def test_token_embedding_shape():
    embedding = TokenEmbedding(vocab_size=1000, hidden_size=64)

    input_ids = torch.randint(0, 1000, (2, 16))
    output = embedding(input_ids)

    assert output.shape == (2, 16, 64)


def test_token_embedding_is_trainable():
    embedding = TokenEmbedding(vocab_size=1000, hidden_size=64)

    assert embedding.embedding.weight.requires_grad
    assert embedding.embedding.weight.shape == (1000, 64)


def test_token_embedding_rejects_non_integer_ids():
    embedding = TokenEmbedding(vocab_size=1000, hidden_size=64)

    input_ids = torch.randn(2, 16)

    with pytest.raises(TypeError):
        embedding(input_ids)


def test_token_embedding_supports_backward():
    embedding = TokenEmbedding(vocab_size=1000, hidden_size=64)

    input_ids = torch.randint(0, 1000, (2, 16))
    output = embedding(input_ids)

    loss = output.mean()
    loss.backward()

    assert embedding.embedding.weight.grad is not None