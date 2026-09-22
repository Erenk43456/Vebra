import torch

from vebra.model.block import TransformerBlock


def create_block() -> TransformerBlock:
    return TransformerBlock(
        hidden_size=64,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=32,
    )


def test_transformer_block_output_shape():
    block = create_block()

    x = torch.randn(2, 16, 64)
    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_supports_backward():
    block = create_block()

    x = torch.randn(
        2,
        16,
        64,
        requires_grad=True,
    )

    output = block(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in block.parameters():
        assert parameter.grad is not None


def test_transformer_block_contains_expected_components():
    block = create_block()

    assert block.attention_norm is not None
    assert block.attention is not None
    assert block.mlp_norm is not None
    assert block.mlp is not None


def test_transformer_block_preserves_batch_and_sequence_dimensions():
    block = create_block()

    x = torch.randn(4, 7, 64)
    output = block(x)

    assert output.shape[0] == 4
    assert output.shape[1] == 7
    assert output.shape[2] == 64