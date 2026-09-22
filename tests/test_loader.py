import pytest
import torch

from vebra.data.dataset import TokenDataset
from vebra.data.loader import create_dataloader


def create_dataset() -> TokenDataset:
    return TokenDataset(
        token_ids=list(range(100)),
        sequence_length=16,
    )


def test_dataloader_returns_expected_batch_shape():
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    input_ids, labels = next(iter(loader))

    assert input_ids.shape == (4, 16)
    assert labels.shape == (4, 16)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_dataloader_preserves_next_token_relationship():
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    input_ids, labels = next(iter(loader))

    assert torch.equal(
        labels,
        input_ids + 1,
    )


def test_dataloader_can_shuffle():
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    input_ids, labels = next(iter(loader))

    assert input_ids.shape == (8, 16)
    assert labels.shape == (8, 16)


def test_dataloader_rejects_invalid_batch_size():
    dataset = create_dataset()

    with pytest.raises(ValueError, match="positive"):
        create_dataloader(
            dataset,
            batch_size=0,
        )


def test_dataloader_handles_final_partial_batch():
    dataset = TokenDataset(
        token_ids=list(range(30)),
        sequence_length=8,
    )

    loader = create_dataloader(
        dataset,
        batch_size=7,
        shuffle=False,
    )

    batches = list(loader)

    assert batches[-1][0].shape[0] < 7

def test_dataloader_can_enable_pin_memory():
    dataset = create_dataset()

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
    )

    assert loader.pin_memory