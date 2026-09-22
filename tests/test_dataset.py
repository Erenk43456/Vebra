import pytest
import torch

from vebra.data.dataset import TokenDataset


def test_token_dataset_returns_correct_shapes():
    dataset = TokenDataset(
        token_ids=list(range(20)),
        sequence_length=8,
    )

    input_ids, labels = dataset[0]

    assert input_ids.shape == (8,)
    assert labels.shape == (8,)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_token_dataset_shifts_labels_by_one():
    dataset = TokenDataset(
        token_ids=list(range(10)),
        sequence_length=4,
    )

    input_ids, labels = dataset[0]

    assert torch.equal(
        input_ids,
        torch.tensor([0, 1, 2, 3]),
    )

    assert torch.equal(
        labels,
        torch.tensor([1, 2, 3, 4]),
    )


def test_token_dataset_next_sample_overlaps_by_one():
    dataset = TokenDataset(
        token_ids=list(range(10)),
        sequence_length=4,
    )

    first_input, first_labels = dataset[0]
    second_input, second_labels = dataset[1]

    assert torch.equal(
        first_input,
        torch.tensor([0, 1, 2, 3]),
    )

    assert torch.equal(
        first_labels,
        torch.tensor([1, 2, 3, 4]),
    )

    assert torch.equal(
        second_input,
        torch.tensor([1, 2, 3, 4]),
    )

    assert torch.equal(
        second_labels,
        torch.tensor([2, 3, 4, 5]),
    )


def test_token_dataset_length():
    dataset = TokenDataset(
        token_ids=list(range(20)),
        sequence_length=8,
    )

    assert len(dataset) == 12


def test_token_dataset_rejects_short_input():
    with pytest.raises(ValueError, match="at least"):
        TokenDataset(
            token_ids=[1, 2, 3, 4],
            sequence_length=4,
        )


def test_token_dataset_rejects_invalid_sequence_length():
    with pytest.raises(ValueError, match="positive"):
        TokenDataset(
            token_ids=list(range(10)),
            sequence_length=0,
        )