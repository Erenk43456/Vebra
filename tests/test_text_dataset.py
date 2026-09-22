import pytest
import torch

from vebra.data.text_dataset import TextDataset
from vebra.tokenizer.byte import ByteTokenizer
from vebra.data.loader import create_dataloader

from vebra.data.text_dataset import (
    TextDataset,
    create_text_dataset_from_file,
)


def create_dataset() -> TextDataset:
    tokenizer = ByteTokenizer()

    return TextDataset(
        text="abcdefghijklmnopqrstuvwxyz",
        tokenizer=tokenizer,
        sequence_length=8,
    )


def test_text_dataset_returns_tensors():
    dataset = create_dataset()

    input_ids, labels = dataset[0]

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long


def test_text_dataset_returns_expected_shape():
    dataset = create_dataset()

    input_ids, labels = dataset[0]

    assert input_ids.shape == (8,)
    assert labels.shape == (8,)


def test_text_dataset_shifts_labels():
    dataset = create_dataset()

    input_ids, labels = dataset[0]

    assert input_ids.tolist() == list(b"abcdefgh")
    assert labels.tolist() == list(b"bcdefghi")


def test_text_dataset_uses_tokenizer():
    tokenizer = ByteTokenizer()

    dataset = TextDataset(
        text="Vebra",
        tokenizer=tokenizer,
        sequence_length=4,
    )

    input_ids, labels = dataset[0]

    assert input_ids.tolist() == list(b"Vebr")
    assert labels.tolist() == list(b"ebra")


def test_text_dataset_length():
    dataset = create_dataset()

    assert len(dataset) == 18


def test_text_dataset_rejects_non_string():
    tokenizer = ByteTokenizer()

    with pytest.raises(TypeError, match="string"):
        TextDataset(
            text=123,
            tokenizer=tokenizer,
            sequence_length=4,
        )


def test_text_dataset_rejects_invalid_sequence_length():
    tokenizer = ByteTokenizer()

    with pytest.raises(ValueError, match="positive"):
        TextDataset(
            text="hello world",
            tokenizer=tokenizer,
            sequence_length=0,
        )


def test_text_dataset_rejects_short_text():
    tokenizer = ByteTokenizer()

    with pytest.raises(ValueError, match="enough"):
        TextDataset(
            text="abc",
            tokenizer=tokenizer,
            sequence_length=8,
        )

def test_text_dataset_works_with_dataloader():
    tokenizer = ByteTokenizer()

    dataset = TextDataset(
        text="abcdefghijklmnopqrstuvwxyz" * 4,
        tokenizer=tokenizer,
        sequence_length=16,
    )

    loader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    input_ids, labels = next(iter(loader))

    assert input_ids.shape == (4, 16)
    assert labels.shape == (4, 16)
    assert torch.equal(
        labels,
        input_ids + 1,
    )

def test_text_dataset_from_file(tmp_path):
    corpus_path = tmp_path / "corpus.txt"

    corpus_path.write_text(
        "abcdefghijklmnopqrstuvwxyz",
        encoding="utf-8",
    )

    tokenizer = ByteTokenizer()

    dataset = create_text_dataset_from_file(
        path=corpus_path,
        tokenizer=tokenizer,
        sequence_length=8,
    )

    input_ids, labels = dataset[0]

    assert input_ids.tolist() == list(b"abcdefgh")
    assert labels.tolist() == list(b"bcdefghi")