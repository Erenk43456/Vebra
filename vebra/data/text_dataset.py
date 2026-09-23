from pathlib import Path

import torch
from torch.utils.data import Dataset

from vebra.data.corpus import read_text_corpus
from vebra.tokenizer.base import Tokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        sequence_length: int,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        token_ids = tokenizer.encode(text)

        if len(token_ids) < sequence_length + 1:
            raise ValueError(
                "text must contain enough tokens for the sequence length"
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.sequence_length + 1

        sequence = self.token_ids[start:end]

        return sequence[:-1], sequence[1:]


def create_text_dataset_from_file(
    path: str | Path,
    tokenizer: Tokenizer,
    sequence_length: int,
) -> TextDataset:
    text = read_text_corpus(path)

    return TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
    )