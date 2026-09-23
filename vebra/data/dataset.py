from torch.utils.data import Dataset
import torch


class TokenDataset(Dataset):
    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if len(token_ids) < sequence_length + 1:
            raise ValueError(
                "token_ids must contain at least sequence_length + 1 tokens"
            )

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
        )
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index
        end = start + self.sequence_length + 1

        sequence = self.token_ids[start:end]

        input_ids = sequence[:-1]
        labels = sequence[1:]

        return input_ids, labels