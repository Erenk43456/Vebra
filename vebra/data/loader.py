from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = False,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )