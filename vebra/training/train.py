from pathlib import Path

import torch

from vebra.data.loader import create_dataloader
from vebra.data.text_dataset import create_text_dataset_from_file
from vebra.model.model import VebraModel
from vebra.tokenizer.bpe import BPETokenizer
from vebra.training.config import VebraTrainingConfig
from vebra.training.trainer import Trainer


def train_from_corpus(
    corpus_path: str | Path,
    checkpoint_path: str | Path,
    config: VebraTrainingConfig,
) -> float:
    tokenizer = BPETokenizer.load(
        config.tokenizer.path,
    )

    if tokenizer.vocab_size != config.tokenizer.vocab_size:
        raise ValueError(
            "loaded tokenizer vocabulary size does not match "
            "configured tokenizer vocabulary size"
        )

    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError(
            "tokenizer vocabulary size does not match model vocabulary size"
        )

    dataset = create_text_dataset_from_file(
        path=corpus_path,
        tokenizer=tokenizer,
        sequence_length=config.model.max_sequence_length,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    model = VebraModel(config.model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device=config.training.device,
        precision=config.training.precision,
    )

    metrics = trainer.train(
        steps=config.training.steps,
    )

    trainer.save_checkpoint(
        checkpoint_path,
    )

    return metrics.mean_loss