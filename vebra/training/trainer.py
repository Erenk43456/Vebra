from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from vebra.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from vebra.training.step import training_step


@dataclass
class TrainingMetrics:
    steps: int
    mean_loss: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device | str = "cpu",
        precision: str = "fp32",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = self._resolve_device(device)

        if precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(
                "precision must be one of: fp32, bf16, fp16"
            )
        
        self.precision = precision

        if (
            self.precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bf16 precision requested but CUDA device does not support BF16"
            )
        
        self.global_step = 0
        self._data_iterator = iter(self.dataloader)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(
        device: torch.device | str,
    ) -> torch.device:
        resolved_device = torch.device(device)

        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available"
                )

        return resolved_device

    def train(
        self,
        steps: int,
    ) -> TrainingMetrics:
        if steps <= 0:
            raise ValueError("steps must be positive")

        total_loss = 0.0

        for _ in range(steps):
            try:
                input_ids, labels = next(self._data_iterator)
            except StopIteration:
                self._data_iterator = iter(self.dataloader)
                input_ids, labels = next(self._data_iterator)

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            loss = training_step(
                model=self.model,
                optimizer=self.optimizer,
                input_ids=input_ids,
                labels=labels,
                device=self.device,
                precision=self.precision,
            )

            total_loss += loss
            self.global_step += 1

        return TrainingMetrics(
            steps=steps,
            mean_loss=total_loss / steps,
        )

    def save_checkpoint(
        self,
        path: str,
    ) -> None:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.global_step,
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> None:
        self.global_step = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
        )