import pytest
import torch

from vebra.data.loader import create_dataloader
from vebra.data.text_dataset import TextDataset
from vebra.model.config import VebraConfig
from vebra.model.model import VebraModel
from vebra.tokenizer.byte import ByteTokenizer
from vebra.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from vebra.training.trainer import Trainer


def create_components():
    tokenizer = ByteTokenizer()

    text = (
        "Vebra checkpoint testing corpus. "
        "The model should be saveable and loadable. "
    ) * 8

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        sequence_length=16,
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    config = VebraConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        intermediate_size=256,
        max_sequence_length=16,
    )

    model = VebraModel(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    return model, optimizer, dataloader


def test_checkpoint_round_trip(tmp_path):
    model, optimizer, dataloader = create_components()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
    )

    trainer.train(steps=3)

    checkpoint_path = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(checkpoint_path)

    assert checkpoint_path.is_file()

    restored_model, restored_optimizer, restored_dataloader = (
        create_components()
    )

    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        dataloader=restored_dataloader,
    )

    restored_trainer.load_checkpoint(checkpoint_path)

    assert restored_trainer.global_step == 3

    for original, restored in zip(
        trainer.model.parameters(),
        restored_trainer.model.parameters(),
    ):
        assert torch.equal(original, restored)


def test_checkpoint_restores_optimizer_state(tmp_path):
    model, optimizer, dataloader = create_components()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
    )

    trainer.train(steps=2)

    checkpoint_path = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(checkpoint_path)

    restored_model, restored_optimizer, restored_dataloader = (
        create_components()
    )

    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        dataloader=restored_dataloader,
    )

    restored_trainer.load_checkpoint(checkpoint_path)

    original_state = optimizer.state_dict()
    restored_state = restored_optimizer.state_dict()

    assert original_state["param_groups"] == restored_state["param_groups"]

    assert original_state["state"].keys() == restored_state["state"].keys()

    for parameter_id in original_state["state"]:
        original_parameters = original_state["state"][parameter_id]
        restored_parameters = restored_state["state"][parameter_id]

        assert original_parameters.keys() == restored_parameters.keys()

        for key in original_parameters:
            original_value = original_parameters[key]
            restored_value = restored_parameters[key]

            if isinstance(original_value, torch.Tensor):
                assert torch.equal(
                    original_value,
                    restored_value,
                )
            else:
                assert original_value == restored_value


def test_checkpoint_rejects_missing_file(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "missing.pt"

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )


def test_checkpoint_rejects_invalid_step(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "invalid.pt"

    torch.save(
        {
            "version": 1,
            "step": -1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="invalid step",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )


def test_checkpoint_rejects_unsupported_version(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "invalid.pt"

    torch.save(
        {
            "version": 999,
            "step": 0,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="unsupported checkpoint version",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

def test_trainer_continues_after_checkpoint(tmp_path):
    model, optimizer, dataloader = create_components()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
    )

    trainer.train(steps=3)

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)

    restored_model, restored_optimizer, restored_dataloader = (
        create_components()
    )

    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        dataloader=restored_dataloader,
    )

    restored_trainer.load_checkpoint(checkpoint_path)

    assert restored_trainer.global_step == 3

    metrics = restored_trainer.train(steps=2)

    assert metrics.steps == 2
    assert restored_trainer.global_step == 5

def test_checkpoint_rejects_missing_step(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "missing_step.pt"

    torch.save(
        {
            "version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint is missing step",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

def test_checkpoint_rejects_missing_model(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "missing_model.pt"

    torch.save(
        {
            "version": 1,
            "step": 0,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint is missing model state",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )


def test_checkpoint_rejects_missing_optimizer(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "missing_optimizer.pt"

    torch.save(
        {
            "version": 1,
            "step": 0,
            "model": model.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(
        ValueError,
        match="checkpoint is missing optimizer state",
    ):
        load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    model_a, optimizer_a, dataloader_a = create_components()

    trainer_a = Trainer(
        model=model_a,
        optimizer=optimizer_a,
        dataloader=dataloader_a,
    )

    trainer_a.train(steps=3)

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer_a.save_checkpoint(checkpoint_path)

    uninterrupted_model, uninterrupted_optimizer, uninterrupted_dataloader = (
        create_components()
    )

    uninterrupted_model.load_state_dict(model_a.state_dict())
    uninterrupted_optimizer.load_state_dict(
        optimizer_a.state_dict()
    )

    uninterrupted_trainer = Trainer(
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        dataloader=uninterrupted_dataloader,
    )

    uninterrupted_trainer.global_step = trainer_a.global_step
    uninterrupted_trainer.train(steps=2)

    resumed_model, resumed_optimizer, resumed_dataloader = (
        create_components()
    )

    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        dataloader=resumed_dataloader,
    )

    resumed_trainer.load_checkpoint(checkpoint_path)
    resumed_trainer.train(steps=2)

    for uninterrupted, resumed in zip(
        uninterrupted_trainer.model.parameters(),
        resumed_trainer.model.parameters(),
    ):
        assert torch.equal(
            uninterrupted,
            resumed,
        )

def test_checkpoint_rejects_incompatible_model(tmp_path):
    model, optimizer, _ = create_components()

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        step=1,
    )

    incompatible_config = VebraConfig(
        vocab_size=256,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        intermediate_size=512,
        max_sequence_length=16,
    )

    incompatible_model = VebraModel(incompatible_config)

    incompatible_optimizer = torch.optim.AdamW(
        incompatible_model.parameters(),
        lr=1e-3,
    )

    with pytest.raises(RuntimeError):
        load_checkpoint(
            path=checkpoint_path,
            model=incompatible_model,
            optimizer=incompatible_optimizer,
        )

def test_checkpoint_loads_optimizer_state_on_model_device(
    tmp_path,
):
    model, optimizer, dataloader = create_components()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        device="cpu",
    )

    trainer.train(steps=1)

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)

    restored_model, restored_optimizer, restored_dataloader = (
        create_components()
    )

    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        dataloader=restored_dataloader,
        device="cpu",
    )

    restored_trainer.load_checkpoint(checkpoint_path)

    model_device = next(
        restored_trainer.model.parameters()
    ).device

    for state in restored_trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device == model_device