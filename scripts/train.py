import argparse

from vebra.training.config import load_training_config
from vebra.training.train import train_from_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Vebra on a text corpus."
    )

    parser.add_argument(
        "corpus",
        help="Path to UTF-8 training corpus",
    )

    parser.add_argument(
        "config",
        help="Path to YAML training configuration",
    )

    parser.add_argument(
        "--checkpoint",
        default="vebra-checkpoint.pt",
        help="Checkpoint output path",
    )

    args = parser.parse_args()

    config = load_training_config(args.config)

    mean_loss = train_from_corpus(
        corpus_path=args.corpus,
        checkpoint_path=args.checkpoint,
        config=config,
    )

    print(f"steps: {config.training.steps}")
    print(f"mean_loss: {mean_loss:.6f}")
    print(f"device: {config.training.device}")
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()