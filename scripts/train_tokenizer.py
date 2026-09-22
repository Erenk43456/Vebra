import argparse

from vebra.tokenizer.train import train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Vebra BPE tokenizer."
    )

    parser.add_argument(
        "corpus",
        help="Path to UTF-8 corpus file",
    )

    parser.add_argument(
        "output",
        help="Path for tokenizer JSON",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    tokenizer = train_tokenizer(
        corpus_path=args.corpus,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    print(
        f"Tokenizer trained: "
        f"{tokenizer.vocab_size} tokens"
    )


if __name__ == "__main__":
    main()