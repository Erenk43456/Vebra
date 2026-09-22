from pathlib import Path

from vebra.tokenizer.bpe import BPETrainer, BPETokenizer


def read_corpus(path: str | Path) -> str:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"corpus file not found: {path}"
        )

    return path.read_text(encoding="utf-8")


def train_tokenizer(
    corpus_path: str | Path,
    output_path: str | Path,
    vocab_size: int,
    min_frequency: int = 2,
) -> BPETokenizer:
    corpus = read_corpus(corpus_path)

    if not corpus:
        raise ValueError("corpus must not be empty")

    trainer = BPETrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    sequences = [
        list(corpus.encode("utf-8"))
    ]

    vocab, merges = trainer.train(sequences)

    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
    )

    tokenizer.save(output_path)

    return tokenizer