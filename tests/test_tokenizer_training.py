from pathlib import Path

import pytest

from vebra.tokenizer.bpe import BPETokenizer
from vebra.tokenizer.train import read_corpus, train_tokenizer


def test_read_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"

    corpus.write_text(
        "Merhaba Vebra",
        encoding="utf-8",
    )

    assert read_corpus(corpus) == "Merhaba Vebra"


def test_read_corpus_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_corpus(tmp_path / "missing.txt")


def test_train_tokenizer_creates_file(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    output = tmp_path / "tokenizer.json"

    corpus.write_text(
        "abababab " * 20,
        encoding="utf-8",
    )

    tokenizer = train_tokenizer(
        corpus_path=corpus,
        output_path=output,
        vocab_size=300,
        min_frequency=2,
    )

    assert output.is_file()
    assert tokenizer.vocab_size <= 300


def test_trained_tokenizer_can_be_reloaded(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    output = tmp_path / "tokenizer.json"

    text = "Merhaba dünya. Vebra bir dil modelidir."

    corpus.write_text(
        text * 10,
        encoding="utf-8",
    )

    train_tokenizer(
        corpus_path=corpus,
        output_path=output,
        vocab_size=300,
        min_frequency=2,
    )

    tokenizer = BPETokenizer.load(output)

    assert tokenizer.decode(
        tokenizer.encode(text)
    ) == text


def test_train_tokenizer_rejects_empty_corpus(tmp_path: Path):
    corpus = tmp_path / "empty.txt"
    output = tmp_path / "tokenizer.json"

    corpus.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty"):
        train_tokenizer(
            corpus_path=corpus,
            output_path=output,
            vocab_size=300,
        )