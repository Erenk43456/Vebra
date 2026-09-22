import pytest

from vebra.data.corpus import read_text_corpus


def test_read_text_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.txt"

    corpus_path.write_text(
        "Vebra training corpus.",
        encoding="utf-8",
    )

    text = read_text_corpus(corpus_path)

    assert text == "Vebra training corpus."


def test_read_text_corpus_preserves_utf8(tmp_path):
    corpus_path = tmp_path / "corpus.txt"

    corpus_path.write_text(
        "Vebra Türkçe bir dil modelidir.",
        encoding="utf-8",
    )

    text = read_text_corpus(corpus_path)

    assert text == "Vebra Türkçe bir dil modelidir."


def test_read_text_corpus_rejects_missing_file(tmp_path):
    corpus_path = tmp_path / "missing.txt"

    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        read_text_corpus(corpus_path)


def test_read_text_corpus_rejects_empty_file(tmp_path):
    corpus_path = tmp_path / "empty.txt"

    corpus_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        read_text_corpus(corpus_path)