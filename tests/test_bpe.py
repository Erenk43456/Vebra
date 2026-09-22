from pathlib import Path
from vebra.tokenizer.bpe import BPETrainer
from vebra.tokenizer.bpe import BPETrainer, BPETokenizer


def test_count_pairs():
    sequences = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
    ]

    counts = BPETrainer.count_pairs(sequences)

    assert counts[(1, 2)] == 3
    assert counts[(2, 3)] == 2
    assert counts[(2, 4)] == 1


def test_count_pairs_handles_empty_sequence():
    counts = BPETrainer.count_pairs(
        [
            [],
            [1],
        ]
    )

    assert len(counts) == 0


def test_merge_pair():
    sequence = [1, 2, 1, 2, 3]

    merged = BPETrainer.merge_pair(
        sequence,
        pair=(1, 2),
        new_token_id=10,
    )

    assert merged == [10, 10, 3]


def test_merge_pair_merges_non_overlapping_pairs():
    sequence = [1, 2, 2, 1, 2]

    merged = BPETrainer.merge_pair(
        sequence,
        pair=(1, 2),
        new_token_id=10,
    )

    assert merged == [10, 2, 10]


def test_merge_pair_does_not_modify_original():
    sequence = [1, 2, 3]

    BPETrainer.merge_pair(
        sequence,
        pair=(1, 2),
        new_token_id=10,
    )

    assert sequence == [1, 2, 3]


def test_trainer_validates_vocab_size():
    try:
        BPETrainer(vocab_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_trainer_validates_min_frequency():
    try:
        BPETrainer(
            vocab_size=300,
            min_frequency=0,
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected ValueError")

def test_bpe_training_creates_new_tokens():
    trainer = BPETrainer(
        vocab_size=260,
        min_frequency=2,
    )

    sequences = [
        list(b"abababab"),
        list(b"abab"),
    ]

    vocab, merges = trainer.train(sequences)

    assert len(vocab) > 256
    assert len(merges) > 0


def test_bpe_training_respects_vocab_size():
    trainer = BPETrainer(
        vocab_size=260,
        min_frequency=1,
    )

    sequences = [
        list(b"abcdefghijklmnopqrstuvwxyz"),
    ]

    vocab, merges = trainer.train(sequences)

    assert len(vocab) <= 260
    assert len(vocab) == 260


def test_bpe_training_stops_when_frequency_is_too_low():
    trainer = BPETrainer(
        vocab_size=300,
        min_frequency=100,
    )

    sequences = [
        list(b"hello"),
    ]

    vocab, merges = trainer.train(sequences)

    assert len(vocab) == 256
    assert merges == []


def test_bpe_training_returns_byte_vocabulary():
    trainer = BPETrainer(
        vocab_size=260,
        min_frequency=2,
    )

    sequences = [
        list(b"abababab"),
    ]

    vocab, _ = trainer.train(sequences)

    for token_id in range(256):
        assert vocab[token_id] == bytes([token_id])


def test_bpe_training_is_deterministic():
    trainer = BPETrainer(
        vocab_size=264,
        min_frequency=2,
    )

    sequences = [
        list(b"abababab"),
        list(b"abab"),
    ]

    vocab_a, merges_a = trainer.train(sequences)
    vocab_b, merges_b = trainer.train(sequences)

    assert vocab_a == vocab_b
    assert merges_a == merges_b


def test_bpe_training_rejects_empty_sequences():
    trainer = BPETrainer(
        vocab_size=300,
    )

    try:
        trainer.train([])
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("expected ValueError")

def create_trained_tokenizer() -> BPETokenizer:
    trainer = BPETrainer(
        vocab_size=264,
        min_frequency=2,
    )

    sequences = [
        list(b"abababab"),
        list(b"abababab"),
        list(b"abab"),
    ]

    vocab, merges = trainer.train(sequences)

    return BPETokenizer(
        vocab=vocab,
        merges=merges,
    )


def test_bpe_tokenizer_vocab_size():
    tokenizer = create_trained_tokenizer()

    assert tokenizer.vocab_size >= 256
    assert tokenizer.vocab_size <= 264
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges)


def test_bpe_tokenizer_encode_returns_token_ids():
    tokenizer = create_trained_tokenizer()

    token_ids = tokenizer.encode("abababab")

    assert token_ids
    assert all(
        isinstance(token_id, int)
        for token_id in token_ids
    )


def test_bpe_tokenizer_round_trip():
    tokenizer = create_trained_tokenizer()

    text = "abababab"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_bpe_tokenizer_supports_utf8():
    trainer = BPETrainer(
        vocab_size=256,
        min_frequency=2,
    )

    vocab, merges = trainer.train(
        [
            list("Merhaba dünya".encode("utf-8")),
        ]
    )

    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
    )

    text = "Merhaba dünya"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_empty_string():
    tokenizer = create_trained_tokenizer()

    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_bpe_tokenizer_rejects_unknown_token():
    tokenizer = create_trained_tokenizer()

    try:
        tokenizer.decode([999999])
    except ValueError as error:
        assert "unknown token" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_bpe_tokenizer_rejects_non_string_input():
    tokenizer = create_trained_tokenizer()

    try:
        tokenizer.encode(123)
    except TypeError as error:
        assert "string" in str(error)
    else:
        raise AssertionError("expected TypeError")

def test_bpe_tokenizer_can_save_and_load(tmp_path: Path):
    tokenizer = create_trained_tokenizer()

    path = tmp_path / "tokenizer.json"

    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges
    assert loaded.vocab_size == tokenizer.vocab_size


def test_saved_tokenizer_preserves_round_trip(tmp_path: Path):
    tokenizer = create_trained_tokenizer()

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    text = "abababab"

    original_ids = tokenizer.encode(text)
    loaded_ids = loaded.encode(text)

    assert loaded_ids == original_ids
    assert loaded.decode(loaded_ids) == text


def test_saved_tokenizer_supports_turkish_text(tmp_path: Path):
    tokenizer = create_trained_tokenizer()

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    text = "Merhaba dünya, şğüİıöç."

    assert loaded.decode(loaded.encode(text)) == text


def test_load_rejects_unsupported_version(tmp_path: Path):
    path = tmp_path / "tokenizer.json"

    path.write_text(
        '{"version": 999, "vocab": {}, "merges": []}',
        encoding="utf-8",
    )

    try:
        BPETokenizer.load(path)
    except ValueError as error:
        assert "version" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_load_rejects_invalid_vocab(tmp_path: Path):
    path = tmp_path / "tokenizer.json"

    path.write_text(
        '{"version": 1, "vocab": [], "merges": []}',
        encoding="utf-8",
    )

    try:
        BPETokenizer.load(path)
    except ValueError as error:
        assert "vocabulary" in str(error)
    else:
        raise AssertionError("expected ValueError")