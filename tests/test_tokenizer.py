import pytest

from vebra.tokenizer.byte import ByteTokenizer


def create_tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


def test_encode_returns_byte_ids():
    tokenizer = create_tokenizer()

    token_ids = tokenizer.encode("Vebra")

    assert token_ids == list(b"Vebra")


def test_decode_returns_original_text():
    tokenizer = create_tokenizer()

    text = "Vebra language model"

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_tokenizer_supports_turkish_text():
    tokenizer = create_tokenizer()

    text = "Merhaba dünya, şğüİıöç."

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text


def test_tokenizer_vocab_size():
    tokenizer = create_tokenizer()

    assert tokenizer.vocab_size == 256


def test_encode_rejects_non_string():
    tokenizer = create_tokenizer()

    with pytest.raises(TypeError, match="string"):
        tokenizer.encode(123)


def test_decode_rejects_invalid_byte():
    tokenizer = create_tokenizer()

    with pytest.raises(ValueError, match="range"):
        tokenizer.decode([256])


def test_decode_rejects_invalid_utf8():
    tokenizer = create_tokenizer()

    with pytest.raises(ValueError, match="UTF-8"):
        tokenizer.decode([0xFF])


def test_empty_string_round_trip():
    tokenizer = create_tokenizer()

    token_ids = tokenizer.encode("")
    decoded = tokenizer.decode(token_ids)

    assert token_ids == []
    assert decoded == ""