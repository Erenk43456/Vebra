from vebra.tokenizer.base import Tokenizer


class ByteTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        if not all(0 <= token_id <= 255 for token_id in token_ids):
            raise ValueError("byte token IDs must be in the range [0, 255]")

        try:
            return bytes(token_ids).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("token IDs do not form valid UTF-8") from error

    @property
    def vocab_size(self) -> int:
        return 256