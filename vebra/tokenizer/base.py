from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError