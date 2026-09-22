import base64
import json
from collections import Counter
from pathlib import Path


class BPETrainer:
    def __init__(
        self,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        if vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256 for byte-level BPE"
            )

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

    @staticmethod
    def count_pairs(
        sequences: list[list[int]],
    ) -> Counter[tuple[int, int]]:
        pair_counts: Counter[tuple[int, int]] = Counter()

        for sequence in sequences:
            for left, right in zip(sequence, sequence[1:]):
                pair_counts[(left, right)] += 1

        return pair_counts

    @staticmethod
    def merge_pair(
        sequence: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        merged: list[int] = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):
                merged.append(new_token_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1

        return merged

    def train(
        self,
        sequences: list[list[int]],
    ) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
        if not sequences:
            raise ValueError("sequences must not be empty")

        working_sequences = [
            list(sequence)
            for sequence in sequences
        ]

        vocab: dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

        merges: list[tuple[int, int]] = []
        next_token_id = 256

        while len(vocab) < self.vocab_size:
            pair_counts = self.count_pairs(working_sequences)

            if not pair_counts:
                break

            candidates = [
                (pair, count)
                for pair, count in pair_counts.items()
                if count >= self.min_frequency
            ]

            if not candidates:
                break

            best_pair, _ = max(
                candidates,
                key=lambda item: (
                    item[1],
                    -item[0][0],
                    -item[0][1],
                ),
            )

            left, right = best_pair

            if left not in vocab or right not in vocab:
                raise ValueError(
                    "pair contains unknown token"
                )

            vocab[next_token_id] = (
                vocab[left] + vocab[right]
            )

            merges.append(best_pair)

            working_sequences = [
                self.merge_pair(
                    sequence,
                    best_pair,
                    next_token_id,
                )
                for sequence in working_sequences
            ]

            next_token_id += 1

        return vocab, merges

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
    ) -> None:
        if not vocab:
            raise ValueError("vocab must not be empty")

        if not merges and len(vocab) > 256:
            raise ValueError(
                "extended vocabulary requires merge rules"
            )

        self.vocab = dict(vocab)
        self.merges = list(merges)

        self._merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }

        self._token_bytes = dict(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text:
            return []

        sequence = list(text.encode("utf-8"))

        for pair in self.merges:
            new_token_id = self._find_token_id(pair)

            if new_token_id is None:
                raise ValueError(
                    "merge rule has no corresponding vocabulary token"
                )

            sequence = BPETrainer.merge_pair(
                sequence,
                pair,
                new_token_id,
            )

        return sequence

    def decode(self, token_ids: list[int]) -> str:
        try:
            byte_sequence = b"".join(
                self._token_bytes[token_id]
                for token_id in token_ids
            )
        except KeyError as error:
            raise ValueError(
                f"unknown token ID: {error.args[0]}"
            ) from error

        try:
            return byte_sequence.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "token IDs do not form valid UTF-8"
            ) from error

    def _find_token_id(
        self,
        pair: tuple[int, int],
    ) -> int | None:
        for token_id, token_bytes in self.vocab.items():
            if token_id < 256:
                continue

            left, right = pair

            if (
                self._token_bytes.get(left) is not None
                and self._token_bytes.get(right) is not None
                and token_bytes
                == self._token_bytes[left] + self._token_bytes[right]
            ):
                return token_id

        return None

    def save(self, path: str | Path) -> None:
        path = Path(path)

        data = {
            "version": 1,
            "vocab": {
                str(token_id): base64.b64encode(token_bytes).decode("ascii")
                for token_id, token_bytes in self.vocab.items()
            },
            "merges": [
                [left, right]
                for left, right in self.merges
            ],
        }

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if data.get("version") != 1:
            raise ValueError("unsupported tokenizer version")

        raw_vocab = data.get("vocab")
        raw_merges = data.get("merges")

        if not isinstance(raw_vocab, dict):
            raise ValueError("invalid tokenizer vocabulary")

        if not isinstance(raw_merges, list):
            raise ValueError("invalid tokenizer merges")

        vocab = {
            int(token_id): base64.b64decode(token_bytes)
            for token_id, token_bytes in raw_vocab.items()
        }

        merges = [
            (int(left), int(right))
            for left, right in raw_merges
        ]

        return cls(
            vocab=vocab,
            merges=merges,
        )