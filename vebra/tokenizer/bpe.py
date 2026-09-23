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

        vocab: dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

        merges: list[tuple[int, int]] = []
        next_token_id = 256

        # Mutable linked representation of every sequence.
        #
        # Each node contains:
        #   token -> token ID
        #   prev  -> previous node index
        #   next  -> next node index
        #   alive -> whether the node is still active
        nodes: list[list[dict[str, int | bool]]] = []

        for sequence in sequences:
            sequence_nodes: list[dict[str, int | bool]] = []

            for index, token in enumerate(sequence):
                sequence_nodes.append(
                    {
                        "token": token,
                        "prev": index - 1,
                        "next": (
                            index + 1
                            if index + 1 < len(sequence)
                            else -1
                        ),
                        "alive": True,
                    }
                )

            nodes.append(sequence_nodes)

        # Current number of occurrences of every adjacent token pair.
        pair_counts: Counter[tuple[int, int]] = Counter()

        # pair -> {(sequence_index, left_node_index), ...}
        #
        # An occurrence identifies the node containing the left token
        # of the pair.
        pair_occurrences: dict[
            tuple[int, int],
            set[tuple[int, int]],
        ] = {}

        for sequence_index, sequence_nodes in enumerate(nodes):
            for index in range(len(sequence_nodes) - 1):
                left = int(sequence_nodes[index]["token"])
                right = int(sequence_nodes[index + 1]["token"])

                pair = (left, right)

                pair_counts[pair] += 1

                pair_occurrences.setdefault(pair, set()).add(
                    (sequence_index, index)
                )

        def remove_occurrence(
            pair: tuple[int, int],
            occurrence: tuple[int, int],
        ) -> None:
            occurrences = pair_occurrences.get(pair)

            if occurrences is None:
                return

            occurrences.discard(occurrence)

            if not occurrences:
                pair_occurrences.pop(pair, None)

        def add_occurrence(
            pair: tuple[int, int],
            occurrence: tuple[int, int],
        ) -> None:
            pair_occurrences.setdefault(pair, set()).add(
                occurrence
            )

        def remove_pair_occurrence(
            sequence_index: int,
            left_index: int,
        ) -> None:
            sequence_nodes = nodes[sequence_index]
            left_node = sequence_nodes[left_index]

            if not bool(left_node["alive"]):
                return

            right_index = int(left_node["next"])

            if right_index == -1:
                return

            right_node = sequence_nodes[right_index]

            if not bool(right_node["alive"]):
                return

            pair = (
                int(left_node["token"]),
                int(right_node["token"]),
            )

            count = pair_counts[pair]

            if count <= 0:
                raise RuntimeError(
                    "internal BPE pair count underflow"
                )

            pair_counts[pair] = count - 1

            remove_occurrence(
                pair,
                (sequence_index, left_index),
            )

        def add_pair_occurrence(
            sequence_index: int,
            left_index: int,
        ) -> None:
            sequence_nodes = nodes[sequence_index]
            left_node = sequence_nodes[left_index]

            if not bool(left_node["alive"]):
                return

            right_index = int(left_node["next"])

            if right_index == -1:
                return

            right_node = sequence_nodes[right_index]

            if not bool(right_node["alive"]):
                return

            pair = (
                int(left_node["token"]),
                int(right_node["token"]),
            )

            pair_counts[pair] += 1

            add_occurrence(
                pair,
                (sequence_index, left_index),
            )

        while len(vocab) < self.vocab_size:
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

            left_token, right_token = best_pair

            if left_token not in vocab or right_token not in vocab:
                raise ValueError(
                    "pair contains unknown token"
                )

            vocab[next_token_id] = (
                vocab[left_token] + vocab[right_token]
            )

            merges.append(best_pair)

            occurrences = pair_occurrences.get(best_pair)

            if not occurrences:
                raise RuntimeError(
                    "best BPE pair has no occurrences"
                )

            # Work from left to right to preserve the exact
            # non-overlapping semantics of merge_pair().
            selected: list[tuple[int, int]] = []

            last_consumed_right: dict[int, int] = {}

            for sequence_index, left_index in sorted(
                occurrences
            ):
                previous_right = last_consumed_right.get(
                    sequence_index,
                    -1,
                )

                # If this node was consumed as the right side of
                # the preceding occurrence, this occurrence overlaps
                # and must be skipped.
                if left_index <= previous_right:
                    continue

                sequence_nodes = nodes[sequence_index]
                left_node = sequence_nodes[left_index]

                if not bool(left_node["alive"]):
                    continue

                right_index = int(left_node["next"])

                if right_index == -1:
                    continue

                right_node = sequence_nodes[right_index]

                if not bool(right_node["alive"]):
                    continue

                if (
                    int(left_node["token"]),
                    int(right_node["token"]),
                ) != best_pair:
                    continue

                selected.append(
                    (sequence_index, left_index)
                )

                last_consumed_right[sequence_index] = right_index

            for sequence_index, left_index in selected:
                sequence_nodes = nodes[sequence_index]
                left_node = sequence_nodes[left_index]

                if not bool(left_node["alive"]):
                    continue

                right_index = int(left_node["next"])

                if right_index == -1:
                    continue

                right_node = sequence_nodes[right_index]

                if not bool(right_node["alive"]):
                    continue

                if (
                    int(left_node["token"]),
                    int(right_node["token"]),
                ) != best_pair:
                    continue

                previous_index = int(left_node["prev"])
                next_index = int(right_node["next"])

                # Remove all pair occurrences affected by the
                # structural change.
                if previous_index != -1:
                    remove_pair_occurrence(
                        sequence_index,
                        previous_index,
                    )

                remove_pair_occurrence(
                    sequence_index,
                    left_index,
                )

                if next_index != -1:
                    remove_pair_occurrence(
                        sequence_index,
                        right_index,
                    )

                # Replace left + right with the new token.
                left_node["token"] = next_token_id

                if next_index != -1:
                    left_node["next"] = next_index
                    sequence_nodes[next_index]["prev"] = left_index
                else:
                    left_node["next"] = -1

                right_node["alive"] = False
                right_node["prev"] = -1
                right_node["next"] = -1

                # Add the newly created neighboring pairs.
                if previous_index != -1:
                    add_pair_occurrence(
                        sequence_index,
                        previous_index,
                    )

                add_pair_occurrence(
                    sequence_index,
                    left_index,
                )

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