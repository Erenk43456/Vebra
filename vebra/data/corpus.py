from pathlib import Path


def read_text_corpus(path: str | Path) -> str:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"corpus file not found: {path}"
        )

    text = path.read_text(encoding="utf-8")

    if not text:
        raise ValueError(
            "corpus must not be empty"
        )

    return text