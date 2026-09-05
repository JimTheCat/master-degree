import json
from pathlib import Path
from typing import Iterator

from src.schema.models import AnnotatedSpeech, GoldStandardEntry, Speech


def load_speeches_jsonl(path: str | Path) -> list[GoldStandardEntry]:
    """Load speeches from a JSONL file."""
    path = Path(path)
    entries = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entries.append(GoldStandardEntry.model_validate(data))
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {path}: {e}") from e
    return entries


def load_gold_standard(path: str | Path) -> list[AnnotatedSpeech]:
    """Load gold standard dataset as AnnotatedSpeech objects."""
    entries = load_speeches_jsonl(path)
    return [entry.to_annotated_speech() for entry in entries]


def load_corpus(path: str | Path) -> Iterator[Speech]:
    """Iterate over corpus speeches from a JSONL file or directory of shards.

    For large corpora, yields one speech at a time to avoid loading everything
    into memory.
    """
    path = Path(path)

    if path.is_dir():
        shard_files = sorted(path.glob("*.jsonl"))
        if not shard_files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
    else:
        shard_files = [path]

    for shard_path in shard_files:
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                entry = GoldStandardEntry.model_validate(data)
                yield entry.to_speech()


def load_split(path: str | Path) -> list[AnnotatedSpeech]:
    """Load a train/val/test split file."""
    return load_gold_standard(path)
