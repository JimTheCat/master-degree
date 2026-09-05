from pathlib import Path

from src.data.loader import load_speeches_jsonl
from src.schema.labels import ALL_LABELS


def validate_gold_standard(path: str | Path) -> dict:
    """Validate a gold standard JSONL file and return statistics.

    Returns a dict with:
        - total: number of entries
        - valid: number of valid entries
        - errors: list of error messages
        - label_distribution: count per label
        - speakers: unique speaker count
        - parties: unique party count
    """
    path = Path(path)
    errors = []
    label_counts: dict[str, int] = {label: 0 for label in ALL_LABELS}
    speakers: set[str] = set()
    parties: set[str] = set()
    total = 0
    valid = 0

    entries = load_speeches_jsonl(path)
    total = len(entries)

    for entry in entries:
        speakers.add(entry.speaker)
        parties.add(entry.party)

        if not entry.text.strip():
            errors.append(f"Speech {entry.speech_id}: empty text")
            continue

        if entry.annotations is None:
            errors.append(f"Speech {entry.speech_id}: missing annotations")
            continue

        for label in entry.annotations.labels:
            if label in label_counts:
                label_counts[label] += 1

        valid += 1

    return {
        "total": total,
        "valid": valid,
        "errors": errors,
        "label_distribution": label_counts,
        "speakers": len(speakers),
        "parties": len(parties),
    }
