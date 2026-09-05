"""Few-shot example selection strategies."""

import random

from src.schema.labels import ALL_LABELS
from src.schema.models import AnnotatedSpeech


def select_examples_random(
    pool: list[AnnotatedSpeech],
    n: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Select random examples from the pool."""
    rng = random.Random(seed)
    selected = rng.sample(pool, min(n, len(pool)))
    return [
        {
            "text": s.speech.text,
            "labels": s.gold.labels if s.gold else [],
        }
        for s in selected
    ]


def select_examples_stratified(
    pool: list[AnnotatedSpeech],
    n: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Select examples that cover as many different labels as possible.

    Greedy approach: iteratively pick the example that adds the most
    not-yet-covered labels.
    """
    rng = random.Random(seed)

    # Shuffle to randomize tie-breaking
    candidates = list(pool)
    rng.shuffle(candidates)

    selected = []
    covered_labels: set[str] = set()

    for _ in range(min(n, len(candidates))):
        best_candidate = None
        best_new_labels = -1

        for candidate in candidates:
            if candidate in selected:
                continue
            if not candidate.gold:
                continue
            new_labels = len(set(candidate.gold.labels) - covered_labels)
            if new_labels > best_new_labels:
                best_new_labels = new_labels
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        if best_candidate.gold:
            covered_labels.update(best_candidate.gold.labels)

    return [
        {
            "text": s.speech.text,
            "labels": s.gold.labels if s.gold else [],
        }
        for s in selected
    ]


def select_examples(
    pool: list[AnnotatedSpeech],
    strategy: str = "stratified",
    n: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Select few-shot examples using the specified strategy."""
    if strategy == "stratified":
        return select_examples_stratified(pool, n, seed)
    elif strategy == "random":
        return select_examples_random(pool, n, seed)
    else:
        raise ValueError(f"Unknown example selection strategy: {strategy}")
