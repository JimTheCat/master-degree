"""Polish display names for labels, used by chart axes and report tables."""

from collections.abc import Iterable

from src.schema.labels import get_label_info


def pl_name(label: str) -> str:
    """Polish display name for a label id, falling back to the id itself."""
    try:
        return get_label_info(label).display_name_pl
    except ValueError:
        # Non-label columns (party, period, ...) pass through unchanged so this
        # can be used directly as a DataFrame rename mapper.
        return label


def pl_names(labels: Iterable[str]) -> list[str]:
    return [pl_name(label) for label in labels]
