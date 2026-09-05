"""ParlaMint-PL corpus TSV → structured records.

Input format (TSV, optional leading index column):
    [idx<TAB>]<ParlaMint-PL_YYYY-MM-DD-{sejm|senat}-NN-N.uXXX><TAB><text>

Speech ID parsed into: date, chamber, sitting, segment, utterance.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

# ParlaMint-PL_2015-12-09-sejm-04-1.u40
PARLAMINT_ID_RE = re.compile(
    r"^ParlaMint-PL_(?P<date>\d{4}-\d{2}-\d{2})"
    r"-(?P<chamber>sejm|senat)"
    r"-(?P<sitting>\d+)"
    r"-(?P<segment>\d+)"
    r"\.u(?P<utterance>\d+)$"
)


def parse_parlamint_id(speech_id: str) -> dict | None:
    """Extract structured metadata from a ParlaMint speech ID. Returns None if invalid."""
    m = PARLAMINT_ID_RE.match(speech_id.strip())
    if not m:
        return None
    return {
        "speech_id": speech_id.strip(),
        "date": m.group("date"),
        "chamber": m.group("chamber"),
        "sitting": int(m.group("sitting")),
        "segment": int(m.group("segment")),
        "utterance": int(m.group("utterance")),
    }


def parse_parlamint_tsv(
    path: str | Path,
    min_text_length: int = 20,
) -> Iterator[dict]:
    """Yield parsed records from a ParlaMint TSV file.

    Each line may have 2 cols (id<TAB>text) or 3 cols (idx<TAB>id<TAB>text).
    Records with unparseable IDs or text shorter than min_text_length are skipped.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                speech_id, text = parts[0], parts[1]
            elif len(parts) >= 3:
                # First column is a row index (e.g. "0", "1", ...). Drop it.
                speech_id, text = parts[1], "\t".join(parts[2:])
            else:
                continue

            parsed = parse_parlamint_id(speech_id)
            if parsed is None:
                continue
            text = text.strip()
            if len(text) < min_text_length:
                continue

            yield {
                **parsed,
                "text": text,
                "source_file": path.name,
                "source_line": line_num,
            }


def load_speaker_metadata(path: str | Path) -> dict[str, dict]:
    """Load speaker → party mapping from a TSV/CSV file.

    Expected columns: speech_id, speaker, party, political_group (optional).
    Format auto-detected by extension. Empty file path → empty mapping.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}

    import csv
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    mapping: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            sid = (row.get("speech_id") or "").strip()
            if not sid:
                continue
            mapping[sid] = {
                "speaker": (row.get("speaker") or "unknown").strip(),
                "party": (row.get("party") or "unknown").strip(),
                "political_group": (row.get("political_group") or "").strip() or None,
            }
    return mapping


def to_corpus_record(
    parsed: dict,
    speaker_metadata: Optional[dict] = None,
) -> dict:
    """Build a JSONL-ready record matching GoldStandardEntry schema."""
    speech_id = parsed["speech_id"]
    meta = (speaker_metadata or {}).get(speech_id, {})
    return {
        "speech_id": speech_id,
        "text": parsed["text"],
        "speaker": meta.get("speaker", "unknown"),
        "party": meta.get("party", "unknown"),
        "political_group": meta.get("political_group"),
        "date": parsed["date"],
        "session_id": f"{parsed['chamber']}-{parsed['sitting']}-{parsed['segment']}",
        "metadata": {
            "source": "parlaMint",
            "chamber": parsed["chamber"],
            "sitting": parsed["sitting"],
            "segment": parsed["segment"],
            "utterance": parsed["utterance"],
            "needs_metadata": speech_id not in (speaker_metadata or {}),
        },
    }
