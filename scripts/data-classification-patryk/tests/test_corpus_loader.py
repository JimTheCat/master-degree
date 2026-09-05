"""Tests for ParlaMint corpus loader."""

from pathlib import Path

import pytest

from src.data.corpus_loader import (
    parse_parlamint_id,
    parse_parlamint_tsv,
    to_corpus_record,
)


def test_parse_parlamint_id_sejm():
    parsed = parse_parlamint_id("ParlaMint-PL_2015-12-09-sejm-04-1.u40")
    assert parsed == {
        "speech_id": "ParlaMint-PL_2015-12-09-sejm-04-1.u40",
        "date": "2015-12-09",
        "chamber": "sejm",
        "sitting": 4,
        "segment": 1,
        "utterance": 40,
    }


def test_parse_parlamint_id_senat():
    parsed = parse_parlamint_id("ParlaMint-PL_2020-08-13-senat-14-3.u115")
    assert parsed["chamber"] == "senat"
    assert parsed["sitting"] == 14
    assert parsed["segment"] == 3
    assert parsed["utterance"] == 115


def test_parse_parlamint_id_invalid():
    assert parse_parlamint_id("not-a-parlamint-id") is None
    assert parse_parlamint_id("") is None
    assert parse_parlamint_id("ParlaMint-PL_2020-08-13-bundestag-14-3.u115") is None


def test_parse_parlamint_tsv_three_columns(tmp_path: Path):
    f = tmp_path / "sample.tsv"
    f.write_text(
        "0\tParlaMint-PL_2015-12-09-sejm-04-1.u40\tDługa wypowiedź na temat ustawy emerytalnej i jej znaczenia.\n"
        "1\tParlaMint-PL_2017-11-23-sejm-52-2.u526\tKolejna długa wypowiedź dotycząca ordynacji wyborczej.\n",
        encoding="utf-8",
    )
    records = list(parse_parlamint_tsv(f))
    assert len(records) == 2
    assert records[0]["speech_id"] == "ParlaMint-PL_2015-12-09-sejm-04-1.u40"
    assert records[0]["chamber"] == "sejm"
    assert "emerytalnej" in records[0]["text"]


def test_parse_parlamint_tsv_two_columns(tmp_path: Path):
    f = tmp_path / "two.tsv"
    f.write_text(
        "ParlaMint-PL_2020-08-13-senat-14-3.u115\tWypowiedź senacka wystarczająco długa do przepuszczenia przez próg.\n",
        encoding="utf-8",
    )
    records = list(parse_parlamint_tsv(f))
    assert len(records) == 1
    assert records[0]["chamber"] == "senat"


def test_parse_parlamint_tsv_skips_short_and_invalid(tmp_path: Path):
    f = tmp_path / "mixed.tsv"
    f.write_text(
        "0\tParlaMint-PL_2015-12-09-sejm-04-1.u40\tx\n"  # too short
        "1\tnot-a-valid-id\tThis text is long enough but the id is wrong.\n"  # invalid id
        "2\tParlaMint-PL_2017-11-23-sejm-52-2.u526\tValid record with sufficient text length here.\n",
        encoding="utf-8",
    )
    records = list(parse_parlamint_tsv(f, min_text_length=20))
    assert len(records) == 1
    assert records[0]["speech_id"] == "ParlaMint-PL_2017-11-23-sejm-52-2.u526"


def test_to_corpus_record_no_metadata():
    parsed = parse_parlamint_id("ParlaMint-PL_2015-12-09-sejm-04-1.u40")
    parsed["text"] = "Treść wypowiedzi."
    record = to_corpus_record(parsed)
    assert record["speech_id"] == "ParlaMint-PL_2015-12-09-sejm-04-1.u40"
    assert record["speaker"] == "unknown"
    assert record["party"] == "unknown"
    assert record["date"] == "2015-12-09"
    assert record["session_id"] == "sejm-4-1"
    assert record["metadata"]["chamber"] == "sejm"
    assert record["metadata"]["needs_metadata"] is True


def test_to_corpus_record_with_metadata():
    parsed = parse_parlamint_id("ParlaMint-PL_2015-12-09-sejm-04-1.u40")
    parsed["text"] = "Treść."
    meta = {
        "ParlaMint-PL_2015-12-09-sejm-04-1.u40": {
            "speaker": "Jan Kowalski",
            "party": "PiS",
            "political_group": "Zjednoczona Prawica",
        }
    }
    record = to_corpus_record(parsed, meta)
    assert record["speaker"] == "Jan Kowalski"
    assert record["party"] == "PiS"
    assert record["political_group"] == "Zjednoczona Prawica"
    assert record["metadata"]["needs_metadata"] is False
