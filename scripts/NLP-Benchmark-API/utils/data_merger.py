#!/usr/bin/env python3
"""
Rekurencyjny merger plików ParlaMint-PL:
- skanuje katalog wejściowy
- scala wszystkie pliki transkrypcji (id \t text) w jeden plik
- scala metadane po id w dwóch wersjach językowych (polska / angielska)
Użycie:
    python data_merger.py <root_dir> <out_transcripts.tsv> <out_meta_pl.tsv> <out_meta_en.tsv>
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
import logging
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

POLISH_DIACRITICS = set("ąćęłńóśżźĄĆĘŁŃÓŚŻŹ")
ENGLISH_KEYWORDS = {"speaker", "party", "age", "gender", "role", "id", "date"}
POLISH_KEYWORDS = {"mówca", "mowca", "partia", "wiek", "płeć", "plec", "id", "data"}


def read_sample_lines(path: str, max_lines: int = 20) -> List[str]:
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        return []
    return lines


def looks_like_transcript(path: str) -> bool:
    sample = read_sample_lines(path)
    if not sample:
        return False
    count_candidates = 0
    for line in sample:
        if "\t" not in line:
            continue
        cols = line.split("\t")
        if len(cols) >= 2:
            second = cols[1].strip()
            # transkrypcja: druga kolumna zawiera zdania (ma spacje) i jest dłuższa
            if " " in second and len(second) >= 10:
                count_candidates += 1
    return count_candidates >= max(1, len(sample) // 3)


def looks_like_metadata(path: str) -> bool:
    sample = read_sample_lines(path, max_lines=2)
    if not sample:
        return False
    header = sample[0]
    if "\t" not in header:
        return False
    # jeśli pierwsza linia zawiera nazwy kolumn -> metadata
    tokens = [t.strip().lower() for t in header.split("\t") if t.strip()]
    # heurystyka: obecność znanych słów kluczowych lub znaków diakrytycznych sugeruje metadane
    if any(tok in POLISH_KEYWORDS for tok in tokens) or any(tok in ENGLISH_KEYWORDS for tok in tokens):
        return True
    if any(ch in POLISH_DIACRITICS for ch in header):
        return True
    # fallback: jeśli nagłówek ma nienumeryczne tokeny i wiele kolumn
    if len(tokens) >= 2:
        return True
    return False


def detect_metadata_language(header_line: str) -> str:
    tokens = [t.strip().lower() for t in header_line.split("\t")]
    if any(any(ch in POLISH_DIACRITICS for ch in tok) for tok in tokens):
        return "pl"
    if any(tok in POLISH_KEYWORDS for tok in tokens):
        return "pl"
    if any(tok in ENGLISH_KEYWORDS for tok in tokens):
        return "en"
    # fallback na podstawie ascii vs non-ascii
    nonascii = sum(1 for ch in header_line if ord(ch) > 127)
    return "pl" if nonascii else "en"


def find_id_col(header: List[str]) -> int:
    # heurystyka: szukaj kolumny o nazwie id lub podobnej (case-insensitive)
    low = [h.strip().lower() for h in header]
    candidates = ["id", "speech_id", "speechid", "utterance_id", "utteranceid"]
    for cand in candidates:
        if cand in low:
            return low.index(cand)
    # fallback: pierwsza kolumna
    return 0


def normalize_header(header: List[str]) -> Tuple[List[str], Dict[str, str]]:
    # zwraca listę canonical names (lower, stripped) oraz mapę canonical->display (pierwsze wystąpienie)
    canonical = []
    display_map: Dict[str, str] = {}
    for h in header:
        disp = h.strip()
        cname = disp.lower()
        canonical.append(cname)
        if cname not in display_map:
            display_map[cname] = disp if disp else cname
    return canonical, display_map


def merge_transcripts_and_metadata(root_dir: str,
                                   out_transcripts: str,
                                   out_meta_pl: str,
                                   out_meta_en: str) -> None:
    transcript_seen: Set[str] = set()
    transcripts: List[Tuple[str, str]] = []
    meta_pl_rows: Dict[str, Dict[str, str]] = {}
    meta_en_rows: Dict[str, Dict[str, str]] = {}
    meta_pl_cols: Set[str] = set()
    meta_en_cols: Set[str] = set()
    meta_headers_seen = {"pl": None, "en": None}
    # map canonical -> display name per language
    meta_display = {"pl": {}, "en": {}}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            lname = fname.lower()
            if not (lname.endswith(".txt") or lname.endswith(".tsv")):
                # próbuj jednak na bazie zawartości, bo rozszerzenia mogą być różne
                pass
            # attempt classification
            try:
                # najpierw heurystyka wg nazwy pliku: "-meta-en.tsv" -> en metadata,
                # "-meta.tsv" (ale nie "-meta-en.tsv") -> pl metadata
                explicit_lang = None
                if lname.endswith("-meta-en.tsv"):
                    explicit_lang = "en"
                elif lname.endswith("-meta.tsv"):
                    explicit_lang = "pl"

                if explicit_lang:
                    # traktuj plik jako metadane zgodnie z nazwą pliku (pomijamy dalsze wykrywanie)
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        reader = csv.reader(fh, delimiter="\t")
                        try:
                            header = next(reader)
                        except StopIteration:
                            continue
                        header = [h.rstrip("\ufeff") for h in header]  # usuń BOM z pierwszego nagłówka
                        canonical_hdr, display_map = normalize_header(header)
                        lang = explicit_lang
                        if meta_headers_seen.get(lang) is None:
                            meta_headers_seen[lang] = canonical_hdr.copy()
                        # zapamiętaj display names (nie nadpisuj istniejących)
                        for k, v in display_map.items():
                            if k not in meta_display[lang]:
                                meta_display[lang][k] = v
                        id_col = find_id_col(header)
                        for r in reader:
                            if not r:
                                continue
                            if len(r) < len(header):
                                r = r + [""] * (len(header) - len(r))
                            # zmapuj wartości na canonical keys
                            row_canonical: Dict[str, str] = {}
                            for i, val in enumerate(r[:len(header)]):
                                key = canonical_hdr[i]
                                row_canonical[key] = val.strip()
                            sid = r[id_col].strip()
                            if not sid:
                                continue
                            target_rows = meta_pl_rows if lang == "pl" else meta_en_rows
                            target_cols = meta_pl_cols if lang == "pl" else meta_en_cols
                            if sid in target_rows:
                                for k, v in row_canonical.items():
                                    if v and not target_rows[sid].get(k):
                                        target_rows[sid][k] = v
                            else:
                                target_rows[sid] = dict(row_canonical)
                            for c in canonical_hdr:
                                target_cols.add(c)
                    continue

                # fallback: content-based detection
                if looks_like_transcript(path):
                    # read all lines and append
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        for raw in fh:
                            line = raw.rstrip("\n")
                            if not line.strip():
                                continue
                            if "\t" not in line:
                                continue
                            cols = line.split("\t")
                            if len(cols) < 2:
                                continue
                            sid = cols[0].strip()
                            text = cols[1].strip()
                            if not sid:
                                continue
                            if sid in transcript_seen:
                                # jeśli ten sam id, sprawdź spójność; jeśli różne teksty, loguj
                                # zachowaj pierwszy
                                # opcjonalnie można logować różnice
                                continue
                            transcript_seen.add(sid)
                            transcripts.append((sid, text))
                elif looks_like_metadata(path):
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        reader = csv.reader(fh, delimiter="\t")
                        try:
                            header = next(reader)
                        except StopIteration:
                            continue
                        header = [h.rstrip("\ufeff") for h in header]
                        canonical_hdr, display_map = normalize_header(header)
                        lang = detect_metadata_language("\t".join(header))
                        if meta_headers_seen.get(lang) is None:
                            meta_headers_seen[lang] = canonical_hdr.copy()
                        for k, v in display_map.items():
                            if k not in meta_display[lang]:
                                meta_display[lang][k] = v
                        id_col = find_id_col(header)
                        for r in reader:
                            if not r:
                                continue
                            # pad row to header length if needed
                            if len(r) < len(header):
                                r = r + [""] * (len(header) - len(r))
                            row_canonical: Dict[str, str] = {}
                            for i in range(len(header)):
                                key = canonical_hdr[i]
                                row_canonical[key] = r[i].strip()
                            sid = r[id_col].strip()
                            if not sid:
                                continue
                            target_rows = meta_pl_rows if lang == "pl" else meta_en_rows
                            target_cols = meta_pl_cols if lang == "pl" else meta_en_cols
                            # if id already present, update missing fields but keep existing values (don't overwrite)
                            if sid in target_rows:
                                for k, v in row_canonical.items():
                                    if v and not target_rows[sid].get(k):
                                        target_rows[sid][k] = v
                            else:
                                target_rows[sid] = dict(row_canonical)
                            for c in canonical_hdr:
                                target_cols.add(c)
                else:
                    # nie rozpoznano: spróbuj jeszcze parsować jako transcript
                    if lname.endswith(".txt") or lname.endswith(".tsv"):
                        if looks_like_transcript(path):
                            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                                for raw in fh:
                                    line = raw.rstrip("\n")
                                    if not line.strip() or "\t" not in line:
                                        continue
                                    cols = line.split("\t")
                                    if len(cols) < 2:
                                        continue
                                    sid = cols[0].strip()
                                    text = cols[1].strip()
                                    if sid and sid not in transcript_seen:
                                        transcript_seen.add(sid)
                                        transcripts.append((sid, text))
            except Exception as e:
                logging.warning("Błąd przy przetwarzaniu pliku %s: %s", path, e)

    # Zapis transkryptów
    os.makedirs(os.path.dirname(out_transcripts) or ".", exist_ok=True)
    with open(out_transcripts, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        for sid, text in transcripts:
            writer.writerow([sid, text])
    logging.info("Zapisano transkrypcje: %s (%d rekordów)", out_transcripts, len(transcripts))

    # Helper to write metadata with union of columns
    def write_metadata(out_path: str, rows: Dict[str, Dict[str, str]], cols: Set[str],
                       header_pref: List[str] = None, display_map: Dict[str, str] = None):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # canonical final columns
        final_cols = []
        if header_pref:
            for c in header_pref:
                if c in cols:
                    final_cols.append(c)
        for c in sorted(cols):
            if c not in final_cols:
                final_cols.append(c)
        if not final_cols:
            # nic do zapisania
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("")  # pusty plik
            return
        # determine display names
        display_map = display_map or {}
        header_to_write = [display_map.get(c, c) for c in final_cols]
        with open(out_path, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.writer(out_f, delimiter="\t")
            writer.writerow(header_to_write)
            for sid, row in rows.items():
                writer.writerow([row.get(c, "") for c in final_cols])

    write_metadata(out_meta_pl, meta_pl_rows, meta_pl_cols, meta_headers_seen.get("pl"), meta_display["pl"])
    write_metadata(out_meta_en, meta_en_rows, meta_en_cols, meta_headers_seen.get("en"), meta_display["en"])
    logging.info("Zapisano metadane (PL): %s (%d rekordów)", out_meta_pl, len(meta_pl_rows))
    logging.info("Zapisano metadane (EN): %s (%d rekordów)", out_meta_en, len(meta_en_rows))


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rekurencyjny merger ParlaMint-PL (transcripts + metadata PL/EN)")
    p.add_argument("root_dir", help="katalog z danymi wejściowymi")
    p.add_argument("out_transcripts", help="ścieżka pliku wyjściowego dla transkryptów (TSV)")
    p.add_argument("out_meta_pl", help="ścieżka pliku wyjściowego dla metadanych PL (TSV)")
    p.add_argument("out_meta_en", help="ścieżka pliku wyjściowego dla metadanych EN (TSV)")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    merge_transcripts_and_metadata(args.root_dir, args.out_transcripts, args.out_meta_pl, args.out_meta_en)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
