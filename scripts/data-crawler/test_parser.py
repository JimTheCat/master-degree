import json
import os
import re

import pandas as pd
from bs4 import BeautifulSoup


def process_old_format(soup0, base):
    """
    Stary format: mówca to <P><B><FONT SIZE="+1">...</FONT></B></P>.
    Zbieramy kolejne <P> aż do następnego takiego znacznika.
    Zwraca listę (speaker, text); speaker może być None dla kontekstu.
    """
    speeches = []
    current_speaker = None
    current_text = []
    buffer_context = []

    def norm(s):
        s = re.sub(r'\xa0+', ' ', s)
        return re.sub(r'\s+', ' ', s).strip()

    def flush_speech():
        nonlocal current_speaker, current_text
        if current_speaker is not None:
            text = norm(' '.join(t for t in current_text if t and t.strip()))
            if text:
                speeches.append((current_speaker, text))
        current_speaker, current_text[:] = None, []

    def flush_context():
        nonlocal buffer_context
        text = norm(' '.join(t for t in buffer_context if t and t.strip()))
        if text:
            speeches.append((None, text))
        buffer_context[:] = []

    def is_speaker_marker(p):
        if not (p.find('b') and p.find('font')):
            return False
        txt = p.get_text(" ", strip=True)
        if txt.endswith(':'):
            return True
        role_prefixes = (
            "Marszałek", "Wicemarszałek", "Poseł", "Sekretarz",
            "Minister", "Podsekretarz", "Sekretarz Stanu", "Prezydent",
            "Prezes", "Główny", "Wiceprezes", "Sprawozdawca"
        )
        return any(txt.startswith(rp) for rp in role_prefixes)

    for p in soup0.find_all('p'):
        txt = p.get_text(" ", strip=True)
        if not txt:
            continue
        if is_speaker_marker(p):
            if current_speaker is not None:
                flush_speech()
            else:
                flush_context()
            current_speaker = re.sub(r':$', '', txt).strip()
            current_text = []
        else:
            (current_text if current_speaker is not None else buffer_context).append(txt)

    if current_speaker is not None:
        flush_speech()
    else:
        flush_context()

    return speeches

def load_metadata(deputies_path: str) -> pd.DataFrame:
    """
    Load deputies metadata into a DataFrame. Supports CSV and TSV.
    Ensures a 'name' column and converts NaN to None for JSON.
    """
    if deputies_path.lower().endswith('.tsv'):
        df = pd.read_csv(deputies_path, sep='\t')
    else:
        df = pd.read_csv(deputies_path)
    if 'name' not in df.columns:
        for col in ['firstLastName', 'Speaker_name', 'speaker']:
            if col in df.columns:
                df['name'] = df[col]
                break
        else:
            raise KeyError(f"Brak kolumny 'name' w metadanych. Dostępne kolumny: {list(df.columns)}")
    df = df.where(pd.notnull(df), None)
    return df


def parse_speech_file(transcript_dir: str, base: str, idx: str) -> list:
    """
    Load individual speech HTML and return list of (speaker, text) tuples.
    Splits on sub-speakers like "Poseł [name]: [text]" if present.
    Preserves parentheses insertions as part of text.
    """
    path = os.path.join(transcript_dir, f"{base}_{int(idx)}.html")
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    header = soup.find('h2', class_='mowca')
    if not header:
        return []
    speaker_main = header.get_text(strip=True).rstrip(':').strip()  # Strip trailing ':' early
    segments = []
    for sib in header.find_next_siblings():
        if sib.name == 'h2':
            break
        if sib.name == 'p':
            text = sib.get_text(separator=' ', strip=True)
            if text:
                segments.append(text)
    full = ' '.join(segments)
    full = re.sub(r"\s+", ' ', full).strip()

    # Split on sub-speakers pattern: "Poseł [name]: [text until next or end]"
    pattern = r"(Poseł\s+[^:]+:\s*[^P]+?(?=\s*Poseł\s+|$))"
    matches = re.findall(pattern, full, re.DOTALL)
    sub_speeches = []
    if matches:
        for match in matches:
            # Extract speaker and text from match
            parts = re.split(r":\s*", match, maxsplit=1)
            if len(parts) == 2:
                sub_speaker = parts[0].rstrip(':').strip()  # Strip trailing ':' if any
                sub_text = re.sub(r"\s+", ' ', parts[1]).strip()
                sub_speeches.append((sub_speaker, sub_text))

    # Check if this is a vow section (many short sub-speeches)
    if sub_speeches and len(sub_speeches) > 10:
        avg_len = sum(len(t) for _, t in sub_speeches) / len(sub_speeches)
        if avg_len < 50:
            return sub_speeches

    # Not a vow section: return full text with main speaker
    return [(speaker_main, full)]


def process_html_transcripts(transcript_dir: str, deputies_path: str, output_dir: str):
    """
    Reads backbone _0.html sequentially. Outputs combined.txt with lines:
    uniqueId<TAB>speaker: text for speeches (or just text for contexts), preserving order.
    Contexts and speeches share a unified numeric key sequence. Contexts are merged per block.
    Splits speech blocks into sub-speeches if detected.
    Also outputs metadata JSON for speeches.
    Handles both new and old formats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Identify backbone file
    htmls = [f for f in os.listdir(transcript_dir) if f.lower().endswith('.html')]
    backbone = next((f for f in htmls if f.endswith('_0.html')), None)
    if not backbone:
        raise FileNotFoundError('Brak backbone *_0.html')
    base = backbone[:-7]

    with open(os.path.join(transcript_dir, backbone), encoding='utf-8') as f:
        soup0 = BeautifulSoup(f, 'html.parser')

    metadata_df = load_metadata(deputies_path)
    combined = []
    metadata_list = []
    seq = 1
    buffer_context = []

    def flush_context():
        nonlocal seq, buffer_context
        if buffer_context:
            merged = ' '.join(buffer_context)
            clean = re.sub(r"\s+", ' ', merged).strip()
            uid = f"{base}_{seq}"
            combined.append(f"{uid}\t{clean}")
            metadata_list.append({'id': uid, 'speaker': None, 'metadata': []})
            seq += 1
            buffer_context.clear()

    # --- poprawione wykrywanie formatu ---
    pattern = re.compile(rf'^{re.escape(base)}_(\d+)\.html$', re.I)
    has_partials_for_base = any((m := pattern.match(f)) and m.group(1) != '0' for f in htmls)
    has_mowca_links = soup0.select_one('p.mowca-link, p.mowca-link1') is not None
    is_new_format = has_partials_for_base or has_mowca_links

    if is_new_format:
        # --- nowy format ---
        for p in soup0.find_all('p'):
            if 'mowca-link' in (p.get('class') or []):
                flush_context()
                a = p.find('a', attrs={'name': True})
                if not a:
                    continue
                idx = a['name']
                speaker_main = a.get_text(strip=True).rstrip(':').strip()
                sub_speeches = parse_speech_file(transcript_dir, base, idx)
                for sub_speaker, sub_text in sub_speeches:
                    uid = f"{base}_{seq}"
                    if sub_speaker:
                        sub_speaker = sub_speaker.rstrip(':').strip()
                        line_text = f"{sub_speaker}: {sub_text}"
                    else:
                        line_text = sub_text
                    combined.append(f"{uid}\t{line_text}")
                    speaker_to_match = sub_speaker if sub_speaker != speaker_main else speaker_main
                    rows = metadata_df[metadata_df['name'].apply(lambda n: n and n in speaker_to_match)]
                    recs = rows.to_dict(orient='records')
                    metadata_list.append({'id': uid, 'speaker': speaker_to_match, 'metadata': recs})
                    seq += 1
            else:
                text = p.get_text(separator=' ', strip=True)
                if text:
                    buffer_context.append(text)
    else:
        # --- stary format ---
        sub_speeches = process_old_format(soup0, base)
        for speaker, text in sub_speeches:
            uid = f"{base}_{seq}"
            if speaker:
                speaker = speaker.rstrip(':').strip()
                line_text = f"{speaker}: {text}"
                speaker_to_match = speaker
            else:
                line_text = text
                speaker_to_match = None
            combined.append(f"{uid}\t{line_text}")
            if speaker_to_match:
                rows = metadata_df[metadata_df['name'].apply(lambda n: n and n in speaker_to_match)]
                recs = rows.to_dict(orient='records')
            else:
                recs = []
            metadata_list.append({'id': uid, 'speaker': speaker_to_match, 'metadata': recs})
            seq += 1

    # final flush
    flush_context()

    # write combined file
    out_txt = os.path.join(output_dir, f"{base}_combined.txt")
    with open(out_txt, 'w', encoding='utf-8') as f:
        for line in combined:
            f.write(line + '\n')

    # write metadata JSON
    out_meta = os.path.join(output_dir, f"{base}_metadata.json")
    with open(out_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


# Przykład wywołania funkcji bez argparse:
if __name__ == '__main__':
    process_html_transcripts(
        transcript_dir='data/transcripts/10/1',
        deputies_path='data/deputies/10/deputies.csv',
        output_dir='output'
    )