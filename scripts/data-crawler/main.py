'''
Sejm Crawler
'''
import json
import os

from test_parser import process_html_transcripts
from transcripts import download_transcripts

# This script is a web crawler that scrapes data from the Polish Sejm website.

# Sprawdzenie istnienia folderu 'data'
if not os.path.exists('data'):
    os.makedirs('data')
    os.makedirs('data/terms')
    os.makedirs('data/deputies')
    os.makedirs('data/transcripts')


def transcripts_process():
    for kadencja_dir in os.listdir('data/transcripts'):
        kadencja_path = os.path.join('data/transcripts', kadencja_dir)
        if os.path.isdir(kadencja_path):
            for posiedzenie_dir in os.listdir(kadencja_path):
                transcript_dir = os.path.join(kadencja_path, posiedzenie_dir)
                if os.path.isdir(transcript_dir):
                    deputies_path = os.path.join('data/deputies', kadencja_dir, 'deputies.csv')
                    if os.path.exists(deputies_path):
                        htmls = [f for f in os.listdir(transcript_dir) if f.endswith('_0.html')]
                        if htmls:
                            base = htmls[0][:-7]  # YYYY-MM-DD
                            year = base[:4]
                            output_dir = os.path.join('output', year)
                            process_html_transcripts(transcript_dir, deputies_path, output_dir)


def merge_all_transcripts(base_output_dir: str, merged_txt_path: str, merged_json_path: str):
    """
    Merge all _combined.txt into one merged.txt with separators, and all _metadata.json into one list in merged.json.
    """
    all_txt_content = []
    all_metadata = []

    for year_dir in os.listdir(base_output_dir):
        year_path = os.path.join(base_output_dir, year_dir)
        if os.path.isdir(year_path):
            for file in os.listdir(year_path):
                if file.endswith('_combined.txt'):
                    txt_path = os.path.join(year_path, file)
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    all_txt_content.append(f"{content}")
                elif file.endswith('_metadata.json'):
                    json_path = os.path.join(year_path, file)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    all_metadata.extend(metadata)

    # Write merged TXT
    with open(merged_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_txt_content))
    # Write merged JSON
    with open(merged_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # download_all_terms()
    # download_transcripts()
    # download_deputies()
    transcripts_process()
    # merge_all_transcripts('output', 'merged_all.txt', 'merged_all.json')
