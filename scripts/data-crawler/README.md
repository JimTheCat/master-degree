# Sejm Crawler

This is a simple crawler for the Polish Sejm website. It scrapes the website for information about the Sejm, including
the list of MPs, their contact information, and their speeches.

## Requirements

- Python 3.12 or higher
- `requests` library
- `beautifulsoup4` library
- `pandas` library

## Installation

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the crawler, simply execute the `main.py` script:

```bash
python main.py
```

This will use the Sejm API and save the data to a CSV file.

For the transcripts, you can run the `main.py` script from the `test_transcript_download` directory:

```bash
python main.py
```

After that you can run parser.py to parse the transcripts:

```bash
python parser.py
```