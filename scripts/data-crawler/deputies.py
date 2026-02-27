import os

import requests

path = 'data/deputies/'


def __get_all_unique_attributes(deputies):
    """
    Get all unique attributes from the deputies data.
    """
    unique_attributes = set()
    for deputy in deputies:
        for key in deputy.keys():
            unique_attributes.add(key)
    return unique_attributes


def __save_deputies_to_csv(deputies, term_number, attributes):
    """
    Save deputies data to a CSV file.
    """
    import pandas as pd

    if deputies is None or len(deputies) == 0:
        print('Brak danych do zapisania.')
        return

    # Create a DataFrame from the deputies data
    df = pd.DataFrame(deputies)

    # Reorder columns based on unique attributes
    df = df[attributes]

    # Save to CSV
    csv_path = f'{path}{term_number}/deputies.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Dane posłów zapisano do pliku: {csv_path}')


def download_deputies():
    """
    Download deputies data from the given URL and save it to a CSV file.
    """

    response = requests.get('https://api.sejm.gov.pl/sejm/term')
    terms = response.json()

    for term in terms:
        term_number = term['num']
        term_from = term['from']
        term_current = term['current']
        # Ustalamy datę zakończenia kadencji
        term_to = 'present'
        if not term_current:
            term_to = term['to']
        print(f'Przetwarzanie kadencji: {term_number} ({term_from} - {term_to})')

        # Tworzenie katalogu dla danej kadencji
        term_dir = f'{path}{term_number}'
        if not os.path.exists(term_dir):
            os.makedirs(term_dir)

        # Download deputies data
        response = requests.get(f'https://api.sejm.gov.pl/sejm/term{term_number}/MP')
        deputies = response.json()

        # Get all unique attributes
        unique_attributes = list(__get_all_unique_attributes(deputies))
        print(f'Znaleziono {len(deputies)} posłów z {len(unique_attributes)} unikalnymi atrybutami.')

        # Save deputies data to CSV
        __save_deputies_to_csv(deputies, term_number, unique_attributes)
