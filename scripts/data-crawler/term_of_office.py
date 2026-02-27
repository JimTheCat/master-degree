import os

import pandas as pd
import requests

path = 'data/terms/'


def download_all_terms():
    """
    Download all terms of office data from the given URL and save it to a CSV file.
    """

    response = requests.get('https://api.sejm.gov.pl/sejm/term')
    terms = response.json()

    if not os.path.exists(path):
        os.makedirs(path)

    # Create a DataFrame from the terms data
    df = pd.DataFrame(terms)

    # Convert prints to a more readable format
    df['prints'] = df['prints'].apply(lambda x: {
        'count': x.get('count') if isinstance(x, dict) else None,
        'lastChanged': x.get('lastChanged') if isinstance(x, dict) else None,
        'link': x.get('link') if isinstance(x, dict) else None
    })
    # Save to CSV
    csv_path = f'{path}terms.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Dane kadencji zapisano do pliku: {csv_path}')
