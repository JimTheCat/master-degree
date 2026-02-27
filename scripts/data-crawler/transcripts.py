import json
import os

import requests

path = "data/transcripts/"


def process_transcripts(rq_data, json_data):
    """
    Przetwarzanie transkryptów z API Sejmu.
    :param json_data: Dane JSON z API Sejmu
    :return: None
    """

    # Sprawdzenie, czy folder na transkrypty istnieje, jeśli nie, to go tworzymy
    if not os.path.exists(path):
        os.makedirs(path)

    # Sprawdzenie, czy dane JSON zawierają transkrypty
    if 'statements' not in json_data:
        print("Brak transkryptów w danych JSON.")
        return

    # Zapisz json_data do pliku jeśli nie istnieje
    # if not os.path.exists(f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/{rq_data.get("date")}.json'):
    json_path = f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/{rq_data.get("date")}.json'
    if not os.path.exists(f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/'):
        os.makedirs(f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/')

    # Jeśli statements są puste, to nie zapisuj pliku
    if not json_data['statements']:
        print("Brak transkryptów w danych JSON.")
        return
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # Iteracja po transkryptach
    for statement in json_data['statements']:
        # Pobierz transkrypt wypowiedzi w html
        if 'num' in statement:
            statement_num = statement['num']

            # Sprawdzenie, czy transkrypt już istnieje
            if os.path.exists(
                    f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/{rq_data.get("date")}_{statement_num}.html'):
                # print(f'Transkrypt {statement_num} już istnieje.')
                continue

            response = requests.get(
                f'https://api.sejm.gov.pl/sejm/term{rq_data.get("term_number")}/proceedings/{rq_data.get("proceeding_num")}/{rq_data.get("date")}/transcripts/{statement_num}'
            )

            if response.status_code == 200:
                html_path = f'{path}{rq_data.get("term_number")}/{rq_data.get("proceeding_num")}/{rq_data.get("date")}_{statement_num}.html'
                with open(html_path, 'wb') as f:
                    f.write(response.content)
                # print(f'Zapisano transkrypt: {html_path}')


def download_transcripts():
    """
    Główna funkcja wykonująca przetwarzanie danych z API Sejmu.
    """
    # Pobieranie wszystkich kadencji z API Sejmu
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

        # Pobieranie posiedzeń dla danej kadencji
        response = requests.get(f'https://api.sejm.gov.pl/sejm/term{term_number}/proceedings')
        proceedings = response.json()

        for proceeding in proceedings:
            proceeding_num = proceeding['number']
            proceeding_dates = proceeding['dates']
            print(f'Przetwarzanie posiedzenia: {proceeding_num} ({proceeding_dates})')

            # Tworzenie katalogu dla danego posiedzenia
            proceeding_dir = f'{term_dir}/{proceeding_num}'
            if not os.path.exists(proceeding_dir):
                os.makedirs(proceeding_dir)
            # Przetwarzanie dat dla danego posiedzenia
            for date in proceeding_dates:
                # print(f'Przetwarzanie daty: {date}')

                # Pominięcie przetwarzania, jeśli plik już istnieje
                if os.path.exists(f'{proceeding_dir}/{date}.pdf'):
                    # print(f'Plik PDF dla {date} już istnieje')
                    continue

                # Pobieranie listy transkryptów
                transcripts = requests.get(
                    f'https://api.sejm.gov.pl/sejm/term{term_number}/proceedings/{proceeding_num}/{date}/transcripts'
                )

                if transcripts.status_code == 200:
                    request_data = {
                        "term_number": term_number,
                        "proceeding_num": proceeding_num,
                        "date": date
                    }

                    process_transcripts(request_data, transcripts.json())
                else:
                    print(f'Nie znaleziono transkryptów dla {date}')
                    continue
