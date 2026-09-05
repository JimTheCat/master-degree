# Raport z analizy korpusu

## Podsumowanie

- Model: `qwen/qwen3.6-35b-a3b`, wariant promptu: `zeroshot`
- Wypowiedzi w korpusie z predykcją: 223,808 (niedopasowanych: 0)
- Po odfiltrowaniu partii ['unknown']: 182,103
- Role mówców: Parlamentarzysta 96,881, Przewodniczący 85,222

## Zróżnicowanie partyjne

Partie uwzględnione w testach (≥50 wypowiedzi): KO, KP-PSL, Konfederacja, Kukiz15, Lewica, PiS, PrzywrócićPrawo, Teraz, UPR

### Etykiety najsilniej różnicujące partie (chi², wg Craméra V)

| Etykieta                     |      chi² |            p |   Cramér V |
|:-----------------------------|----------:|-------------:|-----------:|
| Przypisywanie złych intencji | 13147.9   | 0            |  0.268702  |
| Polaryzacja My-Oni           | 11534.9   | 0            |  0.25168   |
| Strategia strachu            |  7462.91  | 0            |  0.20244   |
| Dowód anegdotyczny           |  5247.01  | 0            |  0.169745  |
| Sofizmat rozszerzenia        |  4223.54  | 0            |  0.152293  |
| Agresja werbalna             |  4058.06  | 0            |  0.14928   |
| Dehumanizacja / pogarda      |  3157.9   | 0            |  0.131686  |
| Oblężona twierdza            |  3116.26  | 0            |  0.130815  |
| Duma i sukces                |  1139.54  | 1.10737e-240 |  0.0791053 |
| Mesjanizm moralny            |  1023.17  | 1.48437e-215 |  0.0749577 |
| Ad hominem                   |   971.56  | 2.05295e-204 |  0.0730426 |
| Whataboutism                 |   476.415 | 8.05289e-98  |  0.0511487 |
| Apel o jedność               |   406.328 | 8.29367e-83  |  0.0472367 |

## Ranking mówców

Filtry: rola `Parlamentarzysta`, ≥50 wypowiedzi, ≥200 znaków na wypowiedź. Uwzględniono 430 mówców (81,269 wypowiedzi).

### Top 10 — nasycenie emocji na wypowiedź

| Mówca                   | Partia       |   Wypowiedzi |   Wskaźnik |
|:------------------------|:-------------|-------------:|-----------:|
| Morawiecki, Mateusz     | PiS          |           52 |      2.385 |
| Kopiec, Maciej          | Lewica       |           62 |      1.823 |
| Niesiołowski, Stefan    | KP-PSL       |           79 |      1.797 |
| Jachira, Klaudia        | KO           |          162 |      1.735 |
| Nowacka, Barbara        | KO           |          103 |      1.728 |
| Budka, Borys            | KO           |          297 |      1.694 |
| Macierewicz, Antoni     | PiS          |           61 |      1.639 |
| Gasiuk-Pihowicz, Kamila | KO           |          282 |      1.638 |
| Winnicki, Robert        | Konfederacja |          453 |      1.594 |
| Błaszczak, Mariusz      | PiS          |           64 |      1.594 |

### Top 10 — nasycenie technik retorycznych na wypowiedź

| Mówca                | Partia       |   Wypowiedzi |   Wskaźnik |
|:---------------------|:-------------|-------------:|-----------:|
| Morawiecki, Mateusz  | PiS          |           52 |      4.231 |
| Niesiołowski, Stefan | KP-PSL       |           79 |      3.19  |
| Sośnierz, Dobromir   | Konfederacja |          231 |      3.134 |
| Budka, Borys         | KO           |          297 |      3.101 |
| Petru, Ryszard       | Teraz        |          121 |      3.05  |
| Konwiński, Zbigniew  | KO           |           68 |      3.029 |
| Kierwiński, Marcin   | KO           |           98 |      2.98  |
| Berkowicz, Konrad    | Konfederacja |           72 |      2.958 |
| Neumann, Sławomir    | KO           |           88 |      2.92  |
| Wilk, Jacek          | Konfederacja |           61 |      2.869 |

## Dynamika czasowa

Granulacja: `Q`, zakres 2015Q4 – 2022Q2. Wartości to odsetek wypowiedzi w okresie, nie liczby bezwzględne — 2015 (od 12 listopada) i 2022 (do 30 czerwca) są niepełne, więc zmiana rok-do-roku liczona jest na 2016–2021.

### Szczyt i zmiana w czasie dla każdej etykiety

| Etykieta                     |   Rok szczytowy |   Szczyt % |   2016 % |   2021 % |   Zmiana p.p. |
|:-----------------------------|----------------:|-----------:|---------:|---------:|--------------:|
| Strategia strachu            |            2021 |       9.88 |     7.74 |     9.88 |          2.14 |
| Dowód anegdotyczny           |            2020 |      12.81 |     9.53 |    11.57 |          2.03 |
| Duma i sukces                |            2021 |      14.52 |    13.32 |    14.52 |          1.2  |
| Przypisywanie złych intencji |            2020 |      18.01 |    15.72 |    16.7  |          0.98 |
| Apel o jedność               |            2021 |       4.02 |     3.13 |     4.02 |          0.88 |
| Agresja werbalna             |            2020 |       9.08 |     7.17 |     7.79 |          0.61 |
| Dehumanizacja / pogarda      |            2020 |       3.8  |     2.88 |     3.25 |          0.36 |
| Mesjanizm moralny            |            2020 |       3.81 |     3.67 |     3.75 |          0.09 |
| Oblężona twierdza            |            2017 |       9.17 |     8.31 |     8.2  |         -0.1  |
| Whataboutism                 |            2020 |       2.38 |     2.29 |     1.92 |         -0.37 |
| Sofizmat rozszerzenia        |            2017 |       2.47 |     2.39 |     1.74 |         -0.65 |
| Ad hominem                   |            2017 |       5.58 |     4.97 |     4.18 |         -0.78 |
| Polaryzacja My-Oni           |            2017 |      33.26 |    31.7  |    30.23 |         -1.47 |

## Koalicja vs opozycja

Pokrycie metadanych `party_status`: 172,520 z 182,103 wypowiedzi (94.7%). Wartości w % wypowiedzi danej grupy.

| Etykieta                     |   Koalicja |   Opozycja |   Cramér V |
|:-----------------------------|-----------:|-----------:|-----------:|
| Agresja werbalna             |       4.94 |      10.52 |      0.102 |
| Strategia strachu            |       3.21 |      12.54 |      0.167 |
| Dehumanizacja / pogarda      |       2.26 |       3.94 |      0.047 |
| Duma i sukces                |      15.2  |      12.81 |      0.034 |
| Mesjanizm moralny            |       3.26 |       3.85 |      0.016 |
| Polaryzacja My-Oni           |      21.97 |      38.34 |      0.176 |
| Ad hominem                   |       4.44 |       5.1  |      0.015 |
| Oblężona twierdza            |       5.37 |      10.81 |      0.097 |
| Przypisywanie złych intencji |       7.48 |      22.92 |      0.209 |
| Whataboutism                 |       2.2  |       2.17 |      0.001 |
| Sofizmat rozszerzenia        |       0.89 |       3.21 |      0.079 |
| Dowód anegdotyczny           |       6.21 |      14.24 |      0.129 |
| Apel o jedność               |       4.22 |       2.85 |      0.037 |

## Kadencje i izby

Pokrycie metadanych `term`: 182,103 z 182,103 wypowiedzi (100.0%). Wartości w % wypowiedzi danej grupy.

| Etykieta                     |   Sejm 8. (2015–2019) |   Senat 9. (2015–2019) |   Sejm 9. (2019–2022) |   Senat 10. (2019–2022) |   Cramér V |
|:-----------------------------|----------------------:|-----------------------:|----------------------:|------------------------:|-----------:|
| Agresja werbalna             |                  9.78 |                   2.96 |                 11.1  |                    2.91 |      0.121 |
| Strategia strachu            |                  8.38 |                   5.23 |                 12.3  |                    5.38 |      0.092 |
| Dehumanizacja / pogarda      |                  3.47 |                   1.61 |                  4.49 |                    1.47 |      0.063 |
| Duma i sukces                |                 14.12 |                  11.49 |                 16.37 |                    9.93 |      0.061 |
| Mesjanizm moralny            |                  3.5  |                   3.42 |                  4.05 |                    3.05 |      0.017 |
| Polaryzacja My-Oni           |                 33.2  |                  26.71 |                 35.88 |                   21.47 |      0.101 |
| Ad hominem                   |                  4.96 |                   4.99 |                  4.75 |                    3.93 |      0.016 |
| Oblężona twierdza            |                  9.49 |                   6.13 |                  9.93 |                    5.25 |      0.063 |
| Przypisywanie złych intencji |                 17.58 |                  12.07 |                 20    |                   10.5  |      0.09  |
| Whataboutism                 |                  2.24 |                   2.12 |                  2.37 |                    1.63 |      0.015 |
| Sofizmat rozszerzenia        |                  2.27 |                   2.3  |                  2.41 |                    1.36 |      0.022 |
| Dowód anegdotyczny           |                 10.86 |                   8.15 |                 14.79 |                    6.64 |      0.086 |
| Apel o jedność               |                  3.3  |                   2.94 |                  4.34 |                    2.66 |      0.031 |

## Ograniczenia

Wszystkie powyższe wskaźniki pochodzą z predykcji jednego modelu (`qwen/qwen3.6-35b-a3b`, `zeroshot`), którego jakość na zbiorze złotego standardu wynosi: macro-F1 **0.532** (emocje 0.580, retoryka 0.503; źródło: `sweep_summary_20260722_212441.csv`).

Konsekwencje, które trzeba czytać razem z każdym wykresem:

- Mierzone jest to, co model wykrywa, a nie zjawisko samo w sobie. Rankingi i różnice między grupami są rzetelne o tyle, o ile błąd modelu rozkłada się równomiernie między mówcami, partiami i latami — czego nie sprawdzono.
- Etykiety o niższym F1 (zwłaszcza retoryczne) mają rankingi obarczone większym błędem niż etykiety emocji.
- Liczba etykiet rośnie z długością wypowiedzi (średnio 0.18 dla wypowiedzi poniżej 500 znaków wobec 2.85 powyżej 3000). Filtr `--min-speech-chars 200` i przeliczanie na wypowiedź ograniczają to obciążenie, ale go nie usuwają: mówcy wygłaszający dłuższe wystąpienia są systematycznie wyżej w rankingach.
- Wypowiedzi o roli `Przewodniczący` (prowadzenie obrad) są wyłączone z rankingu mówców, ale wchodzą do analiz partyjnych i czasowych, gdzie zaniżają wskaźniki.

## Pliki wyjściowe

- `party_label_frequencies.csv`
- `chi_squared_results.csv`
- `party_rhetoric_stacked.png`
- `rhetoric_profiles.png`
- `cooccurrence_heatmap.png`
- `rhetoric_profiles_normalized.csv`
- `speaker_rankings.csv`
- `ranking_emocje.png`
- `ranking_retoryka.png`
- `ranking_agresja.png`
- `temporal_periods.csv`
- `temporal_lines.png`
- `temporal_heatmap.png`
- `temporal_by_party.csv`
- `temporal_party_polaryzacja.png`
- `coalition_vs_opposition.csv`
- `coalition_vs_opposition.png`
- `chi_squared_by_party_status.csv`
- `by_term.csv`
- `by_term.png`
- `chi_squared_by_term.csv`
