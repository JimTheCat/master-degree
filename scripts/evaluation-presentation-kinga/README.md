# Przeglądarka anotacji korpusu ParlaMint-PL

Interaktywna aplikacja do eksploracji tematycznych anotacji polskiego korpusu
parlamentarnego. 228 326 wypowiedzi z Sejmu i Senatu, wygłoszonych między
listopadem 2015 a czerwcem 2022, otrzymało po jednej lub kilku z 15 kategorii
tematycznych przypisanych przez lokalnie uruchomiony model językowy.

Aplikacja łączy te anotacje z metadanymi o mówcach i posiedzeniach dołączonymi
do korpusu, dzięki czemu tematy można rozbić według partii, izby, kadencji,
mówcy, płci i czasu.

To repozytorium zawiera wyłącznie warstwę przeglądową. Sam proces anotacji
przebiegał osobno.

## Wersja online

<!-- Po pierwszym wdrożeniu wstaw tu adres ze Streamlit Cloud. -->
https://s24839-masters-degree-presentation-app.streamlit.app

## Uruchomienie lokalne

```bash
git clone https://github.com/KinTrae/masters-degree-presentation-app.git
cd masters-degree-presentation-app
python -m venv .venv 
source .venv/bin/activate # Linux / macOS
.venv\Scripts\activate # Windows (PowerShell)
pip install -r requirements.txt
streamlit run app.py
```

Wymaga Pythona 3.10 lub nowszego. Aplikacja czyta dwa pliki Parquet z katalogu
`data/` i nie potrzebuje niczego więcej: żadnej bazy danych, kluczy API ani
dostępu do sieci.

## Zakładki

**Przegląd** — rozkład kategorii przy bieżących filtrach, przełączalny między
odsetkiem wypowiedzi a liczbami bezwzględnymi.

**Czas** — trendy kategorii w podziale na miesiące, kwartały lub lata, z
opcjonalnym wygładzaniem średnią kroczącą i liniami odniesienia dla trzech
wydarzeń: początku pandemii, protestów z października 2020 i inwazji na
Ukrainę. Drugi wykres porównuje udziały kategorii w symetrycznych oknach przed
wybranym wydarzeniem i po nim.

**Wymiary** — mapa cieplna kategorii według wybranego wymiaru. Do wyboru:
partia, status koalicja/opozycja, orientacja partii, izba, kadencja, płeć i
rola mówcy oraz długość wypowiedzi.

**Mówcy** — 25 najaktywniejszych mówców z kolorem partii oraz wykres radarowy
porównujący profile tematyczne maksymalnie sześciu wybranych osób.

**Porównanie z ParlaMint** — tabela krzyżowa kategorii modelu wobec własnej,
jednoetykietowej klasyfikacji korpusu (pole `Topic`, 23 kategorie). Schematy
nie mają być zgodne; wykres służy sprawdzeniu, czy odwzorowania idą w
oczekiwaną stronę.

**Jakość anotacji** — diagnostyka przebiegu, niezależna od filtrów w panelu
bocznym: odpowiedzi nierozpoznane przez parser, obcięcia wejścia, naruszenia
reguł, udział kategorii `INNE` według roli mówcy i długości wypowiedzi oraz
zgodność anotatorów zestawiona z częstością kategorii.

## Jak czytać liczby

Kilka właściwości danych może wprowadzić w błąd.

**Udziały nie sumują się do 100%.** Anotacja jest wieloetykietowa, średnio
1,74 kategorii na wypowiedź. Każdy procent w aplikacji odpowiada na pytanie
„jaki odsetek wypowiedzi otrzymał tę kategorię" — mianownikiem są wypowiedzi,
nie etykiety.

**Prowadzący obrady są domyślnie wyłączeni.** Wypowiedzi osób prowadzących
posiedzenie stanowią 46,4% korpusu, a 97,6% z nich to komunikaty proceduralne
oznaczone jako `INNE`. Można je włączyć z powrotem, ale przytłaczają każdy
wykres tematyczny.

**Na osi czasu lepszy jest odsetek niż liczba.** Miesięczna liczba wypowiedzi
waha się w korpusie od 535 do 6 673, więc wartości bezwzględne odzwierciedlają
głównie kalendarz obrad, a nie zmiany w tym, o czym mówiono.

**`ADMINISTRACJA` i `PRAWORZĄDNOŚĆ` to etykiety dodatkowe.** Instrukcja
anotacji każe modelowi dopisywać je obok kategorii tematycznej zawsze, gdy
wypowiedź dotyczy tego, jak państwo coś realizuje. Występują odpowiednio w
31,6% i 24,4% wypowiedzi i miały najniższą zgodność międzyanotatorską w całym
zestawie (kappa Cohena 0,32 i 0,34). Same w sobie nie świadczą o tym, że
administracja zdominowała debatę.

**Partia jest nieznana dla 18,6% wypowiedzi** — głównie prowadzących i gości.
Orientacja partii brakuje dla mniej więcej połowy.

**Korpus kończy się 30 czerwca 2022**, cztery miesiące po inwazji na Ukrainę.
To zbyt krótkie okno, by wnioskować o trwałości trendów.

## Dane

Katalog `data/` zawiera dwa pliki Parquet, łącznie około 6,5 MB:

| Plik | Wierszy | Ziarno |
| --- | --- | --- |
| `wypowiedzi.parquet` | 228 326 | jeden wiersz na wypowiedź; kategorie sklejone przecinkami |
| `kategorie.parquet` | 396 301 | jeden wiersz na parę wypowiedź–kategoria, do wykresów |

Oba mają ten sam komplet kolumn metadanych, więc każdy da się filtrować
niezależnie.

Najważniejsze kolumny: `id`, `data`, `rok`/`miesiac`/`kwartal`, `izba`,
`kadencja`, `mowca`, `partia`, `status_partii`, `orientacja_partii`, `plec`,
`rola`, `temat_parlamint`, `kategoria` (tabela długa) lub `kategorie` (tabela
szeroka), `status`, `znaki`, `dlugosc`.

### Pochodzenie danych

Korpus źródłowy: [ParlaMint-PL](https://www.clarin.si/repository/xmlui/handle/11356/1859),
polska część projektu ParlaMint — porównywalnych korpusów debat
parlamentarnych krajów europejskich, udostępnianych przez CLARIN.

Anotacje tematyczne powstały przy użyciu lokalnie uruchomionego modelu obsługiwanego przez LM Studio, sekwencyjnie, przy temperaturze
0 i ustalonym ziarnie, w ciągu około 16 godzin. Wejście obcinano na 6 000
znaków, co dotknęło 2,78% wypowiedzi. 399 odpowiedzi (0,175%) nie dało się
sparsować do kategorii; wszystkie okazały się komunikatami proceduralnymi i
otrzymały etykietę `INNE`.

Schemat anotacji zwalidowano na 500-wypowiedziowym złotym standardzie,
oznaczonym niezależnie przez kilkoro anotatorów, a następnie uwspólnionym.
Średnia kappa Cohena dla kategorii wyniosła 0,613. Wartości dla poszczególnych
kategorii pokazuje zakładka jakości i należy je brać pod uwagę przy każdym
wniosku wyciąganym z aplikacji.

Warto odnotować, że złoty standard nie zawiera wypowiedzi krótszych niż 121
znaków, podczas gdy 32,7% korpusu produkcyjnego jest poniżej tego progu.
Wyniki ewaluacji nie są ekstrapolowane na to stratum; zakładka jakości
raportuje je osobno.

## Licencja

Kod: MIT. Anotacje: CC BY 4.0, zgodnie z korpusem źródłowym.