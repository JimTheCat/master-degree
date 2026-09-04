"""
Definicje kategorii tematycznych.

UWAGA: treść KATEGORIE_DEF musi być identyczna z tą, na której przeprowadzono
ewaluację (evaluator.py). Zgodność sprawdza suma kontrolna promptu — jeśli
w .env ustawiono OCZEKIWANA_SUMA, klasyfikator odmówi startu przy rozjeździe.
"""

from __future__ import annotations

import hashlib
import json

KATEGORIE_DEF: dict[str, str] = {
    "GOSPODARKA": (
        "budżet, podatki, inflacja, rynek pracy, sektory gospodarki. Wszystko związane z robieniem i wydawaniem piniądza, biznesem, finansami, inwestycjami, przedsiębiorczością."
    ),
    "POLITYKA_SPOŁECZNA": (
        "emerytury, ZUS, świadczenia, rodzina, pomoc społeczna, prawa społeczne obywateli, niepełnosprawności, ubóstwo, bezdomność."
    ),
    "BEZPIECZEŃSTWO": (
        "wojsko, policja, granice, cyberbezpieczeństwo, służby mundurowe, działania na obronność państwa, NATO, wydatki militarne."
    ),
    "ŚWIATOPOGLĄD": (
        "religia, prawa reprodukcyjne, LGBT+, związki partnerskie, etyka, wartości — tylko gdy stanowią główny przedmiot sporu. Sama emocjonalność wypowiedzi lub odwołanie do wartości w argumentacji to za mało."
    ),
    "PRAWORZĄDNOŚĆ": (
        "sądy, konstytucja, prawo, regulacje, ustawy, projekty ustaw"
    ),
    "EDUKACJA_NAUKA": (
        "szkoły, uczelnie, nauka, system oświaty, program nauczania, nauczyciele, badania naukowe, stypendia."
    ),
    "ŚRODOWISKO": (
        "klimat, energia, odpady, przyroda."
    ),
    "SPRAWY_ZAGRANICZNE": (
        "dyplomacja, UE, NATO, relacje międzynarodowe. Nie dotyczy, gdy wspomniany kraj jest tylko przykładem w innym kontekście (np. 'jak w Niemczech...')."
    ),
    "ROLNICTWO": (
        "rolnicy, hodowla, gleba, pestycydy, łowiectwo, rybołóstwo, dopłaty dla rolników, skupy, agrobiznes."
    ),
    "POLITYKA_LOKALNA": (
        "samorząd, regiony, inwestycje lokalne."
    ),
    "ADMINISTRACJA": (
        "działanie organów władzy wykonawczej: ministerstwa, urzędy, agencje, rząd i resorty, procedury publiczne, systemy obsługi obywateli, interpelacje i odpowiedzi na nie, nadzór i kontrola. Przypisuj również jako etykietę dodatkową, ilekroć wypowiedź dotyczy tego, jak państwo coś realizuje lub wdraża — obok kategorii tematycznej. Nie dotyczy prowadzenia obrad Sejmu (→ INNE)."
    ),
    "INFRASTRUKTURA": (
        "drogi, kolej, transport publiczny, budynki publiczne (np. szpitale jako obiekty budowlane), inwestycje infrastrukturalne, poruszanie się po drogach"
    ),
    "HISTORIA_PAMIĘĆ_NARODOWA": (
        "rocznice, IPN, pamięć historyczna, II Wojna Światowa, komunizm, bohaterowie narodowi, muzea, pomniki.tylko gdy pamięć historyczna jest przedmiotem wypowiedzi. Odwołania retoryczne, porównania do przeszłości i przywoływanie postaci historycznych w argumentacji nie kwalifikują."
    ),
    "ZDROWIE": (
        "leczenie, NFZ, lekarze, epidemie."
    ),
    "INNE": (
        "Jeśli wypowiedź dotyczy prowadzenia obrad (otwarcie/zamknięcie posiedzenia, zapowiedź głosowania, udzielanie głosu, wnioski formalne, komunikaty marszałka), przypisz wyłącznie INNE — nawet jeśli pada w niej tytuł ustawy, nazwa ministerstwa czy temat merytoryczny. Sam tytuł omawianego punktu nie czyni wypowiedzi merytoryczną."
    ),
}

KATEGORIE: list[str] = list(KATEGORIE_DEF.keys())


def suma_kontrolna_promptu(szablon: str) -> str:
    """
    Skrót szablonu + definicji kategorii. Identyczny algorytm jak w ewaluatorze
    — pozwala ustalić, na jakiej wersji promptu powstała anotacja.
    """
    tresc = szablon + json.dumps(KATEGORIE_DEF, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(tresc.encode("utf-8")).hexdigest()[:12]
