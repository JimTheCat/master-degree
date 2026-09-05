from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class LabelInfo:
    name: str
    display_name_pl: str
    display_name_en: str
    description_pl: str
    examples: list[str]


class EmotionLabel(str, Enum):
    AGRESJA_WERBALNA = "AGRESJA_WERBALNA"
    STRATEGIA_STRACHU = "STRATEGIA_STRACHU"
    DEHUMANIZACJA_POGARDA = "DEHUMANIZACJA_POGARDA"
    DUMA_I_SUKCES = "DUMA_I_SUKCES"
    MESJANIZM_MORALNY = "MESJANIZM_MORALNY"


class RhetoricalLabel(str, Enum):
    POLARYZACJA_MY_ONI = "POLARYZACJA_MY_ONI"
    AD_HOMINEM = "AD_HOMINEM"
    OBLEZIONA_TWIERDZA = "OBLEZIONA_TWIERDZA"
    PRZYPISYWANIE_ZLYCH_INTENCJI = "PRZYPISYWANIE_ZLYCH_INTENCJI"
    WHATABOUTISM = "WHATABOUTISM"
    SOFIZMAT_ROZSZERZENIA = "SOFIZMAT_ROZSZERZENIA"
    DOWOD_ANEGDOTYCZNY = "DOWOD_ANEGDOTYCZNY"
    APEL_O_JEDNOSC = "APEL_O_JEDNOSC"


EMOTION_INFO: dict[EmotionLabel, LabelInfo] = {
    EmotionLabel.AGRESJA_WERBALNA: LabelInfo(
        name="AGRESJA_WERBALNA",
        display_name_pl="Agresja werbalna",
        display_name_en="Verbal aggression",
        description_pl=(
            "Bezpośrednia ekspresja złości, gniewu, oburzenia. "
            "Inwektywy, krzyk, wulgaryzmy."
        ),
        examples=["Złodzieje!", "Hańba!"],
    ),
    EmotionLabel.STRATEGIA_STRACHU: LabelInfo(
        name="STRATEGIA_STRACHU",
        display_name_pl="Strategia strachu",
        display_name_en="Fear strategy",
        description_pl=(
            "Wyrażanie lęku lub celowe straszenie odbiorców zagrożeniem "
            "(wojną, biedą, zniszczeniem państwa)."
        ),
        examples=["Zniszczą nas", "Idą po wasze pieniądze"],
    ),
    EmotionLabel.DEHUMANIZACJA_POGARDA: LabelInfo(
        name="DEHUMANIZACJA_POGARDA",
        display_name_pl="Dehumanizacja / pogarda",
        display_name_en="Dehumanization / contempt",
        description_pl=(
            "Wyrażanie wyższości, obrzydzenia, traktowanie oponentów "
            'jako "podludzi", "zarazę" lub "szkodników".'
        ),
        examples=["Robactwo", "Patologia", "Moralne dno"],
    ),
    EmotionLabel.DUMA_I_SUKCES: LabelInfo(
        name="DUMA_I_SUKCES",
        display_name_pl="Duma i sukces",
        display_name_en="Pride and success",
        description_pl=(
            "Afirmacja własnej grupy. Wyrażanie satysfakcji, dumy "
            "z osiągnięć, siły i sprawczości."
        ),
        examples=["Wielkie zwycięstwo", "Polska rośnie w siłę"],
    ),
    EmotionLabel.MESJANIZM_MORALNY: LabelInfo(
        name="MESJANIZM_MORALNY",
        display_name_pl="Mesjanizm moralny",
        display_name_en="Moral messianism",
        description_pl=(
            "Poczucie moralnej wyższości i misji dziejowej. "
            "Własna grupa jako jedyny obrońca Dobra/Prawdy."
        ),
        examples=["Bronimy świętości", "Tylko my uratujemy naród"],
    ),
}

RHETORICAL_INFO: dict[RhetoricalLabel, LabelInfo] = {
    RhetoricalLabel.POLARYZACJA_MY_ONI: LabelInfo(
        name="POLARYZACJA_MY_ONI",
        display_name_pl="Polaryzacja My-Oni",
        display_name_en="Us vs Them polarization",
        description_pl=(
            'Fundamentalny podział na dwa zwaśnione obozy ("My"-dobrzy vs "Oni"-źli). '
            "Budowanie barier tożsamościowych."
        ),
        examples=["Prawdziwi Polacy vs zdrajcy"],
    ),
    RhetoricalLabel.AD_HOMINEM: LabelInfo(
        name="AD_HOMINEM",
        display_name_pl="Ad hominem",
        display_name_en="Ad hominem attack",
        description_pl=(
            "Atak na osobę (wygląd, przeszłość, cechy), a nie na jej argumenty."
        ),
        examples=["Lecz się człowieku", "Taki alkoholik nie ma prawa głosu"],
    ),
    RhetoricalLabel.OBLEZIONA_TWIERDZA: LabelInfo(
        name="OBLEZIONA_TWIERDZA",
        display_name_pl="Oblężona twierdza",
        display_name_en="Siege mentality",
        description_pl=(
            "Przedstawianie własnej grupy jako ofiary niesprawiedliwych "
            "ataków, nagonki lub spisku."
        ),
        examples=["Nagonka medialna", "Cały świat się na nas uwziął"],
    ),
    RhetoricalLabel.PRZYPISYWANIE_ZLYCH_INTENCJI: LabelInfo(
        name="PRZYPISYWANIE_ZLYCH_INTENCJI",
        display_name_pl="Przypisywanie złych intencji",
        display_name_en="Attributing malicious intent",
        description_pl=(
            "Sugerowanie, że przeciwnik działa celowo na szkodę "
            "(agentura, zdrada, tajny plan), a nie z błędu."
        ),
        examples=["Realizują obce interesy", "Chcą zniszczyć Polskę"],
    ),
    RhetoricalLabel.WHATABOUTISM: LabelInfo(
        name="WHATABOUTISM",
        display_name_pl="Whataboutism",
        display_name_en="Whataboutism",
        description_pl=(
            "Odwracanie uwagi poprzez wytykanie hipokryzji "
            "lub błędów z przeszłości."
        ),
        examples=["A za waszych rządów...", "Spójrzcie na siebie"],
    ),
    RhetoricalLabel.SOFIZMAT_ROZSZERZENIA: LabelInfo(
        name="SOFIZMAT_ROZSZERZENIA",
        display_name_pl="Sofizmat rozszerzenia",
        display_name_en="Straw man fallacy",
        description_pl=(
            "Atakowanie przeinaczonej, wyolbrzymionej wersji poglądów przeciwnika."
        ),
        examples=["Chcecie, żeby ludzie umierali na ulicach?"],
    ),
    RhetoricalLabel.DOWOD_ANEGDOTYCZNY: LabelInfo(
        name="DOWOD_ANEGDOTYCZNY",
        display_name_pl="Dowód anegdotyczny",
        display_name_en="Anecdotal evidence",
        description_pl=(
            "Używanie pojedynczych historii/plotek przeciwko twardym danym. Populizm."
        ),
        examples=["Znam kogoś, kto...", "Ludzie widzą jak jest, eksperci kłamią"],
    ),
    RhetoricalLabel.APEL_O_JEDNOSC: LabelInfo(
        name="APEL_O_JEDNOSC",
        display_name_pl="Apel o jedność",
        display_name_en="Appeal for unity",
        description_pl=(
            "Wzywanie do zgody (często instrumentalne, by uciszyć krytykę)."
        ),
        examples=["Nie czas na kłótnie", "Bądźmy jednością"],
    ),
}

ALL_LABELS: list[str] = [e.value for e in EmotionLabel] + [r.value for r in RhetoricalLabel]

ALL_LABEL_INFO: dict[str, LabelInfo] = {
    **{e.value: info for e, info in EMOTION_INFO.items()},
    **{r.value: info for r, info in RHETORICAL_INFO.items()},
}


def get_label_info(label_name: str) -> LabelInfo:
    """Get LabelInfo for a label name string."""
    if label_name not in ALL_LABEL_INFO:
        raise ValueError(f"Unknown label: {label_name}. Valid labels: {ALL_LABELS}")
    return ALL_LABEL_INFO[label_name]


def is_emotion(label_name: str) -> bool:
    """Check if a label name is an emotion category."""
    return label_name in {e.value for e in EmotionLabel}


def is_rhetorical(label_name: str) -> bool:
    """Check if a label name is a rhetorical technique category."""
    return label_name in {r.value for r in RhetoricalLabel}
