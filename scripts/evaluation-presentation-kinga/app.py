"""Przeglądarka anotacji korpusu parlamentarnego ParlaMint-PL."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Anotacje korpusu parlamentarnego",
                   layout="wide", initial_sidebar_state="expanded")

KATALOG = Path("data")

KUBELKI_DLUGOSCI = ["0–120", "121–299", "300–699", "700–1199",
                    "1200–1999", "2000–3499", "3500+"]

ETYKIETY_DODATKOWE = ["ADMINISTRACJA", "PRAWORZĄDNOŚĆ"]

WYDARZENIA = {
    "Pandemia COVID-19": "2020-03-01",
    "Protesty po wyroku TK": "2020-10-22",
    "Inwazja na Ukrainę": "2022-02-24",
}

WYMIARY = {
    "partia": "Partia",
    "status_partii": "Koalicja / opozycja",
    "orientacja_partii": "Orientacja partii",
    "izba": "Izba",
    "izba_kadencja": "Izba i kadencja",
    "plec": "Płeć mówcy",
    "rola": "Rola mówcy",
    "czy_minister": "Minister / nie",
    "temat_parlamint": "Temat wg ParlaMint",
    "dlugosc": "Długość wypowiedzi",
}

CZAS = {"miesiac": "Miesiąc", "kwartal": "Kwartał", "rok": "Rok"}

KAPPA_ANOTATOROW = {
    "ROLNICTWO": 0.781, "ZDROWIE": 0.786, "EDUKACJA_NAUKA": 0.739,
    "ŚRODOWISKO": 0.736, "SPRAWY_ZAGRANICZNE": 0.722,
    "HISTORIA_PAMIĘĆ_NARODOWA": 0.684, "INNE": 0.674,
    "POLITYKA_SPOŁECZNA": 0.643, "POLITYKA_LOKALNA": 0.629,
    "BEZPIECZEŃSTWO": 0.589, "INFRASTRUKTURA": 0.581,
    "GOSPODARKA": 0.552, "ŚWIATOPOGLĄD": 0.422,
    "PRAWORZĄDNOŚĆ": 0.342, "ADMINISTRACJA": 0.315,
}


@st.cache_data(show_spinner="Wczytuję dane…")
def wczytaj() -> tuple[pd.DataFrame, pd.DataFrame]:
    szeroki = pd.read_parquet(KATALOG / "wypowiedzi.parquet")
    dlugi = pd.read_parquet(KATALOG / "kategorie.parquet")
    return szeroki, dlugi


def bez_pustych(seria: pd.Series) -> list:
    return sorted(x for x in seria.dropna().unique())


def panel_filtrow(szeroki: pd.DataFrame) -> dict:
    st.sidebar.header("Filtry")

    lata = (int(szeroki["rok"].min()), int(szeroki["rok"].max()))
    zakres = st.sidebar.slider("Zakres lat", lata[0], lata[1], lata,
                               help="Korpus obejmuje 2015-11 … 2022-06")

    role = bez_pustych(szeroki["rola"])
    domyslne_role = [r for r in role if r != "Przewodniczący"]
    wybrane_role = st.sidebar.multiselect(
        "Rola mówcy", role, default=domyslne_role,
        help="Prowadzący obrady są domyślnie wyłączeni — 97,6% ich wypowiedzi "
             "to INNE i przytłoczyłyby wykresy tematyczne.")

    izby = st.sidebar.multiselect("Izba", bez_pustych(szeroki["izba"]),
                                  default=bez_pustych(szeroki["izba"]))

    partie = bez_pustych(szeroki["partia"])
    wybrane_partie = st.sidebar.multiselect(
        "Partia", partie, default=[],
        help="Puste = wszystkie. 18,6% wypowiedzi nie ma przypisanej partii.")

    plcie = st.sidebar.multiselect("Płeć mówcy", bez_pustych(szeroki["plec"]),
                                   default=bez_pustych(szeroki["plec"]))

    st.sidebar.divider()
    st.sidebar.caption("Kategorie")
    pokaz_inne = st.sidebar.checkbox("Pokaż INNE", value=False)
    pokaz_dodatkowe = st.sidebar.checkbox(
        "Pokaż ADMINISTRACJA i PRAWORZĄDNOŚĆ", value=True,
        help="To etykiety dodatkowe, przypisywane obok kategorii tematycznej. "
             "Zgodność anotatorów: kappa 0,32 i 0,34 — najniższa w zestawie.")

    return {
        "zakres": zakres, "role": wybrane_role, "izby": izby,
        "partie": wybrane_partie, "plcie": plcie,
        "pokaz_inne": pokaz_inne, "pokaz_dodatkowe": pokaz_dodatkowe,
    }


def zastosuj(df: pd.DataFrame, f: dict, kategorie: bool = False) -> pd.DataFrame:
    m = df["rok"].between(f["zakres"][0], f["zakres"][1])
    if f["role"]:
        m &= df["rola"].isin(f["role"])
    if f["izby"]:
        m &= df["izba"].isin(f["izby"])
    if f["partie"]:
        m &= df["partia"].isin(f["partie"])
    if f["plcie"]:
        m &= df["plec"].isin(f["plcie"])
    out = df[m]

    if kategorie:
        wyklucz = []
        if not f["pokaz_inne"]:
            wyklucz.append("INNE")
        if not f["pokaz_dodatkowe"]:
            wyklucz += ETYKIETY_DODATKOWE
        if wyklucz:
            out = out[~out["kategoria"].isin(wyklucz)]
    return out


def udzialy(dlugi: pd.DataFrame, szeroki: pd.DataFrame,
            wymiar: str, jako_udzial: bool) -> pd.DataFrame:
    """Tabela wymiar x kategoria. Mianownikiem udziału jest liczba wypowiedzi
    w grupie, nie liczba etykiet."""
    licznik = (dlugi.groupby([wymiar, "kategoria"], observed=True)
               .size().unstack(fill_value=0))
    if not jako_udzial:
        return licznik
    mianownik = szeroki.groupby(wymiar, observed=True).size()
    mianownik = mianownik.reindex(licznik.index).replace(0, pd.NA)
    return licznik.div(mianownik, axis=0) * 100


def dodaj_wydarzenia(fig, pokaz: bool) -> None:
    if not pokaz:
        return
    for nazwa, data in WYDARZENIA.items():
        x = pd.Timestamp(data)
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(dash="dot", color="rgba(120,120,120,0.8)"))
        fig.add_annotation(x=x, y=1, xref="x", yref="paper",
                           text=nazwa, showarrow=False, yanchor="bottom",
                           font=dict(size=10))

def podpis_miary(jako_udzial: bool) -> str:
    return ("% wypowiedzi z daną kategorią" if jako_udzial
            else "liczba wypowiedzi")


def zakladka_przeglad(sz: pd.DataFrame, dl: pd.DataFrame,
                      sz_all: pd.DataFrame) -> None:
    k = st.columns(4)
    k[0].metric("Wypowiedzi po filtrach", f"{len(sz):,}".replace(",", " "))
    k[1].metric("Udział korpusu", f"{len(sz) / len(sz_all) * 100:.1f}%")
    k[2].metric("Mówców", f"{sz['mowca_id'].nunique():,}".replace(",", " "))
    k[3].metric("Śr. etykiet na wypowiedź", f"{sz['n_kategorii'].mean():.2f}")

    if dl.empty:
        st.warning("Filtry nie zwróciły żadnych wypowiedzi.")
        return

    st.subheader("Rozkład kategorii")
    jako_udzial = st.toggle("Pokaż jako odsetek wypowiedzi", value=True,
                            key="przeglad_udzial")
    licz = dl["kategoria"].value_counts()
    war = licz / len(sz) * 100 if jako_udzial else licz
    fig = px.bar(x=war.values, y=war.index, orientation="h",
                 labels={"x": podpis_miary(jako_udzial), "y": ""})
    fig.update_layout(height=480, yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela"):
        tab = (dl.groupby("kategoria", observed=True)
               .size().rename("wypowiedzi").reset_index())
        tab["% wypowiedzi"] = (tab["wypowiedzi"] / len(sz) * 100).round(2)
        st.dataframe(tab.sort_values("wypowiedzi", ascending=False),
                     use_container_width=True, hide_index=True)


def zakladka_czas(sz: pd.DataFrame, dl: pd.DataFrame) -> None:
    st.subheader("Kategorie w czasie")
    c = st.columns([2, 2, 3, 2])
    granulacja = c[0].selectbox("Oś czasu", list(CZAS), format_func=CZAS.get,
                                index=0)
    jako_udzial = c[1].selectbox("Miara", [True, False], index=0,
                                 format_func=lambda x: "Odsetek wypowiedzi"
                                 if x else "Liczba wypowiedzi")
    dostepne = sorted(dl["kategoria"].unique())
    domyslne = [k for k in ["ZDROWIE", "BEZPIECZEŃSTWO", "ŚWIATOPOGLĄD",
                            "POLITYKA_SPOŁECZNA"] if k in dostepne][:4]
    wybrane = c[2].multiselect("Kategorie", dostepne,
                               default=domyslne or dostepne[:3])
    wygladz = c[3].selectbox("Wygładzanie", [1, 3, 6], index=1,
                             format_func=lambda n: "brak" if n == 1
                             else f"średnia z {n} okr.")

    if not wybrane:
        st.info("Wybierz co najmniej jedną kategorię.")
        return

    pokaz_wyd = st.checkbox("Zaznacz wydarzenia", value=True)

    tab = udzialy(dl[dl["kategoria"].isin(wybrane)], sz, granulacja,
                  jako_udzial)
    if wygladz > 1:
        tab = tab.rolling(wygladz, min_periods=1, center=True).mean()

    dane = tab.reset_index().melt(id_vars=granulacja, var_name="kategoria",
                                  value_name="wartosc")
    fig = px.line(dane, x=granulacja, y="wartosc", color="kategoria",
                  labels={"wartosc": podpis_miary(jako_udzial),
                          granulacja: CZAS[granulacja], "kategoria": ""})
    fig.update_layout(height=460, hovermode="x unified",
                      margin=dict(l=0, r=0, t=30, b=0))
    if granulacja != "rok":
        dodaj_wydarzenia(fig, pokaz_wyd)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Uwaga: liczba wypowiedzi w miesiącu waha się w korpusie od 535 do "
        "6 673, dlatego domyślną miarą jest odsetek. Wykres liczb "
        "bezwzględnych odzwierciedla przede wszystkim kalendarz obrad."
    )

    st.divider()
    st.subheader("Przed i po wydarzeniu")
    c = st.columns([3, 2, 2])
    wyd = c[0].selectbox("Wydarzenie", list(WYDARZENIA))
    okno = c[1].slider("Okno (miesiące)", 3, 12, 6)
    data0 = pd.Timestamp(WYDARZENIA[wyd])

    przed = (data0 - pd.DateOffset(months=okno), data0)
    po = (data0, data0 + pd.DateOffset(months=okno))
    wiersze = []
    for kat in wybrane:
        d = dl[dl["kategoria"] == kat]
        for etykieta, (a, b) in (("przed", przed), ("po", po)):
            m_sz = sz["data"].between(a, b).sum()
            m_dl = d["data"].between(a, b).sum()
            wiersze.append({"kategoria": kat, "okres": etykieta,
                            "udzial": m_dl / m_sz * 100 if m_sz else 0,
                            "wypowiedzi": m_sz})
    por = pd.DataFrame(wiersze)
    fig = px.bar(por, x="kategoria", y="udzial", color="okres",
                 barmode="group",
                 labels={"udzial": "% wypowiedzi", "kategoria": ""},
                 category_orders={"okres": ["przed", "po"]})
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Okno ±{okno} mies. wokół {data0.date()}. "
               f"Podstawa: {por[por.okres == 'przed'].wypowiedzi.iloc[0]:,} "
               f"wypowiedzi przed, {por[por.okres == 'po'].wypowiedzi.iloc[0]:,} po."
               .replace(",", " "))


def zakladka_wymiary(sz: pd.DataFrame, dl: pd.DataFrame) -> None:
    st.subheader("Kategorie według wymiaru")
    c = st.columns([3, 2, 2])
    wymiar = c[0].selectbox("Wymiar", list(WYMIARY), format_func=WYMIARY.get)
    jako_udzial = c[1].selectbox("Miara", [True, False], index=0,
                                 format_func=lambda x: "Odsetek wypowiedzi"
                                 if x else "Liczba wypowiedzi",
                                 key="wym_miara")
    minimum = c[2].number_input("Min. wypowiedzi w grupie", 0, 5000, 200, 50)

    licznosc = sz.groupby(wymiar, observed=True).size()
    grupy = licznosc[licznosc >= minimum].index
    sz_f = sz[sz[wymiar].isin(grupy)]
    dl_f = dl[dl[wymiar].isin(grupy)]

    if dl_f.empty:
        st.warning("Brak grup spełniających próg.")
        return

    tab = udzialy(dl_f, sz_f, wymiar, jako_udzial)
    fig = px.imshow(tab.T, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color=podpis_miary(jako_udzial),
                                x=WYMIARY[wymiar], y=""))
    fig.update_layout(height=max(360, 26 * tab.shape[1] + 120),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela"):
        st.dataframe(tab.round(2), use_container_width=True)

    st.caption(
        "Udziały nie sumują się do 100% — anotacja jest wieloetykietowa, "
        "więc jedna wypowiedź może trafić do kilku kategorii."
    )


def zakladka_mowcy(sz: pd.DataFrame, dl: pd.DataFrame) -> None:
    st.subheader("Aktywność mówców")
    minimum = st.slider("Minimalna liczba wypowiedzi", 10, 1000, 100, 10,
                        help="Korpus ma 1 209 mówców, ale tylko 397 ma co "
                             "najmniej 100 wypowiedzi.")
    licz = sz["mowca_id"].value_counts()
    aktywni = licz[licz >= minimum].index
    sz_f = sz[sz["mowca_id"].isin(aktywni)]

    if sz_f.empty:
        st.warning("Żaden mówca nie spełnia progu.")
        return

    st.caption(f"Mówców spełniających próg: {len(aktywni)}")

    top = (sz_f.groupby(["mowca", "partia"], observed=True)
           .size().rename("wypowiedzi").reset_index()
           .sort_values("wypowiedzi", ascending=False).head(25))
    fig = px.bar(top, x="wypowiedzi", y="mowca", color="partia",
                 orientation="h", labels={"mowca": "", "partia": "Partia"})
    fig.update_layout(height=620, yaxis={"categoryorder": "total ascending"},
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Profil tematyczny mówcy")
    lista = sorted(sz_f["mowca"].unique())
    wybrani = st.multiselect("Mówcy (do 6)", lista, default=lista[:2],
                             max_selections=6)
    if not wybrani:
        st.info("Wybierz co najmniej jednego mówcę.")
        return

    dl_w = dl[dl["mowca"].isin(wybrani)]
    sz_w = sz[sz["mowca"].isin(wybrani)]
    tab = udzialy(dl_w, sz_w, "mowca", True)
    dane = tab.reset_index().melt(id_vars="mowca", var_name="kategoria",
                                  value_name="udzial")
    fig = px.line_polar(dane, r="udzial", theta="kategoria", color="mowca",
                        line_close=True)
    fig.update_traces(fill="toself", opacity=0.35)
    fig.update_layout(height=560, margin=dict(l=40, r=40, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)


def zakladka_parlamint(sz: pd.DataFrame, dl: pd.DataFrame) -> None:
    st.subheader("Porównanie z klasyfikacją ParlaMint")
    st.markdown(
        "Korpus zawiera niezależną, jednoetykietową klasyfikację tematyczną "
        "(kolumna `Topic`, 23 kategorie). Poniższa tabela pokazuje, jak "
        "rozkładają się kategorie modelu wewnątrz każdego tematu ParlaMint. "
        "Schematy są różne z założenia — celem nie jest zgodność, tylko "
        "sprawdzenie, czy odwzorowania idą w oczekiwaną stronę."
    )
    kierunek = st.radio(
        "Normalizacja", ["wiersze (temat ParlaMint = 100%)",
                         "kolumny (kategoria modelu = 100%)"],
        horizontal=True)

    tab = pd.crosstab(dl["temat_parlamint"], dl["kategoria"])
    if kierunek.startswith("wiersze"):
        norm = tab.div(tab.sum(axis=1), axis=0) * 100
    else:
        norm = tab.div(tab.sum(axis=0), axis=1) * 100

    fig = px.imshow(norm, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="%", x="Kategoria modelu",
                                y="Temat ParlaMint"))
    fig.update_layout(height=680, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela liczebności"):
        st.dataframe(tab, use_container_width=True)


def zakladka_jakosc(sz_all: pd.DataFrame, dl_all: pd.DataFrame) -> None:
    st.subheader("Jakość anotacji")
    st.caption("Ta zakładka pomija filtry z panelu bocznego — dotyczy "
               "całego przebiegu.")

    k = st.columns(4)
    statusy = sz_all["status"].value_counts()
    k[0].metric("Wypowiedzi", f"{len(sz_all):,}".replace(",", " "))
    k[1].metric("PARSE_FAIL", f"{statusy.get('PARSE_FAIL', 0):,}"
                .replace(",", " "),
                f"{statusy.get('PARSE_FAIL', 0) / len(sz_all) * 100:.3f}%")
    k[2].metric("Obcięte (>6000 zn.)", f"{sz_all['obciety'].sum():,}"
                .replace(",", " "),
                f"{sz_all['obciety'].mean() * 100:.2f}%")
    naruszenia = sz_all["flagi"].str.contains("inne_z_innymi", na=False).sum()
    k[3].metric("INNE z innymi kategoriami",
                f"{naruszenia:,}".replace(",", " "),
                f"{naruszenia / len(sz_all) * 100:.2f}%")

    st.divider()
    c = st.columns(2)

    with c[0]:
        st.markdown("**Udział INNE według roli mówcy**")
        tylko_inne = sz_all["kategorie"].eq("INNE")
        rola = (sz_all.assign(inne=tylko_inne)
                .groupby("rola", observed=True)
                .agg(wypowiedzi=("inne", "size"), udzial=("inne", "mean")))
        rola["udzial"] = (rola["udzial"] * 100).round(1)
        fig = px.bar(rola.reset_index(), x="udzial", y="rola",
                     orientation="h", text="udzial",
                     labels={"udzial": "% wypowiedzi z samym INNE", "rola": ""})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Model nie miał dostępu do roli mówcy — rozróżnienie "
                   "wynika wyłącznie z treści wypowiedzi.")

    with c[1]:
        st.markdown("**Udział INNE według długości wypowiedzi**")
        tylko_inne = sz_all["kategorie"].eq("INNE")
        dl_k = (sz_all.assign(inne=tylko_inne)
                .groupby("dlugosc", observed=True)
                .agg(wypowiedzi=("inne", "size"), udzial=("inne", "mean")))
        dl_k["udzial"] = (dl_k["udzial"] * 100).round(1)
        dl_k = dl_k.reindex([k for k in KUBELKI_DLUGOSCI if k in dl_k.index])
        fig = px.bar(dl_k.reset_index(), x="dlugosc", y="udzial", text="udzial",
                     labels={"udzial": "% wypowiedzi z samym INNE",
                             "dlugosc": "znaków"},
                     category_orders={"dlugosc": KUBELKI_DLUGOSCI})
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Kubełek 0–120 to stratum nieobjęte zbiorem ewaluacyjnym "
                   "(najkrótsza wypowiedź w złotym korpusie miała 121 znaków).")

    st.divider()
    st.markdown("**Zgodność anotatorów (kappa Cohena) a częstość kategorii**")
    st.caption(
        "Kategorie o niskiej zgodności międzyanotatorskiej należy czytać "
        "ostrożnie także na wykresach w pozostałych zakładkach."
    )
    czest = dl_all["kategoria"].value_counts()
    zest = pd.DataFrame({
        "kategoria": list(KAPPA_ANOTATOROW),
        "kappa": list(KAPPA_ANOTATOROW.values()),
        "wypowiedzi": [czest.get(k, 0) for k in KAPPA_ANOTATOROW],
    })
    zest["% wypowiedzi"] = (zest["wypowiedzi"] / len(sz_all) * 100).round(1)
    fig = px.scatter(zest, x="kappa", y="% wypowiedzi", text="kategoria",
                     size="wypowiedzi", size_max=45,
                     labels={"kappa": "kappa anotatorów"})
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    if not (KATALOG / "wypowiedzi.parquet").exists():
        st.error(f"Brak plików w katalogu `{KATALOG}/`. "
                 "Uruchom najpierw `python prepare_data.py`.")
        st.stop()

    szeroki, dlugi = wczytaj()

    st.title("Anotacje korpusu parlamentarnego")
    st.caption(
        f"{len(szeroki):,} wypowiedzi · {szeroki['data'].min().date()} – "
        f"{szeroki['data'].max().date()} · 15 kategorii tematycznych "
        f"przypisanych modelem językowym".replace(",", " ")
    )

    f = panel_filtrow(szeroki)
    sz = zastosuj(szeroki, f)
    dl = zastosuj(dlugi, f, kategorie=True)

    z = st.tabs(["Przegląd", "Czas", "Wymiary", "Mówcy",
                 "Porównanie z ParlaMint", "Jakość anotacji"])
    with z[0]:
        zakladka_przeglad(sz, dl, szeroki)
    with z[1]:
        zakladka_czas(sz, dl)
    with z[2]:
        zakladka_wymiary(sz, dl)
    with z[3]:
        zakladka_mowcy(sz, dl)
    with z[4]:
        zakladka_parlamint(sz, dl)
    with z[5]:
        zakladka_jakosc(szeroki, dlugi)


if __name__ == "__main__":
    main()
