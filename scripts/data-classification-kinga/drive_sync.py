"""
Synchronizacja z Google Drive — wersja bezokienkowa.

Port drive_service.py ze Streamlita: te same operacje (wyszukanie pliku po
nazwie w folderze, files().update), ale konfiguracja z .env i logowanie
zamiast st.error.

Kontrakt: żadna metoda publiczna nigdy nie rzuca wyjątku i nigdy nie blokuje
wątku wywołującego. Wysyłka idzie w osobnym wątku-demonie. Jeśli nowe zlecenie
przyjdzie, zanim poprzednie się wyśle, poprzednie jest porzucane — liczy się
tylko najświeższy stan.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("drive")

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    GOOGLE_DOSTEPNE = True
except ImportError:  # brak bibliotek — program ma działać dalej
    GOOGLE_DOSTEPNE = False

ZAKRESY = ["https://www.googleapis.com/auth/drive"]


class DriveSync:
    """Nieblokująca wysyłka plików na Dysk Google."""

    def __init__(self, sciezka_sa: Optional[str], folder_id: Optional[str]):
        self.service = None
        self.folder_id = folder_id
        self.konto: Optional[str] = None
        self._blad_init: Optional[str] = None
        self._id_plikow: dict[str, str] = {}
        self._ostrzezono_o_tworzeniu = False

        self._zadania: dict[str, tuple[str, bool]] = {}
        self._lock = threading.Lock()
        self._sygnal = threading.Event()
        self._stop = threading.Event()
        self._watek: Optional[threading.Thread] = None

        self._inicjuj(sciezka_sa)

        if self.dostepny:
            self._watek = threading.Thread(target=self._petla, daemon=True)
            self._watek.start()

    # ------------------------------------------------------------------ init

    def _inicjuj(self, sciezka_sa: Optional[str]) -> None:
        if not GOOGLE_DOSTEPNE:
            self._blad_init = "brak bibliotek google-api-python-client"
            return
        if not sciezka_sa or not self.folder_id:
            self._blad_init = "brak GDRIVE_SA_JSON lub GDRIVE_FOLDER_ID"
            return
        if not Path(sciezka_sa).exists():
            self._blad_init = f"nie znaleziono pliku konta serwisowego: {sciezka_sa}"
            return

        # Walidacja zawartości: wszystkie pola, które w Streamlicie rozpisuje się
        # w secrets.toml, muszą być w tym jednym pliku JSON.
        wymagane = ("type", "project_id", "private_key", "client_email",
                    "token_uri")
        try:
            with open(sciezka_sa, encoding="utf-8") as f:
                dane = json.load(f)
        except Exception as exc:
            self._blad_init = f"plik konta serwisowego nie jest poprawnym JSON: {exc}"
            return

        brakujace = [k for k in wymagane if not dane.get(k)]
        if brakujace:
            self._blad_init = f"w JSON brakuje pól: {', '.join(brakujace)}"
            return

        self.konto = dane.get("client_email")

        try:
            creds = service_account.Credentials.from_service_account_file(
                sciezka_sa, scopes=ZAKRESY
            )
            self.service = build("drive", "v3", credentials=creds,
                                 cache_discovery=False)
            log.info("Konto serwisowe: %s", self.konto)
        except Exception as exc:
            self._blad_init = str(exc)
            log.warning("Dysk Google niedostępny: %s", exc)

    @property
    def dostepny(self) -> bool:
        return self.service is not None and self.folder_id is not None

    @property
    def status(self) -> str:
        if self.dostepny:
            return "połączono"
        return f"niedostępny ({self._blad_init})"

    # --------------------------------------------------------------- zlecenia

    def zlec(self, nazwa_zdalna: str, sciezka_lokalna: str,
             spakuj: bool = False) -> None:
        """
        Zleca wysyłkę. Wraca natychmiast. Kolejne zlecenie o tej samej nazwie
        zdalnej nadpisuje poprzednie, jeśli tamto nie zdążyło się wysłać.
        """
        if not self.dostepny:
            return
        with self._lock:
            self._zadania[nazwa_zdalna] = (sciezka_lokalna, spakuj)
        self._sygnal.set()

    def zamknij(self, timeout: float = 120.0) -> None:
        """Czeka na dokończenie ostatniej wysyłki i kończy wątek."""
        if not self._watek:
            return
        self._sygnal.set()
        self._stop.set()
        self._watek.join(timeout=timeout)

    # ------------------------------------------------------------------ wątek

    def _petla(self) -> None:
        while True:
            self._sygnal.wait(timeout=5.0)
            self._sygnal.clear()

            with self._lock:
                partia = dict(self._zadania)
                self._zadania.clear()

            for nazwa, (sciezka, spakuj) in partia.items():
                try:
                    self._wyslij(nazwa, sciezka, spakuj)
                except Exception as exc:  # nic stąd nie wychodzi na zewnątrz
                    log.warning("Wysyłka %s nie powiodła się: %s", nazwa, exc)

            if self._stop.is_set() and not partia:
                return

    def _wyslij(self, nazwa: str, sciezka: str, spakuj: bool) -> None:
        if not Path(sciezka).exists():
            return

        do_wyslania = sciezka
        tymczasowy = None
        if spakuj:
            tymczasowy = f"{sciezka}.tmp.gz"
            with open(sciezka, "rb") as we, gzip.open(tymczasowy, "wb",
                                                      compresslevel=6) as wy:
                shutil.copyfileobj(we, wy, length=1024 * 1024)
            do_wyslania = tymczasowy

        try:
            t0 = time.time()
            mime = "application/gzip" if spakuj else (
                "application/json" if nazwa.endswith(".json") else "text/plain"
            )
            media = MediaFileUpload(do_wyslania, mimetype=mime, resumable=False)
            file_id = self._znajdz_lub_utworz(nazwa, media)
            if file_id:
                self.service.files().update(
                    fileId=file_id, media_body=media, supportsAllDrives=True
                ).execute()
            rozmiar = os.path.getsize(do_wyslania) / 1024
            log.info("Dysk: %s (%.0f kB) w %.1fs", nazwa, rozmiar,
                     time.time() - t0)
        finally:
            if tymczasowy and Path(tymczasowy).exists():
                try:
                    os.remove(tymczasowy)
                except OSError:
                    pass

    # ------------------------------------------------------------------- test

    def sprawdz(self, katalog_roboczy: str = ".") -> bool:
        """
        Synchroniczny test połączenia — uruchom PRZED długim przebiegiem.
        Wysyła mały plik próbny i zwraca True, jeśli się udało.
        """
        if not self.dostepny:
            print(f"  Dysk Google: {self.status}")
            return False

        print(f"  Konto serwisowe : {self.konto}")
        try:
            folder = self.service.files().get(
                fileId=self.folder_id, fields="name,driveId",
                supportsAllDrives=True,
            ).execute()
            typ = "dysk współdzielony" if folder.get("driveId") else "My Drive"
            print(f"  Folder docelowy : {folder.get('name')} ({typ})")
        except Exception as exc:
            print(f"  BŁĄD: nie widzę folderu {self.folder_id} — czy jest "
                  f"udostępniony kontu {self.konto}? ({exc})")
            return False

        probny = Path(katalog_roboczy) / "_proba_drive.tmp"
        probny.parent.mkdir(parents=True, exist_ok=True)
        probny.write_text(json.dumps({"test": True, "ts": time.time()}),
                          encoding="utf-8")

        wszystko_ok = True
        for nazwa, pakuj in (("progress.json", False), ("wyniki.jsonl.gz", True)):
            try:
                self._wyslij(nazwa, str(probny), spakuj=pakuj)
                print(f"  {nazwa:<18}: OK")
            except Exception as exc:
                wszystko_ok = False
                if "storageQuotaExceeded" in str(exc):
                    print(f"  {nazwa:<18}: BRAK PLIKU NA DYSKU. Konto serwisowe "
                          f"nie może go utworzyć.\n"
                          f"                      Utwórz pusty '{nazwa}' lokalnie "
                          f"i wgraj go ręcznie do folderu docelowego.")
                else:
                    print(f"  {nazwa:<18}: NIEUDANA ({type(exc).__name__}: {exc})")
        try:
            probny.unlink()
        except OSError:
            pass
        return wszystko_ok

    def _znajdz_lub_utworz(self, nazwa: str, media) -> Optional[str]:
        if nazwa in self._id_plikow:
            return self._id_plikow[nazwa]

        zapytanie = (
            f"'{self.folder_id}' in parents "
            f"and name = '{nazwa}' "
            f"and trashed = false"
        )
        wynik = self.service.files().list(
            q=zapytanie, spaces="drive",
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        pliki = wynik.get("files", [])

        if pliki:
            self._id_plikow[nazwa] = pliki[0]["id"]
            return self._id_plikow[nazwa]

        # Konto serwisowe nie ma własnego limitu miejsca — tworzenie pliku
        # w zwykłym folderze My Drive zwykle kończy się storageQuotaExceeded.
        # Na dysku współdzielonym (Shared Drive) działa bez problemu.
        try:
            utworzony = self.service.files().create(
                body={"name": nazwa, "parents": [self.folder_id]},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            self._id_plikow[nazwa] = utworzony["id"]
            log.info("Utworzono %s na Dysku", nazwa)
            return None  # create wgrał już treść, update niepotrzebny
        except Exception as exc:
            if not self._ostrzezono_o_tworzeniu:
                self._ostrzezono_o_tworzeniu = True
                log.error(
                    "Nie mogę utworzyć '%s' na Dysku (%s). "
                    "Utwórz ten plik ręcznie w folderze docelowym (może być pusty) "
                    "i udostępnij folder kontu %s z prawem edycji. "
                    "Konta serwisowe nie mają własnego limitu miejsca, "
                    "więc mogą tylko nadpisywać pliki, których właścicielem jesteś Ty.",
                    nazwa, type(exc).__name__, self.konto,
                )
            raise
