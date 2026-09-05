"""
Warstwa MLflow — jeden run na cały przebieg, agregaty co batch.

Kontrakt: żadna metoda nigdy nie rzuca wyjątku i nigdy nie wiesza pętli
głównej. Sam try/except nie wystarcza — martwy serwer odpowiada timeoutem,
a nie odmową połączenia, dlatego limity HTTP ustawiane są przed importem
mlflow. Po kilku nieudanych batchach warstwa wyłącza się na jakiś czas,
żeby nie płacić timeoutu przy każdym raporcie.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger("mlflow")

# Limity muszą być ustawione PRZED importem mlflow.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

try:
    import mlflow

    MLFLOW_DOSTEPNE = True
except ImportError:
    MLFLOW_DOSTEPNE = False

PROG_BLEDOW = 3          # po tylu nieudanych próbach warstwa się wyłącza
PRZERWA_PO_BLEDACH = 1800  # sekund


class MLflowSink:
    def __init__(self, uri: Optional[str], eksperyment: str,
                 nazwa_runu: str, parametry: dict):
        self.aktywny = False
        self.run_id: Optional[str] = None
        self._bledy = 0
        self._wylaczony_do = 0.0

        if not MLFLOW_DOSTEPNE:
            log.warning("MLflow: biblioteka niedostępna, pomijam")
            return
        if not uri:
            log.info("MLflow: brak MLFLOW_URI, pomijam")
            return

        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(eksperyment)
            run = mlflow.start_run(run_name=nazwa_runu)
            self.run_id = run.info.run_id
            self.aktywny = True
            self._loguj_parametry(parametry)
            log.info("MLflow: run %s w %s", self.run_id, uri)
        except Exception as exc:
            log.warning("MLflow: start nieudany (%s), przetwarzanie bez metryk",
                        exc)

    # ---------------------------------------------------------------- pomocne

    def _dostepny(self) -> bool:
        if not self.aktywny:
            return False
        if time.time() < self._wylaczony_do:
            return False
        return True

    def _niepowodzenie(self, exc: Exception) -> None:
        self._bledy += 1
        if self._bledy >= PROG_BLEDOW:
            self._wylaczony_do = time.time() + PRZERWA_PO_BLEDACH
            self._bledy = 0
            log.warning("MLflow: %s — wstrzymuję raportowanie na %d min",
                        exc, PRZERWA_PO_BLEDACH // 60)

    def _loguj_parametry(self, parametry: dict) -> None:
        try:
            mlflow.log_params({k: str(v)[:250] for k, v in parametry.items()})
        except Exception as exc:
            self._niepowodzenie(exc)

    # --------------------------------------------------------------- publiczne

    def loguj(self, metryki: dict, krok: int) -> None:
        if not self._dostepny():
            return
        try:
            czyste = {k: float(v) for k, v in metryki.items()
                      if isinstance(v, (int, float))}
            mlflow.log_metrics(czyste, step=krok)
            self._bledy = 0
        except Exception as exc:
            self._niepowodzenie(exc)

    def tag(self, klucz: str, wartosc: str) -> None:
        if not self._dostepny():
            return
        try:
            mlflow.set_tag(klucz, str(wartosc)[:250])
        except Exception as exc:
            self._niepowodzenie(exc)

    def artefakt(self, sciezka: str) -> None:
        if not self._dostepny():
            return
        try:
            mlflow.log_artifact(sciezka)
        except Exception as exc:
            self._niepowodzenie(exc)

    def zakoncz(self, status: str = "FINISHED") -> None:
        if not self.aktywny:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as exc:
            log.warning("MLflow: zamknięcie runu nieudane (%s)", exc)
