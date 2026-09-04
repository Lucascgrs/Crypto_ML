"""
Entrées / sorties disque du projet.

Deux optimisations importantes par rapport à l'ancienne version :

  1. **Cache Parquet** — les fichiers Excel restent la source visible (on peut
     les ouvrir à la main), mais un jumeau `.parquet` est écrit à côté. La
     lecture est alors 30 à 50× plus rapide (79 Mo d'Excel = plusieurs dizaines
     de secondes ; le parquet équivalent se lit en une fraction de seconde).
     Si le `.xlsx` est plus récent que le `.parquet`, le cache est ignoré puis
     régénéré : impossible de travailler sur des données périmées.

  2. **Cache mémoire** — un même fichier chargé plusieurs fois dans la session
     (analyse, entraînement, prédiction, backtest) n'est lu qu'une fois. C'est
     un usage utile de la RAM disponible.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config

try:
    import pyarrow  # noqa: F401  (import testé, utilisé via pandas)
    PARQUET_OK = True
except ImportError:                                    # pragma: no cover
    PARQUET_OK = False


# Cache mémoire : {chemin: (date de modification, DataFrame)}.
# Plafonné pour ne pas retenir indéfiniment de gros tableaux : au-delà, la plus
# ancienne entrée est évincée (les dictionnaires Python gardent l'ordre
# d'insertion, ce qui suffit à en faire une file).
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
TAILLE_CACHE = 8


def _memoriser(chemin: str, mtime: float, df: pd.DataFrame) -> None:
    """Range un tableau dans le cache mémoire, en évinçant le plus ancien."""
    _CACHE.pop(chemin, None)
    _CACHE[chemin] = (mtime, df)
    while len(_CACHE) > TAILLE_CACHE:
        _CACHE.pop(next(iter(_CACHE)))


def vider_cache() -> None:
    """Vide le cache mémoire (utile après une réécriture massive de fichiers)."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Chemins normalisés
# ---------------------------------------------------------------------------
def chemin_brut(symbole: str, intervalle: str) -> str:
    """data_crypto/BTC_1h.xlsx"""
    return os.path.join(config.DOSSIER_DONNEES, f"{symbole}_{intervalle}.xlsx")


def chemin_analyse(symbole: str, intervalle: str) -> str:
    """analysis_crypto/BTC_1h_analyzed.xlsx"""
    return os.path.join(config.DOSSIER_ANALYSES, f"{symbole}_{intervalle}_analyzed.xlsx")


def chemin_prediction(symbole: str, intervalle: str, horizon: int) -> str:
    """prediction_crypto/BTC_1h_h12_prediction.xlsx"""
    return os.path.join(config.DOSSIER_PREDICTIONS,
                        f"{symbole}_{intervalle}_h{int(horizon)}_prediction.xlsx")


def chemin_modele(symbole: str, intervalle: str, horizon: int) -> str:
    """models/MODELE_BTC_1h_h12.joblib"""
    return os.path.join(config.DOSSIER_MODELES,
                        f"MODELE_{symbole}_{intervalle}_h{int(horizon)}.joblib")


def chemin_meta(symbole: str, intervalle: str, horizon: int) -> str:
    """models/META_BTC_1h_h12.json"""
    return os.path.join(config.DOSSIER_MODELES,
                        f"META_{symbole}_{intervalle}_h{int(horizon)}.json")


def _chemin_parquet(chemin_xlsx: str) -> str:
    return os.path.splitext(chemin_xlsx)[0] + ".parquet"


# ---------------------------------------------------------------------------
# Lecture / écriture des tableaux temporels
# ---------------------------------------------------------------------------
def lire_tableau(chemin: str, cache: bool = True) -> pd.DataFrame | None:
    """
    Charge un tableau indexé par la date (Excel, avec cache Parquet et mémoire).

    Retourne None si le fichier est absent ou illisible.
    """
    if not os.path.exists(chemin):
        return None

    mtime = os.path.getmtime(chemin)
    if cache:
        memo = _CACHE.get(chemin)
        if memo is not None and memo[0] == mtime:
            return memo[1]

    parquet = _chemin_parquet(chemin)
    df = None

    # 1. Cache disque : valide seulement s'il est plus récent que l'Excel.
    if PARQUET_OK and os.path.exists(parquet) and os.path.getmtime(parquet) >= mtime:
        try:
            df = pd.read_parquet(parquet)
        except Exception:                              # cache corrompu -> on relit l'Excel
            df = None

    # 2. Lecture Excel (lente) puis régénération du cache.
    if df is None:
        try:
            df = pd.read_excel(chemin, index_col=0)
        except Exception as err:
            print(f"❌ Lecture impossible ({os.path.basename(chemin)}) : {err}")
            return None
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()]
        _ecrire_parquet(df, parquet)

    df = df.sort_index()
    df.index.name = "Date"
    if cache:
        _memoriser(chemin, mtime, df)
    return df


def ecrire_tableau(df: pd.DataFrame, chemin: str) -> None:
    """Écrit un tableau en Excel (lisible) + son cache Parquet (rapide)."""
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    df.to_excel(chemin)
    _ecrire_parquet(df, _chemin_parquet(chemin))
    _memoriser(chemin, os.path.getmtime(chemin), df)


def _ecrire_parquet(df: pd.DataFrame, chemin: str) -> None:
    """Écrit le cache Parquet, en silence si la librairie manque."""
    if not PARQUET_OK:
        return
    try:
        df.to_parquet(chemin)
    except Exception:
        pass  # le cache est un bonus : son échec ne doit jamais bloquer


# ---------------------------------------------------------------------------
# Inventaire des fichiers disponibles
# ---------------------------------------------------------------------------
def _lister(dossier: str, suffixe: str) -> list[str]:
    """Clés 'SYMBOLE_INTERVALLE' des fichiers d'un dossier portant ce suffixe."""
    if not os.path.isdir(dossier):
        return []
    fin = suffixe + ".xlsx"
    cles = [f[: -len(fin)] for f in os.listdir(dossier)
            if f.endswith(fin) and not f.startswith(("~$", "TOP_"))]
    return sorted(cles)


def lister_donnees_brutes() -> list[str]:
    """['BTC_1h', 'ETH_1h', ...] — fichiers OHLCV téléchargés."""
    return _lister(config.DOSSIER_DONNEES, "")


def lister_analyses() -> list[str]:
    """['BTC_1h', ...] — fichiers enrichis des 8 indicateurs et des variations."""
    return _lister(config.DOSSIER_ANALYSES, "_analyzed")


def lister_predictions() -> list[str]:
    """['BTC_1h_h12', ...] — fichiers de prédiction générés."""
    if not os.path.isdir(config.DOSSIER_PREDICTIONS):
        return []
    fin = "_prediction.xlsx"
    return sorted(f[: -len(fin)] for f in os.listdir(config.DOSSIER_PREDICTIONS)
                  if f.endswith(fin) and not f.startswith("~$"))


def lister_modeles() -> list[str]:
    """['BTC_1h_h12', ...] — modèles entraînés disponibles."""
    if not os.path.isdir(config.DOSSIER_MODELES):
        return []
    cles = [f[len("MODELE_"): -len(".joblib")]
            for f in os.listdir(config.DOSSIER_MODELES)
            if f.startswith("MODELE_") and f.endswith(".joblib")]
    return sorted(cles)


def separer_cle(cle: str) -> tuple[str, str]:
    """'BTC_1h' -> ('BTC', '1h'). Tolère les symboles contenant un underscore."""
    if "_" not in cle:
        return cle, "1h"
    symbole, intervalle = cle.rsplit("_", 1)
    return symbole, intervalle


def separer_cle_modele(cle: str) -> tuple[str, str, int]:
    """'BTC_1h_h12' -> ('BTC', '1h', 12)."""
    if "_h" in cle:
        base, horizon = cle.rsplit("_h", 1)
        if horizon.isdigit():
            symbole, intervalle = separer_cle(base)
            return symbole, intervalle, int(horizon)
    symbole, intervalle = separer_cle(cle)
    return symbole, intervalle, config.HORIZON_DEFAUT
