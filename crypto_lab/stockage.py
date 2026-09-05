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
from dataclasses import dataclass

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
    _APERCUS.clear()


# ---------------------------------------------------------------------------
# Chemins normalisés
# ---------------------------------------------------------------------------
def chemin_brut(symbole: str, intervalle: str) -> str:
    """data_crypto/BTC_1h.xlsx"""
    return os.path.join(config.DOSSIER_DONNEES, f"{symbole}_{intervalle}.xlsx")


def chemin_analyse(symbole: str, intervalle: str) -> str:
    """analysis_crypto/BTC_1h_analyzed.xlsx"""
    return os.path.join(config.DOSSIER_ANALYSES, f"{symbole}_{intervalle}_analyzed.xlsx")


def chemin_exogene(symbole: str, intervalle: str) -> str:
    """data_crypto/EXO_BTC_1h.xlsx — funding rate et open interest cumulés."""
    return os.path.join(config.DOSSIER_DONNEES, f"EXO_{symbole}_{intervalle}.xlsx")


# Marque les fichiers issus d'un walk-forward, pour les distinguer d'une
# prédiction classique tout en gardant le même format de nom.
SUFFIXE_WALKFORWARD = "_wf"

# Objectif historique : ses fichiers gardent le nom court, sans suffixe. Les
# autres objectifs (direction nette, triple barrière, amplitude) s'ajoutent en
# suffixe, ce qui permet de les faire coexister pour un même horizon.
TACHE_SANS_SUFFIXE = "direction"

# Fichier produit par les modèles de régression (quantiles + volatilité +
# espérance de gain). Traité comme un « objectif » supplémentaire côté nommage.
SUFFIXE_ESPERANCE = "esperance"


def _marque_tache(tache: str | None) -> str:
    """'_barriere' pour un objectif alternatif, '' pour la direction simple."""
    if not tache or tache == TACHE_SANS_SUFFIXE:
        return ""
    return f"_{tache}"


def chemin_prediction(symbole: str, intervalle: str, horizon: int,
                      walk_forward: bool = False, tache: str | None = None) -> str:
    """prediction_crypto/BTC_1h_h12_prediction.xlsx (+ suffixes objectif / _wf)"""
    marque = SUFFIXE_WALKFORWARD if walk_forward else ""
    return os.path.join(
        config.DOSSIER_PREDICTIONS,
        f"{symbole}_{intervalle}_h{int(horizon)}"
        f"{_marque_tache(tache)}{marque}_prediction.xlsx")


def chemin_modele(symbole: str, intervalle: str, horizon: int,
                  tache: str | None = None) -> str:
    """models/MODELE_BTC_1h_h12.joblib"""
    return os.path.join(
        config.DOSSIER_MODELES,
        f"MODELE_{symbole}_{intervalle}_h{int(horizon)}{_marque_tache(tache)}.joblib")


def chemin_meta(symbole: str, intervalle: str, horizon: int,
                tache: str | None = None) -> str:
    """models/META_BTC_1h_h12.json"""
    return os.path.join(
        config.DOSSIER_MODELES,
        f"META_{symbole}_{intervalle}_h{int(horizon)}{_marque_tache(tache)}.json")


def chemin_regression(symbole: str, intervalle: str, horizon: int,
                      cible: str) -> str:
    """models/REGRESSION_BTC_1h_h12_volatilite.joblib"""
    return os.path.join(
        config.DOSSIER_MODELES,
        f"REGRESSION_{symbole}_{intervalle}_h{int(horizon)}_{cible}.joblib")


def chemin_meta_regression(symbole: str, intervalle: str, horizon: int,
                           cible: str) -> str:
    """models/METAREG_BTC_1h_h12_volatilite.json"""
    return os.path.join(
        config.DOSSIER_MODELES,
        f"METAREG_{symbole}_{intervalle}_h{int(horizon)}_{cible}.json")


def _chemin_parquet(chemin_xlsx: str) -> str:
    return os.path.splitext(chemin_xlsx)[0] + ".parquet"



# Aperçus déjà calculés, indexés par (chemin, date de modification).
_APERCUS: dict[str, tuple[float, dict]] = {}


def apercu_tableau(chemin: str) -> dict | None:
    """
    Nombre de lignes et bornes de dates d'un tableau, SANS le charger.

    Écrit pour l'interface : décrire un panier de vingt-cinq cryptos avec
    `lire_tableau` obligerait à charger vingt-cinq fichiers complets dans le
    thread graphique, donc à figer la fenêtre plusieurs secondes. Ici on ne lit
    que la colonne d'index du cache Parquet — quelques millisecondes par
    fichier, et le résultat est mémorisé tant que le fichier ne change pas.

    Retourne None si le fichier est absent ou illisible.
    """
    if not os.path.exists(chemin):
        return None

    mtime = os.path.getmtime(chemin)
    memo = _APERCUS.get(chemin)
    if memo is not None and memo[0] == mtime:
        return memo[1]

    # Le tableau est peut-être déjà en mémoire : inutile de toucher au disque.
    en_cache = _CACHE.get(chemin)
    df = en_cache[1] if en_cache is not None and en_cache[0] == mtime else None

    if df is None:
        parquet = _chemin_parquet(chemin)
        if PARQUET_OK and os.path.exists(parquet) and os.path.getmtime(parquet) >= mtime:
            try:
                # `columns=[]` ne rapatrie que l'index : c'est tout ce qu'il faut.
                df = pd.read_parquet(parquet, columns=[])
            except Exception:                          # noqa: BLE001
                df = None

    if df is None:
        # Pas de cache Parquet exploitable : on retombe sur la lecture normale,
        # qui alimentera le cache pour les fois suivantes.
        df = lire_tableau(chemin)
    # Attention : `df.empty` serait vrai ici. La lecture « index seul » rend un
    # tableau à zéro colonne, et pandas juge vide tout tableau dont un axe l'est.
    if df is None or len(df) == 0:
        return None

    index = pd.to_datetime(df.index, errors="coerce")
    index = index[index.notna()]
    if not len(index):
        return None

    apercu = {"lignes": int(len(index)),
              "debut": index.min(), "fin": index.max()}
    _APERCUS[chemin] = (mtime, apercu)
    while len(_APERCUS) > 128:
        _APERCUS.pop(next(iter(_APERCUS)))
    return apercu

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
            if f.endswith(fin) and not f.startswith(("~$", "TOP_", "EXO_"))]
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


def lister_regressions() -> list[str]:
    """['BTC_1h_h12_volatilite', ...] — modèles de régression entraînés."""
    if not os.path.isdir(config.DOSSIER_MODELES):
        return []
    cles = [f[len("REGRESSION_"): -len(".joblib")]
            for f in os.listdir(config.DOSSIER_MODELES)
            if f.startswith("REGRESSION_") and f.endswith(".joblib")]
    return sorted(cles)


def separer_cle(cle: str) -> tuple[str, str]:
    """'BTC_1h' -> ('BTC', '1h'). Tolère les symboles contenant un underscore."""
    if "_" not in cle:
        return cle, "1h"
    symbole, intervalle = cle.rsplit("_", 1)
    return symbole, intervalle


def est_walkforward(cle: str) -> bool:
    """Vrai si la clé désigne un fichier produit par un walk-forward."""
    return cle.endswith(SUFFIXE_WALKFORWARD)


@dataclass(frozen=True)
class CleModele:
    """Tout ce qu'un nom de fichier de modèle ou de prédiction encode."""

    symbole: str
    intervalle: str
    horizon: int
    tache: str = TACHE_SANS_SUFFIXE
    walk_forward: bool = False

    @property
    def esperance(self) -> bool:
        """Vrai pour un fichier de régression (quantiles + espérance de gain)."""
        return self.tache == SUFFIXE_ESPERANCE


# Suffixes d'objectif reconnus dans les noms de fichiers. Constitué
# dynamiquement pour qu'un nouvel objectif dans `cibles.TACHES` soit lu sans
# rien toucher ici — l'import est tardif afin d'éviter une boucle d'imports.
def _suffixes_connus() -> tuple[str, ...]:
    from . import cibles
    return tuple(cibles.TACHES) + (SUFFIXE_ESPERANCE,)


def analyser_cle(cle: str) -> CleModele:
    """
    'BTC_1h_h12', 'BTC_1h_h3_barriere_wf', 'BTC_1h_h6_esperance' -> CleModele.

    Tolère les symboles contenant un underscore et les clés incomplètes : à
    défaut d'horizon lisible, celui par défaut est retenu.
    """
    walk_forward = cle.endswith(SUFFIXE_WALKFORWARD)
    if walk_forward:
        cle = cle[: -len(SUFFIXE_WALKFORWARD)]

    tache = TACHE_SANS_SUFFIXE
    for suffixe in _suffixes_connus():
        if cle.endswith("_" + suffixe):
            tache = suffixe
            cle = cle[: -len(suffixe) - 1]
            break

    horizon = config.HORIZON_DEFAUT
    if "_h" in cle:
        base, valeur = cle.rsplit("_h", 1)
        if valeur.isdigit():
            cle, horizon = base, int(valeur)

    symbole, intervalle = separer_cle(cle)
    return CleModele(symbole=symbole, intervalle=intervalle, horizon=horizon,
                     tache=tache, walk_forward=walk_forward)


def separer_cle_modele(cle: str) -> tuple[str, str, int]:
    """Raccourci historique : (symbole, intervalle, horizon) seulement."""
    infos = analyser_cle(cle)
    return infos.symbole, infos.intervalle, infos.horizon


def paniers_contenant(symbole: str, intervalle: str, horizon: int,
                      tache: str | None = None) -> list[str]:
    """
    Modèles de panier entraînés qui incluent cette crypto.

    Sert de repli : quand on demande l'importance des features ou une
    prédiction pour BTC sans que BTC ait son propre modèle, un panier
    « PANIER-BTC-ETH-SOL » fait parfaitement l'affaire — c'est même l'usage
    prévu. Sans ce repli, l'interface renverrait « modèle introuvable » alors
    qu'un modèle applicable existe juste à côté.
    """
    trouves = []
    for cle in lister_modeles():
        infos = analyser_cle(cle)
        if (infos.intervalle != intervalle or infos.horizon != horizon
                or infos.walk_forward):
            continue
        if tache is not None and infos.tache != tache:
            continue
        if symbole in config.symboles_panier(infos.symbole):
            trouves.append(infos.symbole)
    return sorted(set(trouves))


def taches_entrainees(symbole: str, intervalle: str, horizon: int) -> list[str]:
    """
    Objectifs de classification réellement entraînés pour cette crypto et cet horizon.

    Sert à ne pas envoyer l'utilisateur chercher un modèle « direction » quand
    il vient d'entraîner « direction_nette » : le fichier sur le disque fait
    foi, pas le réglage affiché.
    """
    trouvees = []
    for cle in lister_modeles():
        infos = analyser_cle(cle)
        if (infos.symbole == symbole and infos.intervalle == intervalle
                and infos.horizon == horizon and not infos.walk_forward):
            trouvees.append(infos.tache)
    return sorted(set(trouvees))
