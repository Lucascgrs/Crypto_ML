"""
Définition des cibles : ce qu'on demande exactement au modèle d'apprendre.

Toutes les cibles se construisent à partir des colonnes DÉJÀ présentes dans le
fichier analysé (`variation_x`, `High`, `Low`, `Close`, `ATR_Pct`). Aucune
donnée supplémentaire n'est nécessaire — l'amplitude du mouvement était là
depuis le début, on ne s'en servait simplement pas.

Quatre objectifs, tous ramenés à une classification pour réutiliser la même
machinerie de confiance (calibration, seuil, table de précision) :

    direction         signe de variation_h  — la question historique
    direction_nette   signe, mais seulement au-delà des frais  — solution 5
    barriere          take-profit / stop-loss / temps  — solution 6
    amplitude         5 classes graduées par l'ATR  — solution 3

La régression (intervalle de prix, volatilité attendue) vit dans `amplitude.py` :
elle ne rend pas une classe mais une valeur, et sa machinerie est différente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import config

# Étiquettes de direction utilisées partout (fichiers de prédiction, backtest,
# graphiques). « NEUTRE » signifie « aucune position à prendre ».
HAUSSE = "HAUSSE"
BAISSE = "BAISSE"
NEUTRE = "NEUTRE"


# ===========================================================================
# BRIQUES DE CALCUL
# ===========================================================================
def _variation(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Colonne `variation_h` en pourcentage, nettoyée."""
    colonne = config.colonne_variation(horizon)
    if colonne not in df.columns:
        raise ValueError(f"Colonne {colonne} absente : relance l'étape Analyse.")
    variation = pd.to_numeric(df[colonne], errors="coerce")
    return variation.replace([np.inf, -np.inf], np.nan)


def echelle_attendue(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Amplitude « normale » d'un mouvement sur `horizon` périodes, en pourcentage.

    C'est l'ATR de la bougie, dilaté en racine du temps : sur un marché sans
    mémoire, l'amplitude cumulée croît comme √h. Sert d'unité de mesure aux
    classes d'amplitude et aux barrières — un mouvement de 1 % n'a pas du tout
    le même sens en marché calme et en pleine tempête.
    """
    atr = pd.to_numeric(df["ATR_Pct"], errors="coerce")
    return atr * np.sqrt(max(1, int(horizon))) * 100


def triple_barriere(df: pd.DataFrame, horizon: int,
                    tp_atr: float = config.BARRIERE_TP_ATR,
                    sl_atr: float = config.BARRIERE_SL_ATR) -> pd.Series:
    """
    Étiquetage par triple barrière (López de Prado).

    Trois issues possibles à partir de chaque bougie :
      * le prix touche d'abord la barrière HAUTE (take-profit)  -> 1
      * il touche d'abord la barrière BASSE (stop-loss)         -> 0
      * il n'en touche aucune avant l'échéance                  -> signe final

    Les barrières sont posées à `tp_atr` / `sl_atr` fois l'ATR de la bougie
    d'entrée : elles s'écartent donc automatiquement en marché agité. C'est ce
    qui rend l'étiquette bien plus proche d'un vrai trade que le simple signe
    de la variation finale — un mouvement qui monte de 3 % avant de revenir à
    son point de départ n'est plus compté comme « stagnation ».

    Quand les deux barrières sont franchies dans la MÊME bougie, l'ordre réel
    est inconnu (on n'a pas le détail intra-bougie) : on retombe alors sur le
    signe de la variation finale plutôt que d'inventer une issue.
    """
    horizon = max(1, int(horizon))
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    haut = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    bas = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(df["ATR_Pct"], errors="coerce").to_numpy(dtype=float)

    barriere_haute = close * (1 + tp_atr * atr)
    barriere_basse = close * (1 - sl_atr * atr)

    n = len(close)
    premier_tp = np.full(n, np.inf)
    premier_sl = np.full(n, np.inf)

    # Une passe vectorisée par décalage : `horizon` passes au lieu de n × horizon
    # itérations Python. Sur 75 000 bougies et un horizon de 12, c'est instantané.
    for decalage in range(1, horizon + 1):
        futur_haut = np.concatenate([haut[decalage:], np.full(decalage, np.nan)])
        futur_bas = np.concatenate([bas[decalage:], np.full(decalage, np.nan)])
        premier_tp = np.where(np.isinf(premier_tp) & (futur_haut >= barriere_haute),
                              decalage, premier_tp)
        premier_sl = np.where(np.isinf(premier_sl) & (futur_bas <= barriere_basse),
                              decalage, premier_sl)

    variation = _variation(df, horizon).to_numpy(dtype=float)
    etiquette = np.where(
        premier_tp < premier_sl, 1.0,
        np.where(premier_sl < premier_tp, 0.0,
                 np.where(variation > 0, 1.0, 0.0)))

    # Aucune barrière atteinte ET issue finale inconnue : le futur n'est pas
    # encore arrivé, on ne peut rien étiqueter.
    inconnu = np.isinf(premier_tp) & np.isinf(premier_sl) & np.isnan(variation)
    etiquette[inconnu] = np.nan
    return pd.Series(etiquette, index=df.index)


# ===========================================================================
# LES QUATRE CIBLES
# ===========================================================================
def cible_direction(df: pd.DataFrame, horizon: int) -> pd.Series:
    """1 si le prix est plus haut dans `horizon` périodes, 0 sinon."""
    variation = _variation(df, horizon)
    return pd.Series(np.where(variation > 0, 1.0, 0.0),
                     index=df.index).mask(variation.isna())


def cible_direction_nette(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Direction, mais uniquement pour les mouvements qui dépassent les frais.

    Les bougies dont la variation tient dans le coût d'un aller-retour
    (0.30 % par défaut) sont mises de côté : elles ne sont ni apprises, ni
    évaluées. Le modèle cesse ainsi d'user sa capacité sur du bruit qu'on ne
    pourrait de toute façon pas exploiter, et sa précision devient
    économiquement lisible — « quand il dit hausse, le mouvement paie ».
    """
    variation = _variation(df, horizon)
    cout = config.COUT_ALLER_RETOUR_PCT
    cible = pd.Series(np.nan, index=df.index)
    cible[variation > cout] = 1.0
    cible[variation < -cout] = 0.0
    return cible


def cible_barriere(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Triple barrière : 1 si le take-profit est touché avant le stop-loss."""
    return triple_barriere(df, horizon)


def bornes_amplitude(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Bornes des 5 classes, en variation normalisée par l'amplitude attendue.

    Ce sont les quintiles empiriques de `variation / (ATR × √h)`. Chaque classe
    pèse donc 20 % par construction, ce qui est indispensable : avec des bornes
    fixes, la classe « neutre » écrasait tout et le modèle se contentait de la
    répondre systématiquement (voir `config.QUANTILES_CLASSES`).

    RÉSERVE À CONNAÎTRE. Ces quantiles sont calculés sur tout l'historique
    disponible, blocs de test compris. C'est une fuite au sens strict, mais de
    la même nature qu'une standardisation de features : elle ne transmet que
    l'étendue globale de la distribution, jamais l'ordre des observations ni le
    lien entre features et cible. Les calculer sur le seul bloc d'apprentissage
    déplacerait les frontières entre les blocs, et la cible ne voudrait plus
    dire la même chose d'un bout à l'autre du fichier — ce qui serait pire.
    """
    variation = _variation(df, horizon)
    echelle = echelle_attendue(df, horizon).replace(0, np.nan)
    normalisee = (variation / echelle).dropna()
    if normalisee.empty:
        return np.array(config.QUANTILES_CLASSES, dtype=float)
    return normalisee.quantile(list(config.QUANTILES_CLASSES)).to_numpy()


def cible_amplitude(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Cinq classes graduées, des fortes baisses aux fortes hausses.

    La variation est d'abord normalisée par l'amplitude attendue (ATR dilaté en
    √h) : une classe dit donc « ce mouvement est grand PAR RAPPORT au régime de
    volatilité du moment », et garde le même sens en marché calme comme agité,
    et d'une crypto à l'autre.

    L'intérêt : chaque signal porte désormais une AMPLITUDE. « Forte hausse à
    62 % de confiance » et « hausse à 62 % » ne valent pas la même position.
    """
    variation = _variation(df, horizon)
    echelle = echelle_attendue(df, horizon).replace(0, np.nan)
    normalisee = variation / echelle

    bornes = bornes_amplitude(df, horizon)
    classe = pd.Series(np.nan, index=df.index, dtype=float)
    valides = normalisee.notna()
    classe[valides] = np.digitize(normalisee[valides].to_numpy(), bornes).astype(float)
    return classe


def centres_amplitude(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """
    Variation normalisée représentative de chaque classe (son centre).

    Sert à convertir une classe prédite en amplitude exprimée en pourcentage.
    Les classes extrêmes étant ouvertes, leur centre est extrapolé en
    prolongeant la largeur de la tranche voisine : c'est un ordre de grandeur
    destiné au dimensionnement des positions, pas une prévision fine.
    """
    bornes = bornes_amplitude(df, horizon)
    largeur_basse = max(1e-6, bornes[1] - bornes[0])
    largeur_haute = max(1e-6, bornes[-1] - bornes[-2])
    return np.concatenate([
        [bornes[0] - largeur_basse / 2],
        (bornes[:-1] + bornes[1:]) / 2,
        [bornes[-1] + largeur_haute / 2]])


# ===========================================================================
# CATALOGUE DES OBJECTIFS
# ===========================================================================
@dataclass(frozen=True)
class Tache:
    """Un objectif d'apprentissage : sa cible, ses classes, sa lecture."""

    cle: str
    libelle: str                     # texte affiché dans le menu
    description: str                 # explication sous le menu
    classes: tuple[str, ...]         # étiquettes, dans l'ordre des indices
    sens: tuple[str, ...]            # direction de trading associée à chaque classe
    construire: Callable[[pd.DataFrame, int], pd.Series]
    # Cible utilisée pour ÉVALUER, quand elle diffère de celle qu'on apprend.
    # Voir `construire_evaluation` : sans cette distinction, « direction nette »
    # afficherait une précision mesurée uniquement sur les gros mouvements,
    # c'est-à-dire conditionnée à un événement inconnu au moment de décider.
    evaluer: Callable[[pd.DataFrame, int], pd.Series] | None = None

    @property
    def construire_evaluation(self) -> Callable[[pd.DataFrame, int], pd.Series]:
        """
        Cible servant à mesurer la performance, sur TOUTES les bougies.

        Elle coïncide avec la cible d'apprentissage pour trois objectifs sur
        quatre. « Direction nette » fait exception : elle n'apprend que sur les
        mouvements dépassant les frais, mais doit être jugée sur l'ensemble —
        parce qu'en situation réelle, on ne sait pas d'avance si le mouvement
        sera grand. Mesurée sur le seul sous-ensemble filtré, sa précision
        atteignait 63.5 % sur BTC 1h à l'horizon 3, alors que les mêmes signaux
        rejoués en backtest donnaient 34 % de trades gagnants : l'écart venait
        entièrement de ce biais de sélection.
        """
        return self.evaluer or self.construire

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def binaire(self) -> bool:
        return self.n_classes == 2

    @property
    def confiance_neutre(self) -> float:
        """Confiance d'un modèle qui répondrait au hasard (1/2, 1/5…)."""
        return 1.0 / self.n_classes

    def indices_hausse(self) -> list[int]:
        """Indices des classes qui correspondent à une position acheteuse."""
        return [i for i, s in enumerate(self.sens) if s == HAUSSE]

    def indices_baisse(self) -> list[int]:
        return [i for i, s in enumerate(self.sens) if s == BAISSE]


TACHES: dict[str, Tache] = {
    "direction": Tache(
        cle="direction",
        libelle="Direction — hausse ou baisse",
        description="La question de base : dans X périodes, le prix sera-t-il plus "
                    "haut ou plus bas ? Toutes les bougies comptent, y compris les "
                    "mouvements minuscules.",
        classes=(BAISSE, HAUSSE),
        sens=(BAISSE, HAUSSE),
        construire=cible_direction,
    ),
    "direction_nette": Tache(
        cle="direction_nette",
        libelle="Direction nette — au-delà des frais",
        description=f"Même question, mais les mouvements inférieurs à "
                    f"{config.COUT_ALLER_RETOUR_PCT:.2f} % (le coût d'un aller-retour) "
                    f"sont ignorés À L'APPRENTISSAGE. Le modèle n'use pas sa "
                    f"capacité sur du bruit inexploitable. L'évaluation, elle, "
                    f"porte sur toutes les bougies : en pratique on ne sait pas "
                    f"d'avance si le mouvement sera grand.",
        classes=(BAISSE, HAUSSE),
        sens=(BAISSE, HAUSSE),
        construire=cible_direction_nette,
        evaluer=cible_direction,
    ),
    "barriere": Tache(
        cle="barriere",
        libelle="Triple barrière — TP / SL / temps",
        description="On simule un vrai trade : le prix touche-t-il d'abord le "
                    "take-profit (+1 ATR) ou le stop-loss (−1 ATR) ? À défaut, on "
                    "prend le signe à l'échéance. Beaucoup plus proche de ce qui "
                    "se passe en pratique que le simple signe final.",
        classes=(BAISSE, HAUSSE),
        sens=(BAISSE, HAUSSE),
        construire=cible_barriere,
    ),
    "amplitude": Tache(
        cle="amplitude",
        libelle="Amplitude — 5 classes",
        description="Cinq classes d'égale fréquence, graduées en variation "
                    "rapportée à l'ATR : forte baisse, baisse, neutre, hausse, "
                    "forte hausse. Chaque signal porte enfin une amplitude, et "
                    "pas seulement un sens. La confiance part ici de 20 %.",
        classes=tuple(config.CLASSES_AMPLITUDE),
        sens=(BAISSE, BAISSE, NEUTRE, HAUSSE, HAUSSE),
        construire=cible_amplitude,
    ),
}

TACHE_DEFAUT = "direction"


def obtenir(cle: str | None) -> Tache:
    """Tâche correspondant à une clé, avec repli sur la direction simple."""
    return TACHES.get(cle or TACHE_DEFAUT, TACHES[TACHE_DEFAUT])


def par_libelle(libelle: str) -> Tache:
    """Tâche correspondant au libellé affiché dans l'interface."""
    for tache in TACHES.values():
        if tache.libelle == libelle:
            return tache
    return TACHES[TACHE_DEFAUT]


def libelles() -> list[str]:
    """Libellés des objectifs, dans l'ordre du menu."""
    return [tache.libelle for tache in TACHES.values()]


# ===========================================================================
# LECTURE DES PROBABILITÉS
# ===========================================================================
def lire_probabilites(probas: np.ndarray, tache: Tache) -> dict:
    """
    Traduit une matrice de probabilités en décisions lisibles.

    Retourne un dictionnaire de tableaux alignés sur les lignes de `probas` :

        classe        indice de la classe la plus probable
        etiquette     son nom (« Forte hausse », « HAUSSE »…)
        sens          HAUSSE / BAISSE / NEUTRE — ce qu'on ferait en pratique
        confiance     probabilité de la classe retenue
        proba_hausse  probabilité cumulée des classes haussières

    En binaire, `confiance` retrouve exactement max(p, 1−p) et `proba_hausse`
    vaut p : la lecture historique est un cas particulier de celle-ci.
    """
    probas = np.atleast_2d(np.asarray(probas, dtype=float))
    if probas.shape[1] != tache.n_classes:
        raise ValueError(f"{probas.shape[1]} colonnes de probabilité pour "
                         f"{tache.n_classes} classes attendues.")

    classe = probas.argmax(axis=1)
    confiance = probas.max(axis=1)
    indices_hausse = tache.indices_hausse()
    proba_hausse = probas[:, indices_hausse].sum(axis=1) if indices_hausse else np.zeros(len(probas))

    etiquettes = np.array(tache.classes, dtype=object)[classe]
    sens = np.array(tache.sens, dtype=object)[classe]

    return {"classe": classe, "etiquette": etiquettes, "sens": sens,
            "confiance": confiance, "proba_hausse": proba_hausse}
