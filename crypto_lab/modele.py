"""
Entraînement et prédiction (objectifs de CLASSIFICATION).

QUESTION POSÉE AU MODÈLE
------------------------
« Dans X périodes (X entre 1 et 24), que va-t-il se passer ? »

La forme exacte de la question dépend de l'objectif choisi (voir `cibles.py`) :

    direction         le prix sera-t-il plus haut ou plus bas ?
    direction_nette   … en ignorant les mouvements qui ne couvrent pas les frais
    barriere          touchera-t-on le take-profit avant le stop-loss ?
    amplitude         dans laquelle des 5 classes d'amplitude tombera-t-on ?

Quel que soit l'objectif, le modèle renvoie une matrice de probabilités dont on
tire toujours les trois mêmes colonnes :

    Sens_Predit = HAUSSE / BAISSE / NEUTRE  (ce qu'on ferait en pratique)
    Confiance   = probabilité de la classe retenue
    Proba_Hausse = probabilité cumulée des classes haussières

En binaire, `Confiance` retrouve exactement max(p, 1−p) : la lecture historique
est un cas particulier de celle-ci, et rien ne change pour l'objectif par défaut.

La prédiction reste **bidirectionnelle** : une probabilité de 0.15 est un signal
de baisse à 85 % de confiance, aussi exploitable qu'une probabilité de 0.85. Le
seuil de confiance sert ensuite à ne retenir que les prédictions dont le modèle
est sûr, dans un sens comme dans l'autre.

Les objectifs de RÉGRESSION (intervalle de prix attendu, volatilité, espérance
de gain) vivent dans `amplitude.py` : ils ne rendent pas une classe mais une
valeur, et leur évaluation n'a rien à voir.

CE QUI EST AUTOMATIQUE (l'interface n'expose que crypto / modèle / horizon / seuil)
----------------------------------------------------------------------------------
  * Découpage chronologique 70 / 15 / 15 avec embargo (voir `_decouper`).
  * Rééquilibrage des classes si l'une dépasse 55 %.
  * Réglage des hyperparamètres : 3 configurations par défaut (18 ou 40 sur
    demande, voir `config.MODES_RECHERCHE`), départagées en validation croisée
    purgée ; à égalité statistique, la plus simple est retenue.
  * Early stopping sur la validation pour les modèles à boosting.
  * Calibration des probabilités (isotonique ou Platt selon la taille).
  * Utilité de chaque feature mesurée sur la validation, et filtrable au
    prochain entraînement (`seuil_utilite`).
  * Dimensionnement RAM / CPU (voir `ressources.py`).
  * Progression, courbe d'apprentissage et arrêt propre publiés dans
    `suivi.MONITEUR` — sans coût quand personne n'écoute.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import cibles, config, ressources, stockage, suivi

# --- Librairies optionnelles ------------------------------------------------
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:                                    # pragma: no cover
    XGBOOST_OK = False

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    LIGHTGBM_OK = True
except ImportError:                                    # pragma: no cover
    LIGHTGBM_OK = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:                                    # pragma: no cover
    CATBOOST_OK = False


# ===========================================================================
# CATALOGUE DES MODÈLES
# ===========================================================================
# Pour chaque modèle : 3 configurations couvrant « prudent / équilibré /
# souple ». Elles sont toutes entraînées puis départagées sur la validation.
# Trois essais suffisent : avec 8 features seulement, le gain d'une grille
# exhaustive est négligeable face au temps qu'elle coûte.
CONFIGURATIONS = {
    "XGBoost": [
        {"max_depth": 3, "learning_rate": 0.02, "subsample": 0.7,
         "colsample_bytree": 0.7, "min_child_weight": 30, "reg_lambda": 5.0},
        {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 2.0},
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.9,
         "colsample_bytree": 0.9, "min_child_weight": 3, "reg_lambda": 1.0},
    ],
    "LightGBM": [
        {"num_leaves": 15, "learning_rate": 0.02, "min_child_samples": 200,
         "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 5.0},
        {"num_leaves": 31, "learning_rate": 0.03, "min_child_samples": 60,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20,
         "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    ],
    "CatBoost": [
        {"depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 6.0},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 1.0},
    ],
    "RandomForest": [
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 100,
         "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 14, "min_samples_leaf": 30,
         "max_features": 0.6},
        {"n_estimators": 800, "max_depth": None, "min_samples_leaf": 10,
         "max_features": 0.8},
    ],
    "RegressionLogistique": [
        {"C": 0.05}, {"C": 0.5}, {"C": 5.0},
    ],
}


# ===========================================================================
# RECHERCHE ÉTENDUE D'HYPERPARAMÈTRES
# ===========================================================================
# Les trois configurations ci-dessus sont des points de départ écrits à la
# main. Les grilles suivantes définissent l'espace dans lequel les modes
# « approfondie » et « exhaustive » tirent des candidates supplémentaires (voir
# `config.MODES_RECHERCHE`).
#
# Tirage ALÉATOIRE et non balayage complet : sur huit paramètres à six valeurs,
# un balayage exhaustif ferait 1,7 million de combinaisons. La recherche
# aléatoire trouve un réglage équivalent en quelques dizaines d'essais, parce
# que la performance ne dépend fortement que de deux ou trois paramètres et que
# le tirage les explore tous en même temps (Bergstra & Bengio, 2012).
#
# Le tirage est semé par `config.SEED` : deux entraînements identiques testent
# exactement les mêmes candidates, sinon comparer deux exécutions n'aurait
# aucun sens.
GRILLES = {
    "XGBoost": {
        "max_depth": [2, 3, 4, 5, 6, 7, 8],
        "learning_rate": [0.008, 0.012, 0.02, 0.03, 0.05, 0.08],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 10, 20, 30, 60, 120, 250],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
        "reg_alpha": [0.0, 0.1, 0.5, 2.0],
        "gamma": [0.0, 0.1, 0.5, 2.0],
    },
    "LightGBM": {
        "num_leaves": [7, 15, 31, 63, 127],
        "learning_rate": [0.008, 0.012, 0.02, 0.03, 0.05, 0.08],
        "min_child_samples": [10, 20, 60, 120, 250, 500],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
        "reg_alpha": [0.0, 0.1, 0.5, 2.0],
    },
    "CatBoost": {
        "depth": [3, 4, 5, 6, 7, 8],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "l2_leaf_reg": [1.0, 2.0, 3.0, 6.0, 10.0, 20.0],
        "random_strength": [0.5, 1.0, 2.0, 4.0],
    },
    "RandomForest": {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [6, 8, 12, 16, 24, None],
        "min_samples_leaf": [5, 10, 30, 60, 100, 200],
        "max_features": ["sqrt", 0.4, 0.6, 0.8, 1.0],
    },
    "RegressionLogistique": {
        "C": [0.005, 0.02, 0.05, 0.2, 0.5, 1.0, 5.0, 20.0],
    },
}

# Poids de « capacité » : combien chaque paramètre augmente (positif) ou réduit
# (négatif) le risque de surapprentissage. Ils ne servent qu'à RANGER les
# candidates de la plus prudente à la plus souple.
#
# Pourquoi cet ordre compte : la règle du 1 écart-type retient la PREMIÈRE
# configuration statistiquement à égalité avec la meilleure. Sans tri, cette
# « première » serait un tirage au sort ; avec le tri, c'est la plus sobre —
# celle qui a le plus de chances de tenir hors échantillon.
CAPACITE = {
    "XGBoost": {"max_depth": 1.0, "learning_rate": 25.0, "subsample": 1.0,
                "colsample_bytree": 1.5, "min_child_weight": -0.015,
                "reg_lambda": -0.10, "reg_alpha": -0.30, "gamma": -0.60},
    "LightGBM": {"num_leaves": 0.06, "learning_rate": 25.0, "subsample": 1.0,
                 "colsample_bytree": 1.5, "min_child_samples": -0.008,
                 "reg_lambda": -0.10, "reg_alpha": -0.30},
    "CatBoost": {"depth": 1.0, "learning_rate": 25.0, "l2_leaf_reg": -0.15,
                 "random_strength": -0.20},
    "RandomForest": {"max_depth": 0.15, "min_samples_leaf": -0.02,
                     "max_features": 1.0, "n_estimators": 0.0},
    "RegressionLogistique": {"C": 0.4},
}

# Profondeur attribuée à `max_depth=None` (arbres non bornés) dans le calcul
# de capacité : c'est la configuration la plus souple qui soit.
PROFONDEUR_INFINIE = 40


def _capacite(nom_modele: str, params: dict) -> float:
    """
    Score de souplesse d'une configuration. Plus il est haut, plus le modèle
    peut coller au bruit.

    Volontairement grossier : il ne sert pas à prédire la performance, mais à
    départager deux candidates dont les AUC sont indiscernables.
    """
    poids = CAPACITE.get(nom_modele, {})
    score = 0.0
    for cle, facteur in poids.items():
        valeur = params.get(cle)
        if valeur is None and cle == "max_depth":
            valeur = PROFONDEUR_INFINIE          # arbres non bornés
        if isinstance(valeur, str):
            # `max_features="sqrt"` : la plus économe des options.
            valeur = 0.2 if valeur == "sqrt" else 0.5
        if valeur is None:
            continue
        score += facteur * float(valeur)
    return score


def _signature(params: dict) -> tuple:
    """Empreinte comparable d'une configuration, pour éviter les doublons."""
    return tuple(sorted((str(cle), str(valeur)) for cle, valeur in params.items()))


def configurations_candidates(nom_modele: str, mode: str | None = None) -> list[dict]:
    """
    Configurations à tester pour ce modèle et ce mode de recherche.

    Le mode rapide rend les trois configurations de référence, dans leur ordre
    d'origine. Les modes plus longs y ajoutent des tirages aléatoires puis
    trient l'ensemble de la plus prudente à la plus souple.

    Les trois références font toujours partie du lot : une recherche longue ne
    peut donc jamais rendre un résultat pire qu'une recherche courte, elle peut
    seulement ne rien trouver de mieux.
    """
    base = [dict(params) for params in CONFIGURATIONS[nom_modele]]
    cible = int(config.mode_recherche(mode)["n_configurations"])
    grille = GRILLES.get(nom_modele)

    if cible <= len(base) or not grille:
        return base

    tirage = random.Random(config.SEED)
    vues = {_signature(params) for params in base}
    candidates = list(base)

    # Une grille peut être plus petite que la cible (la régression logistique
    # n'a qu'un paramètre) : le compteur d'essais évite la boucle infinie.
    essais = 0
    while len(candidates) < cible and essais < cible * 60:
        essais += 1
        params = {cle: tirage.choice(valeurs) for cle, valeurs in grille.items()}
        empreinte = _signature(params)
        if empreinte in vues:
            continue
        vues.add(empreinte)
        candidates.append(params)

    return sorted(candidates, key=lambda params: _capacite(nom_modele, params))


# ===========================================================================
# COURBE D'APPRENTISSAGE EN DIRECT
# ===========================================================================
# Les callbacks ci-dessous publient, tous les `suivi.PAS_COURBE` arbres, la
# valeur de la métrique d'arrêt sur l'apprentissage et sur la validation. Ils
# servent deux choses à la fois :
#
#   * MONTRER l'entraînement — l'écart qui se creuse entre les deux courbes est
#     le surapprentissage, en direct ;
#   * PERMETTRE L'ARRÊT — chacun consulte le drapeau du moniteur à chaque
#     arbre, si bien qu'un « Arrêter » se traduit par un arrêt en une fraction
#     de seconde au lieu d'attendre la fin de l'ajustement en cours.
#
# Tout est enveloppé dans des `try` : une version de librairie dont l'API de
# callback aurait changé doit dégrader le suivi, jamais casser l'entraînement.
NOMS_JEUX = {"validation_0": "apprentissage", "validation_1": "validation",
             "training": "apprentissage", "valid_0": "validation",
             "valid_1": "validation", "learn": "apprentissage"}


def _nom_jeu(brut: str, total: int) -> str:
    """Nom lisible d'un jeu d'évaluation ('apprentissage' ou 'validation')."""
    if total <= 1:
        return "validation"
    return NOMS_JEUX.get(str(brut), str(brut))


def _callback_xgb(etiquette: str):
    """Callback XGBoost : publie la courbe et respecte la demande d'arrêt."""
    try:
        from xgboost.callback import TrainingCallback
    except ImportError:                                # pragma: no cover
        return None

    class _Courbe(TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch % suivi.PAS_COURBE == 0 and evals_log:
                total = len(evals_log)
                valeurs = {}
                for jeu, metriques in evals_log.items():
                    for historique in metriques.values():
                        if len(historique):
                            valeurs[_nom_jeu(jeu, total)] = float(historique[-1])
                        break
                suivi.MONITEUR.courbe(epoch, valeurs, etiquette)
            # Retourner True demande à XGBoost de s'arrêter proprement.
            return suivi.MONITEUR.arret_demande()

    return _Courbe()


def _callback_lgbm(etiquette: str):
    """Callback LightGBM, même rôle que `_callback_xgb`."""
    try:
        from lightgbm.callback import EarlyStopException
    except ImportError:                                # pragma: no cover
        return None

    def rappel(env):
        resultats = list(env.evaluation_result_list or [])
        if env.iteration % suivi.PAS_COURBE == 0 and resultats:
            noms = {ligne[0] for ligne in resultats}
            valeurs = {_nom_jeu(ligne[0], len(noms)): float(ligne[2])
                       for ligne in resultats}
            suivi.MONITEUR.courbe(env.iteration, valeurs, etiquette)
        if suivi.MONITEUR.arret_demande():
            raise EarlyStopException(env.iteration, resultats)

    rappel.order = 40
    return rappel


def _callback_catboost(etiquette: str):
    """Callback CatBoost. Renvoyer False interrompt l'apprentissage."""

    class _Courbe:
        def after_iteration(self, info):
            metriques = getattr(info, "metrics", None) or {}
            iteration = int(getattr(info, "iteration", 0))
            if iteration % suivi.PAS_COURBE == 0 and metriques:
                total = len(metriques)
                valeurs = {}
                for jeu, mesures in metriques.items():
                    for historique in mesures.values():
                        if len(historique):
                            valeurs[_nom_jeu(jeu, total)] = float(historique[-1])
                        break
                suivi.MONITEUR.courbe(iteration, valeurs, etiquette)
            return not suivi.MONITEUR.arret_demande()

    return _Courbe()


# Modèles capables de s'arrêter tout seuls quand la validation ne progresse plus.
AVEC_EARLY_STOPPING = {"XGBoost", "LightGBM", "CatBoost"}

# Description courte affichée dans l'interface sous le sélecteur de modèle.
DESCRIPTIONS_MODELES = {
    "XGBoost": "Référence sur données tabulaires. Rapide, robuste au bruit, "
               "s'arrête tout seul quand il n'apprend plus. Choix par défaut.",
    "LightGBM": "Très rapide sur gros historiques, souvent au niveau de XGBoost. "
                "Un peu plus sensible au surapprentissage sur peu de données.",
    "CatBoost": "Excellent sans réglage, très résistant au surapprentissage. "
                "Entraînement plus lent que les deux autres.",
    "RandomForest": "Simple et stable, difficile à faire surapprendre. "
                    "Généralement un cran en dessous du boosting en précision.",
    "RegressionLogistique": "Modèle linéaire : la baseline de référence. "
                            "Si le boosting ne la bat pas, le signal est très faible.",
}


def modeles_disponibles() -> list[str]:
    """Modèles réellement utilisables compte tenu des librairies installées."""
    dispo = []
    if XGBOOST_OK:
        dispo.append("XGBoost")
    if LIGHTGBM_OK:
        dispo.append("LightGBM")
    if CATBOOST_OK:
        dispo.append("CatBoost")
    dispo += ["RandomForest", "RegressionLogistique"]
    return dispo


def _construire(nom: str, params: dict, res: ressources.Ressources,
                poids_positif: float | None, n_classes: int = 2) -> object:
    """
    Instancie un classifieur non entraîné.

    Les réglages « machine » (nombre de threads, finesse des histogrammes)
    viennent de `ressources.py` : c'est là que la RAM libre est convertie en
    précision et en vitesse.

    `n_classes` bascule l'objectif et la métrique d'arrêt : au-delà de deux
    classes (objectif « amplitude »), c'est le logloss multi-classe qui pilote
    l'early stopping, et le rééquilibrage passe par des poids par classe plutôt
    que par un simple `scale_pos_weight`.
    """
    equilibre = poids_positif is not None
    multi = n_classes > 2

    if nom == "XGBoost":
        specifiques = ({"objective": "multi:softprob", "eval_metric": "mlogloss",
                        "num_class": n_classes} if multi else
                       {"objective": "binary:logistic", "eval_metric": "logloss",
                        "scale_pos_weight": poids_positif if equilibre else 1.0})
        return XGBClassifier(
            tree_method="hist", n_estimators=config.ARBRES_MAX,
            early_stopping_rounds=config.PATIENCE_EARLY_STOP,
            max_bin=res.max_bin, n_jobs=res.n_jobs, random_state=config.SEED,
            verbosity=0, **specifiques, **params)

    if nom == "LightGBM":
        specifiques = ({"objective": "multiclass", "num_class": n_classes,
                        "class_weight": "balanced" if equilibre else None} if multi else
                       {"objective": "binary",
                        "scale_pos_weight": poids_positif if equilibre else 1.0})
        return LGBMClassifier(
            n_estimators=config.ARBRES_MAX,
            max_bin=res.max_bin, n_jobs=res.n_jobs, random_state=config.SEED,
            verbose=-1, histogram_pool_size=res.pool_histogramme_mo,
            **specifiques, **params)

    if nom == "CatBoost":
        return CatBoostClassifier(
            iterations=config.ARBRES_MAX,
            eval_metric="MultiClass" if multi else "Logloss",
            loss_function="MultiClass" if multi else "Logloss",
            border_count=min(254, res.max_bin),   # CatBoost CPU plafonne à 254
            thread_count=res.n_jobs, random_seed=config.SEED, verbose=0,
            auto_class_weights="Balanced" if equilibre else None,
            **params)

    if nom == "RandomForest":
        return RandomForestClassifier(
            n_jobs=res.n_jobs, random_state=config.SEED,
            class_weight="balanced" if equilibre else None,
            **params)

    if nom == "RegressionLogistique":
        # Un modèle linéaire exige des variables à la même échelle : le RSI
        # varie de 0 à 100, l'ATR% autour de 0.01. D'où le StandardScaler.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=config.SEED,
                               class_weight="balanced" if equilibre else None,
                               **params))

    raise ValueError(f"Modèle inconnu : {nom}")


# ===========================================================================
# CALIBRATION DES PROBABILITÉS
# ===========================================================================
# Nombre minimal d'observations derrière chaque valeur de probabilité calibrée.
# C'est ce qui empêche d'annoncer « 95 % de confiance » sur la foi de 4 bougies.
SUPPORT_MINIMUM = 250

# Taille de bloc de calibration à partir de laquelle les fréquences observées
# sont jugées fiables. En dessous, elles sont ramenées vers 0.5 au prorata (voir
# `Calibrateur._facteur_confiance`) : sur 230 bougies, une « fréquence de hausse
# de 67 % » est indiscernable du hasard, et la laisser passer produirait un
# seuil de confiance élevé mais faux — le pire des cas pour un outil de tri.
CALIBRATION_FIABLE = 1250

# Seuils de confiance évalués après l'entraînement. La grille est resserrée
# entre 0.50 et 0.60 : c'est là que se joue tout sur des marchés très bruités,
# où un modèle honnête dépasse rarement 0.60 de confiance.
SEUILS_TESTES = (0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65, 0.70, 0.80)

# Un seuil n'est conseillé que s'il laisse passer assez de signaux pour être
# mesurable, et s'il gagne vraiment sur l'absence de filtrage.
SIGNAUX_MINIMUM = 200
GAIN_MINIMUM = 0.005


class Calibrateur:
    """
    Corrige les probabilités brutes du modèle.

    Un modèle qui annonce « 70 % » n'a pas forcément raison 70 % du temps : le
    boosting est réputé pour ses probabilités mal calibrées. La correction est
    apprise sur un bloc de validation que le modèle n'a jamais vu.

    Deux méthodes, choisies selon la taille de l'échantillon :

    * **isotonique par quantiles** — les probabilités brutes sont regroupées en
      tranches d'au moins `SUPPORT_MINIMUM` observations, et la régression
      isotonique est ajustée sur la fréquence réelle de hausse de chaque
      tranche. Sans ce regroupement, l'isotonique classique produit des valeurs
      extrêmes (0 ou 1) appuyées sur une poignée de points : le seuil de
      confiance deviendrait alors trompeur, ce qui ruinerait justement l'usage
      qu'on veut en faire.
    * **Platt / sigmoïde** — deux paramètres seulement, utilisée quand
      l'échantillon est trop petit pour former des tranches fiables.

    Dans les deux cas, la sortie est **bornée par ce qui a été observé** : on ne
    peut jamais annoncer une confiance supérieure à la meilleure fréquence de
    réussite réellement constatée sur les données de calibration. Sans cette
    borne, Platt extrapole allègrement — mesuré sur BTC en 1d, il annonçait
    68 % de confiance pour 47 % de réussite réelle, ce qui rend le seuil non
    seulement inutile mais trompeur.
    """

    def __init__(self):
        self.methode = "aucune"
        self._modele = None
        self.plage = (0.001, 0.999)   # bornes soutenues par les observations
        self.facteur_confiance = 1.0  # < 1 quand l'échantillon est trop petit

    def entrainer(self, probas: np.ndarray, verite: np.ndarray) -> "Calibrateur":
        probas = np.asarray(probas, dtype=float)
        verite = np.asarray(verite, dtype=float)

        if len(np.unique(verite)) < 2:
            self.methode = "aucune"                    # une seule classe : rien à calibrer
            return self

        self.facteur_confiance = self._facteur_confiance(len(probas))

        n_tranches = len(probas) // SUPPORT_MINIMUM
        if n_tranches >= 5:
            self.methode = "isotonique"
            centres, frequences, effectifs = self._regrouper(probas, verite, n_tranches)
            self._modele = IsotonicRegression(out_of_bounds="clip")
            self._modele.fit(centres, self._attenuer(frequences),
                             sample_weight=effectifs)
        else:
            self.methode = "platt"
            self._modele = LogisticRegression(max_iter=1000)
            self._modele.fit(probas.reshape(-1, 1), verite)

        self.plage = self._plage_observee(probas, verite)

        if self.facteur_confiance < 1:
            print(f"⚠️  Seulement {len(probas):,} bougies pour calibrer : les écarts "
                  f"observés sont ramenés vers 50 % (×{self.facteur_confiance:.2f}).")
            print("   La confiance affichée restera basse — c'est volontaire : sur si "
                  "peu de données, un chiffre élevé serait du hasard.")
        return self

    @staticmethod
    def _facteur_confiance(n: int) -> float:
        """
        Crédit accordé aux fréquences observées, selon la taille de l'échantillon.

        1.0 dès `CALIBRATION_FIABLE` observations, proportionnel en dessous. Un
        écart à 50 % mesuré sur 230 bougies a une marge d'erreur de plusieurs
        points : l'atténuer évite d'afficher une confiance que rien ne soutient.
        """
        return float(min(1.0, n / CALIBRATION_FIABLE))

    def _attenuer(self, frequences: np.ndarray) -> np.ndarray:
        """Rapproche les fréquences de 0.5 au prorata de la fiabilité de l'échantillon."""
        return 0.5 + (np.asarray(frequences) - 0.5) * self.facteur_confiance

    def _plage_observee(self, probas: np.ndarray,
                        verite: np.ndarray) -> tuple[float, float]:
        """
        Fréquences de hausse extrêmes réellement constatées, en tranches larges.

        Ces deux valeurs bornent la sortie du calibrateur : elles disent jusqu'où
        le modèle a effectivement démontré qu'il savait trancher. Les tranches
        sont volontairement larges (au moins 250 points, 8 au maximum) pour que
        chaque borne repose sur un échantillon solide, puis atténuées comme les
        autres — c'est ce qui empêche Platt d'extrapoler librement.
        """
        n_tranches = min(8, max(2, len(probas) // SUPPORT_MINIMUM))
        groupes = np.array_split(np.argsort(probas), n_tranches)
        frequences = [verite[g].mean() for g in groupes if len(g)]
        if not frequences:
            return 0.001, 0.999
        frequences = self._attenuer(np.array(frequences))
        # Marge de jeu autour des fréquences observées, elle aussi proportionnelle
        # à la fiabilité : un échantillon faible n'a droit à aucune latitude.
        marge = 0.02 * self.facteur_confiance
        return (max(0.001, float(frequences.min()) - marge),
                min(0.999, float(frequences.max()) + marge))

    @staticmethod
    def _regrouper(probas: np.ndarray, verite: np.ndarray,
                   n_tranches: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Découpe les probabilités en tranches de même effectif.

        Retourne, pour chaque tranche : probabilité brute moyenne, fréquence de
        hausse observée, et effectif (qui sert de poids à la régression).
        """
        n_tranches = min(n_tranches, 20)
        rangs = np.argsort(probas)
        groupes = np.array_split(rangs, n_tranches)

        centres, frequences, effectifs = [], [], []
        for groupe in groupes:
            if len(groupe) == 0:
                continue
            centres.append(probas[groupe].mean())
            frequences.append(verite[groupe].mean())
            effectifs.append(len(groupe))
        return np.array(centres), np.array(frequences), np.array(effectifs, dtype=float)

    def appliquer(self, probas: np.ndarray) -> np.ndarray:
        """Applique la correction apprise, bornée aux fréquences observées."""
        if self._modele is None:
            return probas
        if self.methode == "isotonique":
            corrigees = self._modele.predict(probas)
        else:
            corrigees = self._modele.predict_proba(np.asarray(probas).reshape(-1, 1))[:, 1]
        return np.clip(corrigees, *self.plage)


class CalibrateurMulti:
    """
    Calibration d'une matrice de probabilités, quel que soit le nombre de classes.

    Principe « un contre tous » : chaque colonne est calibrée séparément par un
    `Calibrateur` — « quand le modèle annonce 40 % de forte hausse, à quelle
    fréquence est-ce vraiment une forte hausse ? » — puis les lignes sont
    renormalisées pour que les probabilités somment à 1.

    Le cas binaire est traité à part et reste **exactement** celui d'avant :
    un seul calibrateur sur la probabilité de hausse, la baisse étant son
    complément. Changer d'objectif ne change donc rien à la direction simple.
    """

    def __init__(self, n_classes: int = 2):
        self.n_classes = int(n_classes)
        self._calibrateurs: list[Calibrateur] = []

    @property
    def methode(self) -> str:
        return self._calibrateurs[0].methode if self._calibrateurs else "aucune"

    @property
    def facteur_confiance(self) -> float:
        if not self._calibrateurs:
            return 1.0
        return min(c.facteur_confiance for c in self._calibrateurs)

    def entrainer(self, probas: np.ndarray, verite: np.ndarray) -> "CalibrateurMulti":
        probas = np.atleast_2d(np.asarray(probas, dtype=float))
        verite = np.asarray(verite, dtype=int)

        if self.n_classes == 2:
            self._calibrateurs = [Calibrateur().entrainer(probas[:, 1], verite)]
            return self

        self._calibrateurs = [
            Calibrateur().entrainer(probas[:, classe], (verite == classe).astype(float))
            for classe in range(self.n_classes)]
        return self

    def appliquer(self, probas: np.ndarray) -> np.ndarray:
        """Renvoie une matrice calibrée (n, n_classes) dont chaque ligne somme à 1."""
        probas = np.atleast_2d(np.asarray(probas, dtype=float))
        if not self._calibrateurs:
            return probas

        if self.n_classes == 2:
            hausse = self._calibrateurs[0].appliquer(probas[:, 1])
            return np.column_stack([1.0 - hausse, hausse])

        corrigees = np.column_stack([
            calibrateur.appliquer(probas[:, classe])
            for classe, calibrateur in enumerate(self._calibrateurs)])
        # Les calibrateurs travaillent indépendamment : rien ne garantit que la
        # somme fasse 1. La renormalisation rétablit une vraie distribution.
        totaux = corrigees.sum(axis=1, keepdims=True)
        totaux[totaux <= 0] = 1.0
        return corrigees / totaux


def seuils_testes(n_classes: int = 2) -> tuple[float, ...]:
    """
    Grille de seuils de confiance adaptée au nombre de classes.

    Avec deux classes, la confiance part de 0.50 (le hasard) ; avec cinq, elle
    part de 0.20. Utiliser la même grille dans les deux cas ne retiendrait
    aucun signal en multi-classe et donnerait l'illusion d'un modèle muet.
    """
    if n_classes <= 2:
        return SEUILS_TESTES
    base = 1.0 / n_classes
    ecarts = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
    return tuple(round(min(0.95, base + ecart), 4) for ecart in ecarts)


# ===========================================================================
# PRÉPARATION DES DONNÉES
# ===========================================================================
@dataclass
class Jeu:
    """
    Features et cibles alignées, prêtes pour l'entraînement.

    Deux cibles cohabitent parce qu'elles répondent à deux besoins distincts :

      `y`             ce que le modèle APPREND ;
      `y_evaluation`  ce sur quoi on le JUGE, défini pour toutes les bougies.

    Elles ne diffèrent que pour l'objectif « direction nette », qui n'apprend
    que sur les mouvements dépassant les frais mais doit être évalué sur
    l'ensemble — sinon sa précision serait conditionnée à un événement inconnu
    au moment de décider (voir `cibles.Tache.construire_evaluation`).

    `apprenable` marque les lignes utilisables pour l'apprentissage. Les autres
    restent présentes pour que le découpage chronologique et l'évaluation
    portent sur la même série de bougies.
    """

    X: pd.DataFrame               # les 8 indicateurs (+ contexte si disponible)
    y: pd.Series                  # indice de classe appris (1 = hausse en binaire)
    horizon: int                  # nombre de périodes prédites
    tache: cibles.Tache = None    # objectif appris (direction par défaut)
    y_evaluation: pd.Series = None
    apprenable: pd.Series = None
    regime: pd.Series = None      # rang de volatilité, pour l'analyse par régime
    origine: pd.Series = None     # crypto d'origine de chaque ligne (panier)

    def __post_init__(self):
        if self.y_evaluation is None:
            self.y_evaluation = self.y
        if self.apprenable is None:
            self.apprenable = pd.Series(True, index=self.X.index)

    @property
    def n_classes(self) -> int:
        return (self.tache or cibles.obtenir(None)).n_classes

    @property
    def panier(self) -> bool:
        """Vrai si le jeu mélange plusieurs cryptos."""
        return self.origine is not None and self.origine.nunique() > 1


@dataclass
class Decoupage:
    """Indices positionnels des trois blocs chronologiques, et leurs dates."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    embargo: int
    date_validation: pd.Timestamp = None
    date_test: pd.Timestamp = None
    purgees: int = 0


def lire_analyse(symbole: str, intervalle: str) -> pd.DataFrame:
    """Charge le fichier analysé, avec un message explicite s'il manque."""
    df = stockage.lire_tableau(stockage.chemin_analyse(symbole, intervalle))
    if df is None or df.empty:
        raise FileNotFoundError(
            f"Fichier analysé introuvable pour {symbole} ({intervalle}). "
            f"Lance d'abord l'étape Analyse.")
    return df


def charger_jeu(symbole: str, intervalle: str, horizon: int,
                tache: str | cibles.Tache | None = None,
                contexte: bool = True) -> Jeu:
    """
    Prépare features et cible pour un horizon et un objectif donnés.

    Features : **liste blanche stricte**. Les 8 indicateurs de base, plus les
    colonnes de contexte réellement présentes dans le fichier (multi-timeframe
    en `_MTF`, funding, open interest). Aucun prix, aucune colonne
    `variation_*` ne peut se retrouver en entrée du modèle, même par accident.

    Cibles : construites par `cibles.py` selon l'objectif. Les bougies dont le
    futur n'est pas encore connu sont écartées. Celles que l'objectif exclut
    volontairement de l'APPRENTISSAGE (mouvements sous le coût des frais) sont
    conservées mais marquées non apprenables : elles serviront quand même à
    l'évaluation, puisqu'en pratique on ne sait pas d'avance dans quel cas on
    se trouve.
    """
    df = lire_analyse(symbole, intervalle)
    tache = tache if isinstance(tache, cibles.Tache) else cibles.obtenir(tache)

    features = config.features_disponibles(df.columns, contexte=contexte)
    manquants = [c for c in config.INDICATEURS if c not in features]
    if manquants:
        raise ValueError(
            f"Indicateurs absents du fichier analysé : {manquants}. "
            f"Relance l'étape Analyse (le format a changé).")

    X = df[features].astype("float32")
    y = tache.construire(df, horizon)
    y_evaluation = tache.construire_evaluation(df, horizon)

    valides = X.notna().all(axis=1) & y_evaluation.notna()
    apprenable = y[valides].notna()

    return Jeu(X=X[valides], y=y[valides].fillna(-1).astype(int),
               horizon=horizon, tache=tache,
               y_evaluation=y_evaluation[valides].astype(int),
               apprenable=apprenable,
               regime=_regime(df)[valides])


def _regime(df: pd.DataFrame) -> pd.Series:
    """
    Rang de volatilité de chaque bougie, entre 0 (calme) et 1 (agité).

    Sert uniquement à VENTILER l'évaluation : un modèle peut afficher 55 % de
    justesse en moyenne tout en n'ayant d'avantage que dans un seul régime de
    marché — auquel cas la bonne décision n'est pas de le jeter, mais de ne
    trader que dans ce régime. C'est la version économe des « modèles par
    régime » : on ne découpe pas les données, on découpe la LECTURE.

    Reconstruit à la volée pour les fichiers analysés avant l'ajout de la
    colonne, afin que la ventilation reste disponible partout.
    """
    if "Regime_Volatilite" in df.columns:
        return pd.to_numeric(df["Regime_Volatilite"], errors="coerce").fillna(0.5)
    if "ATR_Pct" in df.columns:
        rang = df["ATR_Pct"].rolling(config.FENETRE_REGIME, min_periods=50).rank(pct=True)
        return rang.fillna(0.5)
    return pd.Series(0.5, index=df.index)


def _decouper(dates: pd.Index, horizon: int,
              intervalle: str = "1h") -> Decoupage:
    """
    Découpe chronologique 70 / 15 / 15, aux mêmes DATES pour tout le monde.

    Le test est TOUJOURS la période la plus récente : c'est la seule façon
    honnête d'estimer ce que vaudra le modèle demain.

    Les frontières sont d'abord repérées sur le nombre de lignes (pour obtenir
    un vrai 70/15/15 en volume), puis converties en DATES et appliquées comme
    telles. Sur une seule crypto, cela revient exactement au découpage
    positionnel précédent. Sur un panier de plusieurs cryptos, c'est ce qui
    garantit que toutes basculent en validation puis en test le même jour :
    chacune apprend sur tout l'historique dont elle dispose avant cette date —
    une crypto née en 2018 s'entraîne de 2018 à la frontière, une née en 2015
    depuis 2015 — et toutes sont jugées sur exactement la même période de
    marché. Sans cela, on comparerait des modèles évalués sur des marchés
    différents.

    L'embargo purge `horizon` PÉRIODES à la fin de chaque bloc. Sans lui, les
    dernières lignes du train ont une cible qui regarde déjà dans la période de
    validation : le modèle « connaîtrait » une partie de ce sur quoi on
    l'évalue, et les scores seraient artificiellement bons.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    if n == 0:
        raise ValueError("Aucune ligne à découper.")

    date_validation = dates[min(int(n * config.PART_TRAIN), n - 1)]
    date_test = dates[min(int(n * (1 - config.PART_TEST)), n - 1)]
    marge = pd.Timedelta(hours=config.heures(intervalle) * max(0, int(horizon)))

    positions = np.arange(n)
    train = positions[dates < date_validation - marge]
    validation = positions[(dates >= date_validation) & (dates < date_test - marge)]
    test = positions[dates >= date_test]

    purgees = n - (len(train) + len(validation) + len(test))
    return Decoupage(train=train, validation=validation, test=test,
                     embargo=max(0, int(horizon)), date_validation=date_validation,
                     date_test=date_test, purgees=purgees)


def taille_effective(dates: pd.Index, horizon: int) -> float:
    """
    Nombre d'observations réellement INDÉPENDANTES parmi `dates`.

    Deux raisons de ne pas compter les lignes une par une :

      * **Étiquettes chevauchantes.** À l'horizon 5, la cible de la bougie de
        10 h regarde jusqu'à 15 h, celle de 11 h jusqu'à 16 h : cinq lignes
        consécutives décrivent en grande partie le même morceau de futur. Elles
        comptent pour une seule observation, pas cinq.
      * **Cryptos corrélées.** Dans un panier, BTC et ETH à la même heure
        bougent ensemble à plus de 80 % : deux lignes, mais presque une seule
        information. Ne compter que les horodatages DISTINCTS règle les deux
        cas d'un coup.

    Conséquence directe et inconfortable : une précision de 55 % mesurée sur
    1 200 signaux à l'horizon 5 ne repose que sur ~240 observations, et sa
    marge d'erreur est plus du double de ce qu'un comptage naïf annoncerait.
    C'est exactement ce qui sépare un avantage réel d'un mirage.
    """
    if dates is None or len(dates) == 0:
        return 0.0
    horizon = max(1, int(horizon))
    distinctes = len(pd.unique(pd.DatetimeIndex(dates).to_numpy()))
    return max(1.0, distinctes / horizon)


def marge_proportion(part: float, n_effectif: float, z: float = 1.96) -> float:
    """
    Demi-largeur de l'intervalle de confiance à 95 % d'une proportion.

    Appliquée à la taille EFFECTIVE et non au nombre de lignes : sans cette
    correction, toutes les précisions affichées par le projet paraîtraient deux
    à trois fois plus solides qu'elles ne le sont.
    """
    if n_effectif <= 1:
        return 1.0
    part = float(np.clip(part, 0.0, 1.0))
    return float(z * np.sqrt(max(part * (1 - part), 1e-12) / n_effectif))


def _sous_ensemble(jeu: Jeu, indices: np.ndarray, apprenable: bool):
    """
    (X, y) d'un bloc, avec la cible qui convient à son usage.

    `apprenable=True`  -> cible d'apprentissage, lignes exclues par l'objectif retirées.
    `apprenable=False` -> cible d'évaluation, toutes les lignes conservées.
    """
    X = jeu.X.iloc[indices]
    if not apprenable:
        return X, jeu.y_evaluation.iloc[indices]
    masque = jeu.apprenable.iloc[indices].to_numpy()
    return X[masque], jeu.y.iloc[indices][masque]


def _poids_classe_positive(y: pd.Series, n_classes: int = 2) -> float | None:
    """
    Poids à donner à la classe minoritaire pour rééquilibrer l'apprentissage.

    Retourne None si les classes sont déjà équilibrées (aucun rééquilibrage
    appliqué) : forcer un rééquilibrage inutile dégrade la calibration.

    En binaire, la valeur est le rapport baisses/hausses passé à
    `scale_pos_weight`. Au-delà, elle sert seulement de drapeau : les modèles
    multi-classes utilisent leur propre pondération « balanced », qui traite
    les cinq classes d'un coup.

    Le seuil de déclenchement est relatif au nombre de classes : 55 % pour deux
    classes, mais une classe qui pèse 55 % sur cinq est déjà écrasante. On
    compare donc à la part attendue (1/k) multipliée par le même facteur.
    """
    effectifs = y.value_counts()
    if len(effectifs) < 2:
        return None

    part_majoritaire = float(effectifs.max() / effectifs.sum())
    limite = config.SEUIL_DESEQUILIBRE if n_classes <= 2 else \
        min(0.90, config.SEUIL_DESEQUILIBRE * 2 / n_classes)
    if part_majoritaire < limite:
        return None

    if n_classes > 2:
        return 1.0                       # drapeau : « rééquilibrage demandé »
    n_hausse = int((y == 1).sum())
    n_baisse = int((y == 0).sum())
    return (n_baisse / n_hausse) if n_hausse else None


# ===========================================================================
# ENTRAÎNEMENT
# ===========================================================================
def _poids_echantillons(y, n_classes: int, equilibrer: bool):
    """
    Poids par ligne rééquilibrant les classes, ou None.

    XGBoost n'expose pas de `class_weight` : en multi-classe, son
    `scale_pos_weight` ne s'applique pas et le rééquilibrage doit passer par
    des poids d'échantillon explicites. Sans cela, le modèle « amplitude »
    apprend simplement à toujours répondre « Neutre », qui pèse 60 % des
    bougies — il obtient alors exactement le score de la réponse constante.
    """
    if not equilibrer or n_classes <= 2:
        return None
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)


def _detacher_callbacks(modele) -> None:
    """
    Retire les callbacks de suivi après l'ajustement.

    Indispensable : un callback garde une référence au moniteur, donc à des
    objets `threading.Event` et `queue.Queue` que `joblib.dump` ne sait pas
    sérialiser. Sans ce nettoyage, la sauvegarde du modèle échouerait — et
    seulement quand l'interface est ouverte, ce qui est le pire des bugs.
    """
    try:
        modele.set_params(callbacks=None)
    except (TypeError, ValueError, AttributeError):     # pragma: no cover
        pass


def _ajuster(modele, nom: str, X_tr, y_tr, X_val, y_val, n_classes: int = 2,
             poids_lignes=None, etiquette: str = "", courbe_train: bool = False):
    """
    Entraîne un modèle, avec early stopping sur la validation si supporté.

    `etiquette` non vide et moniteur actif : la courbe d'apprentissage est
    publiée en direct et la demande d'arrêt est consultée à chaque arbre.
    `courbe_train` ajoute l'apprentissage aux jeux évalués, ce qui rend visible
    l'écart avec la validation — c'est-à-dire le surapprentissage. Cela coûte
    une évaluation supplémentaire par arbre, donc c'est réservé à l'ajustement
    final, pas aux dizaines d'ajustements de la validation croisée.
    """
    suivre = bool(etiquette) and suivi.MONITEUR.actif

    if nom == "XGBoost":
        jeux = ([(X_tr, y_tr), (X_val, y_val)] if suivre and courbe_train
                else [(X_val, y_val)])
        if suivre:
            rappel = _callback_xgb(etiquette)
            if rappel is not None:
                try:
                    modele.set_params(callbacks=[rappel])
                except (TypeError, ValueError):         # pragma: no cover
                    pass
        modele.fit(X_tr, y_tr, sample_weight=poids_lignes,
                   eval_set=jeux, verbose=False)
        _detacher_callbacks(modele)
        return int(getattr(modele, "best_iteration", 0) or 0)

    if nom == "LightGBM":
        metrique = "multi_logloss" if n_classes > 2 else "binary_logloss"
        rappels = [early_stopping(config.PATIENCE_EARLY_STOP, verbose=False),
                   log_evaluation(0)]
        jeux = [(X_val, y_val)]
        if suivre:
            if courbe_train:
                jeux = [(X_tr, y_tr), (X_val, y_val)]
            rappel = _callback_lgbm(etiquette)
            if rappel is not None:
                rappels.append(rappel)
        modele.fit(X_tr, y_tr, eval_set=jeux, eval_metric=metrique,
                   callbacks=rappels)
        return int(getattr(modele, "best_iteration_", 0) or 0)

    if nom == "CatBoost":
        commun = dict(eval_set=(X_val, y_val), verbose=False,
                      early_stopping_rounds=config.PATIENCE_EARLY_STOP)
        rappel = _callback_catboost(etiquette) if suivre else None
        if rappel is not None:
            try:
                modele.fit(X_tr, y_tr, callbacks=[rappel], **commun)
                return int(modele.get_best_iteration() or 0)
            except Exception:                           # noqa: BLE001
                # API de callback indisponible sur cette version : on réajuste
                # sans suivi plutôt que de faire échouer l'entraînement.
                pass
        modele.fit(X_tr, y_tr, **commun)
        return int(modele.get_best_iteration() or 0)

    modele.fit(X_tr, y_tr)
    return int(getattr(modele, "n_estimators", 0) or 0)


def _auc(modele, X, y, n_classes: int = 2) -> float:
    """
    Pouvoir de séparation du modèle sur un jeu donné.

    En binaire, l'AUC-ROC classique. Au-delà, la moyenne des AUC « une classe
    contre toutes les autres » : même échelle de lecture (0.5 = hasard), et
    toujours comparable d'un objectif à l'autre.
    """
    presentes = np.unique(y)
    if len(presentes) < 2:
        return 0.5
    probas = modele.predict_proba(X)
    if n_classes <= 2:
        return float(roc_auc_score(y, probas[:, 1]))
    if len(presentes) < n_classes:
        # Une classe absente du bloc rendrait le calcul multi-classe impossible :
        # on se rabat sur les classes réellement observées.
        probas = probas[:, presentes.astype(int)]
        probas = probas / np.clip(probas.sum(axis=1, keepdims=True), 1e-12, None)
    return float(roc_auc_score(y, probas, multi_class="ovr", average="macro",
                               labels=presentes))


def _essayer(nom_modele, params, res, poids, X_tr, y_tr, X_sel, y_sel, n_classes=2,
             etiquette=""):
    """Entraîne UNE configuration et la note. Retourne (modèle, AUC, params, arbres)."""
    candidat = _construire(nom_modele, params, res, poids, n_classes)
    poids_lignes = _poids_echantillons(y_tr, n_classes, poids is not None)
    iterations = _ajuster(candidat, nom_modele, X_tr, y_tr, X_sel, y_sel,
                          n_classes, poids_lignes, etiquette=etiquette)
    return candidat, _auc(candidat, X_sel, y_sel, n_classes), params, iterations


def _tester_configurations(nom_modele, res, poids, X_tr, y_tr, X_sel, y_sel,
                           n_classes: int = 2, mode: str | None = None,
                           progression=None):
    """
    Entraîne les trois configurations candidates, l'une après l'autre.

    Les essais sont indépendants et pourraient tourner en parallèle, un
    processus chacun. Mesuré sur BTC 1h (horizon 12, 16 cœurs), c'est pourtant
    plus lent : XGBoost 4.1 s contre 3.5 s, et surtout RandomForest 131 s
    contre 46 s. La raison est que chaque modèle exploite DÉJÀ tous les cœurs
    en interne ; les répartir entre trois processus ne fait que diviser les
    threads de chacun et ajouter le coût de démarrage des processus.

    Le parallélisme reste utile là où le calcul est réellement séquentiel :
    l'importance par permutation (voir `importance_indicateurs`).
    """
    configurations = configurations_candidates(nom_modele, mode)
    total = len(configurations)
    print(f"🔎 Réglage automatique : {total} configurations testées "
          f"({res.n_jobs} threads chacune)…")

    essais = []
    for numero, params in enumerate(configurations, start=1):
        suivi.verifier()
        if progression is not None:
            progression((numero - 1) / total, f"configuration {numero}/{total}")
        essai = _essayer(nom_modele, params, res, poids, X_tr, y_tr, X_sel, y_sel,
                         n_classes, etiquette=f"config {numero}/{total}")
        suivi.MONITEUR.configuration(numero, total, params, essai[1])
        essais.append(essai)
    return essais


def _erreur_type_auc(auc: float, verite: np.ndarray,
                     reduction: float = 1.0) -> float:
    """
    Marge d'erreur d'une AUC, formule de Hanley & McNeil (1982).

    Sert à savoir si l'écart entre deux configurations est réel ou s'il tient
    dans le bruit d'échantillonnage. Sur 5 000 points, elle vaut environ 0.008 —
    soit bien plus que les écarts habituellement observés entre nos trois
    candidates, d'où la règle de sélection ci-dessous.

    `reduction` (entre 0 et 1) ramène les effectifs à leur taille EFFECTIVE :
    à l'horizon 5, cinq lignes consécutives décrivent le même morceau de futur
    et ne valent qu'une seule observation. Sans cette correction, la marge est
    sous-estimée d'un facteur racine de h, et la règle du 1 écart-type retient
    des modèles trop complexes sur des écarts qui n'existent pas.
    """
    verite = np.asarray(verite)
    classes = np.unique(verite)
    reduction = float(np.clip(reduction, 1e-6, 1.0))

    # Objectif multi-classe : l'AUC affichée est une moyenne « une contre
    # toutes ». Sa marge d'erreur est donc la moyenne des marges de chaque
    # découpage binaire correspondant.
    if len(classes) > 2:
        marges = [_erreur_type_auc(auc, (verite == classe).astype(int), reduction)
                  for classe in classes]
        return float(np.mean(marges)) if marges else 1.0

    n_pos = float(np.sum(verite == 1)) * reduction
    n_neg = float(np.sum(verite == 0)) * reduction
    if n_pos < 2 or n_neg < 2:
        return 1.0

    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    variance = (auc * (1 - auc)
                + (n_pos - 1) * (q1 - auc ** 2)
                + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(variance, 0.0)))


def blocs_cv(dates: pd.Index, horizon: int, intervalle: str,
             n_blocs: int = config.BLOCS_CV) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Blocs chronologiques successifs pour la validation croisée PURGÉE.

    Pourquoi pas un K-fold ordinaire : mélanger les lignes ferait apprendre au
    modèle des périodes postérieures à celles qu'on lui demande de prédire. Sur
    une série temporelle, la seule validation croisée honnête avance dans le
    temps — chaque bloc est évalué par un modèle entraîné uniquement sur ce qui
    le précède, exactement comme en situation réelle.

    « Purgée » : les `horizon` périodes qui précèdent immédiatement un bloc
    d'évaluation sont retirées de l'apprentissage, sinon leur cible empiète
    déjà sur le bloc évalué.

    Ce que ça change concrètement : au lieu d'une AUC mesurée sur un seul
    morceau de marché — dont les trois millièmes d'écart entre configurations
    ne veulent rien dire — on obtient N mesures sur N époques différentes, donc
    un écart-type réel. La configuration retenue est celle qui tient sur
    plusieurs marchés, pas celle qui a eu de la chance sur un seul.
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    if n < config.LIGNES_MIN_BLOC_CV * 2:
        return []

    n_blocs = max(2, int(n_blocs))
    # n_blocs + 1 tranches : la première ne sert qu'à amorcer l'apprentissage.
    frontieres = [int(n * i / (n_blocs + 1)) for i in range(n_blocs + 2)]
    marge = pd.Timedelta(hours=config.heures(intervalle) * max(0, int(horizon)))
    positions = np.arange(n)

    decoupes = []
    for i in range(1, n_blocs + 1):
        debut, fin = frontieres[i], frontieres[i + 1]
        if fin - debut < config.LIGNES_MIN_BLOC_CV:
            continue
        evaluation = positions[debut:fin]
        # Tout ce qui précède le bloc, purgé de l'horizon qui le touche.
        apprentissage = positions[dates < dates[debut] - marge]
        if len(apprentissage) < config.LIGNES_MIN_BLOC_CV:
            continue
        decoupes.append((apprentissage, evaluation))
    return decoupes


def _essayer_cv(nom_modele, params, res, poids, X, y, decoupes, n_classes=2,
                etiquette="", progression=None):
    """
    Une configuration, évaluée sur tous les blocs de validation croisée.

    Retourne (AUC de chaque bloc, nombre moyen d'arbres). L'arrêt anticipé se
    règle sur les 10 derniers pour cent de l'apprentissage de chaque bloc : le
    bloc évalué ne doit jamais servir à décider quand s'arrêter, sinon son AUC
    est juge et partie.
    """
    aucs, arbres = [], []
    for numero, (apprentissage, evaluation) in enumerate(decoupes, start=1):
        suivi.verifier()
        if progression is not None:
            progression((numero - 1) / len(decoupes))
        coupe = max(1, int(len(apprentissage) * 0.9))
        indices_tr, indices_stop = apprentissage[:coupe], apprentissage[coupe:]
        if len(indices_stop) < 50:
            indices_tr, indices_stop = apprentissage, apprentissage[-50:]

        candidat = _construire(nom_modele, params, res, poids, n_classes)
        y_tr = y.iloc[indices_tr]
        poids_lignes = _poids_echantillons(y_tr, n_classes, poids is not None)
        iterations = _ajuster(candidat, nom_modele, X.iloc[indices_tr], y_tr,
                              X.iloc[indices_stop], y.iloc[indices_stop],
                              n_classes, poids_lignes,
                              etiquette=f"{etiquette} · bloc {numero}/{len(decoupes)}"
                              if etiquette else "")
        aucs.append(_auc(candidat, X.iloc[evaluation], y.iloc[evaluation], n_classes))
        arbres.append(iterations)
    return aucs, int(np.mean(arbres)) if arbres else 0


def _tester_configurations_cv(nom_modele, res, poids, X, y, decoupes,
                              n_classes: int = 2, mode: str | None = None,
                              progression=None):
    """Les configurations candidates, chacune sur tous les blocs."""
    configurations = configurations_candidates(nom_modele, mode)
    total = len(configurations)
    print(f"🔎 Réglage automatique — validation croisée purgée : "
          f"{total} configurations x {len(decoupes)} blocs "
          f"({res.n_jobs} threads)…")

    resultats = []
    for numero, params in enumerate(configurations, start=1):
        suivi.verifier()
        etiquette = f"config {numero}/{total}"

        def avancer(part_bloc, _numero=numero):
            """Progression fine : configuration entamée + blocs déjà faits."""
            if progression is not None:
                progression((_numero - 1 + part_bloc) / total,
                            f"configuration {_numero}/{total}")

        avancer(0.0)
        aucs, arbres = _essayer_cv(nom_modele, params, res, poids, X, y,
                                   decoupes, n_classes, etiquette=etiquette,
                                   progression=avancer)
        moyenne = float(np.mean(aucs))
        ecart = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        detail = " / ".join(f"{a:.4f}" for a in aucs)
        print(f"   {numero}/{total} — AUC {moyenne:.4f} "
              f"+/- {ecart:.4f} (blocs : {detail})")
        suivi.MONITEUR.configuration(numero, total, params, moyenne, ecart, aucs)
        resultats.append({"params": params, "aucs": aucs, "moyenne": moyenne,
                          "ecart": ecart, "arbres": arbres})
    return resultats


def _choisir_par_cv(resultats, marge_theorique: float, bavard: bool = False):
    """
    Retient la configuration la plus simple parmi celles à égalité statistique.

    La marge retenue est la PLUS GRANDE de deux estimations : la dispersion
    entre blocs (mesurée) et la marge théorique d'une AUC sur la taille
    effective de l'échantillon (calculée). Les blocs étant très corrélés entre
    eux, la première seule sous-estimerait l'incertitude ; la seconde ignore
    les changements de régime. Prendre le maximum, c'est refuser de départager
    deux configurations tant que l'écart n'est pas franchement au-dessus du
    bruit — donc garder la plus prudente, qui généralise mieux.
    """
    meilleur = max(range(len(resultats)), key=lambda i: resultats[i]["moyenne"])
    blocs = max(1, len(resultats[meilleur]["aucs"]))
    dispersion = resultats[meilleur]["ecart"] / np.sqrt(blocs)
    marge = max(float(dispersion), float(marge_theorique))
    reference = resultats[meilleur]["moyenne"]

    for indice, resultat in enumerate(resultats):
        if resultat["moyenne"] >= reference - marge:
            if bavard and indice != meilleur:
                print(f"   ⚖️  Règle du 1 écart-type (+/-{marge:.4f}) : configuration "
                      f"{indice + 1} retenue au lieu de {meilleur + 1} "
                      f"— écart non significatif, on garde la plus simple.")
            return resultat, marge
    return resultats[meilleur], marge


def _choisir_configuration(essais, y_validation, bavard: bool = False,
                           reduction: float = 1.0):
    """
    Retient la configuration la plus SIMPLE parmi celles statistiquement à
    égalité avec la meilleure (règle dite « à un écart-type »).

    Pourquoi : les trois candidates obtiennent des AUC de validation séparées
    par quelques millièmes, alors que la marge d'erreur d'une AUC sur ce volume
    de données est de l'ordre du centième. Prendre systématiquement le maximum
    revient donc à choisir le modèle le plus complexe sur du bruit — et c'est
    exactement ce qui creuse l'écart entre l'apprentissage et le test.

    Les configurations de `CONFIGURATIONS` sont rangées de la plus prudente à la
    plus souple : à égalité statistique, on garde la première.
    """
    meilleur_indice = max(range(len(essais)), key=lambda i: essais[i][1])
    auc_max = essais[meilleur_indice][1]
    marge = _erreur_type_auc(auc_max, np.asarray(y_validation), reduction)

    for indice, essai in enumerate(essais):
        if essai[1] >= auc_max - marge:
            if bavard and indice != meilleur_indice:
                print(f"   ⚖️  Règle du 1 écart-type (±{marge:.4f}) : configuration "
                      f"{indice + 1} retenue au lieu de {meilleur_indice + 1} "
                      f"— écart non significatif, on garde la plus simple.")
            return essai
    return essais[meilleur_indice]


def _table_confiance(probas: np.ndarray, verite: np.ndarray,
                     tache: cibles.Tache | None = None,
                     reduction: float = 1.0) -> list[dict]:
    """
    Précision obtenue en ne gardant que les prédictions les plus sûres.

    C'est le tableau qui sert à choisir son seuil : pour chaque niveau de
    confiance, combien de bougies sont retenues, quelle part est correcte, et
    surtout AVEC QUELLE MARGE D'ERREUR.

    La marge est calculée sur la taille effective (`reduction`), pas sur le
    nombre de lignes : à l'horizon 5, 1 200 signaux consécutifs ne valent que
    ~240 observations indépendantes, et une précision de 55 % y est encore
    compatible avec le hasard. Sans cette colonne, le tableau invite à choisir
    le seuil le plus flatteur, qui est presque toujours le plus bruité.
    """
    tache = tache or cibles.obtenir(None)
    lecture = cibles.lire_probabilites(probas, tache)
    confiance = lecture["confiance"]
    correct = (lecture["classe"] == np.asarray(verite))
    est_hausse = np.isin(lecture["classe"], tache.indices_hausse())

    lignes = []
    for seuil in seuils_testes(tache.n_classes):
        retenus = confiance >= seuil
        nb = int(retenus.sum())
        precision = float(correct[retenus].mean()) if nb else None
        lignes.append({
            "seuil": seuil,
            "n": nb,
            "n_effectif": round(nb * reduction, 1),
            "couverture": float(nb / len(confiance)) if len(confiance) else 0.0,
            "precision": precision,
            "marge": (marge_proportion(precision, nb * reduction)
                      if precision is not None else None),
            "part_hausse": float(est_hausse[retenus].mean()) if nb else None,
        })
    return lignes


def _seuil_conseille(table: list[dict], reference: float) -> dict | None:
    """
    Meilleur seuil exploitable d'après la table de confiance.

    Deux conditions pour qu'un seuil soit retenu :
      * il laisse passer assez de signaux pour que le chiffre veuille dire
        quelque chose (`SIGNAUX_MINIMUM`) ;
      * il fait réellement mieux que ne pas filtrer du tout.

    Retourne None quand aucun seuil n'apporte rien — auquel cas il vaut mieux le
    dire clairement que laisser l'utilisateur chercher un réglage qui n'existe pas.
    """
    candidats = [ligne for ligne in table
                 if ligne["n"] >= SIGNAUX_MINIMUM
                 and ligne["precision"] is not None
                 and ligne["precision"] > reference + GAIN_MINIMUM]
    if not candidats:
        return None

    # À gain égal, on préfère le seuil dont l'écart au sans-filtre dépasse sa
    # propre marge d'erreur : un seuil qui affiche 62 % sur 60 signaux n'est
    # pas meilleur qu'un seuil à 56 % sur 1 500, il est seulement plus bruyant.
    solides = [ligne for ligne in candidats
               if ligne.get("marge") is not None
               and ligne["precision"] - ligne["marge"] > reference]
    choisi = dict(max(solides or candidats, key=lambda ligne: ligne["precision"]))
    choisi["solide"] = bool(solides)
    return choisi


def _auc_probas(probas: np.ndarray, verite: np.ndarray, n_classes: int) -> float:
    """AUC (binaire ou macro « une contre toutes ») à partir de probabilités déjà calculées."""
    presentes = np.unique(verite)
    if len(presentes) < 2:
        return 0.5
    if n_classes <= 2:
        return float(roc_auc_score(verite, probas[:, 1]))
    colonnes = probas[:, presentes.astype(int)] if len(presentes) < n_classes else probas
    colonnes = colonnes / np.clip(colonnes.sum(axis=1, keepdims=True), 1e-12, None)
    return float(roc_auc_score(verite, colonnes, multi_class="ovr",
                               average="macro", labels=presentes))


def _table_regimes(probas: np.ndarray, verite: np.ndarray, regime, tache,
                   reduction: float = 1.0) -> list[dict]:
    """
    Justesse ventilée par régime de volatilité (quatre quarts).

    Une justesse moyenne de 53 % peut recouvrir 58 % en marché calme et 48 %
    en marché agité. Dans ce cas le modèle n'est pas mauvais : il est
    conditionnel, et la bonne décision est de ne le suivre que dans le régime
    où il fonctionne. C'est l'information que la moyenne globale détruit.
    """
    if regime is None or len(regime) != len(verite):
        return []

    regime = np.asarray(regime, dtype=float)
    lecture = cibles.lire_probabilites(probas, tache)
    correct = (lecture["classe"] == np.asarray(verite)).astype(float)

    noms = ["Calme", "Normal", "Actif", "Agité"]
    lignes = []
    for indice, nom in enumerate(noms):
        bas, haut = indice / 4, (indice + 1) / 4
        dedans = (regime >= bas) & (regime < haut if indice < 3 else regime <= 1.0)
        nb = int(dedans.sum())
        if nb < 30:
            continue
        justesse = float(correct[dedans].mean())
        lignes.append({
            "regime": nom,
            "n": nb,
            "part": float(nb / len(regime)),
            "justesse": justesse,
            "marge": marge_proportion(justesse, nb * reduction),
            "confiance_moyenne": float(lecture["confiance"][dedans].mean()),
        })
    return lignes


def _evaluer(probas: np.ndarray, verite: np.ndarray,
             tache: cibles.Tache | None = None,
             reduction: float = 1.0, regime=None) -> dict:
    """Métriques complètes sur un jeu (test en principe)."""
    tache = tache or cibles.obtenir(None)
    probas = np.atleast_2d(np.asarray(probas, dtype=float))
    verite = np.asarray(verite)

    lecture = cibles.lire_probabilites(probas, tache)
    prediction = lecture["classe"]
    confiance = lecture["confiance"]

    effectifs = np.bincount(verite, minlength=tache.n_classes) / max(1, len(verite))
    indices_hausse = tache.indices_hausse()

    rapport = classification_report(
        verite, prediction, labels=list(range(tache.n_classes)),
        target_names=list(tache.classes), output_dict=True, zero_division=0)

    table = _table_confiance(probas, verite, tache, reduction)
    accuracy = float(accuracy_score(verite, prediction))
    n_effectif = len(verite) * reduction
    marge = marge_proportion(accuracy, n_effectif)

    return {
        "tache": tache.cle,
        "classes": list(tache.classes),
        "accuracy": accuracy,
        "n_effectif": round(n_effectif, 1),
        "marge_accuracy": marge,
        # L'écart à la réponse constante dépasse-t-il sa propre marge d'erreur ?
        # C'est la seule question qui décide si le modèle a appris quelque chose.
        "significatif": bool(accuracy - marge > float(effectifs.max())),
        "regimes": _table_regimes(probas, verite, regime, tache, reduction),
        "auc": _auc_probas(probas, verite, tache.n_classes),
        "rapport": rapport,
        "baseline_majoritaire": float(effectifs.max()),
        "part_hausse_reelle": float(effectifs[indices_hausse].sum()),
        "repartition_reelle": [float(part) for part in effectifs],
        "table_confiance": table,
        # Plafond de confiance réellement atteint : au-delà, régler le seuil plus
        # haut ne produit tout simplement aucun signal.
        "confiance_max": float(confiance.max()),
        "confiance_p99": float(np.quantile(confiance, 0.99)),
        "confiance_neutre": tache.confiance_neutre,
        "seuil_conseille": _seuil_conseille(table, accuracy),
    }


def entrainer(symbole: str, intervalle: str, horizon: int,
              nom_modele: str = config.MODELE_DEFAUT,
              tache: str | None = None,
              contexte: bool = True,
              mode_recherche: str | None = None,
              seuil_utilite: float = 0.0) -> dict:
    """
    Entraîne un modèle pour une crypto, un intervalle, un horizon et un objectif.

    `symbole` accepte aussi un NOM DE PANIER (`PANIER-BTC-ETH-SOL`) : le modèle
    est alors entraîné sur les cryptos empilées, avec des frontières de blocs
    exprimées en dates pour que toutes soient jugées sur la même période de
    marché. Voir `panier.py`.

    Déroulé complet :
      1. Chargement du jeu (features en liste blanche, cible selon l'objectif).
      2. Filtrage éventuel des features par leur utilité mesurée (`seuil_utilite`).
      3. Découpage chronologique 70/15/15 aux mêmes dates pour tout le monde.
      4. Rééquilibrage automatique si nécessaire.
      5. N configurations comparées en validation croisée PURGÉE ; la plus
         simple à égalité statistique est retenue. N dépend de `mode_recherche`
         (3, 18 ou 40 — voir `config.MODES_RECHERCHE`).
      6. Calibration des probabilités sur la seconde moitié de la validation.
      7. Mesure de l'utilité de chaque feature, sur la validation.
      8. Évaluation sur le test (jamais vu) : seuils de confiance, marges
         d'erreur sur la taille effective, ventilation par régime.
      9. Sauvegarde du modèle (.joblib) et de ses métadonnées (.json).

    `seuil_utilite` est une perte d'AUC (0.001 = un millième). Au-dessus de 0,
    seules les features dont l'utilité mesurée au PRÉCÉDENT entraînement atteint
    ce seuil sont conservées. Sans entraînement précédent, le seuil est ignoré
    et l'entraînement se fait sur toutes les features — il faut bien une
    première mesure.

    L'avancement est publié dans `suivi.MONITEUR` (progression, courbe
    d'apprentissage, configurations évaluées) et la demande d'arrêt est
    consultée entre chaque étape. Sans interface à l'écoute, cela ne coûte rien.

    Retourne le dictionnaire de métadonnées.
    """
    config.preparer_dossiers()
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))
    objectif = cibles.obtenir(tache)
    est_panier = config.est_panier(symbole)
    mode = config.recherche_par_libelle(mode_recherche) \
        if mode_recherche in config.libelles_recherche() else \
        (mode_recherche or config.RECHERCHE_DEFAUT)
    infos_mode = config.mode_recherche(mode)
    debut_horloge = datetime.now()

    print(f"\n🤖 Entraînement — {symbole} ({intervalle}) | modèle : {nom_modele} | "
          f"horizon : {horizon} période(s)")
    print(f"🎯 Objectif : {objectif.libelle}")
    print(f"🔧 Recherche : {infos_mode['libelle'].lower()}.")

    # --- 1. Données -------------------------------------------------------
    suivi.etape("Chargement des données", 0.02)
    suivi.verifier()
    if est_panier:
        from . import panier as module_panier
        membres = config.symboles_panier(symbole)
        print(f"🧺 Panier de {len(membres)} cryptos : {', '.join(membres)}")
        jeu = module_panier.charger(membres, intervalle, horizon, objectif, contexte)
        normalisation = module_panier.NORMALISATION_RANG
    else:
        jeu = charger_jeu(symbole, intervalle, horizon, objectif, contexte)
        normalisation = None
    res = ressources.detecter(int(jeu.X.memory_usage(deep=True).sum()))
    print(res.resume())

    # --- 1 bis. Filtrage des features par leur utilité mesurée ------------
    # Le seuil s'applique à la mesure du PRÉCÉDENT entraînement : c'est ce qui
    # permet un aller-retour « j'entraîne, je regarde ce qui sert, je relance
    # sans le reste » sans jamais faire dépendre le choix des features du bloc
    # de test.
    seuil_utilite = max(0.0, float(seuil_utilite or 0.0))
    features_candidates = list(jeu.X.columns)
    utilite_precedente = {}
    ecartees_utilite = []

    if seuil_utilite > 0:
        utilite_precedente = utilite_enregistree(symbole, intervalle, horizon,
                                                 objectif.cle)
        if not utilite_precedente:
            print("⚠️  Aucune utilité mesurée pour ce modèle : le filtre est "
                  "ignoré. Entraîne une première fois sans seuil, le curseur "
                  "servira ensuite.")
            seuil_utilite = 0.0
        else:
            gardees, ecartees_utilite = trier_par_utilite(
                utilite_precedente, seuil_utilite, disponibles=jeu.X.columns)
            ordre = [c for c in jeu.X.columns if c in set(gardees)]
            print(f"🎚️  Utilité minimale {seuil_utilite:.4f} d'AUC : "
                  f"{len(ordre)} features gardées sur {len(features_candidates)}.")
            if ecartees_utilite:
                print(f"   Écartées : {', '.join(ecartees_utilite)}")
            jeu.X = jeu.X[ordre]

    n_classes = objectif.n_classes
    part_hausse = float(jeu.y_evaluation.isin(objectif.indices_hausse()).mean())
    contexte = [c for c in jeu.X.columns if c not in config.INDICATEURS]
    base = [c for c in jeu.X.columns if c in config.INDICATEURS]
    print(f"📐 {len(jeu.X):,} lignes exploitables | {len(jeu.X.columns)} features "
          f"({len(base)} indicateurs"
          + (f" + {len(contexte)} de contexte : {', '.join(contexte)}" if contexte else "")
          + ")")
    _afficher_repartition(jeu.y[jeu.apprenable], objectif)

    # --- 2. Découpage -----------------------------------------------------
    decoupage = _decouper(jeu.X.index, horizon, intervalle)
    if min(len(decoupage.train), len(decoupage.validation), len(decoupage.test)) < 50:
        raise ValueError("Historique trop court pour entraîner un modèle fiable "
                         "(il faut au moins quelques centaines de bougies).")

    # Apprentissage, sélection et calibration ne voient que les lignes
    # « apprenables » ; le TEST, lui, garde toutes les bougies. C'est ce qui
    # rend la précision annoncée comparable à ce qu'on obtiendra en situation
    # réelle, où l'on ignore d'avance si le mouvement sera exploitable.
    X_tr, y_tr = _sous_ensemble(jeu, decoupage.train, apprenable=True)
    X_val, y_val = _sous_ensemble(jeu, decoupage.validation, apprenable=True)
    X_test, y_test = _sous_ensemble(jeu, decoupage.test, apprenable=False)

    if min(len(X_tr), len(X_val)) < 50:
        raise ValueError("Trop peu de bougies apprenables pour cet objectif : "
                         "essaie un horizon plus long ou un autre objectif.")

    # La validation sert à deux choses différentes, sur deux moitiés distinctes :
    # choisir/arrêter le modèle d'un côté, calibrer ses probabilités de l'autre.
    milieu = len(X_val) // 2
    X_sel, y_sel = X_val.iloc[:milieu], y_val.iloc[:milieu]
    X_cal, y_cal = X_val.iloc[milieu:], y_val.iloc[milieu:]

    ecartees = len(decoupage.train) - len(X_tr)
    print(f"✂️  Train {len(X_tr):,} | Validation {len(X_val):,} "
          f"(sélection {len(X_sel):,} + calibration {len(X_cal):,}) | Test {len(X_test):,}")
    print(f"📅 Frontières : validation à partir du "
          f"{decoupage.date_validation:%Y-%m-%d}, test à partir du "
          f"{decoupage.date_test:%Y-%m-%d}"
          + (" — les mêmes dates pour toutes les cryptos du panier."
             if est_panier else "."))
    if ecartees:
        print(f"🚫 {ecartees:,} bougies d'apprentissage écartées par l'objectif "
              f"(mouvements trop petits) — mais TOUTES comptent dans l'évaluation.")
    print(f"🚧 Embargo : {decoupage.embargo} période(s), soit "
          f"{decoupage.purgees:,} ligne(s) purgée(s) entre les blocs (anti-fuite).")

    # Taille EFFECTIVE : le nombre d'observations réellement indépendantes.
    # Toutes les marges d'erreur affichées plus bas en dépendent — et c'est
    # souvent la différence entre « le modèle a un avantage » et « le modèle a
    # eu de la chance ».
    reduction_sel = taille_effective(X_sel.index, horizon) / max(1, len(X_sel))
    reduction_test = taille_effective(X_test.index, horizon) / max(1, len(X_test))
    print(f"🧮 Taille effective du test : {len(X_test) * reduction_test:,.0f} "
          f"observations indépendantes pour {len(X_test):,} lignes "
          f"(chevauchement des cibles sur {horizon} période(s)"
          + (", cryptos corrélées" if est_panier else "") + ").")

    # --- 3. Rééquilibrage automatique ------------------------------------
    poids = _poids_classe_positive(y_tr, n_classes)
    if poids is not None:
        detail = f" (poids {poids:.2f})" if n_classes <= 2 else " (pondération par classe)"
        print(f"⚖️  Classes déséquilibrées → rééquilibrage automatique{detail}.")

    # --- 4. Sélection de la configuration --------------------------------
    # Validation croisée purgée quand l'historique le permet : trois
    # configurations séparées par quelques millièmes d'AUC sur un seul bloc,
    # c'est un tirage au sort. Sur quatre époques successives, c'est une mesure.
    suivi.verifier()
    decoupes = blocs_cv(X_tr.index, horizon, intervalle)
    validation_croisee = len(decoupes) >= 2

    # La recherche occupe l'essentiel du temps : elle occupe donc l'essentiel
    # de la barre de progression (de 15 % à 72 %).
    def avancement_recherche(part, libelle=""):
        suivi.etape(f"Réglage — {libelle}" if libelle else "Réglage",
                    0.15 + 0.57 * float(min(max(part, 0.0), 1.0)))

    if validation_croisee:
        resultats = _tester_configurations_cv(nom_modele, res, poids,
                                              X_tr, y_tr, decoupes, n_classes,
                                              mode=mode,
                                              progression=avancement_recherche)
        marge_theorique = _erreur_type_auc(
            max(r["moyenne"] for r in resultats), y_tr.to_numpy(),
            taille_effective(X_tr.index, horizon) / max(1, len(X_tr)))
        choisi, marge_selection = _choisir_par_cv(resultats, marge_theorique,
                                                  bavard=True)
        meilleurs_params = choisi["params"]
        aucs_blocs = choisi["aucs"]
        meilleure_auc = choisi["moyenne"]
        print(f"🏆 Configuration retenue : AUC {meilleure_auc:.4f} "
              f"+/- {choisi['ecart']:.4f} sur {len(aucs_blocs)} blocs.")

        # Réentraînement final sur tout le bloc d'apprentissage, l'arrêt
        # anticipé se réglant sur la moitié « sélection » de la validation.
        suivi.verifier()
        suivi.etape("Entraînement du modèle retenu", 0.74)
        poids_lignes = _poids_echantillons(y_tr, n_classes, poids is not None)
        meilleur = _construire(nom_modele, meilleurs_params, res, poids, n_classes)
        meilleures_iterations = _ajuster(meilleur, nom_modele, X_tr, y_tr,
                                         X_sel, y_sel, n_classes, poids_lignes,
                                         etiquette="modèle retenu",
                                         courbe_train=True)
    else:
        print("ℹ️  Historique trop court pour une validation croisée : "
              "sélection sur un seul bloc de validation.")
        essais = _tester_configurations(nom_modele, res, poids,
                                        X_tr, y_tr, X_sel, y_sel, n_classes,
                                        mode=mode,
                                        progression=avancement_recherche)
        for numero, essai in enumerate(essais, start=1):
            print(f"   {numero}/{len(essais)} — AUC validation {essai[1]:.4f} ({essai[2]})")
        meilleur, meilleure_auc, meilleurs_params, meilleures_iterations = \
            _choisir_configuration(essais, y_sel, bavard=True, reduction=reduction_sel)
        aucs_blocs, marge_selection = [], None
        print(f"🏆 Configuration retenue : AUC validation {meilleure_auc:.4f}")

    if nom_modele in AVEC_EARLY_STOPPING:
        print(f"   {meilleures_iterations} arbres (early stopping).")

    # --- 5. Calibration ---------------------------------------------------
    suivi.verifier()
    suivi.etape("Calibration des probabilités", 0.82)
    calibrateur = CalibrateurMulti(n_classes).entrainer(
        meilleur.predict_proba(X_cal), y_cal.to_numpy())
    print(f"📐 Calibration des probabilités : méthode {calibrateur.methode}"
          + (f", une par classe ({n_classes})." if n_classes > 2 else "."))

    # --- 6. Utilité des features -----------------------------------------
    # Mesurée sur la moitié « calibration » de la validation : le modèle brut
    # ne l'a vue ni pour apprendre, ni pour décider quand s'arrêter. C'est le
    # bloc le plus propre disponible SANS toucher au test.
    suivi.verifier()
    suivi.etape("Mesure de l'utilité des features", 0.86)
    mesures = _mesurer_utilite(meilleur, X_cal, y_cal.to_numpy(), objectif, res)
    if mesures:
        classement = sorted(mesures.items(), key=lambda paire: paire[1],
                            reverse=True)
        tete = " · ".join(f"{nom} {valeur:+.4f}" for nom, valeur in classement[:4])
        inutiles = [nom for nom, valeur in classement if valeur <= 0]
        print(f"🏅 Utilité des features (perte d'AUC si on les mélange) : {tete}")
        if inutiles:
            print(f"   {len(inutiles)} feature(s) sans apport mesurable : "
                  f"{', '.join(inutiles)}")
        suivi.MONITEUR.importance(mesures, bloc="validation")

    # Les features écartées par le filtre gardent leur dernière valeur connue,
    # sinon abaisser le curseur ne pourrait jamais les faire revenir.
    utilite_complete = dict(utilite_precedente)
    utilite_complete.update(mesures)

    # --- 7. Évaluation sur le test ---------------------------------------
    suivi.verifier()
    suivi.etape("Évaluation sur le bloc test", 0.94)
    probas_test = calibrateur.appliquer(meilleur.predict_proba(X_test))
    regime_test = (jeu.regime.iloc[decoupage.test].to_numpy()
                   if jeu.regime is not None else None)
    metriques = _evaluer(probas_test, y_test.to_numpy(), objectif,
                         reduction=reduction_test, regime=regime_test)
    _afficher_evaluation(metriques, symbole, intervalle, horizon, objectif)

    # --- 8. Sauvegarde ----------------------------------------------------
    suivi.etape("Sauvegarde", 0.98)
    meta = {
        "symbole": symbole,
        "intervalle": intervalle,
        "horizon": horizon,
        "modele": nom_modele,
        "tache": objectif.cle,
        "tache_libelle": objectif.libelle,
        "classes": list(objectif.classes),
        "date_entrainement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(jeu.X.columns),
        "features_contexte": contexte,
        "hyperparametres": {str(k): v for k, v in meilleurs_params.items()},
        "mode_recherche": mode,
        "mode_recherche_libelle": infos_mode["libelle"],
        "n_configurations": len(configurations_candidates(nom_modele, mode)),
        "n_arbres": meilleures_iterations,
        "calibration": calibrateur.methode,
        "reequilibrage": poids is not None,
        "embargo": decoupage.embargo,
        "n_total": len(jeu.X),
        "n_train": len(X_tr),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "part_hausse": part_hausse,
        "auc_validation": float(meilleure_auc),
        "validation_croisee": validation_croisee,
        "auc_blocs": [float(a) for a in aucs_blocs],
        "marge_selection": (float(marge_selection)
                            if marge_selection is not None else None),
        "seuil_utilite": seuil_utilite,
        "features_candidates": features_candidates,
        "features_ecartees": ecartees_utilite,
        "utilite": {
            "bloc": "validation",
            "n": len(X_cal),
            "repetitions": config.REPETITIONS_UTILITE,
            "valeurs": utilite_complete,
            "mesurees": sorted(mesures),
        },
        "duree_secondes": round((datetime.now() - debut_horloge).total_seconds(), 1),
        "panier": config.symboles_panier(symbole) if est_panier else [],
        "normalisation": normalisation,
        "contexte": bool(contexte),
        "taille_effective_test": round(len(X_test) * reduction_test, 1),
        "periode_debut": str(jeu.X.index.min()),
        "periode_fin": str(jeu.X.index.max()),
        "debut_validation": str(X_val.index.min()),
        "debut_test": str(X_test.index.min()),
        "metriques": metriques,
        "ressources": {
            "coeurs": res.coeurs,
            "ram_totale_go": round(res.ram_totale_go, 1),
            "budget_go": round(res.budget_go, 1),
            "max_bin": res.max_bin,
        },
    }

    chemin_modele = stockage.chemin_modele(symbole, intervalle, horizon, objectif.cle)
    joblib.dump(
        {"modele": meilleur, "calibrateur": calibrateur,
         "features": list(jeu.X.columns), "horizon": horizon,
         "nom_modele": nom_modele, "tache": objectif.cle,
         "normalisation": normalisation, "meta": meta},
        chemin_modele, compress=3)

    with open(stockage.chemin_meta(symbole, intervalle, horizon, objectif.cle), "w",
              encoding="utf-8") as fichier:
        json.dump(meta, fichier, indent=2, ensure_ascii=False)

    print(f"💾 Modèle sauvegardé : {chemin_modele}")
    print(f"⏱️  Durée totale : {meta['duree_secondes']:,.0f} s "
          f"({meta['n_configurations']} configuration(s) testée(s)).")
    suivi.etape("Terminé", 1.0)
    memoire = ressources.memoire_processus_go()
    if memoire:
        print(f"🧠 Mémoire utilisée par le processus : {memoire:.2f} Go "
              f"sur {res.budget_go:.1f} Go de budget.")
    return meta


def _afficher_repartition(y: pd.Series, tache: cibles.Tache) -> None:
    """Répartition réelle des classes, avant tout apprentissage."""
    effectifs = y.value_counts(normalize=True).sort_index()
    detail = " / ".join(f"{tache.classes[int(indice)]} {part:.1%}"
                        for indice, part in effectifs.items()
                        if 0 <= int(indice) < tache.n_classes)
    print(f"⚖️  Répartition réelle — {detail}")


def _afficher_evaluation(metriques: dict, symbole: str, intervalle: str,
                         horizon: int, tache: cibles.Tache | None = None) -> None:
    """Rapport console de fin d'entraînement."""
    tache = tache or cibles.obtenir(metriques.get("tache"))
    accuracy = metriques["accuracy"]
    baseline = metriques["baseline_majoritaire"]
    verdict = "✅ battue" if accuracy > baseline else "❌ NON battue"

    print(f"\n📊 ===== ÉVALUATION SUR LE TEST — {symbole} {intervalle} (h={horizon}, "
          f"{tache.libelle}) =====")
    marge = metriques.get("marge_accuracy")
    effectif = metriques.get("n_effectif")
    if marge is not None and effectif:
        print(f"   Justesse : {accuracy:.2%} +/- {marge:.2%}   |   "
              f"AUC-ROC : {metriques['auc']:.4f}")
        print(f"   (marge à 95 % calculée sur {effectif:,.0f} observations "
              f"INDÉPENDANTES — les cibles se chevauchent sur l'horizon, "
              f"le nombre de lignes surestime la précision de la mesure.)")
    else:
        print(f"   Justesse : {accuracy:.2%}   |   AUC-ROC : {metriques['auc']:.4f}")
    print(f"   Baseline « toujours la même réponse » : {baseline:.2%}  ({verdict})")
    print(f"   Confiance maximale atteinte : {metriques['confiance_max']:.1%} "
          f"— régler le seuil au-dessus ne produirait aucun signal.")

    print("\n   Précision selon le seuil de confiance (les deux sens confondus) :")
    print("   seuil │ prédictions retenues │ couverture │ précision")
    for ligne in metriques["table_confiance"]:
        if ligne["precision"] is None:
            continue
        marge_ligne = ligne.get("marge")
        suffixe = f" +/- {marge_ligne:.2%}" if marge_ligne is not None else ""
        print(f"   {ligne['seuil']:.2f}  │ {ligne['n']:>20,} │ "
              f"{ligne['couverture']:>9.1%} │ {ligne['precision']:>9.2%}{suffixe}")
    print("   " + "-" * 60)

    conseil = metriques.get("seuil_conseille")
    if conseil:
        print(f"   👉 Seuil conseillé : {conseil['seuil']:.2f} — "
              f"{conseil['n']:,} signaux ({conseil['couverture']:.1%} du temps), "
              f"{conseil['precision']:.2%} de justesse contre {accuracy:.2%} sans filtre.")
        if conseil.get("solide") is False:
            print(f"   ⚠️  Ce seuil est le plus flatteur, pas le plus solide : "
                  f"avec {conseil['n']:,} signaux sa marge d'erreur "
                  f"(+/-{conseil.get('marge', 0):.2%}) recouvre le score sans "
                  f"filtre. Filtrer fort sur peu de données produit toujours de "
                  f"beaux chiffres — préfère un seuil plus bas et mieux fourni.")
    else:
        print("   ⚠️  Aucun seuil n'améliore nettement la justesse sur ce modèle : "
              "essaie un horizon plus court, ou un autre modèle.")

    if metriques.get("marge_accuracy") is not None:
        if metriques.get("significatif"):
            print("   ✅ L'écart à la réponse constante dépasse sa marge d'erreur : "
                  "le modèle a bien appris quelque chose.")
        else:
            print("   ⚠️  L'écart à la réponse constante tient dans la marge "
                  "d'erreur : à ce stade, rien ne prouve que le modèle fasse "
                  "mieux que le hasard. Plus de données (panier de cryptos) ou "
                  "un horizon plus long sont les deux vrais leviers.")

    _afficher_regimes(metriques.get("regimes") or [])


def _afficher_regimes(regimes: list) -> None:
    """
    Ventilation de la justesse par régime de volatilité.

    Une moyenne globale de 53 % peut cacher 58 % en marché calme et 48 % en
    marché agité : dans ce cas le modèle est utilisable, mais seulement la
    moitié du temps. C'est exactement l'information qu'une moyenne détruit, et
    c'est aussi ce qui remplace avantageusement l'entraînement d'un modèle
    séparé par régime — lequel diviserait les données au moment où elles
    manquent déjà.
    """
    if not regimes:
        return
    print()
    print("   Justesse par régime de volatilité (quartiles de l'ATR récent) :")
    print("   régime  │ bougies │ justesse          │ confiance moyenne")
    for ligne in regimes:
        print(f"   {ligne['regime']:<7} │ {ligne['n']:>7,} │ "
              f"{ligne['justesse']:>7.2%} +/- {ligne['marge']:.2%} │ "
              f"{ligne['confiance_moyenne']:>16.1%}")

    meilleur = max(regimes, key=lambda ligne: ligne["justesse"])
    pire = min(regimes, key=lambda ligne: ligne["justesse"])
    if meilleur["justesse"] - pire["justesse"] > meilleur["marge"] + pire["marge"]:
        print(f"   👉 Le modèle marche nettement mieux en régime "
              f"« {meilleur['regime']} » ({meilleur['justesse']:.2%}) qu'en "
              f"« {pire['regime']} » ({pire['justesse']:.2%}) : ne le suivre "
              f"que dans le premier vaut mieux que de le suivre partout.")


# ===========================================================================
# PRÉDICTION
# ===========================================================================
def preparer_features(df: pd.DataFrame, paquet: dict) -> pd.DataFrame:
    """
    Extrait les features attendues par un modèle, dans l'état où il les a apprises.

    Un modèle de panier a été entraîné sur des features converties en rang de
    percentile glissant, crypto par crypto. Lui présenter des valeurs brutes
    reviendrait à lui parler une autre langue : il rendrait des probabilités
    apparemment normales et complètement fausses. La transformation est donc
    stockée avec le modèle et rejouée à l'identique ici.
    """
    X = df[list(paquet["features"])].astype("float32")
    if paquet.get("normalisation") == "rang":
        from . import panier
        X = panier.normaliser(X)
    return X


def charger_modele(symbole: str, intervalle: str, horizon: int,
                   tache: str | None = None) -> dict:
    """Charge le paquet {modèle, calibrateur, features, meta} depuis le disque."""
    objectif = cibles.obtenir(tache)
    chemin = stockage.chemin_modele(symbole, intervalle, horizon, objectif.cle)
    try:
        paquet = joblib.load(chemin)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Aucun modèle « {objectif.libelle} » pour {symbole} ({intervalle}) à "
            f"l'horizon {horizon}. Lance d'abord l'entraînement.") from None

    # Les modèles entraînés avant l'ajout des objectifs n'ont ni clé « tache »
    # ni calibrateur multi-classe : on complète pour rester compatible.
    paquet.setdefault("tache", cibles.TACHE_DEFAUT)
    if isinstance(paquet.get("calibrateur"), Calibrateur):
        ancien = paquet["calibrateur"]
        enveloppe = CalibrateurMulti(2)
        enveloppe._calibrateurs = [ancien]              # noqa: SLF001
        paquet["calibrateur"] = enveloppe
    return paquet


def predire(symbole: str, intervalle: str, horizon: int,
            seuil_confiance: float = config.SEUIL_DEFAUT,
            symbole_modele: str | None = None,
            tache: str | None = None) -> pd.DataFrame:
    """
    Applique un modèle entraîné à tout l'historique d'une crypto.

    `symbole_modele` permet d'appliquer le modèle d'une crypto à une autre
    (les features étant les mêmes partout, un modèle est transposable).

    Colonnes produites :
      Prix              cours de clôture de la bougie
      Proba_Hausse      probabilité calibrée d'une issue haussière
      Confiance         probabilité de la classe retenue (= max(p, 1-p) en binaire)
      Sens_Predit       HAUSSE / BAISSE / NEUTRE — la position à prendre
      Classe_Predite    étiquette détaillée (identique au sens en binaire)
      Amplitude_Prevue  variation attendue en %, pour l'objectif « amplitude »
      Retenu            1 si Confiance ≥ seuil (c'est ce filtre qu'on exploite)
      Variation_Reelle  variation observée après l'horizon, en % (NaN si futur)
      Sens_Reel         HAUSSE / BAISSE observé
      Correct           1 / 0 (NaN tant que l'issue est inconnue)
      Bloc              train / validation / test — pour un backtest honnête

    Pour les objectifs autres que la direction simple, `Correct` compare la
    CLASSE prédite à la classe réellement observée : c'est bien l'objectif
    appris qui est évalué, pas une direction reconstruite après coup.
    """
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))
    source_modele = symbole_modele or symbole
    paquet = charger_modele(source_modele, intervalle, horizon, tache)
    objectif = cibles.obtenir(paquet.get("tache"))

    etiquette = (symbole if source_modele == symbole
                 else f"{symbole} ← modèle {source_modele}")
    print(f"\n🔮 Prédiction — {etiquette} ({intervalle}) | horizon {horizon} | "
          f"objectif « {objectif.libelle} » | seuil {seuil_confiance:.0%}")

    # On repart du fichier analysé complet : contrairement à l'entraînement, on
    # garde ici les toutes dernières bougies dont l'issue est encore inconnue,
    # puisque ce sont justement celles qui nous intéressent pour décider.
    df = lire_analyse(symbole, intervalle)

    colonne_variation = config.colonne_variation(horizon)
    attendues = list(paquet["features"]) + [colonne_variation, "Close"]
    manquantes = [c for c in attendues if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du fichier analysé de {symbole} : {manquantes}. "
            f"Relance l'étape Analyse pour le régénérer au format attendu.")

    X = preparer_features(df, paquet)
    valides = X.notna().all(axis=1)
    X = X[valides]
    if X.empty:
        raise ValueError("Aucune ligne exploitable après nettoyage.")
    if paquet.get("normalisation") == "rang":
        print("   Features converties en rang glissant, comme à l'entraînement "
              "du panier.")

    probas = paquet["calibrateur"].appliquer(paquet["modele"].predict_proba(X))
    lecture = cibles.lire_probabilites(probas, objectif)

    variation = pd.to_numeric(df.loc[valides, colonne_variation], errors="coerce")
    sens_reel = pd.Series(np.where(variation > 0, cibles.HAUSSE, cibles.BAISSE),
                          index=X.index)
    sens_reel[variation.isna()] = None

    resultats = pd.DataFrame({
        "Prix": pd.to_numeric(df.loc[valides, "Close"], errors="coerce"),
        "Proba_Hausse": np.round(lecture["proba_hausse"], 4),
        "Confiance": np.round(lecture["confiance"], 4),
        "Sens_Predit": lecture["sens"],
        "Classe_Predite": lecture["etiquette"],
        "Retenu": (lecture["confiance"] >= seuil_confiance).astype(int),
        "Variation_Reelle": np.round(variation, 4),
        "Sens_Reel": sens_reel,
    }, index=X.index)

    if not objectif.binaire:
        resultats["Amplitude_Prevue"] = np.round(
            _amplitude_de_classe(lecture["classe"], df.loc[valides], horizon), 4)

    # Vérité au sens de l'objectif, mais définie sur TOUTES les bougies : c'est
    # la seule mesure comparable à ce qu'on obtiendra en situation réelle, où
    # l'on ignore d'avance si le mouvement sera exploitable.
    verite = objectif.construire_evaluation(df.loc[valides], horizon)
    resultats["Correct"] = np.where(
        verite.isna(), np.nan, (lecture["classe"] == verite.fillna(-1)).astype(float))
    resultats["Bloc"] = _etiqueter_blocs(resultats.index, paquet["meta"])

    chemin = stockage.chemin_prediction(symbole, intervalle, horizon,
                                        tache=objectif.cle)
    stockage.ecrire_tableau(resultats, chemin)
    print(f"💾 Prédictions sauvegardées : {chemin}")

    _rapport_prediction(resultats, seuil_confiance)
    return resultats


def _amplitude_de_classe(classes: np.ndarray, df: pd.DataFrame,
                         horizon: int) -> np.ndarray:
    """
    Variation attendue (en %) associée à chaque classe d'amplitude.

    Le centre de chaque tranche, exprimé en variation normalisée, est
    reconverti en pourcentage en le multipliant par l'amplitude attendue de la
    bougie. Une même classe correspond donc à un mouvement plus ample en marché
    agité qu'en marché calme — ce qui est précisément l'intérêt de normaliser
    par l'ATR.
    """
    centres = cibles.centres_amplitude(df, horizon)
    echelle = cibles.echelle_attendue(df, horizon).to_numpy(dtype=float)
    return centres[np.asarray(classes, dtype=int)] * echelle


def _etiqueter_blocs(index: pd.Index, meta: dict) -> pd.Series:
    """
    Marque chaque ligne : train / validation / test.

    Indispensable pour un backtest honnête — seul le bloc « test » correspond à
    des données que le modèle n'a jamais vues.

    Les frontières viennent des dates d'entraînement du MODÈLE. Quand on
    applique le modèle d'une crypto à une autre, elles restent donc celles de
    la crypto d'origine : c'est le choix prudent, puisque le modèle a bien vu
    ces périodes de marché, même sur un autre actif.
    """
    blocs = pd.Series("train", index=index, dtype=object)
    try:
        blocs[index >= pd.Timestamp(meta["debut_validation"])] = "validation"
        blocs[index >= pd.Timestamp(meta["debut_test"])] = "test"
    except (KeyError, ValueError, TypeError):
        pass
    return blocs


def _compter_sens(resultats: pd.DataFrame) -> tuple[int, int]:
    """(nombre de signaux haussiers, nombre de baissiers) — les neutres exclus."""
    sens = resultats["Sens_Predit"]
    return int((sens == cibles.HAUSSE).sum()), int((sens == cibles.BAISSE).sum())


def _rapport_prediction(resultats: pd.DataFrame, seuil: float) -> None:
    """Résumé console : dernier signal + performance des prédictions retenues."""
    derniere = resultats.iloc[-1]
    detail = derniere.get("Classe_Predite", derniere["Sens_Predit"])
    print(f"\n🎯 Dernière bougie ({resultats.index[-1]:%Y-%m-%d %H:%M}) : "
          f"{detail} — confiance {derniere['Confiance']:.1%} "
          f"| prix {derniere['Prix']:,.2f} $")

    evaluables = resultats.dropna(subset=["Correct"])
    if evaluables.empty:
        print("   (aucune issue connue pour évaluer la performance)")
        return

    # Les blocs sont séparés à dessein : sur « train », le modèle rejoue des
    # données qu'il a apprises, donc ses scores y sont flatteurs et sans valeur.
    # Seul le bloc « test » dit ce que vaut vraiment le modèle.
    print(f"\n📊 Performance sur {len(evaluables):,} bougies dont l'issue est connue :")
    print("   bloc       │ prédictions │ retenues │ justesse retenues │ ▲ hausse / ▼ baisse")

    for bloc in ("train", "validation", "test"):
        sous_ensemble = evaluables[evaluables["Bloc"] == bloc]
        if sous_ensemble.empty:
            continue
        retenues = sous_ensemble[sous_ensemble["Retenu"] == 1]
        if retenues.empty:
            print(f"   {bloc:<10} │ {len(sous_ensemble):>11,} │ {0:>8} │ "
                  f"{'—':>17} │ —")
            continue
        hausses, baisses = _compter_sens(retenues)
        print(f"   {bloc:<10} │ {len(sous_ensemble):>11,} │ {len(retenues):>8,} │ "
              f"{retenues['Correct'].mean():>16.2%} │ "
              f"{hausses:,} / {baisses:,}")

    print(f"   (seuil de confiance : {seuil:.0%} — seule la ligne « test » "
          f"reflète des données jamais vues)")


# ===========================================================================
# WALK-FORWARD : PRÉDICTIONS HORS ÉCHANTILLON SUR TOUT L'HISTORIQUE
# ===========================================================================
def _entrainer_fenetre(nom_modele, res, X_passe, y_passe, horizon, n_classes=2):
    """
    Entraîne un modèle complet sur un bloc de passé, et rend (modèle, calibrateur).

    C'est la même recette que `entrainer`, appliquée à une fenêtre : la fin du
    passé est réservée à la sélection puis à la calibration, avec l'embargo
    habituel. Rien de ce qui suit la fenêtre n'est utilisé — c'est toute la
    raison d'être du procédé.
    """
    n = len(X_passe)
    embargo = max(0, int(horizon))

    # 20 % du passé pour choisir et calibrer, 80 % pour apprendre.
    fin_apprentissage = max(1, int(n * 0.8) - embargo)
    debut_reglage = int(n * 0.8)
    milieu = (n + debut_reglage) // 2

    X_tr, y_tr = X_passe.iloc[:fin_apprentissage], y_passe.iloc[:fin_apprentissage]
    X_sel, y_sel = X_passe.iloc[debut_reglage:milieu], y_passe.iloc[debut_reglage:milieu]
    X_cal, y_cal = X_passe.iloc[milieu:], y_passe.iloc[milieu:]

    poids = _poids_classe_positive(y_tr, n_classes)
    essais = _tester_configurations(nom_modele, res, poids, X_tr, y_tr,
                                    X_sel, y_sel, n_classes)
    modele_retenu, auc_validation, _, _ = _choisir_configuration(essais, y_sel)

    calibrateur = CalibrateurMulti(n_classes).entrainer(
        modele_retenu.predict_proba(X_cal), y_cal.to_numpy())
    return modele_retenu, calibrateur, auc_validation


def walk_forward(symbole: str, intervalle: str, horizon: int,
                 nom_modele: str = config.MODELE_DEFAUT,
                 n_fenetres: int = config.FENETRES_WALKFORWARD,
                 seuil_confiance: float = config.SEUIL_DEFAUT,
                 tache: str | None = None) -> pd.DataFrame:
    """
    Réentraîne le modèle en avançant dans le temps, et ne prédit que l'inconnu.

    Le problème que ça résout : un backtest lancé sur tout l'historique inclut
    la période d'apprentissage, où le modèle rejoue ce qu'il a mémorisé. Le
    rendement y est spectaculaire et ne veut rien dire.

    Le principe : l'historique est coupé en `n_fenetres` tranches successives.
    Pour chacune, on réentraîne entièrement le modèle sur TOUT le passé
    disponible à cette date — sélection de configuration et calibration
    comprises — puis on prédit uniquement la tranche suivante, jamais vue. En
    recollant les tranches on obtient une courbe couvrant la moitié de
    l'historique dont **chaque point est hors échantillon**.

    C'est plus honnête qu'un simple découpage 70/15/15 : le modèle est évalué
    sur une douzaine de périodes différentes (hausse, baisse, stagnation), ce
    qui montre aussi sa stabilité et pas seulement sa moyenne.

    Le fichier produit est directement exploitable par le Backtest.
    """
    config.preparer_dossiers()
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))
    objectif = cibles.obtenir(tache)

    print(f"\n📏 Walk-forward — {symbole} ({intervalle}) | modèle : {nom_modele} | "
          f"horizon : {horizon} | {n_fenetres} fenêtres")
    print(f"🎯 Objectif : {objectif.libelle}")

    if config.est_panier(symbole):
        raise ValueError(
            "Le walk-forward ne s'applique pas à un panier : il réentraîne le "
            "modèle une douzaine de fois, ce qui prendrait des heures sur "
            "plusieurs cryptos. Lance-le sur une crypto précise.")

    jeu = charger_jeu(symbole, intervalle, horizon, objectif)
    n_classes = objectif.n_classes
    res = ressources.detecter(int(jeu.X.memory_usage(deep=True).sum()))
    print(res.resume())

    # La première moitié sert d'amorçage : il faut du passé avant de prédire.
    n = len(jeu.X)
    debut = n // 2
    taille_fenetre = max(1, (n - debut) // n_fenetres)
    embargo = max(0, int(horizon))

    if taille_fenetre < 50:
        raise ValueError("Historique trop court pour un walk-forward : réduis le "
                         "nombre de fenêtres ou télécharge plus de données.")

    print(f"📐 {n:,} lignes | amorçage sur les {debut:,} premières | "
          f"fenêtres de {taille_fenetre:,} bougies")

    morceaux, journal = [], []
    for numero in range(n_fenetres):
        # Le walk-forward est l'operation la plus longue du projet : une
        # douzaine d'entrainements complets. Elle doit donc etre suivie et
        # interruptible comme n'importe quelle autre.
        suivi.verifier()
        suivi.etape(f"Walk-forward — fenetre {numero + 1}/{n_fenetres}",
                    numero / max(1, n_fenetres))
        depart = debut + numero * taille_fenetre
        arrivee = n if numero == n_fenetres - 1 else depart + taille_fenetre
        if depart >= n:
            break

        # Embargo : le passé s'arrête `horizon` lignes avant la fenêtre prédite,
        # sinon les dernières cibles d'apprentissage chevauchent ce qu'on évalue.
        fin_passe = max(1, depart - embargo)
        # Le passé n'apprend que sur les bougies apprenables ; la fenêtre
        # prédite, elle, est jugée sur toutes (voir `_sous_ensemble`).
        X_passe, y_passe = _sous_ensemble(jeu, np.arange(fin_passe), apprenable=True)
        X_futur, y_futur = _sous_ensemble(jeu, np.arange(depart, arrivee),
                                          apprenable=False)
        if len(X_futur) < 10 or len(X_passe) < 100 or y_passe.nunique() < 2:
            continue

        modele_fenetre, calibrateur, auc_val = _entrainer_fenetre(
            nom_modele, res, X_passe, y_passe, horizon, n_classes)

        probas = calibrateur.appliquer(modele_fenetre.predict_proba(X_futur))
        verite = y_futur.to_numpy()
        auc = _auc_probas(probas, verite, n_classes)
        justesse = float(np.mean(probas.argmax(axis=1) == verite))

        morceau = pd.DataFrame(probas, index=X_futur.index,
                               columns=[f"p{classe}" for classe in range(n_classes)])
        morceau["Fenetre"] = numero + 1
        morceaux.append(morceau)
        journal.append({"fenetre": numero + 1,
                        "debut": str(X_futur.index[0]), "fin": str(X_futur.index[-1]),
                        "n": len(X_futur), "auc_validation": auc_val,
                        "auc": auc, "justesse": justesse})

        print(f"   Fenêtre {numero + 1:2d}/{n_fenetres} | "
              f"{X_futur.index[0]:%Y-%m-%d} → {X_futur.index[-1]:%Y-%m-%d} | "
              f"{len(X_futur):>6,} bougies | AUC {auc:.4f} | justesse {justesse:.2%}")

    if not morceaux:
        raise ValueError("Aucune fenêtre exploitable.")

    suivi.etape("Assemblage des fenetres", 0.96)
    resultats = _assembler_walkforward(pd.concat(morceaux), symbole, intervalle,
                                       horizon, seuil_confiance, objectif)
    _rapport_walkforward(journal, resultats, seuil_confiance)
    suivi.etape("Termine", 1.0)
    return resultats


def _assembler_walkforward(predictions: pd.DataFrame, symbole: str, intervalle: str,
                           horizon: int, seuil: float,
                           objectif: cibles.Tache) -> pd.DataFrame:
    """
    Met les prédictions walk-forward au format habituel, puis les sauvegarde.

    Le bloc est marqué « test » pour toutes les lignes : c'est exact, elles sont
    toutes hors échantillon. Le backtest et les graphiques les traitent donc
    correctement sans le moindre cas particulier.
    """
    df = lire_analyse(symbole, intervalle)
    contexte = df.loc[predictions.index]
    colonne = config.colonne_variation(horizon)

    colonnes_proba = [f"p{classe}" for classe in range(objectif.n_classes)]
    probas = predictions[colonnes_proba].to_numpy(dtype=float)
    lecture = cibles.lire_probabilites(probas, objectif)

    variation = pd.to_numeric(contexte[colonne], errors="coerce")
    sens_reel = pd.Series(np.where(variation > 0, cibles.HAUSSE, cibles.BAISSE),
                          index=predictions.index)
    sens_reel[variation.isna()] = None

    resultats = pd.DataFrame({
        "Prix": pd.to_numeric(contexte["Close"], errors="coerce"),
        "Proba_Hausse": np.round(lecture["proba_hausse"], 4),
        "Confiance": np.round(lecture["confiance"], 4),
        "Sens_Predit": lecture["sens"],
        "Classe_Predite": lecture["etiquette"],
        "Retenu": (lecture["confiance"] >= seuil).astype(int),
        "Variation_Reelle": np.round(variation, 4),
        "Sens_Reel": sens_reel,
        "Fenetre": predictions["Fenetre"].to_numpy(),
        "Bloc": "test",
    }, index=predictions.index)

    if not objectif.binaire:
        resultats["Amplitude_Prevue"] = np.round(
            _amplitude_de_classe(lecture["classe"], contexte, horizon), 4)

    verite = objectif.construire_evaluation(contexte, horizon)
    resultats["Correct"] = np.where(
        verite.isna(), np.nan, (lecture["classe"] == verite.fillna(-1)).astype(float))

    chemin = stockage.chemin_prediction(symbole, intervalle, horizon,
                                        walk_forward=True, tache=objectif.cle)
    stockage.ecrire_tableau(resultats, chemin)
    print(f"\n💾 Prédictions walk-forward sauvegardées : {chemin}")
    return resultats


def _rapport_walkforward(journal: list[dict], resultats: pd.DataFrame,
                         seuil: float) -> None:
    """Bilan console : stabilité entre fenêtres et gain apporté par le seuil."""
    aucs = np.array([f["auc"] for f in journal])
    ecarts = np.array([f["auc_validation"] - f["auc"] for f in journal])

    print(f"\n📊 ===== BILAN WALK-FORWARD ({len(journal)} fenêtres) =====")
    print(f"   AUC hors échantillon : {aucs.mean():.4f} ± {aucs.std():.4f} "
          f"(min {aucs.min():.4f} / max {aucs.max():.4f})")
    print(f"   Fenêtres au-dessus du hasard : {int((aucs > 0.5).sum())}/{len(aucs)}")
    print(f"   Écart moyen validation → réel : {ecarts.mean():+.4f} "
          f"(un écart important = surapprentissage)")

    evaluables = resultats.dropna(subset=["Correct"])
    retenues = evaluables[evaluables["Retenu"] == 1]
    print(f"\n   Toutes prédictions : {len(evaluables):,} | "
          f"justesse {evaluables['Correct'].mean():.2%}")
    if len(retenues):
        hausses, baisses = _compter_sens(retenues)
        print(f"   Retenues (≥ {seuil:.0%}) : {len(retenues):,} "
              f"({len(retenues) / len(evaluables):.1%} du temps) | "
              f"justesse {retenues['Correct'].mean():.2%} "
              f"| {hausses:,} ▲ / {baisses:,} ▼")
    else:
        print(f"   Aucune prédiction n'atteint le seuil de {seuil:.0%}.")
    print("   " + "-" * 60)
    print("   Chaque point est hors échantillon : ce fichier peut être backtesté")
    print("   sur toute sa durée sans fausser le résultat.")


# ===========================================================================
# IMPORTANCE DES INDICATEURS
# ===========================================================================
def _modele_applicable(symbole: str, intervalle: str, horizon: int,
                       tache: str | None) -> tuple[dict, str]:
    """
    Charge le modèle de cette crypto, ou à défaut un panier qui la contient.

    Un modèle de panier est fait pour s'appliquer à chacune de ses cryptos :
    refuser de le trouver automatiquement obligerait à le désigner à la main
    partout. Le nom réellement utilisé est retourné pour être affiché.
    """
    try:
        return charger_modele(symbole, intervalle, horizon, tache), symbole
    except FileNotFoundError:
        objectif = cibles.obtenir(tache)
        paniers = stockage.paniers_contenant(symbole, intervalle, horizon,
                                             objectif.cle)
        if not paniers:
            raise
        source = paniers[0]
        print(f"ℹ️  Pas de modèle propre à {symbole} : j'utilise « {source} », "
              f"qui le contient.")
        return charger_modele(source, intervalle, horizon, tache), source



# ===========================================================================
# UTILITÉ MESURÉE DES FEATURES
# ===========================================================================
# À la fin de chaque entraînement, on mélange chaque colonne au hasard et on
# regarde de combien l'AUC tombe. Une feature dont l'AUC ne bouge pas ne sert à
# rien ; une feature dont l'AUC MONTE quand on la détruit nuit activement — le
# modèle avait appris à s'appuyer dessus sur l'apprentissage, et cet appui se
# retourne contre lui ailleurs.
#
# CE BLOC EST MESURÉ SUR LA VALIDATION, JAMAIS SUR LE TEST. C'est le point de
# méthode qui rend le curseur « utilité minimale » honnête : choisir ses
# features d'après le test puis annoncer une performance sur ce même test
# reviendrait à se noter soi-même. Le graphique de la page Visualisation, lui,
# reste mesuré sur le test — il sert à diagnostiquer, pas à décider.
def _mesurer_utilite(modele_ml, X, y, objectif, res,
                     repetitions: int | None = None) -> dict:
    """
    Perte d'AUC par permutation, feature par feature.

    Retourne un dictionnaire {feature: perte d'AUC}. Un dictionnaire vide en
    cas d'échec : c'est un diagnostic, il ne doit jamais faire échouer un
    entraînement par ailleurs réussi.
    """
    from sklearn.inspection import permutation_importance

    repetitions = int(repetitions or config.REPETITIONS_UTILITE)
    y = np.asarray(y)
    if len(X) < 200 or len(np.unique(y)) < 2:
        return {}

    score = "roc_auc" if objectif.binaire else "roc_auc_ovr"
    try:
        resultat = permutation_importance(
            modele_ml, X, y, scoring=score, n_repeats=repetitions,
            random_state=config.SEED, n_jobs=res.n_jobs_recherche)
    except Exception as err:                            # noqa: BLE001
        print(f"⚠️  Utilité des features non mesurable ici ({err}).")
        return {}

    return {str(nom): float(valeur)
            for nom, valeur in zip(X.columns, resultat.importances_mean)}


def utilite_enregistree(symbole: str, intervalle: str, horizon: int,
                        tache: str | None = None) -> dict:
    """
    Utilité par feature relevée au dernier entraînement de ce modèle.

    Dictionnaire vide s'il n'y a pas encore de modèle : le curseur d'utilité
    minimale reste alors sans effet, ce qui est le comportement attendu — on ne
    peut pas trier des features avant de les avoir mesurées.
    """
    objectif = cibles.obtenir(tache)
    chemin = stockage.chemin_meta(symbole, intervalle, horizon, objectif.cle)
    if not os.path.exists(chemin):
        return {}
    try:
        with open(chemin, encoding="utf-8") as fichier:
            meta = json.load(fichier)
    except (OSError, ValueError):                       # pragma: no cover
        return {}
    valeurs = (meta.get("utilite") or {}).get("valeurs") or {}
    return {str(nom): float(v) for nom, v in valeurs.items()}


def trier_par_utilite(valeurs: dict, seuil: float,
                      disponibles=None) -> tuple[list[str], list[str]]:
    """
    Sépare les features en (gardées, écartées) pour un seuil d'utilité donné.

    Deux garde-fous :
      * on ne descend jamais sous `config.FEATURES_MINIMUM` features, même si
        le seuil est absurdement haut — un modèle sans entrée n'est pas un
        modèle ;
      * une feature présente dans les données mais jamais mesurée est
        CONSERVÉE. Le doute profite à la donnée : c'est le cas d'une colonne
        ajoutée depuis le dernier entraînement.
    """
    connues = {str(nom): float(v) for nom, v in (valeurs or {}).items()}
    if disponibles is not None:
        colonnes = [str(c) for c in disponibles]
        inconnues = [c for c in colonnes if c not in connues]
        connues = {c: connues[c] for c in colonnes if c in connues}
    else:
        inconnues = []

    ordonne = sorted(connues.items(), key=lambda paire: paire[1], reverse=True)
    gardees = [nom for nom, valeur in ordonne if valeur >= seuil]

    minimum = max(1, int(config.FEATURES_MINIMUM) - len(inconnues))
    if len(gardees) < minimum:
        gardees = [nom for nom, _ in ordonne[:minimum]]

    retenues = set(gardees)
    ecartees = [nom for nom, _ in ordonne if nom not in retenues]
    return gardees + inconnues, ecartees


def apercu_utilite(symbole: str, intervalle: str, horizon: int,
                   tache: str | None = None, seuil: float = 0.0) -> dict:
    """
    Ce que donnerait un seuil d'utilité, sans rien recalculer.

    Sert au curseur de l'interface : il doit pouvoir annoncer « 14 features
    gardées sur 29 » à chaque cran, donc sans jamais toucher au disque autrement
    que pour relire un petit fichier JSON.
    """
    valeurs = utilite_enregistree(symbole, intervalle, horizon, tache)
    if not valeurs:
        return {"mesure": False, "gardees": [], "ecartees": [], "valeurs": {}}
    gardees, ecartees = trier_par_utilite(valeurs, seuil)
    return {"mesure": True, "gardees": gardees, "ecartees": ecartees,
            "valeurs": valeurs}


BLOC_TEST = "test"
BLOC_VALIDATION = "validation"


def importance_indicateurs(symbole: str, intervalle: str, horizon: int,
                           n_repetitions: int = 5,
                           tache: str | None = None,
                           bloc: str = BLOC_TEST) -> pd.Series:
    """
    Importance par permutation : on mélange une colonne, on mesure la perte d'AUC.

    Plus la perte est forte, plus l'indicateur était utile. Une perte NÉGATIVE
    signifie que détruire la colonne améliore le modèle : cette feature nuit.
    La méthode fonctionne avec n'importe quel modèle (y compris la régression
    logistique) et se lit directement, sans dépendance lourde type SHAP.

    Deux blocs possibles, et la différence n'est pas cosmétique :

      `test` (défaut)  DIAGNOSTIC. Sur quoi le modèle s'appuie réellement quand
                       il travaille sur des données jamais vues. C'est ce que
                       montre la page Visualisation.
      `validation`     DÉCISION. Le bloc utilisé pour choisir quelles features
                       garder (curseur « utilité minimale »). Sélectionner ses
                       features d'après le test puis annoncer un résultat sur ce
                       même test reviendrait à se noter soi-même.

    C'est aussi ici qu'on vérifie si le contexte multi-timeframe et les données
    exogènes apportent vraiment quelque chose : s'ils apparaissent en bas du
    classement, ils ne servent qu'à diluer le signal.
    """
    from sklearn.inspection import permutation_importance

    paquet, _source = _modele_applicable(symbole, intervalle, horizon, tache)
    objectif = cibles.obtenir(paquet.get("tache"))
    jeu = charger_jeu(symbole, intervalle, horizon, objectif)
    decoupage = _decouper(jeu.X.index, horizon, intervalle)

    if str(bloc) == BLOC_VALIDATION:
        # Seconde moitié de la validation : celle qui n'a servi ni à
        # l'apprentissage ni à l'arrêt anticipé.
        indices = decoupage.validation[len(decoupage.validation) // 2:]
        X_test, y_test = _sous_ensemble(jeu, indices, apprenable=False)
    else:
        X_test, y_test = _sous_ensemble(jeu, decoupage.test, apprenable=False)
    X_test = X_test[paquet["features"]]
    if paquet.get("normalisation") == "rang":
        # Les features doivent être dans le même état qu'à l'entraînement,
        # sinon la perte d'AUC mesurée n'a aucun rapport avec le modèle réel.
        from . import panier
        X_test = panier.normaliser(X_test)

    # Le score doit suivre l'objectif : l'AUC binaire n'a pas de sens sur cinq
    # classes, où c'est la moyenne « une contre toutes » qui fait référence.
    score = "roc_auc" if objectif.binaire else "roc_auc_ovr"

    res = ressources.detecter(int(X_test.memory_usage(deep=True).sum()))
    resultat = permutation_importance(
        paquet["modele"], X_test, y_test, scoring=score,
        n_repeats=n_repetitions, random_state=config.SEED, n_jobs=res.n_jobs_recherche)

    importance = pd.Series(resultat.importances_mean, index=X_test.columns)
    return importance.sort_values(ascending=False)
