"""
Entraînement et prédiction.

QUESTION POSÉE AU MODÈLE
------------------------
« Dans X périodes (X entre 1 et 24), le prix sera-t-il plus haut ou plus bas
qu'aujourd'hui ? »

La cible est simplement le signe de la colonne `variation_X` du fichier
analysé. Le modèle renvoie une probabilité de hausse `p`, dont on tire :

    Sens      = HAUSSE si p > 0.5, BAISSE sinon
    Confiance = max(p, 1 - p)          -> toujours entre 0.5 et 1

La prédiction est donc **bidirectionnelle** : une probabilité de 0.15 est un
signal de baisse à 85 % de confiance, aussi exploitable qu'une probabilité de
0.85. Le seuil de confiance sert ensuite à ne retenir que les prédictions dont
le modèle est sûr, dans un sens comme dans l'autre.

CE QUI EST AUTOMATIQUE (l'interface n'expose que crypto / modèle / horizon / seuil)
----------------------------------------------------------------------------------
  * Découpage chronologique 70 / 15 / 15 avec embargo (voir `_decouper`).
  * Rééquilibrage des classes si l'une dépasse 55 %.
  * Réglage des hyperparamètres : 3 configurations testées, la meilleure sur la
    validation est retenue.
  * Early stopping sur la validation pour les modèles à boosting.
  * Calibration des probabilités (isotonique ou Platt selon la taille).
  * Dimensionnement RAM / CPU (voir `ressources.py`).
"""

from __future__ import annotations

import json
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

from . import config, ressources, stockage

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
                poids_positif: float | None) -> object:
    """
    Instancie un classifieur non entraîné.

    Les réglages « machine » (nombre de threads, finesse des histogrammes)
    viennent de `ressources.py` : c'est là que la RAM libre est convertie en
    précision et en vitesse.
    """
    equilibre = poids_positif is not None

    if nom == "XGBoost":
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=config.ARBRES_MAX, early_stopping_rounds=config.PATIENCE_EARLY_STOP,
            max_bin=res.max_bin, n_jobs=res.n_jobs, random_state=config.SEED,
            verbosity=0, scale_pos_weight=poids_positif if equilibre else 1.0,
            **params)

    if nom == "LightGBM":
        return LGBMClassifier(
            objective="binary", n_estimators=config.ARBRES_MAX,
            max_bin=res.max_bin, n_jobs=res.n_jobs, random_state=config.SEED,
            verbose=-1, histogram_pool_size=res.pool_histogramme_mo,
            scale_pos_weight=poids_positif if equilibre else 1.0,
            **params)

    if nom == "CatBoost":
        return CatBoostClassifier(
            iterations=config.ARBRES_MAX, eval_metric="Logloss",
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
    """

    def __init__(self):
        self.methode = "aucune"
        self._modele = None

    def entrainer(self, probas: np.ndarray, verite: np.ndarray) -> "Calibrateur":
        probas = np.asarray(probas, dtype=float)
        verite = np.asarray(verite, dtype=float)

        if len(np.unique(verite)) < 2:
            self.methode = "aucune"                    # une seule classe : rien à calibrer
            return self

        n_tranches = len(probas) // SUPPORT_MINIMUM
        if n_tranches >= 5:
            self.methode = "isotonique"
            centres, frequences, effectifs = self._regrouper(probas, verite, n_tranches)
            self._modele = IsotonicRegression(out_of_bounds="clip")
            self._modele.fit(centres, frequences, sample_weight=effectifs)
        else:
            self.methode = "platt"
            self._modele = LogisticRegression(max_iter=1000)
            self._modele.fit(probas.reshape(-1, 1), verite)
        return self

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
        """Applique la correction apprise ; renvoie des probabilités dans ]0, 1[."""
        if self._modele is None:
            return probas
        if self.methode == "isotonique":
            corrigees = self._modele.predict(probas)
        else:
            corrigees = self._modele.predict_proba(np.asarray(probas).reshape(-1, 1))[:, 1]
        return np.clip(corrigees, 0.001, 0.999)


# ===========================================================================
# PRÉPARATION DES DONNÉES
# ===========================================================================
@dataclass
class Jeu:
    """Features et cible alignées, prêtes pour l'entraînement."""

    X: pd.DataFrame          # les 8 indicateurs
    y: pd.Series             # 1 = hausse, 0 = baisse
    horizon: int             # nombre de périodes prédites


@dataclass
class Decoupage:
    """Indices positionnels des trois blocs chronologiques."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    embargo: int


def charger_jeu(symbole: str, intervalle: str, horizon: int) -> Jeu:
    """
    Prépare features et cible pour un horizon donné.

    Features : **liste blanche stricte** = les 8 colonnes de
    `config.INDICATEURS`. Aucun prix, aucune colonne `variation_*` ne peut
    donc se retrouver en entrée du modèle, même par accident.

    Cible : signe de `variation_{horizon}` (1 = hausse, 0 = baisse).
    Les lignes trop récentes pour connaître leur variation sont écartées ici,
    mais restent disponibles pour la prédiction (voir `predire`).
    """
    chemin = stockage.chemin_analyse(symbole, intervalle)
    df = stockage.lire_tableau(chemin)
    if df is None or df.empty:
        raise FileNotFoundError(
            f"Fichier analysé introuvable pour {symbole} ({intervalle}). "
            f"Lance d'abord l'étape Analyse.")

    colonne_cible = config.colonne_variation(horizon)
    manquantes = [c for c in config.INDICATEURS + [colonne_cible] if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du fichier analysé : {manquantes}. "
            f"Relance l'étape Analyse (le format a changé).")

    X = df[config.INDICATEURS].astype("float32")
    variation = pd.to_numeric(df[colonne_cible], errors="coerce")

    valides = X.notna().all(axis=1) & variation.notna() & np.isfinite(variation)
    X, variation = X[valides], variation[valides]

    return Jeu(X=X, y=(variation > 0).astype(int), horizon=horizon)


def _decouper(n: int, horizon: int) -> Decoupage:
    """
    Découpe chronologique 70 / 15 / 15 avec embargo.

    Le test est TOUJOURS la période la plus récente : c'est la seule façon
    honnête d'estimer ce que vaudra le modèle demain.

    L'embargo purge `horizon` lignes à la fin de chaque bloc. Sans lui, les
    dernières lignes du train ont une cible qui regarde déjà dans la période de
    validation : le modèle « connaîtrait » une partie de ce sur quoi on
    l'évalue, et les scores seraient artificiellement bons.
    """
    embargo = max(0, int(horizon))
    fin_train = int(n * config.PART_TRAIN)
    fin_val = int(n * (1 - config.PART_TEST))

    train = np.arange(0, max(1, fin_train - embargo))
    validation = np.arange(fin_train, max(fin_train + 1, fin_val - embargo))
    test = np.arange(fin_val, n)
    return Decoupage(train=train, validation=validation, test=test, embargo=embargo)


def _poids_classe_positive(y: pd.Series) -> float | None:
    """
    Poids à donner à la classe « hausse » pour rééquilibrer l'apprentissage.

    Retourne None si les classes sont déjà équilibrées (aucun rééquilibrage
    appliqué) : forcer un rééquilibrage inutile dégrade la calibration.
    """
    n_hausse = int((y == 1).sum())
    n_baisse = int((y == 0).sum())
    if n_hausse == 0 or n_baisse == 0:
        return None
    part_majoritaire = max(n_hausse, n_baisse) / (n_hausse + n_baisse)
    if part_majoritaire < config.SEUIL_DESEQUILIBRE:
        return None
    return n_baisse / n_hausse


# ===========================================================================
# ENTRAÎNEMENT
# ===========================================================================
def _ajuster(modele, nom: str, X_tr, y_tr, X_val, y_val):
    """Entraîne un modèle, avec early stopping sur la validation si supporté."""
    if nom == "XGBoost":
        modele.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return int(getattr(modele, "best_iteration", 0) or 0)

    if nom == "LightGBM":
        modele.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                   callbacks=[early_stopping(config.PATIENCE_EARLY_STOP, verbose=False),
                              log_evaluation(0)])
        return int(getattr(modele, "best_iteration_", 0) or 0)

    if nom == "CatBoost":
        modele.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                   early_stopping_rounds=config.PATIENCE_EARLY_STOP, verbose=False)
        return int(modele.get_best_iteration() or 0)

    modele.fit(X_tr, y_tr)
    return int(getattr(modele, "n_estimators", 0) or 0)


def _auc(modele, X, y) -> float:
    """AUC-ROC d'un modèle sur un jeu donné (0.5 si une seule classe présente)."""
    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, modele.predict_proba(X)[:, 1]))


def _essayer(nom_modele, params, res, poids, X_tr, y_tr, X_sel, y_sel):
    """Entraîne UNE configuration et la note. Retourne (modèle, AUC, params, arbres)."""
    candidat = _construire(nom_modele, params, res, poids)
    iterations = _ajuster(candidat, nom_modele, X_tr, y_tr, X_sel, y_sel)
    return candidat, _auc(candidat, X_sel, y_sel), params, iterations


def _tester_configurations(nom_modele, res, poids, X_tr, y_tr, X_sel, y_sel):
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
    configurations = CONFIGURATIONS[nom_modele]
    print(f"🔎 Réglage automatique : {len(configurations)} configurations testées "
          f"({res.n_jobs} threads chacune)…")
    return [_essayer(nom_modele, params, res, poids, X_tr, y_tr, X_sel, y_sel)
            for params in configurations]


def _table_confiance(probas: np.ndarray, verite: np.ndarray) -> list[dict]:
    """
    Précision obtenue en ne gardant que les prédictions les plus sûres.

    C'est le tableau qui sert à choisir son seuil : pour chaque niveau de
    confiance, combien de bougies sont retenues et quelle part est correcte.
    Le sens prédit est pris en compte dans les DEUX directions.
    """
    sens_predit = (probas > 0.5).astype(int)
    confiance = np.maximum(probas, 1 - probas)
    correct = (sens_predit == verite)

    lignes = []
    for seuil in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
        retenus = confiance >= seuil
        nb = int(retenus.sum())
        lignes.append({
            "seuil": seuil,
            "n": nb,
            "couverture": float(nb / len(probas)) if len(probas) else 0.0,
            "precision": float(correct[retenus].mean()) if nb else None,
            "part_hausse": float(sens_predit[retenus].mean()) if nb else None,
        })
    return lignes


def _evaluer(probas: np.ndarray, verite: np.ndarray) -> dict:
    """Métriques complètes sur un jeu (test en principe)."""
    prediction = (probas > 0.5).astype(int)
    taux_hausse = float(np.mean(verite))

    rapport = classification_report(
        verite, prediction, labels=[0, 1], target_names=["Baisse", "Hausse"],
        output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy_score(verite, prediction)),
        "auc": float(roc_auc_score(verite, probas)) if len(np.unique(verite)) > 1 else 0.5,
        "rapport": rapport,
        "baseline_majoritaire": float(max(taux_hausse, 1 - taux_hausse)),
        "part_hausse_reelle": taux_hausse,
        "table_confiance": _table_confiance(probas, verite),
    }


def entrainer(symbole: str, intervalle: str, horizon: int,
              nom_modele: str = config.MODELE_DEFAUT) -> dict:
    """
    Entraîne un modèle pour une crypto, un intervalle et un horizon donnés.

    Déroulé complet :
      1. Chargement du jeu (8 features, cible = signe de variation_horizon).
      2. Découpage chronologique 70/15/15 avec embargo.
      3. Rééquilibrage automatique si nécessaire.
      4. Trois configurations entraînées ; la meilleure AUC de validation gagne.
      5. Calibration des probabilités sur la seconde moitié de la validation.
      6. Évaluation sur le test (jamais vu) + table des seuils de confiance.
      7. Sauvegarde du modèle (.joblib) et de ses métadonnées (.json).

    Retourne le dictionnaire de métadonnées.
    """
    config.preparer_dossiers()
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))

    print(f"\n🤖 Entraînement — {symbole} ({intervalle}) | modèle : {nom_modele} | "
          f"horizon : {horizon} période(s)")

    # --- 1. Données -------------------------------------------------------
    jeu = charger_jeu(symbole, intervalle, horizon)
    res = ressources.detecter(int(jeu.X.memory_usage(deep=True).sum()))
    print(res.resume())

    part_hausse = float(jeu.y.mean())
    print(f"📐 {len(jeu.X):,} lignes exploitables | {len(config.INDICATEURS)} indicateurs")
    print(f"⚖️  Répartition réelle — hausse {part_hausse:.1%} / baisse {1 - part_hausse:.1%}")

    # --- 2. Découpage -----------------------------------------------------
    decoupage = _decouper(len(jeu.X), horizon)
    if min(len(decoupage.train), len(decoupage.validation), len(decoupage.test)) < 50:
        raise ValueError("Historique trop court pour entraîner un modèle fiable "
                         "(il faut au moins quelques centaines de bougies).")

    X_tr, y_tr = jeu.X.iloc[decoupage.train], jeu.y.iloc[decoupage.train]
    X_val, y_val = jeu.X.iloc[decoupage.validation], jeu.y.iloc[decoupage.validation]
    X_test, y_test = jeu.X.iloc[decoupage.test], jeu.y.iloc[decoupage.test]

    # La validation sert à deux choses différentes, sur deux moitiés distinctes :
    # choisir/arrêter le modèle d'un côté, calibrer ses probabilités de l'autre.
    milieu = len(X_val) // 2
    X_sel, y_sel = X_val.iloc[:milieu], y_val.iloc[:milieu]
    X_cal, y_cal = X_val.iloc[milieu:], y_val.iloc[milieu:]

    print(f"✂️  Train {len(X_tr):,} | Validation {len(X_val):,} "
          f"(sélection {len(X_sel):,} + calibration {len(X_cal):,}) | Test {len(X_test):,}")
    print(f"🚧 Embargo : {decoupage.embargo} ligne(s) purgée(s) entre les blocs (anti-fuite).")

    # --- 3. Rééquilibrage automatique ------------------------------------
    poids = _poids_classe_positive(y_tr)
    if poids is not None:
        print(f"⚖️  Classes déséquilibrées → rééquilibrage automatique (poids {poids:.2f}).")

    # --- 4. Sélection de la configuration --------------------------------
    essais = _tester_configurations(nom_modele, res, poids,
                                    X_tr, y_tr, X_sel, y_sel)
    for numero, essai in enumerate(essais, start=1):
        print(f"   {numero}/{len(essais)} — AUC validation {essai[1]:.4f} ({essai[2]})")

    meilleur, meilleure_auc, meilleurs_params, meilleures_iterations = max(
        essais, key=lambda essai: essai[1])
    print(f"🏆 Configuration retenue : AUC validation {meilleure_auc:.4f}"
          + (f" | {meilleures_iterations} arbres (early stopping)"
             if nom_modele in AVEC_EARLY_STOPPING else ""))

    # --- 5. Calibration ---------------------------------------------------
    calibrateur = Calibrateur().entrainer(
        meilleur.predict_proba(X_cal)[:, 1], y_cal.to_numpy())
    print(f"📐 Calibration des probabilités : méthode {calibrateur.methode}.")

    # --- 6. Évaluation sur le test ---------------------------------------
    probas_test = calibrateur.appliquer(meilleur.predict_proba(X_test)[:, 1])
    metriques = _evaluer(probas_test, y_test.to_numpy())
    _afficher_evaluation(metriques, symbole, intervalle, horizon)

    # --- 7. Sauvegarde ----------------------------------------------------
    meta = {
        "symbole": symbole,
        "intervalle": intervalle,
        "horizon": horizon,
        "modele": nom_modele,
        "date_entrainement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(config.INDICATEURS),
        "hyperparametres": {str(k): v for k, v in meilleurs_params.items()},
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

    joblib.dump(
        {"modele": meilleur, "calibrateur": calibrateur,
         "features": list(config.INDICATEURS), "horizon": horizon,
         "nom_modele": nom_modele, "meta": meta},
        stockage.chemin_modele(symbole, intervalle, horizon), compress=3)

    with open(stockage.chemin_meta(symbole, intervalle, horizon), "w",
              encoding="utf-8") as fichier:
        json.dump(meta, fichier, indent=2, ensure_ascii=False)

    print(f"💾 Modèle sauvegardé : {stockage.chemin_modele(symbole, intervalle, horizon)}")
    memoire = ressources.memoire_processus_go()
    if memoire:
        print(f"🧠 Mémoire utilisée par le processus : {memoire:.2f} Go "
              f"sur {res.budget_go:.1f} Go de budget.")
    return meta


def _afficher_evaluation(metriques: dict, symbole: str, intervalle: str,
                         horizon: int) -> None:
    """Rapport console de fin d'entraînement."""
    accuracy = metriques["accuracy"]
    baseline = metriques["baseline_majoritaire"]
    verdict = "✅ battue" if accuracy > baseline else "❌ NON battue"

    print(f"\n📊 ===== ÉVALUATION SUR LE TEST — {symbole} {intervalle} (h={horizon}) =====")
    print(f"   Justesse : {accuracy:.2%}   |   AUC-ROC : {metriques['auc']:.4f}")
    print(f"   Baseline « toujours la même réponse » : {baseline:.2%}  ({verdict})")
    print("\n   Précision selon le seuil de confiance (les deux sens confondus) :")
    print("   seuil │ prédictions retenues │ couverture │ précision")
    for ligne in metriques["table_confiance"]:
        if ligne["precision"] is None:
            continue
        print(f"   {ligne['seuil']:.2f}  │ {ligne['n']:>20,} │ "
              f"{ligne['couverture']:>9.1%} │ {ligne['precision']:>9.2%}")
    print("   " + "-" * 60)


# ===========================================================================
# PRÉDICTION
# ===========================================================================
def charger_modele(symbole: str, intervalle: str, horizon: int) -> dict:
    """Charge le paquet {modèle, calibrateur, features, meta} depuis le disque."""
    chemin = stockage.chemin_modele(symbole, intervalle, horizon)
    try:
        return joblib.load(chemin)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Aucun modèle entraîné pour {symbole} ({intervalle}) à l'horizon "
            f"{horizon}. Lance d'abord l'entraînement.") from None


def predire(symbole: str, intervalle: str, horizon: int,
            seuil_confiance: float = config.SEUIL_DEFAUT,
            symbole_modele: str | None = None) -> pd.DataFrame:
    """
    Applique un modèle entraîné à tout l'historique d'une crypto.

    `symbole_modele` permet d'appliquer le modèle d'une crypto à une autre
    (les 8 features étant les mêmes partout, un modèle est transposable).

    Colonnes produites :
      Prix              cours de clôture de la bougie
      Proba_Hausse      probabilité calibrée que le prix monte
      Confiance         max(p, 1-p) — certitude, quel que soit le sens
      Sens_Predit       HAUSSE / BAISSE
      Retenu            1 si Confiance ≥ seuil (c'est ce filtre qu'on exploite)
      Variation_Reelle  variation observée après l'horizon, en % (NaN si futur)
      Sens_Reel         HAUSSE / BAISSE observé
      Correct           1 / 0 (NaN tant que l'issue est inconnue)
      Bloc              train / validation / test — pour un backtest honnête
    """
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))
    source_modele = symbole_modele or symbole
    paquet = charger_modele(source_modele, intervalle, horizon)

    etiquette = (symbole if source_modele == symbole
                 else f"{symbole} ← modèle {source_modele}")
    print(f"\n🔮 Prédiction — {etiquette} ({intervalle}) | horizon {horizon} | "
          f"seuil de confiance {seuil_confiance:.0%}")

    # On repart du fichier analysé complet : contrairement à l'entraînement, on
    # garde ici les toutes dernières bougies dont l'issue est encore inconnue,
    # puisque ce sont justement celles qui nous intéressent pour décider.
    df = stockage.lire_tableau(stockage.chemin_analyse(symbole, intervalle))
    if df is None or df.empty:
        raise FileNotFoundError(f"Fichier analysé introuvable pour {symbole} ({intervalle}).")

    colonne_variation = config.colonne_variation(horizon)
    attendues = list(paquet["features"]) + [colonne_variation, "Close"]
    manquantes = [c for c in attendues if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes absentes du fichier analysé de {symbole} : {manquantes}. "
            f"Relance l'étape Analyse pour le régénérer au format attendu.")

    X = df[paquet["features"]].astype("float32")
    valides = X.notna().all(axis=1)
    X = X[valides]
    if X.empty:
        raise ValueError("Aucune ligne exploitable après nettoyage.")

    probas = paquet["calibrateur"].appliquer(paquet["modele"].predict_proba(X)[:, 1])
    confiance = np.maximum(probas, 1 - probas)
    sens_predit = np.where(probas > 0.5, "HAUSSE", "BAISSE")

    variation = pd.to_numeric(df.loc[valides, colonne_variation], errors="coerce")
    sens_reel = pd.Series(np.where(variation > 0, "HAUSSE", "BAISSE"), index=X.index)
    sens_reel[variation.isna()] = None

    resultats = pd.DataFrame({
        "Prix": pd.to_numeric(df.loc[valides, "Close"], errors="coerce"),
        "Proba_Hausse": np.round(probas, 4),
        "Confiance": np.round(confiance, 4),
        "Sens_Predit": sens_predit,
        "Retenu": (confiance >= seuil_confiance).astype(int),
        "Variation_Reelle": np.round(variation, 4),
        "Sens_Reel": sens_reel,
    }, index=X.index)
    resultats["Correct"] = np.where(
        resultats["Sens_Reel"].isna(), np.nan,
        (resultats["Sens_Predit"] == resultats["Sens_Reel"]).astype(float))
    resultats["Bloc"] = _etiqueter_blocs(resultats.index, paquet["meta"])

    chemin = stockage.chemin_prediction(symbole, intervalle, horizon)
    stockage.ecrire_tableau(resultats, chemin)
    print(f"💾 Prédictions sauvegardées : {chemin}")

    _rapport_prediction(resultats, seuil_confiance)
    return resultats


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


def _rapport_prediction(resultats: pd.DataFrame, seuil: float) -> None:
    """Résumé console : dernier signal + performance des prédictions retenues."""
    derniere = resultats.iloc[-1]
    print(f"\n🎯 Dernière bougie ({resultats.index[-1]:%Y-%m-%d %H:%M}) : "
          f"{derniere['Sens_Predit']} — confiance {derniere['Confiance']:.1%} "
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
        hausses = int((retenues["Sens_Predit"] == "HAUSSE").sum())
        print(f"   {bloc:<10} │ {len(sous_ensemble):>11,} │ {len(retenues):>8,} │ "
              f"{retenues['Correct'].mean():>16.2%} │ "
              f"{hausses:,} / {len(retenues) - hausses:,}")

    print(f"   (seuil de confiance : {seuil:.0%} — seule la ligne « test » "
          f"reflète des données jamais vues)")


# ===========================================================================
# IMPORTANCE DES INDICATEURS
# ===========================================================================
def importance_indicateurs(symbole: str, intervalle: str, horizon: int,
                           n_repetitions: int = 5) -> pd.Series:
    """
    Importance par permutation, calculée sur le jeu de test.

    Principe : on mélange une colonne au hasard et on mesure la perte d'AUC.
    Plus la perte est forte, plus l'indicateur était utile. Cette méthode
    fonctionne avec n'importe quel modèle (y compris la régression logistique)
    et se lit directement, sans dépendance lourde type SHAP.
    """
    from sklearn.inspection import permutation_importance

    paquet = charger_modele(symbole, intervalle, horizon)
    jeu = charger_jeu(symbole, intervalle, horizon)
    decoupage = _decouper(len(jeu.X), horizon)

    X_test = jeu.X.iloc[decoupage.test][paquet["features"]]
    y_test = jeu.y.iloc[decoupage.test]

    res = ressources.detecter(int(X_test.memory_usage(deep=True).sum()))
    resultat = permutation_importance(
        paquet["modele"], X_test, y_test, scoring="roc_auc",
        n_repeats=n_repetitions, random_state=config.SEED, n_jobs=res.n_jobs_recherche)

    importance = pd.Series(resultat.importances_mean, index=X_test.columns)
    return importance.sort_values(ascending=False)
