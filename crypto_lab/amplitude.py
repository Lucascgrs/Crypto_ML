"""
Prédire COMBIEN, et pas seulement dans quel sens.

Le modèle de direction répond à « ça monte ou ça descend ? ». Il ne dit rien de
l'ampleur du mouvement — or un signal juste à 55 % sur un mouvement de 3 % ne
vaut pas du tout un signal juste à 60 % sur 0,2 %. Ce module ajoute les deux
briques qui manquaient, puis les combine.

1 · RÉGRESSION QUANTILE  (`cible="amplitude"`)
    Trois modèles estiment les quantiles 10 / 50 / 90 de `variation_h`. La
    sortie n'est plus un point mais un INTERVALLE : « dans 3 h, entre −0.8 % et
    +1.6 %, 8 fois sur 10 ». C'est littéralement la volatilité du mouvement
    attendu. LightGBM, XGBoost et CatBoost ont tous les trois l'objectif
    quantile en natif ; sklearn sert de repli toujours disponible.

2 · RÉGRESSION DE VOLATILITÉ  (`cible="volatilite"`)
    Un modèle sur |variation_h|. C'est là qu'il y a du vrai signal : la
    direction est quasi imprévisible, l'amplitude beaucoup moins. Le clustering
    de volatilité — une bougie agitée en annonce d'autres — est le fait stylisé
    le plus robuste de la finance de marché. On attend un R² nettement positif,
    là où la direction plafonne autour de zéro.

3 · ESPÉRANCE DE GAIN  (`esperance()`)
    Le score qui pilote vraiment une décision :

        espérance = (2 × P(hausse) − 1) × amplitude attendue − frais

    Le premier facteur vient du modèle de direction, le second du modèle de
    volatilité. Une position longue rapporte +A avec la probabilité p et −A
    avec 1−p : son espérance vaut donc (2p−1)·A. Retrancher le coût de
    l'aller-retour transforme enfin le signal en décision économique.

ÉVALUATION — ce qu'on regarde ici, et pourquoi ce n'est pas l'AUC
-----------------------------------------------------------------
  * **perte pinball** : la métrique propre d'une régression quantile, comparée
    à celle du quantile constant (le « ne rien prédire » de référence).
  * **couverture** : quelle part des issues réelles tombe entre Q10 et Q90 ?
    Doit valoir 80 %. Moins, l'intervalle est trop optimiste ; plus, il est
    inutilement large.
  * **R² et corrélation de rang** pour la volatilité, face à une référence
    naïve « l'ATR dilaté en √h » — qui est déjà un très bon prédicteur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from . import cibles, config, modele, ressources, stockage

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:                                    # pragma: no cover
    XGBOOST_OK = False

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    LIGHTGBM_OK = True
except ImportError:                                    # pragma: no cover
    LIGHTGBM_OK = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_OK = True
except ImportError:                                    # pragma: no cover
    CATBOOST_OK = False


# ===========================================================================
# CATALOGUE
# ===========================================================================
# Mêmes philosophies que pour la classification : prudent / équilibré / souple.
# Les trois sont entraînées puis départagées sur la validation, avec la même
# règle du « un écart-type » qui préfère la plus simple à égalité statistique.
CONFIGURATIONS = {
    "XGBoost": [
        {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.7,
         "colsample_bytree": 0.7, "min_child_weight": 30, "reg_lambda": 5.0},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 2.0},
        {"max_depth": 7, "learning_rate": 0.05, "subsample": 0.9,
         "colsample_bytree": 0.9, "min_child_weight": 3, "reg_lambda": 1.0},
    ],
    "LightGBM": [
        {"num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 200,
         "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 5.0},
        {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 60,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 20,
         "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    ],
    "CatBoost": [
        {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 6.0},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 1.0},
    ],
    "GradientBoosting": [
        {"max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 200,
         "l2_regularization": 5.0},
        {"max_depth": 6, "learning_rate": 0.05, "min_samples_leaf": 60,
         "l2_regularization": 2.0},
        {"max_depth": None, "learning_rate": 0.08, "min_samples_leaf": 20,
         "l2_regularization": 1.0},
    ],
}

DESCRIPTIONS = {
    "XGBoost": "Objectif quantile natif (reg:quantileerror). Rapide et solide.",
    "LightGBM": "Objectif quantile natif, le plus rapide sur gros historiques.",
    "CatBoost": "Objectif quantile natif, très résistant au surapprentissage.",
    "GradientBoosting": "Repli scikit-learn (HistGradientBoosting). Toujours "
                        "disponible, sans dépendance supplémentaire.",
}

# La régression logistique et la forêt aléatoire ne font pas de régression
# quantile : elles ne figurent donc pas ici. Le repli sklearn tient ce rôle.
ORDRE_MODELES = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]


def modeles_disponibles() -> list[str]:
    """Modèles de régression réellement utilisables sur cette machine."""
    dispo = []
    if XGBOOST_OK:
        dispo.append("XGBoost")
    if LIGHTGBM_OK:
        dispo.append("LightGBM")
    if CATBOOST_OK:
        dispo.append("CatBoost")
    dispo.append("GradientBoosting")           # sklearn : toujours présent
    return dispo


# ===========================================================================
# CONSTRUCTION DES RÉGRESSEURS
# ===========================================================================
# Puissance de la loi de Tweedie utilisée pour la volatilité. Entre 1 et 2, elle
# interpole entre Poisson et Gamma : exactement le profil d'une amplitude, à
# savoir positive, très asymétrique à droite et à queue épaisse.
PUISSANCE_TWEEDIE = 1.5


def _construire(nom: str, params: dict, res: ressources.Ressources,
                quantile: float | None) -> object:
    """
    Instancie un régresseur non entraîné.

    `quantile` fixe l'objectif :

      * une valeur entre 0 et 1 demande la **régression quantile**
        correspondante — la question « quelle valeur ne sera pas dépassée dans
        90 % des cas ? » n'est pas la même que « quelle valeur moyenne ? » ;

      * None demande la régression de **volatilité**, avec une perte de Tweedie.
        Le choix de la perte est ici décisif et n'a rien d'un détail : l'erreur
        absolue estimerait la MÉDIANE, or |variation| est très asymétrique
        (médiane 0.44 %, moyenne 0.79 % sur BTC 1h à l'horizon 3). Un modèle
        médian sous-estimerait donc systématiquement l'amplitude, et
        l'espérance de gain — qui a besoin de E[amplitude] — serait faussée
        vers le bas. L'erreur au carré viserait bien la moyenne, mais serait
        dominée par trois krachs. Tweedie vise la moyenne tout en tenant compte
        de l'asymétrie : c'est la perte faite pour ce genre de cible.
    """
    if nom == "XGBoost":
        specifiques = ({"objective": "reg:quantileerror", "quantile_alpha": quantile}
                       if quantile is not None else
                       {"objective": "reg:tweedie",
                        "tweedie_variance_power": PUISSANCE_TWEEDIE})
        return XGBRegressor(
            tree_method="hist", n_estimators=config.ARBRES_MAX,
            early_stopping_rounds=config.PATIENCE_EARLY_STOP,
            max_bin=res.max_bin, n_jobs=res.n_jobs, random_state=config.SEED,
            verbosity=0, **specifiques, **params)

    if nom == "LightGBM":
        specifiques = ({"objective": "quantile", "alpha": quantile}
                       if quantile is not None else
                       {"objective": "tweedie",
                        "tweedie_variance_power": PUISSANCE_TWEEDIE})
        return LGBMRegressor(
            n_estimators=config.ARBRES_MAX, max_bin=res.max_bin, n_jobs=res.n_jobs,
            random_state=config.SEED, verbose=-1,
            histogram_pool_size=res.pool_histogramme_mo, **specifiques, **params)

    if nom == "CatBoost":
        perte = (f"Quantile:alpha={quantile}" if quantile is not None
                 else f"Tweedie:variance_power={PUISSANCE_TWEEDIE}")
        return CatBoostRegressor(
            iterations=config.ARBRES_MAX, loss_function=perte, eval_metric=perte,
            border_count=min(254, res.max_bin), thread_count=res.n_jobs,
            random_seed=config.SEED, verbose=0, **params)

    if nom == "GradientBoosting":
        # sklearn n'expose pas Tweedie sur cet estimateur ; la loi Gamma en est
        # le cas limite (puissance 2) et vise elle aussi la moyenne.
        specifiques = ({"loss": "quantile", "quantile": quantile}
                       if quantile is not None else {"loss": "gamma"})
        return HistGradientBoostingRegressor(
            max_iter=config.ARBRES_MAX, early_stopping=True,
            n_iter_no_change=config.PATIENCE_EARLY_STOP, validation_fraction=0.15,
            max_bins=min(255, res.max_bin), random_state=config.SEED,
            **specifiques, **params)

    raise ValueError(f"Modèle de régression inconnu : {nom}. "
                     f"Disponibles : {', '.join(modeles_disponibles())}.")


def _ajuster(regresseur, nom: str, X_tr, y_tr, X_val, y_val) -> int:
    """Entraîne un régresseur, avec early stopping sur la validation si supporté."""
    if nom == "XGBoost":
        regresseur.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return int(getattr(regresseur, "best_iteration", 0) or 0)

    if nom == "LightGBM":
        regresseur.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                       callbacks=[early_stopping(config.PATIENCE_EARLY_STOP, verbose=False),
                                  log_evaluation(0)])
        return int(getattr(regresseur, "best_iteration_", 0) or 0)

    if nom == "CatBoost":
        regresseur.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                       early_stopping_rounds=config.PATIENCE_EARLY_STOP, verbose=False)
        return int(regresseur.get_best_iteration() or 0)

    # HistGradientBoosting gère son propre early stopping en interne.
    regresseur.fit(X_tr, y_tr)
    return int(getattr(regresseur, "n_iter_", 0) or 0)


# ===========================================================================
# MÉTRIQUES
# ===========================================================================
def perte_pinball(verite: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    """
    Perte pinball — la métrique propre d'une régression quantile.

    Elle pénalise différemment les écarts au-dessus et en dessous : pour le
    quantile 90, sous-estimer coûte 9 fois plus que surestimer. C'est ce qui
    force le modèle à produire un vrai quantile plutôt qu'une moyenne.
    """
    ecart = np.asarray(verite, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.mean(np.maximum(quantile * ecart, (quantile - 1) * ecart)))


def _erreur_type(pertes: np.ndarray) -> float:
    """Marge d'erreur d'une perte moyenne : écart-type / √n."""
    pertes = np.asarray(pertes, dtype=float)
    if len(pertes) < 2:
        return float("inf")
    return float(pertes.std(ddof=1) / np.sqrt(len(pertes)))


def _pertes_individuelles(verite, prediction, quantile: float | None) -> np.ndarray:
    """
    Perte point par point, pour en tirer la moyenne ET sa marge d'erreur.

    Pinball pour les quantiles. Erreur absolue pour la volatilité : les trois
    candidates partagent déjà la même perte d'entraînement (Tweedie), la
    départager sur une métrique robuste évite qu'un seul krach du bloc de
    validation ne décide du modèle retenu.
    """
    ecart = np.asarray(verite, dtype=float) - np.asarray(prediction, dtype=float)
    if quantile is None:
        return np.abs(ecart)
    return np.maximum(quantile * ecart, (quantile - 1) * ecart)


def _choisir_configuration(essais, bavard: bool = False):
    """
    Règle du « un écart-type », version régression.

    Identique à celle de la classification : parmi les configurations dont la
    perte tient dans la marge d'erreur de la meilleure, on garde la plus simple
    (les configurations sont rangées de la plus prudente à la plus souple).
    Sans elle, on choisirait systématiquement le modèle le plus complexe sur
    des écarts de perte qui ne sont que du bruit d'échantillonnage.
    """
    meilleur_indice = min(range(len(essais)), key=lambda i: essais[i][1])
    perte_min = essais[meilleur_indice][1]
    marge = essais[meilleur_indice][4]

    for indice, essai in enumerate(essais):
        if essai[1] <= perte_min + marge:
            if bavard and indice != meilleur_indice:
                print(f"   ⚖️  Règle du 1 écart-type (±{marge:.4f}) : configuration "
                      f"{indice + 1} retenue au lieu de {meilleur_indice + 1}.")
            return essai
    return essais[meilleur_indice]


# ===========================================================================
# PRÉPARATION DES DONNÉES
# ===========================================================================
@dataclass
class JeuRegression:
    """Features et cible continue, alignées."""

    X: pd.DataFrame
    y: pd.Series
    horizon: int
    cible: str
    reference: pd.Series      # référence naïve (ATR dilaté), même index que y


def charger_jeu(symbole: str, intervalle: str, horizon: int,
                cible: str = "volatilite") -> JeuRegression:
    """
    Prépare features et cible continue.

    `cible="amplitude"`  -> variation_h signée, en %  (pour les quantiles)
    `cible="volatilite"` -> |variation_h|, en %       (pour l'ampleur)

    La colonne `reference` porte l'estimation naïve « ATR dilaté en √h ».
    C'est la barre à battre : sans elle, un R² de 0.30 pourrait n'être qu'un
    autre nom pour « l'ATR est un bon prédicteur », ce qu'on sait déjà.
    """
    if cible not in config.CIBLES_REGRESSION:
        raise ValueError(f"Cible de régression inconnue : {cible}. "
                         f"Attendu : {', '.join(config.CIBLES_REGRESSION)}.")

    df = modele.lire_analyse(symbole, intervalle)
    features = config.features_disponibles(df.columns)

    X = df[features].astype("float32")
    variation = pd.to_numeric(df[config.colonne_variation(horizon)], errors="coerce")
    variation = variation.replace([np.inf, -np.inf], np.nan)

    # Les pertes de Tweedie et Gamma exigent une cible strictement positive :
    # un plancher minuscule évite l'erreur sans changer quoi que ce soit à la
    # distribution (une bougie parfaitement plate est rarissime).
    y = variation if cible == "amplitude" else variation.abs().clip(lower=1e-4)

    echelle = cibles.echelle_attendue(df, horizon)
    valides = X.notna().all(axis=1) & y.notna() & echelle.notna()

    return JeuRegression(X=X[valides], y=y[valides], horizon=horizon, cible=cible,
                         reference=echelle[valides])


# ===========================================================================
# ENTRAÎNEMENT
# ===========================================================================
def entrainer(symbole: str, intervalle: str, horizon: int,
              nom_modele: str = "LightGBM",
              cible: str = "volatilite") -> dict:
    """
    Entraîne les modèles de régression pour une crypto et un horizon.

    Pour `cible="amplitude"`, TROIS modèles sont entraînés — un par quantile
    (10 / 50 / 90) — car un quantile est une question différente à chaque fois.
    Pour `cible="volatilite"`, un seul suffit.

    Découpage, embargo et règle du « un écart-type » sont exactement ceux de la
    classification : les deux familles de modèles restent comparables.
    """
    config.preparer_dossiers()
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))

    print(f"\n📈 Régression « {cible} » — {symbole} ({intervalle}) | "
          f"modèle : {nom_modele} | horizon : {horizon}")
    print(f"   {config.CIBLES_REGRESSION[cible]}")

    jeu = charger_jeu(symbole, intervalle, horizon, cible)
    res = ressources.detecter(int(jeu.X.memory_usage(deep=True).sum()))
    print(res.resume())

    decoupage = modele._decouper(len(jeu.X), horizon)       # noqa: SLF001
    if min(len(decoupage.train), len(decoupage.validation), len(decoupage.test)) < 50:
        raise ValueError("Historique trop court pour une régression fiable.")

    X_tr, y_tr = jeu.X.iloc[decoupage.train], jeu.y.iloc[decoupage.train]
    X_val, y_val = jeu.X.iloc[decoupage.validation], jeu.y.iloc[decoupage.validation]
    X_test, y_test = jeu.X.iloc[decoupage.test], jeu.y.iloc[decoupage.test]

    print(f"📐 {len(jeu.X):,} lignes | {len(jeu.X.columns)} features")
    print(f"✂️  Train {len(X_tr):,} | Validation {len(X_val):,} | Test {len(X_test):,} "
          f"| embargo {decoupage.embargo}")
    print(f"📊 Cible — moyenne {y_tr.mean():.3f} % | médiane {y_tr.median():.3f} % "
          f"| écart-type {y_tr.std():.3f} %")

    quantiles = config.QUANTILES if cible == "amplitude" else (None,)
    modeles, iterations, params_retenus = {}, {}, {}

    for quantile in quantiles:
        etiquette = (f"quantile {quantile:.0%}" if quantile is not None
                     else "erreur absolue")
        print(f"\n🔎 Réglage automatique — {etiquette} "
              f"({len(CONFIGURATIONS[nom_modele])} configurations)…")

        essais = []
        for params in CONFIGURATIONS[nom_modele]:
            candidat = _construire(nom_modele, params, res, quantile)
            n_arbres = _ajuster(candidat, nom_modele, X_tr, y_tr, X_val, y_val)
            pertes = _pertes_individuelles(y_val, candidat.predict(X_val), quantile)
            essais.append((candidat, float(pertes.mean()), params, n_arbres,
                           _erreur_type(pertes)))
            print(f"   perte {pertes.mean():.4f} ± {_erreur_type(pertes):.4f} ({params})")

        retenu, perte, params, n_arbres, _ = _choisir_configuration(essais, bavard=True)
        cle = config.COLONNES_QUANTILES.get(quantile, "Volatilite")
        modeles[cle] = retenu
        iterations[cle] = n_arbres
        params_retenus[cle] = {str(k): v for k, v in params.items()}
        print(f"🏆 Retenu — perte validation {perte:.4f}"
              + (f" | {n_arbres} arbres" if n_arbres else ""))

    # --- Évaluation sur le test (jamais vu) -------------------------------
    predictions = {cle: m.predict(X_test) for cle, m in modeles.items()}
    metriques = (_evaluer_quantiles(predictions, y_test.to_numpy())
                 if cible == "amplitude" else
                 _evaluer_volatilite(predictions["Volatilite"], y_test.to_numpy(),
                                     jeu.reference.iloc[decoupage.test].to_numpy(),
                                     y_tr.to_numpy(),
                                     jeu.reference.iloc[decoupage.train].to_numpy()))
    _afficher_evaluation(metriques, cible, symbole, intervalle, horizon)

    # --- Sauvegarde -------------------------------------------------------
    meta = {
        "symbole": symbole, "intervalle": intervalle, "horizon": horizon,
        "cible": cible, "modele": nom_modele,
        "date_entrainement": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(jeu.X.columns),
        "features_contexte": [c for c in jeu.X.columns if c not in config.INDICATEURS],
        "quantiles": [q for q in quantiles if q is not None],
        "hyperparametres": params_retenus,
        "n_arbres": iterations,
        "embargo": decoupage.embargo,
        "n_total": len(jeu.X), "n_train": len(X_tr),
        "n_validation": len(X_val), "n_test": len(X_test),
        "periode_debut": str(jeu.X.index.min()), "periode_fin": str(jeu.X.index.max()),
        "debut_validation": str(X_val.index.min()), "debut_test": str(X_test.index.min()),
        "metriques": metriques,
    }

    chemin = stockage.chemin_regression(symbole, intervalle, horizon, cible)
    joblib.dump({"modeles": modeles, "features": list(jeu.X.columns),
                 "horizon": horizon, "cible": cible, "nom_modele": nom_modele,
                 "meta": meta}, chemin, compress=3)
    with open(stockage.chemin_meta_regression(symbole, intervalle, horizon, cible),
              "w", encoding="utf-8") as fichier:
        json.dump(meta, fichier, indent=2, ensure_ascii=False)

    print(f"💾 Modèle de régression sauvegardé : {chemin}")
    return meta


# ===========================================================================
# ÉVALUATION
# ===========================================================================
def _evaluer_quantiles(predictions: dict, verite: np.ndarray) -> dict:
    """
    Qualité d'un intervalle de prédiction.

    Deux questions, deux réponses :
      * la perte pinball fait-elle mieux que le quantile constant (« je prédis
        toujours la même chose ») ? C'est le gain réel du modèle.
      * la COUVERTURE est-elle honnête ? Si 80 % des issues tombent bien entre
        Q10 et Q90, l'intervalle veut dire ce qu'il annonce. C'est la métrique
        la plus importante ici, et la plus facile à lire.
    """
    detail = {}
    for quantile in config.QUANTILES:
        cle = config.COLONNES_QUANTILES[quantile]
        prevu = np.asarray(predictions[cle], dtype=float)
        constante = np.quantile(verite, quantile)
        perte = perte_pinball(verite, prevu, quantile)
        reference = perte_pinball(verite, np.full_like(verite, constante), quantile)
        detail[cle] = {
            "quantile": quantile,
            "perte_pinball": perte,
            "perte_reference": float(reference),
            "gain": float(1 - perte / reference) if reference else 0.0,
            "moyenne_prevue": float(prevu.mean()),
        }

    bas = np.asarray(predictions[config.COLONNES_QUANTILES[config.QUANTILES[0]]])
    haut = np.asarray(predictions[config.COLONNES_QUANTILES[config.QUANTILES[-1]]])
    couverture = float(np.mean((verite >= bas) & (verite <= haut)))
    attendue = config.QUANTILES[-1] - config.QUANTILES[0]

    return {
        "type": "quantiles",
        "detail": detail,
        "couverture": couverture,
        "couverture_attendue": attendue,
        "largeur_moyenne": float(np.mean(haut - bas)),
        "largeur_mediane": float(np.median(haut - bas)),
        # Un intervalle qui ne varie jamais n'apporte rien : sa largeur doit
        # bouger avec le régime de marché.
        "largeur_ecart_type": float(np.std(haut - bas)),
    }


# Constantes de la loi normale repliée, utilisées pour le plafond de R².
#   E|X| = σ·√(2/π) ≈ 0.7979 σ      Var|X| = σ²(1 − 2/π) ≈ 0.3634 σ²
ESPERANCE_ABSOLUE = float(np.sqrt(2 / np.pi))
VARIANCE_ABSOLUE = 1 - 2 / np.pi


def plafond_r2(prevu: np.ndarray) -> float:
    """
    R² maximal atteignable en prédisant |variation|, même avec un modèle parfait.

    Pourquoi ce plafond existe. On ne prédit pas une quantité déterministe mais
    l'ÉCHELLE d'un tirage aléatoire : même en connaissant exactement la
    volatilité σ de chaque bougie, |variation| reste un tirage unique autour de
    cette échelle, et cette part-là est irréductible. Un R² de 0.09 ne se
    compare donc pas à 1, mais à ce plafond.

    En modélisant la variation comme normale d'écart-type σ :

        R²_max = Var(prévu) / (Var(prévu) + 0.571 × E[prévu²])

    L'estimation utilise les prédictions du modèle comme approximation de σ.
    Elle est donc CONSERVATRICE dans le bon sens : un modèle qui sous-estime la
    variabilité de σ sous-estime aussi son propre plafond. Et comme les
    rendements crypto ont des queues plus épaisses qu'une gaussienne, le
    plafond réel est encore un peu plus bas — ce qui rend la comparaison
    d'autant plus flatteuse pour la référence, jamais pour le modèle.
    """
    prevu = np.asarray(prevu, dtype=float)
    variance_prevue = float(np.var(prevu))
    bruit = (VARIANCE_ABSOLUE / ESPERANCE_ABSOLUE ** 2) * float(np.mean(prevu ** 2))
    total = variance_prevue + bruit
    return float(variance_prevue / total) if total > 0 else 0.0


def _evaluer_volatilite(prevu, verite, reference_test, y_train,
                        reference_train) -> dict:
    """
    Qualité d'une prévision d'amplitude, face à deux références.

    La bonne référence n'est pas « la moyenne » mais **l'ATR dilaté en √h** :
    c'est déjà un excellent prédicteur de volatilité, et battre la moyenne sans
    battre l'ATR ne prouverait rien. Le facteur d'échelle de cette référence est
    calibré sur le TRAIN uniquement, jamais sur le test.
    """
    prevu = np.asarray(prevu, dtype=float)
    verite = np.asarray(verite, dtype=float)

    facteur = (float(np.mean(y_train) / np.mean(reference_train))
               if np.mean(reference_train) else 1.0)
    naif = np.asarray(reference_test, dtype=float) * facteur

    rangs_prevu = pd.Series(prevu).rank()
    rangs_verite = pd.Series(verite).rank()

    r2 = float(r2_score(verite, prevu))
    plafond = plafond_r2(prevu)

    return {
        "type": "volatilite",
        "r2": r2,
        "r2_naif": float(r2_score(verite, naif)),
        "r2_plafond": plafond,
        "part_du_plafond": float(r2 / plafond) if plafond > 0 else 0.0,
        "mae": float(mean_absolute_error(verite, prevu)),
        "mae_naif": float(mean_absolute_error(verite, naif)),
        "mae_moyenne": float(mean_absolute_error(verite, np.full_like(verite, y_train.mean()))),
        "correlation_rang": float(rangs_prevu.corr(rangs_verite)),
        "facteur_naif": facteur,
        "moyenne_reelle": float(verite.mean()),
        "moyenne_prevue": float(prevu.mean()),
    }


def _afficher_evaluation(metriques: dict, cible: str, symbole: str,
                         intervalle: str, horizon: int) -> None:
    """Rapport console de fin d'entraînement d'une régression."""
    print(f"\n📊 ===== ÉVALUATION SUR LE TEST — {symbole} {intervalle} "
          f"(h={horizon}, {cible}) =====")

    if metriques["type"] == "quantiles":
        print("   quantile │ perte pinball │ référence │ gain")
        for cle, detail in metriques["detail"].items():
            print(f"   {cle:>8} │ {detail['perte_pinball']:>13.4f} │ "
                  f"{detail['perte_reference']:>9.4f} │ {detail['gain']:>+6.1%}")
        couverture = metriques["couverture"]
        attendue = metriques["couverture_attendue"]
        verdict = ("✅ honnête" if abs(couverture - attendue) < 0.03 else
                   "⚠️ intervalle trop étroit" if couverture < attendue else
                   "⚠️ intervalle trop large")
        print(f"\n   Couverture de l'intervalle Q10–Q90 : {couverture:.1%} "
              f"(attendu {attendue:.0%})  {verdict}")
        print(f"   Largeur moyenne : {metriques['largeur_moyenne']:.2f} % "
              f"| médiane {metriques['largeur_mediane']:.2f} % "
              f"| écart-type {metriques['largeur_ecart_type']:.2f} %")
        print("   Un écart-type élevé est une BONNE nouvelle : l'intervalle "
              "s'élargit vraiment en marché agité.")
        return

    print(f"   R² du modèle : {metriques['r2']:>7.4f}   "
          f"|   R² de l'ATR naïf : {metriques['r2_naif']:.4f}")
    print(f"   Plafond théorique : {metriques['r2_plafond']:.4f} "
          f"— le modèle en atteint {metriques['part_du_plafond']:.0%}.")
    print("   (|variation| reste un tirage unique autour de la volatilité : même "
          "un modèle parfait\n    ne dépasserait pas ce plafond. C'est à lui "
          "qu'il faut comparer le R², pas à 1.)")
    print(f"   MAE du modèle : {metriques['mae']:.4f} %  "
          f"|  ATR naïf : {metriques['mae_naif']:.4f} %  "
          f"|  moyenne constante : {metriques['mae_moyenne']:.4f} %")
    print(f"   Corrélation de rang : {metriques['correlation_rang']:.4f}")

    if metriques["r2"] > metriques["r2_naif"]:
        print("   ✅ Le modèle bat l'ATR dilaté : il apprend quelque chose que "
              "l'ATR seul ne dit pas.")
    else:
        print("   ⚠️ Le modèle ne bat pas l'ATR dilaté. La volatilité reste "
              "prévisible, mais l'ATR suffit à la prédire — inutile de "
              "complexifier.")
    print("   À comparer au R² de la DIRECTION, qui tourne autour de 0 : "
          "l'amplitude est nettement\n   plus prévisible que le sens, "
          "et c'est tout l'intérêt de ce modèle.")


# ===========================================================================
# PRÉDICTION
# ===========================================================================
def charger_modele(symbole: str, intervalle: str, horizon: int, cible: str) -> dict:
    """Charge le paquet {modèles, features, meta} d'une régression."""
    chemin = stockage.chemin_regression(symbole, intervalle, horizon, cible)
    try:
        return joblib.load(chemin)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Aucune régression « {cible} » pour {symbole} ({intervalle}) à "
            f"l'horizon {horizon}. Lance d'abord l'entraînement.") from None


def predire(symbole: str, intervalle: str, horizon: int,
            cible: str = "volatilite") -> pd.DataFrame:
    """
    Applique une régression entraînée à tout l'historique.

    Retourne un tableau indexé par date, contenant selon la cible :
      * `Q10`, `Q50`, `Q90` et `Largeur_Prevue` pour l'amplitude ;
      * `Volatilite_Prevue` pour la volatilité.
    """
    paquet = charger_modele(symbole, intervalle, horizon, cible)
    df = modele.lire_analyse(symbole, intervalle)

    manquantes = [c for c in paquet["features"] if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier analysé : {manquantes}. "
                         f"Relance l'étape Analyse.")

    X = df[paquet["features"]].astype("float32")
    X = X[X.notna().all(axis=1)]
    if X.empty:
        raise ValueError("Aucune ligne exploitable après nettoyage.")

    colonnes = {cle: np.round(m.predict(X), 4) for cle, m in paquet["modeles"].items()}
    resultat = pd.DataFrame(colonnes, index=X.index)

    if cible == "amplitude":
        bas = config.COLONNES_QUANTILES[config.QUANTILES[0]]
        haut = config.COLONNES_QUANTILES[config.QUANTILES[-1]]
        # Les trois quantiles sont estimés séparément : rien ne garantit
        # mathématiquement qu'ils soient ordonnés. Un tri par ligne rétablit la
        # cohérence (« croisement de quantiles », défaut connu de la méthode).
        ordonnees = np.sort(resultat.to_numpy(), axis=1)
        resultat = pd.DataFrame(ordonnees, index=resultat.index,
                                columns=list(resultat.columns))
        resultat["Largeur_Prevue"] = resultat[haut] - resultat[bas]
    else:
        # Une amplitude ne peut pas être négative : le modèle peut le prédire,
        # la réalité non.
        resultat["Volatilite_Prevue"] = resultat["Volatilite"].clip(lower=0)
        resultat = resultat.drop(columns=["Volatilite"])

    return resultat


# ===========================================================================
# ESPÉRANCE DE GAIN — direction × amplitude
# ===========================================================================
def esperance(symbole: str, intervalle: str, horizon: int,
              seuil_confiance: float = config.SEUIL_DEFAUT,
              tache: str | None = None,
              cout: float = config.COUT_ALLER_RETOUR_PCT) -> pd.DataFrame:
    """
    Combine direction et amplitude en un score de décision unique.

        espérance brute = (2 × P(hausse) − 1) × amplitude attendue
        espérance nette = |espérance brute| − frais

    Une position longue rapporte +A avec la probabilité p, −A avec 1−p : son
    espérance vaut (2p−1)·A. Le signe donne le sens à prendre, la valeur absolue
    l'intérêt du trade, et les frais tranchent : en dessous de zéro, le signal
    est correct mais ne paie pas.

    C'est la réponse au vrai problème : un signal à 55 % sur un mouvement de
    3 % vaut mieux qu'un signal à 60 % sur 0.2 %, et jusqu'ici rien ne
    permettait de les départager.

    Requiert un modèle de direction ET une régression de volatilité au même
    horizon. Le fichier produit est directement backtestable — il contient les
    colonnes attendues par le simulateur, plus la volatilité prévue qui sert au
    dimensionnement proportionnel des positions.
    """
    horizon = int(np.clip(horizon, 1, config.HORIZON_MAX))
    print(f"\n🧮 Espérance de gain — {symbole} ({intervalle}) | horizon {horizon}")

    direction = modele.predire(symbole, intervalle, horizon, seuil_confiance,
                               tache=tache)
    volatilite = predire(symbole, intervalle, horizon, "volatilite")

    quantiles = None
    try:
        quantiles = predire(symbole, intervalle, horizon, "amplitude")
    except FileNotFoundError:
        print("ℹ️  Pas de régression quantile entraînée : l'intervalle Q10–Q90 "
              "ne sera pas ajouté (l'espérance, elle, ne l'utilise pas).")

    commun = direction.index.intersection(volatilite.index)
    if len(commun) == 0:
        raise ValueError("Aucune date commune entre direction et volatilité.")

    resultat = direction.loc[commun].copy()
    amplitude = volatilite.loc[commun, "Volatilite_Prevue"]
    resultat["Volatilite_Prevue"] = np.round(amplitude, 4)

    if quantiles is not None:
        for colonne in quantiles.columns:
            resultat[colonne] = np.round(quantiles.loc[commun, colonne], 4)

    brute = (2 * resultat["Proba_Hausse"] - 1) * amplitude
    resultat["Esperance_Pct"] = np.round(brute, 4)
    resultat["Esperance_Nette"] = np.round(brute.abs() - cout, 4)

    # Décision finale : il faut être assez sûr ET que le mouvement paie.
    resultat["Retenu"] = ((resultat["Confiance"] >= seuil_confiance)
                          & (resultat["Esperance_Nette"] > 0)
                          & (resultat["Sens_Predit"] != cibles.NEUTRE)).astype(int)

    chemin = stockage.chemin_prediction(symbole, intervalle, horizon,
                                        tache=stockage.SUFFIXE_ESPERANCE)
    stockage.ecrire_tableau(resultat, chemin)
    print(f"💾 Espérance sauvegardée : {chemin}")

    _rapport_esperance(resultat, seuil_confiance, cout)
    return resultat


def _rapport_esperance(resultat: pd.DataFrame, seuil: float, cout: float) -> None:
    """Bilan console : combien de signaux paient réellement, et sur quel bloc."""
    derniere = resultat.iloc[-1]
    print(f"\n🎯 Dernière bougie ({resultat.index[-1]:%Y-%m-%d %H:%M}) : "
          f"{derniere['Sens_Predit']} à {derniere['Confiance']:.1%} | "
          f"amplitude attendue {derniere['Volatilite_Prevue']:.2f} % | "
          f"espérance nette {derniere['Esperance_Nette']:+.3f} %")
    if {"Q10", "Q90"}.issubset(resultat.columns):
        print(f"   Intervalle 80 % : {derniere['Q10']:+.2f} % → {derniere['Q90']:+.2f} %")

    evaluables = resultat.dropna(subset=["Correct"])
    if evaluables.empty:
        return

    print(f"\n📊 Effet du filtre « l'espérance couvre les frais » "
          f"(seuil {seuil:.0%}, coût {cout:.2f} %) :")
    print("   bloc       │ bougies │ retenues │ justesse │ gain moyen par trade")
    for bloc in ("train", "validation", "test"):
        sous = evaluables[evaluables["Bloc"] == bloc]
        if sous.empty:
            continue
        retenues = sous[sous["Retenu"] == 1]
        if retenues.empty:
            print(f"   {bloc:<10} │ {len(sous):>7,} │ {0:>8} │ {'—':>8} │ —")
            continue
        print(f"   {bloc:<10} │ {len(sous):>7,} │ {len(retenues):>8,} │ "
              f"{retenues['Correct'].mean():>7.2%} │ "
              f"{_gain_reel(retenues, cout).mean():>+8.3f} %")
    print("   (seule la ligne « test » porte sur des données jamais vues)")

    _table_esperance(evaluables[evaluables["Bloc"] == "test"], cout)


def _gain_reel(retenues: pd.DataFrame, cout: float) -> np.ndarray:
    """Gain effectif d'un trade : variation observée, orientée par le sens, nette de frais."""
    sens = np.where(retenues["Sens_Predit"] == cibles.HAUSSE, 1.0, -1.0)
    return sens * retenues["Variation_Reelle"].to_numpy(dtype=float) - cout


def _table_esperance(test: pd.DataFrame, cout: float) -> None:
    """
    Ce que rapporte réellement chaque niveau d'espérance, sur le bloc test.

    L'équivalent de la table des seuils de confiance, mais en unité
    ÉCONOMIQUE. C'est le tableau à regarder quand aucun signal ne passe le
    filtre par défaut : il montre à quelle distance de la rentabilité on se
    trouve, et si l'espérance prédite est au moins bien ORDONNÉE — c'est-à-dire
    si les signaux annoncés comme les meilleurs sont vraiment les meilleurs.
    """
    test = test[test["Sens_Predit"] != cibles.NEUTRE]
    if test.empty:
        return

    esperance = test["Esperance_Nette"].to_numpy(dtype=float)
    print(f"\n   Rendement réel selon l'espérance nette annoncée (bloc test, "
          f"{len(test):,} bougies) :")
    print("   espérance nette ≥ │ signaux │ justesse │ gain moyen réel │ verdict")

    for niveau in (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20):
        retenues = test[esperance >= niveau]
        if len(retenues) < 30:
            continue
        gain = _gain_reel(retenues, cout)
        verdict = "rentable" if gain.mean() > 0 else "perdant"
        print(f"   {niveau:>17.2f} % │ {len(retenues):>7,} │ "
              f"{retenues['Correct'].mean():>7.2%} │ {gain.mean():>+14.3f} % │ {verdict}")

    meilleur = float(np.max(esperance))
    if meilleur <= 0:
        print(f"   ⚠️  Meilleure espérance nette atteinte : {meilleur:+.3f} % — "
              f"AUCUN signal ne couvre les frais à cet horizon.")
        print("   Ce n'est pas un défaut du modèle mais un constat économique : "
              "l'avantage\n   existe, il est simplement plus petit que le coût "
              "d'un aller-retour. Essaie un\n   horizon plus long (l'amplitude "
              "croît en √h, les frais eux ne bougent pas).")
