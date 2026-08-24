"""
============================================================================
 CRYPTO LAB — Tableau de bord graphique unifié
============================================================================
Interface graphique (CustomTkinter) qui orchestre les 3 étapes du projet :

    1. EXTRACTION   -> GatherData.py      (Binance / Yahoo / Top CoinGecko)
    2. ANALYSE      -> CryptoAnalysis.py  (feature engineering + Fear & Greed)
    3. PRÉDICTION   -> Predict.py         (XGBoost : train / predict / SHAP)

Bonus :
    4. VISUALISATION -> graphiques prix + signaux, exports SHAP
    5. BACKTEST      -> simulateur de trades (TP/SL, frais, equity curve)

Lancement :
    cd Crypto
    pip install -r requirements.txt
    python Dashboard.py
============================================================================
"""

import os
import sys
import json
import queue
import threading
import traceback
from datetime import datetime

# --- Forcer le dossier de travail sur celui du script -----------------------
# Les scripts d'origine utilisent des dossiers relatifs (data_crypto, etc.).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --- Dépendances graphiques -------------------------------------------------
try:
    import customtkinter as ctk
    import tkinter as tk
    from tkinter import ttk, messagebox
except ImportError:
    print("❌ CustomTkinter requis : pip install customtkinter")
    sys.exit(1)

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- Import des modules métier du projet (peut échouer si deps manquantes) ---
try:
    from GatherData import CryptoDataManager
    from CryptoAnalysis import (
        CryptoDataLoader,
        CryptoFeatureEngineer,
        get_fear_and_greed_history,
    )
    from Predict import (
        ModelConfig,
        MLDataManager,
        WalkForwardValidator,
        FeatureSelector,
        CryptoModelTrainer,
        CryptoPredictor,
        CryptoVisualizer,
        PortefeuilleDatasetBuilder,
        modeles_disponibles,
    )
    IMPORT_OK = True
    IMPORT_ERR = None
except Exception as exc:  # noqa: BLE001
    IMPORT_OK = False
    IMPORT_ERR = exc


# ===========================================================================
# PALETTE & CONSTANTES VISUELLES
# ===========================================================================
COULEURS = {
    "fond":          "#1a1a1a",
    "panneau":       "#242424",
    "carte":         "#2b2b2b",
    "accent":        "#1f6aa5",
    "accent_clair":  "#2e86c1",
    "vert":          "#2ecc71",
    "rouge":         "#e74c3c",
    "orange":        "#f39c12",
    "bleu":          "#00bcd4",
    "texte":         "#e6e6e6",
    "texte_doux":    "#9aa0a6",
    "axe":           "#1e1e1e",
}

INTERVALLES = ["1h", "2h", "4h", "6h", "12h", "1d"]

# Textes d'aide affichés au survol des pastilles « ⓘ »
AIDES = {
    # --- Sections (page Prédiction) ---
    "sec_config":  "Choisis la crypto à modéliser et règle la cible que le modèle doit apprendre à prédire.",
    "sec_actions": "Lance les différentes étapes du modèle, de la validation jusqu'à la prédiction finale.",
    "sec_signal":  "Recommandation du modèle sur la toute dernière bougie disponible : faut-il acheter maintenant ?",

    # --- Paramètres (page Prédiction) ---
    "crypto_analysee": "Fichier issu de l'étape Analyse (ex : BTC_1h).\n"
                       "C'est sur ces données enrichies d'indicateurs que le modèle apprend, puis est testé.",
    "horizon": "Combien de périodes dans le futur on cherche à prédire.\n"
               "Ex : intervalle 1h + horizon 24 → « le prix sera-t-il plus haut dans 24 h ? »",
    "seuil_atr": "Filtre anti-bruit basé sur la volatilité.\n"
                 "L'ATR mesure l'amplitude moyenne d'une bougie. Le modèle n'apprend que sur les "
                 "mouvements dont l'ampleur dépasse ATR × ce coefficient.\n"
                 "0.5 = on ignore les variations < 50 % de la volatilité normale.\n"
                 "Plus haut = on ne garde que les gros mouvements (moins de données, plus nets).",
    "taille_test": "Part des données les plus récentes mises de côté pour le test final,\n"
                   "jamais vues pendant l'entraînement. 0.2 = 20 %.\n"
                   "Mesure la vraie performance sur du futur inconnu (évite de tricher).",
    "folds": "Nombre de fenêtres temporelles pour la validation Walk-Forward.\n"
             "Le modèle est testé sur N périodes successives (hausse, baisse, range)\n"
             "pour vérifier qu'il reste stable et pas simplement chanceux sur une seule période.",
    "force_retrain": "Coché : réentraîne le modèle même s'il existe déjà.\n"
                     "Décoché : réutilise le modèle sauvegardé (instantané).",
    "feature_selection": "Coché : ne garde que les ~20 indicateurs les plus influents (via SHAP)\n"
                         "pour réduire le bruit. Décoché : utilise tous les indicateurs.\n"
                         "Nécessite d'avoir lancé « Features SHAP » au moins une fois.",
    "seuil_signal": "Confiance minimale pour déclencher un signal d'achat.\n"
                    "0.60 = on n'achète que si le modèle estime ≥ 60 % de chances de hausse.\n"
                    "Plus haut = moins de trades, mais plus fiables.",

    # --- Boutons (page Prédiction) ---
    "btn_walkforward": "Évalue la robustesse du modèle sur plusieurs périodes successives\n"
                       "AVANT l'entraînement définitif. Donne une AUC moyenne et son écart-type\n"
                       "(écart-type bas = modèle stable).",
    "btn_features": "Calcule et sauvegarde les indicateurs les plus importants (SHAP).\n"
                    "À lancer une fois avant d'entraîner avec « Sélection features » coché.",
    "btn_train": "Entraînement complet : recherche des meilleurs hyperparamètres (GridSearch),\n"
                 "entraînement XGBoost avec early stopping, puis calibration des probabilités.",
    "btn_predict": "Applique le modèle entraîné sur les données récentes\n"
                   "et génère les signaux de trading (réutilisés ensuite par le Backtest).",

    # --- Paramètres (page Backtest) ---
    "bt_capital": "Montant de départ simulé pour la stratégie.",
    "bt_seuil":   "Probabilité calibrée minimale pour entrer en position (acheter).",
    "bt_tp":      "Take Profit : on revend dès que le gain atteint ce pourcentage.",
    "bt_sl":      "Stop Loss : on revend dès que la perte atteint ce pourcentage.",
    "bt_duree":   "Durée maximale de détention (en périodes) avant de sortir d'office.",
    "bt_frais":   "Frais appliqués à chaque transaction (entrée + sortie), en pourcentage.",
    "bt_periode": "Restreint le backtest à une plage de dates (AAAA-MM-JJ).\n"
                  "Laisse vide pour utiliser toute la période disponible.\n"
                  "Pratique pour tester la stratégie sur un bull market, un bear market "
                  "ou une année précise.",
    "bt_slippage": "Glissement de prix subi à chaque transaction, en %.\n"
                   "Simule l'écart entre le prix affiché et le prix réellement obtenu "
                   "(carnet d'ordres, latence). S'ajoute aux frais.",

    # --- Page Prédiction : nouveaux contrôles ---
    "model_type": "Algorithme d'apprentissage utilisé.\n"
                  "Les forces / faiblesses du modèle choisi s'affichent juste en dessous.\n"
                  "Conseil : commence par 'LogisticRegression' comme baseline, puis compare.",
    "target_type": "Façon de définir ce qu'on prédit :\n"
                   "• Seuil ATR : ignore les petits mouvements (bruit)\n"
                   "• Directionnel : monte/baisse sans filtre\n"
                   "• Triple-barrier : take-profit / stop-loss / temps (le plus réaliste)",
    "embargo": "Nombre de lignes purgées entre le train et le test (anti-fuite).\n"
               "Vu que la cible regarde l'horizon dans le futur, les dernières lignes du "
               "train chevauchent le test. Vide = embargo automatique (= horizon).",
    "elagage": "Supprime les features trop corrélées entre elles (redondantes).\n"
               "Réduit le bruit et la dimensionnalité. Le seuil est la corrélation "
               "au-delà de laquelle une des deux features est supprimée.",
    "tb_tp": "Triple-barrier : barrière de take-profit, en multiples de l'ATR.\n"
             "Ex. 1.5 = on étiquette 'gagnant' si le prix monte de 1.5 × volatilité.",
    "tb_sl": "Triple-barrier : barrière de stop-loss, en multiples de l'ATR.\n"
             "Ex. 1.0 = on étiquette 'perdant' si le prix chute de 1 × volatilité.",

    # --- Refonte qualité modèle ---
    "class_weights": "Rééquilibre les classes Hausse/Baisse pendant l'entraînement.\n"
                     "Utile quand une classe domine (fréquent avec le filtre ATR) : évite\n"
                     "que le modèle prédise tout le temps la classe majoritaire.\n"
                     "Active scale_pos_weight (XGBoost/LightGBM) ou class_weight='balanced'.",
    "btn_topn": "Mode Portefeuille : télécharge le Top N CoinGecko en une fois\n"
                "(OHLCV + order-flow Binance). Ces données serviront à construire\n"
                "un dataset multi-cryptos pour un modèle qui généralise mieux.",
    "btn_portefeuille": "Concatène toutes les cryptos analysées en un seul dataset 'MULTI'\n"
                        "avec des features cross-sectionnelles (rang vs pairs, écart à BTC).\n"
                        "Entraîne ensuite le modèle sur 'MULTI' : plus de données, plus de\n"
                        "régimes → meilleure généralisation qu'un modèle mono-crypto.",
    "orderflow": "Données de microstructure Binance conservées à l'extraction :\n"
                 "pression acheteuse agressive (taker buy ratio), nombre de trades, volume\n"
                 "quote. Transformées en features stationnaires très informatives.",
    "use_multi": "Coché : applique le modèle 'MULTI' (entraîné sur plusieurs cryptos) à la\n"
                 "crypto sélectionnée, au lieu de son modèle dédié. Génère un fichier de\n"
                 "prédiction normal → backtestable comme d'habitude.\n"
                 "Nécessite d'avoir entraîné un modèle MULTI au préalable.",
}

# Forces / faiblesses de chaque modèle (affichées dynamiquement sous le menu)
MODELES_INFO = {
    "XGBoost": "✅ Très performant sur données tabulaires, gère le bruit, early stopping.\n"
               "⚠️ Beaucoup d'hyperparamètres ; peut surapprendre sans régularisation.",
    "LightGBM": "✅ Très rapide, excellent sur gros volumes, souvent ≥ XGBoost.\n"
                "⚠️ Sensible au surapprentissage sur petits jeux ; num_leaves délicat à régler.",
    "RandomForest": "✅ Robuste, peu de réglages, peu d'overfit, bonne baseline d'ensemble.\n"
                    "⚠️ Moins précis que le boosting ; pas d'early stopping ; gourmand en mémoire.",
    "LogisticRegression": "✅ Simple, rapide, interprétable — LA baseline de référence.\n"
                          "⚠️ Ne capte que des relations linéaires (même après normalisation).",
    "CatBoost": "✅ Excellent par défaut, résiste bien au surapprentissage, robuste.\n"
                "⚠️ Entraînement plus lent ; librairie lourde.",
}

# Forces / faiblesses de chaque type de cible
CIBLES_INFO = {
    "Seuil ATR": "✅ Filtre le bruit (mouvements < volatilité). Bon compromis signal/bruit.\n"
                 "⚠️ Exclut une partie des données (les périodes 'plates').",
    "Directionnel": "✅ Simple, garde toutes les données.\n"
                    "⚠️ Entraîne le modèle sur beaucoup de bruit (micro-mouvements).",
    "Triple-barrier": "✅ Take-profit / stop-loss / temps : le plus aligné au trading réel.\n"
                      "⚠️ Dépend du réglage TP/SL ; un peu plus lent à calculer.",
}


# Explications détaillées des statistiques d'un modèle (onglet Évaluation).
# Pour chaque stat : un libellé lisible + une note (rôle / interprétation / levier).
STATS_INFO = {
    # --- Performance ---
    "accuracy": {
        "label": "Accuracy (justesse globale)",
        "aide": "À quoi ça sert : pourcentage de prédictions correctes (hausse ou baisse).\n"
                "Interprétation : 0.50 = niveau du hasard sur un problème équilibré. En crypto, "
                "0.53–0.58 est déjà bon (marché très bruité).\n"
                "Levier : meilleures features, plus de données, filtre de bruit (seuil ATR plus haut). "
                "Une accuracy > 0.70 cache souvent une fuite de données.",
    },
    "auc": {
        "label": "AUC-ROC (pouvoir de séparation)",
        "aide": "À quoi ça sert : capacité à classer une hausse au-dessus d'une baisse, "
                "indépendamment du seuil de décision. C'est LA métrique reine d'un signal.\n"
                "Interprétation : 0.5 = hasard, 1.0 = parfait.\n"
                "Levier : features plus prédictives, tuning des hyperparamètres, plus d'historique. "
                "> 0.75 en crypto = vérifier le data leakage.",
    },
    "precision_hausse": {
        "label": "Precision — Hausse",
        "aide": "À quoi ça sert : parmi tous les signaux d'ACHAT émis, combien étaient justes.\n"
                "Interprétation : haute = peu de faux achats (crucial si tu trades pour de vrai).\n"
                "Levier : monter le seuil de signal (proba ≥ 0.6/0.7) → moins de trades mais plus fiables.",
    },
    "recall_hausse": {
        "label": "Recall — Hausse",
        "aide": "À quoi ça sert : parmi toutes les vraies hausses, combien ont été détectées.\n"
                "Interprétation : haut = peu d'opportunités ratées, mais souvent plus de faux signaux.\n"
                "Levier : baisser le seuil de signal. Recall et precision évoluent en sens inverse.",
    },
    "f1_hausse": {
        "label": "F1-score — Hausse",
        "aide": "À quoi ça sert : moyenne harmonique entre precision et recall sur les hausses.\n"
                "Interprétation : équilibre global de la détection des hausses (1.0 = parfait).\n"
                "Levier : améliorer à la fois qualité ET couverture des signaux (meilleures features).",
    },
    "precision_baisse": {
        "label": "Precision — Baisse",
        "aide": "À quoi ça sert : parmi tous les signaux de BAISSE, combien étaient justes.\n"
                "Interprétation : haute = peu de fausses alertes de baisse.\n"
                "Levier : symétrique de la precision Hausse (jeu sur le seuil).",
    },
    "recall_baisse": {
        "label": "Recall — Baisse",
        "aide": "À quoi ça sert : parmi toutes les vraies baisses, combien ont été détectées.\n"
                "Interprétation : haut = le modèle anticipe bien les chutes.\n"
                "Levier : seuil de décision et qualité des features.",
    },
    "f1_baisse": {
        "label": "F1-score — Baisse",
        "aide": "À quoi ça sert : équilibre precision/recall sur la classe Baisse.\n"
                "Interprétation : 1.0 = parfait.\n"
                "Levier : meilleures features et données moins bruitées.",
    },
    "balance_hausse": {
        "label": "Balance des classes (% hausse)",
        "aide": "À quoi ça sert : proportion de hausses vs baisses dans les données d'entraînement.\n"
                "Interprétation : ~50 % = équilibré (idéal). Très déséquilibré → le modèle peut tricher "
                "en prédisant toujours la classe majoritaire.\n"
                "Levier : l'horizon et le seuil ATR modifient cette balance.",
    },

    # --- Configuration / hyperparamètres ---
    "horizon": {
        "label": "Horizon de prédiction",
        "aide": "À quoi ça sert : nombre de périodes dans le futur que le modèle prédit.\n"
                "Interprétation : court (6) = réactif mais bruité ; long (72) = tendance plus lisible "
                "mais signaux plus rares.\n"
                "Levier : paramètre 'Horizon' avant l'entraînement.",
    },
    "seuil_atr": {
        "label": "Seuil ATR (filtre de bruit)",
        "aide": "À quoi ça sert : ignore à l'entraînement les mouvements < ATR × ce coefficient.\n"
                "Interprétation : plus haut = on n'apprend que sur les gros mouvements (plus net, "
                "moins de données).\n"
                "Levier : paramètre 'Seuil ATR ×'.",
    },
    "test_size": {
        "label": "Taille du jeu de test",
        "aide": "À quoi ça sert : part des données récentes réservées au test final.\n"
                "Interprétation : 0.2 = 20 %. Trop petit → métriques peu fiables ; trop grand → "
                "moins de données pour apprendre.",
    },
    "n_features": {
        "label": "Nombre de features",
        "aide": "À quoi ça sert : nombre d'indicateurs en entrée du modèle.\n"
                "Interprétation : trop de features bruitées peut nuire ; la sélection SHAP en garde ~20.\n"
                "Levier : activer la sélection de features (SHAP).",
    },
    "n_estimators_retenus": {
        "label": "Nombre d'arbres retenus",
        "aide": "À quoi ça sert : nombre d'arbres effectivement gardés (l'early stopping a arrêté là).\n"
                "Interprétation : peu = signal vite épuisé ; beaucoup = le modèle continue d'apprendre. "
                "Pas de 'bonne' valeur absolue.",
    },
    "max_depth": {
        "label": "max_depth (profondeur des arbres)",
        "aide": "À quoi ça sert : profondeur maximale de chaque arbre.\n"
                "Interprétation : plus profond = capte des interactions complexes mais risque d'overfit.\n"
                "Levier : valeurs testées par le GridSearch (4/6/8). Réduire si surapprentissage.",
    },
    "learning_rate": {
        "label": "learning_rate (taux d'apprentissage)",
        "aide": "À quoi ça sert : vitesse à laquelle chaque arbre corrige les précédents.\n"
                "Interprétation : bas (0.01) = lent mais robuste (souvent meilleur) ; haut = rapide "
                "mais instable. Un LR bas demande plus d'arbres.",
    },
    "subsample": {
        "label": "subsample (échantillon de lignes)",
        "aide": "À quoi ça sert : fraction des lignes utilisées par arbre.\n"
                "Interprétation : < 1 ajoute de l'aléatoire → réduit l'overfit. 0.7–0.9 est typique.",
    },
    "colsample_bytree": {
        "label": "colsample_bytree (échantillon de colonnes)",
        "aide": "À quoi ça sert : fraction des features utilisées par arbre.\n"
                "Interprétation : < 1 décorrèle les arbres et limite l'overfit.",
    },
    "gamma": {
        "label": "gamma (régularisation)",
        "aide": "À quoi ça sert : gain minimal requis pour créer une nouvelle division.\n"
                "Interprétation : plus haut = arbres plus prudents, modèle plus simple (anti-overfit).",
    },
    "min_child_weight": {
        "label": "min_child_weight",
        "aide": "À quoi ça sert : poids minimal d'observations dans une feuille.\n"
                "Interprétation : plus haut = feuilles plus 'peuplées', modèle plus régularisé.",
    },
    "embargo": {
        "label": "Embargo (anti-fuite)",
        "aide": "À quoi ça sert : lignes purgées entre train et test pour éviter que les "
                "labels (qui regardent le futur) ne fuitent du train vers le test.\n"
                "Interprétation : généralement = horizon. 0 = pas d'embargo (risque de fuite).",
    },
    "num_leaves": {
        "label": "num_leaves (LightGBM)",
        "aide": "À quoi ça sert : nombre max de feuilles par arbre (LightGBM).\n"
                "Interprétation : plus grand = modèle plus complexe (risque d'overfit). 31 par défaut.",
    },
    "depth": {
        "label": "depth (CatBoost)",
        "aide": "À quoi ça sert : profondeur des arbres CatBoost.\n"
                "Interprétation : plus profond = plus de capacité mais risque d'overfit.",
    },
    "l2_leaf_reg": {
        "label": "l2_leaf_reg (CatBoost)",
        "aide": "À quoi ça sert : régularisation L2 des feuilles (CatBoost).\n"
                "Interprétation : plus haut = modèle plus simple, moins d'overfit.",
    },
    "logisticregression__C": {
        "label": "C (Régression logistique)",
        "aide": "À quoi ça sert : inverse de la force de régularisation.\n"
                "Interprétation : petit C = forte régularisation (modèle simple) ; "
                "grand C = ajuste plus aux données (risque d'overfit).",
    },
    "n_estimators": {
        "label": "n_estimators (nb d'arbres)",
        "aide": "À quoi ça sert : nombre d'arbres de la forêt / du boosting.\n"
                "Interprétation : plus = plus de capacité (et de temps). Pour RandomForest, "
                "plus n'overfitte pas mais sature.",
    },
    "min_samples_leaf": {
        "label": "min_samples_leaf (RandomForest)",
        "aide": "À quoi ça sert : nb minimal d'échantillons dans une feuille.\n"
                "Interprétation : plus haut = feuilles plus peuplées, modèle plus régularisé.",
    },
    "max_features": {
        "label": "max_features (RandomForest)",
        "aide": "À quoi ça sert : nb de features tirées au hasard par split.\n"
                "Interprétation : <1 décorrèle les arbres et réduit l'overfit.",
    },
}

# Couleurs associées au niveau de qualité d'une stat
NIVEAUX_COULEUR = {
    "bon":     COULEURS["vert"],
    "moyen":   "#f1c40f",
    "faible":  COULEURS["orange"],
    "mauvais": COULEURS["rouge"],
}


def commenter_stat(cle, valeur):
    """
    Renvoie (commentaire, niveau) pour une statistique donnée selon sa valeur.
    niveau ∈ {bon, moyen, faible, mauvais} ; chaîne vide = pas de commentaire.
    """
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return "", "moyen"

    if cle == "accuracy":
        if v < 0.50:  return "Sous le hasard : le modèle se trompe plus qu'il ne réussit.", "mauvais"
        if v < 0.53:  return "À peine mieux que pile ou face.", "faible"
        if v < 0.58:  return "Correct pour de la crypto (marché très bruité).", "moyen"
        if v <= 0.70: return "Bonne justesse pour ce type de données.", "bon"
        return "Très élevé — vérifie l'absence de fuite de données (data leakage).", "moyen"

    if cle == "auc":
        if v < 0.50:  return "Sous le hasard : signal inversé ou surapprentissage.", "mauvais"
        if v < 0.55:  return "Très faible pouvoir de séparation.", "faible"
        if v < 0.60:  return "Signal faible mais réel, exploitable avec prudence.", "moyen"
        if v <= 0.75: return "Bon pouvoir prédictif pour de la crypto.", "bon"
        return "Excellent — mais méfie-toi d'un éventuel data leakage.", "moyen"

    if cle in ("precision_hausse", "precision_baisse"):
        if v < 0.50:  return "Plus de faux signaux que de bons.", "faible"
        if v < 0.55:  return "Fiabilité modérée des signaux.", "moyen"
        return "Signaux globalement fiables.", "bon"

    if cle in ("recall_hausse", "recall_baisse"):
        if v < 0.40:  return "Beaucoup d'opportunités manquées.", "faible"
        if v < 0.60:  return "Couverture moyenne des mouvements.", "moyen"
        return "Bonne couverture des mouvements réels.", "bon"

    if cle in ("f1_hausse", "f1_baisse"):
        if v < 0.45:  return "Équilibre precision/recall faible.", "faible"
        if v < 0.55:  return "Équilibre correct.", "moyen"
        return "Bon équilibre precision/recall.", "bon"

    if cle == "balance_hausse":
        ecart = abs(v - 0.5)
        if ecart < 0.05: return "Classes bien équilibrées (idéal).", "bon"
        if ecart < 0.12: return "Léger déséquilibre, acceptable.", "moyen"
        return "Fort déséquilibre : surveille la classe majoritaire.", "faible"

    if cle == "learning_rate":
        if v <= 0.02: return "Apprentissage lent et robuste (souvent le meilleur choix).", "bon"
        if v <= 0.10: return "Compromis vitesse/stabilité classique.", "moyen"
        return "Rapide mais potentiellement instable.", "faible"

    if cle == "max_depth":
        if v <= 4:    return "Arbres peu profonds : modèle simple, peu d'overfit.", "bon"
        if v <= 6:    return "Profondeur modérée, bon compromis.", "moyen"
        return "Arbres profonds : surveille le surapprentissage.", "faible"

    if cle in ("subsample", "colsample_bytree"):
        if v < 1.0:   return "Sous-échantillonnage actif : aide à réduire l'overfit.", "bon"
        return "Aucun sous-échantillonnage (= 1.0).", "moyen"

    return "", "moyen"


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ===========================================================================
# REDIRECTION DE LA CONSOLE  (stdout -> file d'attente -> widget Tk)
# ===========================================================================
class FluxConsole:
    """Capture les print() du code métier et les pousse dans une file thread-safe."""

    def __init__(self, file_attente):
        self.file = file_attente
        self.terminal = sys.__stdout__

    def write(self, message):
        self.file.put(message)
        try:
            self.terminal.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass


# ===========================================================================
# INFO-BULLE (tooltip affichée au survol)
# ===========================================================================
class InfoBulle:
    """
    Petite bulle d'aide affichée au survol d'un widget (après un court délai).
    Utilisée par les pastilles « ⓘ » pour expliquer les paramètres techniques.
    """

    def __init__(self, widget, texte, delai=300):
        self.widget = widget
        self.texte = texte
        self.delai = delai
        self.bulle = None
        self.apres_id = None
        widget.bind("<Enter>", self._programmer, add="+")
        widget.bind("<Leave>", self._cacher, add="+")

    def _programmer(self, _event=None):
        self._annuler()
        self.apres_id = self.widget.after(self.delai, self._afficher)

    def _annuler(self):
        if self.apres_id is not None:
            try:
                self.widget.after_cancel(self.apres_id)
            except Exception:
                pass
            self.apres_id = None

    def _afficher(self):
        if self.bulle is not None or not self.texte:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except Exception:
            return
        self.bulle = tk.Toplevel(self.widget)
        self.bulle.wm_overrideredirect(True)
        self.bulle.wm_geometry(f"+{x}+{y}")
        self.bulle.configure(bg=COULEURS["accent"])
        # Bordure fine via un padding de 1px sur le fond accent
        corps = tk.Frame(self.bulle, bg=COULEURS["carte"])
        corps.pack(padx=1, pady=1)
        tk.Label(
            corps, text=self.texte, justify="left", wraplength=360,
            bg=COULEURS["carte"], fg=COULEURS["texte"],
            font=("Segoe UI", 10), padx=12, pady=9,
        ).pack()

    def _cacher(self, _event=None):
        self._annuler()
        if self.bulle is not None:
            try:
                self.bulle.destroy()
            except Exception:
                pass
            self.bulle = None


# ===========================================================================
# SIMULATEUR DE TRADES (BACKTEST)
# ===========================================================================
class SimulateurTrades:
    """
    Backtest long-only sur le fichier de prédictions.

    Logique : on entre en position quand la proba calibrée >= seuil_entree.
    On sort sur Take Profit, Stop Loss ou durée maximale de détention.
    Les frais sont appliqués à l'entrée ET à la sortie.
    """

    def __init__(self, df, capital_initial=1000.0, seuil_entree=0.60,
                 take_profit=0.05, stop_loss=0.03, duree_max=24, frais=0.001,
                 slippage=0.0):
        self.df = df
        self.capital_initial = capital_initial
        self.seuil_entree = seuil_entree
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.duree_max = duree_max
        self.frais = frais
        self.slippage = slippage
        # Coût total par trade : frais + slippage, appliqués à l'entrée ET à la sortie
        self.cout_aller_retour = 2 * (frais + slippage)

    def simuler(self):
        prix = self.df["Price"].values.astype(float)
        proba = self.df["Proba_Calibree"].values.astype(float)
        dates = self.df.index

        capital = self.capital_initial          # equity réalisée (en cash, hors position)
        equity = []                              # courbe d'equity (mark-to-market)
        trades = []

        en_position = False
        prix_entree = 0.0
        idx_entree = 0
        date_entree = None

        for i in range(len(prix)):
            p = prix[i]

            if en_position:
                variation = (p - prix_entree) / prix_entree
                duree = i - idx_entree

                raison = None
                if variation >= self.take_profit:
                    raison = "Take Profit"
                elif variation <= -self.stop_loss:
                    raison = "Stop Loss"
                elif duree >= self.duree_max:
                    raison = "Durée max"

                if raison:
                    rendement_net = variation - self.cout_aller_retour
                    capital_avant = capital
                    capital = capital * (1 + rendement_net)
                    trades.append({
                        "Entrée": date_entree.strftime("%Y-%m-%d %H:%M"),
                        "Sortie": dates[i].strftime("%Y-%m-%d %H:%M"),
                        "Prix entrée": round(prix_entree, 2),
                        "Prix sortie": round(p, 2),
                        "Rendement %": round(rendement_net * 100, 2),
                        "P&L": round(capital - capital_avant, 2),
                        "Raison": raison,
                    })
                    en_position = False
                    equity.append(capital)
                else:
                    equity.append(capital * (1 + variation))
            else:
                equity.append(capital)
                if proba[i] >= self.seuil_entree:
                    en_position = True
                    prix_entree = p
                    idx_entree = i
                    date_entree = dates[i]

        # Clôture forcée si on est encore en position à la fin
        if en_position:
            variation = (prix[-1] - prix_entree) / prix_entree
            rendement_net = variation - self.cout_aller_retour
            capital_avant = capital
            capital = capital * (1 + rendement_net)
            trades.append({
                "Entrée": date_entree.strftime("%Y-%m-%d %H:%M"),
                "Sortie": dates[-1].strftime("%Y-%m-%d %H:%M"),
                "Prix entrée": round(prix_entree, 2),
                "Prix sortie": round(prix[-1], 2),
                "Rendement %": round(rendement_net * 100, 2),
                "P&L": round(capital - capital_avant, 2),
                "Raison": "Fin de période",
            })
            equity[-1] = capital

        equity = pd.Series(equity, index=dates)
        buy_hold = self.capital_initial * (prix / prix[0])
        buy_hold = pd.Series(buy_hold, index=dates)

        # --- Métriques ---
        df_trades = pd.DataFrame(trades)
        nb_trades = len(df_trades)
        gagnants = int((df_trades["P&L"] > 0).sum()) if nb_trades else 0
        win_rate = (gagnants / nb_trades * 100) if nb_trades else 0.0

        gains = df_trades.loc[df_trades["P&L"] > 0, "P&L"].sum() if nb_trades else 0.0
        pertes = abs(df_trades.loc[df_trades["P&L"] < 0, "P&L"].sum()) if nb_trades else 0.0
        profit_factor = (gains / pertes) if pertes > 0 else float("inf")

        pic = equity.cummax()
        drawdown = (equity - pic) / pic
        max_dd = drawdown.min() * 100 if len(drawdown) else 0.0

        rendement_total = (capital / self.capital_initial - 1) * 100

        # --- Métriques de risque (Sharpe / Sortino / Calmar) ---
        rendements = equity.pct_change().dropna()
        # Facteur d'annualisation déduit de la fréquence réelle de l'index
        periodes_an = 252.0
        if len(equity.index) > 2:
            delta = (equity.index[-1] - equity.index[0]).total_seconds() / (len(equity) - 1)
            if delta > 0:
                periodes_an = (365.25 * 24 * 3600) / delta
        racine = np.sqrt(periodes_an)

        if len(rendements) > 2 and rendements.std() > 0:
            sharpe = rendements.mean() / rendements.std() * racine
        else:
            sharpe = 0.0
        negatifs = rendements[rendements < 0]
        if len(negatifs) > 1 and negatifs.std() > 0:
            sortino = rendements.mean() / negatifs.std() * racine
        else:
            sortino = 0.0
        calmar = (rendement_total / abs(max_dd)) if max_dd != 0 else 0.0

        return {
            "trades": df_trades,
            "equity": equity,
            "buy_hold": buy_hold,
            "prix": pd.Series(prix, index=dates),
            "capital_final": capital,
            "rendement_total": rendement_total,
            "rendement_bh": (buy_hold.iloc[-1] / self.capital_initial - 1) * 100,
            "nb_trades": nb_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
        }


# ===========================================================================
# APPLICATION PRINCIPALE
# ===========================================================================
class CryptoDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Crypto Lab — Tableau de bord IA")
        self.geometry("1320x880")
        self.minsize(1120, 740)
        self.configure(fg_color=COULEURS["fond"])

        self.tache_en_cours = False
        self.file_console = queue.Queue()
        self.pages = {}
        self.boutons_nav = {}
        self.rafraichisseurs = {}
        self.canvas_actifs = {}     # pour nettoyer les graphes matplotlib

        self._configurer_styles_ttk()

        # --- Grille racine : sidebar | contenu  /  console  /  statut ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._construire_sidebar()
        self._construire_zone_contenu()
        self._construire_console()
        self._construire_barre_statut()

        # --- Construction des pages ---
        self._page_donnees()
        self._page_analyse()
        self._page_modele()
        self._page_evaluation()
        self._page_visualisation()
        self._page_backtest()

        self.afficher_page("Données")

        # --- Redirection console + boucle de lecture ---
        sys.stdout = FluxConsole(self.file_console)
        sys.stderr = FluxConsole(self.file_console)
        self._lire_console()

        self.protocol("WM_DELETE_WINDOW", self._fermer)

        if not IMPORT_OK:
            self.log("⚠️ Modules métier non chargés : " + str(IMPORT_ERR))
            self.log("   Vérifie : pip install -r requirements.txt")
        else:
            self.log("✅ Crypto Lab prêt. Modules métier chargés.")

    # ----------------------------------------------------------------------
    # STYLES TTK (tableaux dark mode)
    # ----------------------------------------------------------------------
    def _configurer_styles_ttk(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=COULEURS["carte"],
            foreground=COULEURS["texte"],
            fieldbackground=COULEURS["carte"],
            rowheight=26,
            borderwidth=0,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Treeview.Heading",
            background=COULEURS["accent"],
            foreground="white",
            relief="flat",
            font=("Segoe UI Semibold", 10),
        )
        style.map("Treeview", background=[("selected", COULEURS["accent_clair"])])
        style.map("Treeview.Heading", background=[("active", COULEURS["accent_clair"])])

    # ----------------------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------------------
    def _construire_sidebar(self):
        barre = ctk.CTkFrame(self, width=210, corner_radius=0, fg_color=COULEURS["panneau"])
        barre.grid(row=0, column=0, sticky="nsew")
        barre.grid_propagate(False)

        ctk.CTkLabel(
            barre, text="⚡ CRYPTO LAB",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(26, 4), padx=20)
        ctk.CTkLabel(
            barre, text="Pipeline IA de trading",
            font=ctk.CTkFont(size=12), text_color=COULEURS["texte_doux"],
        ).pack(pady=(0, 24))

        items = [
            ("Données",       "📥  1 · Extraction"),
            ("Analyse",       "🔬  2 · Analyse"),
            ("Modèle",        "🧠  3 · Prédiction"),
            ("Évaluation",    "📋  4 · Évaluation"),
            ("Visualisation", "📊  5 · Visualisation"),
            ("Backtest",      "💰  6 · Backtest"),
        ]
        for nom, libelle in items:
            btn = ctk.CTkButton(
                barre, text=libelle, anchor="w", height=42,
                corner_radius=8, fg_color="transparent",
                font=ctk.CTkFont(size=14),
                hover_color=COULEURS["accent"],
                command=lambda n=nom: self.afficher_page(n),
            )
            btn.pack(fill="x", padx=12, pady=4)
            self.boutons_nav[nom] = btn

        # Bas de sidebar : sélecteur d'apparence
        bas = ctk.CTkFrame(barre, fg_color="transparent")
        bas.pack(side="bottom", fill="x", padx=12, pady=18)
        ctk.CTkLabel(bas, text="Apparence", text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w")
        ctk.CTkOptionMenu(
            bas, values=["Dark", "Light", "System"],
            command=lambda v: ctk.set_appearance_mode(v.lower()),
        ).pack(fill="x", pady=(4, 0))

    # ----------------------------------------------------------------------
    # ZONE DE CONTENU (pages empilées)
    # ----------------------------------------------------------------------
    def _construire_zone_contenu(self):
        self.conteneur = ctk.CTkFrame(self, fg_color="transparent")
        self.conteneur.grid(row=0, column=1, sticky="nsew", padx=16, pady=(16, 8))
        self.conteneur.grid_rowconfigure(0, weight=1)
        self.conteneur.grid_columnconfigure(0, weight=1)

    def _nouvelle_page(self, nom, scrollable=True):
        if scrollable:
            page = ctk.CTkScrollableFrame(self.conteneur, fg_color="transparent")
        else:
            page = ctk.CTkFrame(self.conteneur, fg_color="transparent")
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_remove()  # masquée par défaut, affichée à la demande
        self.pages[nom] = page
        return page

    def afficher_page(self, nom):
        # On masque réellement toutes les pages et on n'affiche que l'active.
        # grid_remove()/grid() est fiable, contrairement à tkraise() quand on
        # mélange CTkFrame et CTkScrollableFrame dans une même cellule.
        for n, frame in self.pages.items():
            if n == nom:
                frame.grid()
            else:
                frame.grid_remove()
        for n, btn in self.boutons_nav.items():
            btn.configure(fg_color=COULEURS["accent"] if n == nom else "transparent")
        if nom in self.rafraichisseurs:
            try:
                self.rafraichisseurs[nom]()
            except Exception as e:  # noqa: BLE001
                self.log(f"⚠️ Rafraîchissement {nom}: {e}")

    # ----------------------------------------------------------------------
    # CONSOLE
    # ----------------------------------------------------------------------
    def _construire_console(self):
        cadre = ctk.CTkFrame(self, height=170, corner_radius=10, fg_color=COULEURS["panneau"])
        cadre.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=4)
        cadre.grid_propagate(False)
        cadre.grid_columnconfigure(0, weight=1)

        entete = ctk.CTkFrame(cadre, fg_color="transparent")
        entete.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        ctk.CTkLabel(entete, text="🖥️  Console",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(entete, text="Effacer", width=70, height=26,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._vider_console).pack(side="right")

        self.txt_console = ctk.CTkTextbox(
            cadre, font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=COULEURS["axe"], text_color="#cfd8dc", wrap="word",
        )
        self.txt_console.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        cadre.grid_rowconfigure(1, weight=1)
        self.txt_console.configure(state="disabled")

    def _vider_console(self):
        self.txt_console.configure(state="normal")
        self.txt_console.delete("1.0", "end")
        self.txt_console.configure(state="disabled")

    def log(self, message):
        self.file_console.put(str(message) + "\n")

    def _lire_console(self):
        try:
            while True:
                msg = self.file_console.get_nowait()
                self._inserer_console(msg.replace("\r", "\n"))
        except queue.Empty:
            pass
        self.after(80, self._lire_console)

    def _inserer_console(self, message):
        self.txt_console.configure(state="normal")
        self.txt_console.insert("end", message)
        self.txt_console.see("end")
        self.txt_console.configure(state="disabled")

    # ----------------------------------------------------------------------
    # BARRE DE STATUT
    # ----------------------------------------------------------------------
    def _construire_barre_statut(self):
        barre = ctk.CTkFrame(self, height=34, corner_radius=10, fg_color=COULEURS["panneau"])
        barre.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))
        barre.grid_propagate(False)
        barre.grid_columnconfigure(0, weight=1)

        self.lbl_statut = ctk.CTkLabel(barre, text="● Prêt", text_color=COULEURS["vert"],
                                       font=ctk.CTkFont(size=12))
        self.lbl_statut.grid(row=0, column=0, sticky="w", padx=14)

        self.progression = ctk.CTkProgressBar(barre, width=220, mode="indeterminate")
        self.progression.grid(row=0, column=1, sticky="e", padx=14)
        self.progression.set(0)

    def maj_statut(self, texte, en_cours=False):
        if en_cours:
            self.lbl_statut.configure(text=f"● {texte}", text_color=COULEURS["orange"])
            self.progression.start()
        else:
            self.lbl_statut.configure(text=f"● {texte}", text_color=COULEURS["vert"])
            self.progression.stop()
            self.progression.set(0)

    # ----------------------------------------------------------------------
    # GESTION DES TÂCHES EN ARRIÈRE-PLAN
    # ----------------------------------------------------------------------
    def executer(self, libelle, fonction, on_success=None):
        if not IMPORT_OK:
            self.log("❌ Modules métier indisponibles, action impossible.")
            return
        if self.tache_en_cours:
            self.log("⚠️ Une tâche est déjà en cours, patiente…")
            return

        self.tache_en_cours = True
        self.maj_statut(libelle + "…", en_cours=True)
        self.log(f"\n▶️  {libelle}")

        def worker():
            erreur, resultat = None, None
            try:
                resultat = fonction()
            except Exception as e:  # noqa: BLE001
                erreur = e
                traceback.print_exc()
            self.after(0, lambda: self._fin_tache(libelle, resultat, erreur, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _fin_tache(self, libelle, resultat, erreur, on_success):
        self.tache_en_cours = False
        if erreur:
            self.maj_statut("Erreur", en_cours=False)
            self.lbl_statut.configure(text_color=COULEURS["rouge"])
            self.log(f"❌ {libelle} : {erreur}")
        else:
            self.maj_statut("Prêt", en_cours=False)
            self.log(f"✅ {libelle} — terminé.")
            if on_success:
                try:
                    on_success(resultat)
                except Exception as e:  # noqa: BLE001
                    self.log(f"⚠️ Post-traitement : {e}")

    # ======================================================================
    # HELPERS UI
    # ======================================================================
    def _titre_page(self, parent, titre, sous_titre):
        cadre = ctk.CTkFrame(parent, fg_color="transparent")
        cadre.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(cadre, text=titre,
                     font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(cadre, text=sous_titre, text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=13)).pack(anchor="w")

    def _badge_info(self, parent, texte):
        """Petite pastille « ⓘ » qui affiche une info-bulle explicative au survol."""
        badge = ctk.CTkLabel(parent, text="ⓘ", width=16,
                             font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=COULEURS["accent_clair"], cursor="question_arrow")
        InfoBulle(badge, texte)
        return badge

    @staticmethod
    def _niveau_couleur(niveau):
        return NIVEAUX_COULEUR.get(niveau, COULEURS["texte"])

    @staticmethod
    def _fmt_stat(cle, valeur):
        """Formate une valeur de stat selon son type (ratio %, entier, brut)."""
        ratios = {"accuracy", "auc", "precision_hausse", "recall_hausse", "f1_hausse",
                  "precision_baisse", "recall_baisse", "f1_baisse", "balance_hausse"}
        try:
            if cle in ratios:
                return f"{float(valeur):.3f}  ({float(valeur) * 100:.1f} %)"
            if cle in {"horizon", "n_features", "n_estimators_retenus"}:
                return f"{int(valeur)}"
        except (TypeError, ValueError):
            pass
        return str(valeur)

    def _ligne_stat(self, parent, label, valeur, aide="", commentaire="", niveau="moyen"):
        """Affiche une stat : nom + pastille d'aide + valeur (colorée) + commentaire."""
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["carte"], corner_radius=10)
        carte.pack(fill="x", pady=4)

        haut = ctk.CTkFrame(carte, fg_color="transparent")
        haut.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(haut, text=label, font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        if aide:
            self._badge_info(haut, aide).pack(side="left", padx=(6, 0))
        couleur = self._niveau_couleur(niveau) if commentaire else COULEURS["texte"]
        ctk.CTkLabel(haut, text=str(valeur), font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=couleur).pack(side="right")

        if commentaire:
            bas = ctk.CTkFrame(carte, fg_color="transparent")
            bas.pack(fill="x", padx=14, pady=(0, 10))
            ctk.CTkLabel(bas, text="• " + commentaire, font=ctk.CTkFont(size=12),
                         text_color=self._niveau_couleur(niveau),
                         wraplength=820, justify="left").pack(anchor="w")
        return carte

    def _stat_avec_note(self, parent, cle, valeur):
        """Crée une ligne de stat à partir de STATS_INFO + commentaire contextuel."""
        info = STATS_INFO.get(cle, {"label": cle, "aide": ""})
        if valeur is None:
            self._ligne_stat(parent, info["label"], "—", info["aide"])
            return
        valeur_txt = self._fmt_stat(cle, valeur)
        commentaire, niveau = commenter_stat(cle, valeur)
        self._ligne_stat(parent, info["label"], valeur_txt, info["aide"], commentaire, niveau)

    def _section(self, parent, titre, aide=None):
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["panneau"], corner_radius=12)
        carte.pack(fill="x", pady=8)
        entete = ctk.CTkFrame(carte, fg_color="transparent")
        entete.pack(anchor="w", fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(entete, text=titre,
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        if aide:
            self._badge_info(entete, aide).pack(side="left", padx=(6, 0))
        corps = ctk.CTkFrame(carte, fg_color="transparent")
        corps.pack(fill="x", padx=18, pady=(0, 16))
        return corps

    def _champ(self, parent, libelle, valeur_defaut="", largeur=140, aide=None):
        col = ctk.CTkFrame(parent, fg_color="transparent")
        entete = ctk.CTkFrame(col, fg_color="transparent")
        entete.pack(anchor="w")
        ctk.CTkLabel(entete, text=libelle, font=ctk.CTkFont(size=12),
                     text_color=COULEURS["texte_doux"]).pack(side="left")
        if aide:
            self._badge_info(entete, aide).pack(side="left", padx=(4, 0))
        entree = ctk.CTkEntry(col, width=largeur)
        entree.insert(0, str(valeur_defaut))
        entree.pack(anchor="w", pady=(2, 0))
        return col, entree

    def _menu(self, parent, libelle, valeurs, largeur=160, aide=None):
        col = ctk.CTkFrame(parent, fg_color="transparent")
        entete = ctk.CTkFrame(col, fg_color="transparent")
        entete.pack(anchor="w")
        ctk.CTkLabel(entete, text=libelle, font=ctk.CTkFont(size=12),
                     text_color=COULEURS["texte_doux"]).pack(side="left")
        if aide:
            self._badge_info(entete, aide).pack(side="left", padx=(4, 0))
        var = ctk.StringVar(value=valeurs[0] if valeurs else "")
        menu = ctk.CTkOptionMenu(col, values=valeurs or [""], variable=var, width=largeur)
        menu.pack(anchor="w", pady=(2, 0))
        return col, var, menu

    def _carte_metrique(self, parent, titre):
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["carte"], corner_radius=10)
        ctk.CTkLabel(carte, text=titre, text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=14, pady=(10, 0))
        val = ctk.CTkLabel(carte, text="—", font=ctk.CTkFont(size=22, weight="bold"))
        val.pack(anchor="w", padx=14, pady=(0, 12))
        return carte, val

    def _creer_tableau(self, parent, hauteur=8):
        cadre = tk.Frame(parent, bg=COULEURS["carte"], highlightthickness=0)
        tree = ttk.Treeview(cadre, show="headings", height=hauteur)
        vsb = ttk.Scrollbar(cadre, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return cadre, tree

    def _remplir_tableau(self, tree, df, max_lignes=300):
        tree.delete(*tree.get_children())
        if df is None or df.empty:
            tree["columns"] = ["Info"]
            tree.heading("Info", text="Info")
            tree.column("Info", width=300, anchor="center")
            tree.insert("", "end", values=["Aucune donnée"])
            return
        cols = [str(c) for c in df.columns]
        tree["columns"] = cols
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=max(90, min(160, len(c) * 11)), anchor="center")
        for _, row in df.head(max_lignes).iterrows():
            tree.insert("", "end", values=[self._fmt(v) for v in row.values])

    @staticmethod
    def _fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
        return str(v)

    @staticmethod
    def _valider_date(texte):
        """Retourne la date nettoyée si au format AAAA-MM-JJ, sinon None."""
        texte = (texte or "").strip()
        try:
            datetime.strptime(texte, "%Y-%m-%d")
            return texte
        except ValueError:
            return None

    def _zone_graphe(self, parent, cle, hauteur=360):
        cadre = ctk.CTkFrame(parent, fg_color=COULEURS["panneau"], corner_radius=12,
                             height=hauteur)
        self.canvas_actifs[cle] = cadre
        return cadre

    def _afficher_figure(self, cle, fig, toolbar=True):
        cadre = self.canvas_actifs[cle]
        for w in cadre.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=cadre)
        canvas.draw()
        if toolbar:
            # Barre d'outils matplotlib : zoom (loupe), déplacement (croix),
            # retour vue initiale (maison), sauvegarde PNG…
            try:
                barre = NavigationToolbar2Tk(canvas, cadre, pack_toolbar=False)
                barre.update()
                self._styliser_toolbar(barre)
                barre.pack(side="top", fill="x", padx=8, pady=(4, 0))
            except TypeError:
                barre = NavigationToolbar2Tk(canvas, cadre)  # versions anciennes
                barre.update()
                self._styliser_toolbar(barre)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    @staticmethod
    def _styliser_toolbar(barre):
        """Accorde la barre d'outils matplotlib au thème sombre (best effort)."""
        try:
            barre.configure(background=COULEURS["panneau"])
            for enfant in barre.winfo_children():
                try:
                    enfant.configure(background=COULEURS["panneau"])
                except Exception:
                    pass
        except Exception:
            pass

    def _nouvelle_figure(self, figsize=(9, 4)):
        fig = Figure(figsize=figsize, dpi=100, facecolor=COULEURS["panneau"])
        return fig

    @staticmethod
    def _styliser(ax):
        ax.set_facecolor(COULEURS["axe"])
        ax.tick_params(colors="#bbbbbb", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("#bbbbbb")
        ax.yaxis.label.set_color("#bbbbbb")
        ax.grid(alpha=0.15, color="#888888")

    # ======================================================================
    # LISTE DES FICHIERS DISPONIBLES
    # ======================================================================
    @staticmethod
    def _lister(dossier, suffixe):
        if not os.path.isdir(dossier):
            return []
        res = []
        for f in os.listdir(dossier):
            if f.startswith("~$") or f.startswith("TOP_"):
                continue
            if f.endswith(suffixe + ".xlsx"):
                cle = f.replace(suffixe + ".xlsx", "")
                res.append(cle)
        return sorted(res)

    def lister_brutes(self):
        return self._lister("data_crypto", "")

    def lister_analyses(self):
        return self._lister("analysis_crypto", "_analyzed")

    def lister_predictions(self):
        return self._lister("prediction_crypto", "_prediction")

    @staticmethod
    def lister_modeles():
        """Liste les modèles entraînés (fichiers XGB_*.joblib) -> ['BTC_1h', ...]."""
        dossier = "models"
        if not os.path.isdir(dossier):
            return []
        res = []
        for f in os.listdir(dossier):
            if f.startswith("XGB_") and f.endswith(".joblib"):
                res.append(f[len("XGB_"):-len(".joblib")])
        return sorted(res)

    @staticmethod
    def _separer(cle):
        """'BTC_1h' -> ('BTC', '1h')"""
        if "_" not in cle:
            return cle, "1h"
        sym, intv = cle.rsplit("_", 1)
        return sym, intv

    @staticmethod
    def _maj_menu(menu, var, valeurs):
        valeurs = valeurs or ["(aucun)"]
        menu.configure(values=valeurs)
        if var.get() not in valeurs:
            var.set(valeurs[0])

    # ======================================================================
    # PAGE 1 — DONNÉES (EXTRACTION)
    # ======================================================================
    def _page_donnees(self):
        page = self._nouvelle_page("Données")
        self._titre_page(page, "📥  Extraction des données",
                         "Récupère les chandeliers OHLCV via Binance / Yahoo et le Top CoinGecko.")

        # --- Top CoinGecko ---
        corps = self._section(page, "Classement du marché (CoinGecko)")
        ligne = ctk.CTkFrame(corps, fg_color="transparent")
        ligne.pack(fill="x")
        _, self.don_topn = self._champ(ligne, "Top N cryptos", "5", largeur=100)
        _.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🌍 Récupérer le Top N", command=self._action_top_n,
                      ).pack(side="left", pady=(18, 0))
        btn_pf = ctk.CTkButton(ligne, text="📦 Télécharger l'historique du Top N",
                               fg_color=COULEURS["bleu"], hover_color="#0097a7",
                               command=self._action_portefeuille_extraction)
        btn_pf.pack(side="left", padx=10, pady=(18, 0))
        InfoBulle(btn_pf, AIDES["btn_topn"])

        cadre_top, self.tab_top = self._creer_tableau(corps, hauteur=6)
        cadre_top.pack(fill="x", pady=(12, 0))
        self.tab_top.bind("<Double-1>", self._selectionner_crypto_top)

        # --- Téléchargement OHLCV ---
        corps2 = self._section(page, "Téléchargement de l'historique OHLCV")
        ligne2 = ctk.CTkFrame(corps2, fg_color="transparent")
        ligne2.pack(fill="x")

        col_sym, self.don_symbole = self._champ(ligne2, "Symbole", "BTC", largeur=110)
        col_sym.pack(side="left", padx=(0, 12))

        col_src = ctk.CTkFrame(ligne2, fg_color="transparent")
        ctk.CTkLabel(col_src, text="Source", font=ctk.CTkFont(size=12),
                     text_color=COULEURS["texte_doux"]).pack(anchor="w")
        self.don_source = ctk.StringVar(value="Binance")
        ctk.CTkSegmentedButton(col_src, values=["Binance", "Yahoo"],
                               variable=self.don_source).pack(anchor="w", pady=(2, 0))
        col_src.pack(side="left", padx=(0, 12))

        col_int, self.don_interval, _ = self._menu(ligne2, "Intervalle", INTERVALLES, largeur=100)
        col_int.pack(side="left", padx=(0, 12))

        col_d, self.don_debut = self._champ(ligne2, "Début", "2022-01-01", largeur=120)
        col_d.pack(side="left", padx=(0, 12))
        col_f, self.don_fin = self._champ(ligne2, "Fin", "2026-01-01", largeur=120)
        col_f.pack(side="left", padx=(0, 12))

        boutons = ctk.CTkFrame(corps2, fg_color="transparent")
        boutons.pack(fill="x", pady=(14, 0))
        ctk.CTkButton(boutons, text="⬇️  Lancer l'extraction", height=40,
                      command=self._action_extraction).pack(side="left")
        ctk.CTkButton(boutons, text="🚀  Pipeline complet (extr.→analyse→train→prédict)",
                      height=40, fg_color=COULEURS["orange"], hover_color="#d68910",
                      command=self._action_pipeline).pack(side="left", padx=10)

        cadre_prev, self.tab_donnees = self._creer_tableau(corps2, hauteur=8)
        cadre_prev.pack(fill="x", pady=(14, 0))

    def _action_top_n(self):
        try:
            n = int(self.don_topn.get())
        except ValueError:
            self.log("❌ Top N invalide.")
            return

        def tache():
            mgr = CryptoDataManager()
            df = mgr.get_top_cryptos(n)
            mgr.save_top_to_excel(df, n)
            return df

        self.executer(f"Top {n} CoinGecko", tache,
                      on_success=lambda df: self._remplir_tableau(self.tab_top, df))

    def _action_portefeuille_extraction(self):
        """Télécharge l'historique complet (OHLCV + order-flow) des Top N cryptos."""
        try:
            n = int(self.don_topn.get())
        except ValueError:
            self.log("❌ Top N invalide.")
            return
        intervalle = self.don_interval.get()
        debut = self._valider_date(self.don_debut.get())
        fin = self._valider_date(self.don_fin.get())
        if debut is None or fin is None:
            self.log("❌ Date invalide. Format attendu : AAAA-MM-JJ (ex : 2024-01-01).")
            return

        def tache():
            mgr = CryptoDataManager()
            return mgr.telecharger_top_n(n, debut, fin, intervalle, source="Binance")

        self.executer(
            f"Portefeuille Top {n} ({intervalle})", tache,
            on_success=lambda syms: self.log(
                f"📦 {len(syms) if syms else 0} cryptos téléchargées. "
                f"Analyse-les (📦 Analyser TOUT le dossier) puis construis le dataset Portefeuille."))

    def _selectionner_crypto_top(self, _event):
        sel = self.tab_top.selection()
        if not sel:
            return
        valeurs = self.tab_top.item(sel[0], "values")
        cols = self.tab_top["columns"]
        if "symbol" in cols:
            idx = list(cols).index("symbol")
            self.don_symbole.delete(0, "end")
            self.don_symbole.insert(0, valeurs[idx])
            self.log(f"➡️  Symbole sélectionné : {valeurs[idx]}")

    def _action_extraction(self):
        symbole = self.don_symbole.get().strip().upper()
        source = self.don_source.get()
        intervalle = self.don_interval.get()
        debut = self._valider_date(self.don_debut.get())
        fin = self._valider_date(self.don_fin.get())

        if not symbole:
            self.log("❌ Renseigne un symbole.")
            return
        if debut is None or fin is None:
            self.log("❌ Date invalide. Format attendu : AAAA-MM-JJ (ex : 2024-01-01).")
            return

        def tache():
            mgr = CryptoDataManager()
            if source == "Binance":
                df = mgr.fetch_data_binance(symbole, debut, fin, intervalle)
            else:
                df = mgr.fetch_data_yahoo(symbole, debut, fin, intervalle)
            mgr.save_ohlcv_to_excel(df, symbole, intervalle, overwrite=True)
            return df

        def apres(df):
            if df is not None and not df.empty:
                apercu = df.tail(10).reset_index()
                self._remplir_tableau(self.tab_donnees, apercu)
                self.log(f"📈 {len(df)} lignes récupérées pour {symbole} ({intervalle}).")

        self.executer(f"Extraction {symbole} ({source})", tache, on_success=apres)

    def _action_pipeline(self):
        symbole = self.don_symbole.get().strip().upper()
        source = self.don_source.get()
        intervalle = self.don_interval.get()
        debut = self._valider_date(self.don_debut.get())
        fin = self._valider_date(self.don_fin.get())
        if not symbole:
            self.log("❌ Renseigne un symbole.")
            return
        if debut is None or fin is None:
            self.log("❌ Date invalide. Format attendu : AAAA-MM-JJ (ex : 2024-01-01).")
            return

        def tache():
            # 1. Extraction
            mgr = CryptoDataManager()
            if source == "Binance":
                df = mgr.fetch_data_binance(symbole, debut, fin, intervalle)
            else:
                df = mgr.fetch_data_yahoo(symbole, debut, fin, intervalle)
            mgr.save_ohlcv_to_excel(df, symbole, intervalle, overwrite=True)

            # 2. Analyse
            loader = CryptoDataLoader()
            fng = get_fear_and_greed_history()
            brut = loader.load_crypto_data(symbole, intervalle)
            analyse = CryptoFeatureEngineer(brut).process_all(fng_data=fng)
            loader.save_to_excel(analyse, symbole, intervalle)

            # 3. Entraînement
            CryptoModelTrainer(symbole, intervalle).train(
                force_retrain=True, use_feature_selection=False)

            # 4. Prédiction
            CryptoPredictor(symbole, intervalle).run_inference(threshold=0.60)
            return None

        self.executer(f"Pipeline complet {symbole} ({intervalle})", tache,
                      on_success=lambda _: self.log("🎉 Pipeline terminé — va voir Backtest !"))

    # ======================================================================
    # PAGE 2 — ANALYSE
    # ======================================================================
    def _page_analyse(self):
        page = self._nouvelle_page("Analyse")
        self._titre_page(page, "🔬  Analyse & Feature Engineering",
                         "Calcule les indicateurs techniques, le Fear & Greed et les cibles.")

        corps = self._section(page, "Paramètres d'analyse")
        ligne = ctk.CTkFrame(corps, fg_color="transparent")
        ligne.pack(fill="x")

        col, self.ana_fichier, self.ana_menu = self._menu(ligne, "Fichier brut", ["(aucun)"], 200)
        col.pack(side="left", padx=(0, 16))

        self.ana_fng = ctk.CTkCheckBox(ligne, text="Inclure Fear & Greed")
        self.ana_fng.select()
        self.ana_fng.pack(side="left", pady=(18, 0), padx=(0, 16))

        ctk.CTkButton(ligne, text="🔄", width=40, command=lambda: self.afficher_page("Analyse")
                      ).pack(side="left", pady=(18, 0))

        boutons = ctk.CTkFrame(corps, fg_color="transparent")
        boutons.pack(fill="x", pady=(14, 0))
        ctk.CTkButton(boutons, text="🔬 Analyser ce fichier", height=40,
                      command=self._action_analyse).pack(side="left")
        ctk.CTkButton(boutons, text="📦 Analyser TOUT le dossier", height=40,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._action_analyse_batch).pack(side="left", padx=10)
        btn_pf = ctk.CTkButton(boutons, text="🧺 Construire le dataset Portefeuille", height=40,
                               fg_color=COULEURS["bleu"], hover_color="#0097a7",
                               command=self._action_portefeuille_dataset)
        btn_pf.pack(side="left")
        InfoBulle(btn_pf, AIDES["btn_portefeuille"])

        # Résumé
        corps2 = self._section(page, "Résultat")
        self.cartes_analyse = ctk.CTkFrame(corps2, fg_color="transparent")
        self.cartes_analyse.pack(fill="x")
        self.ana_c_lignes = self._carte_metrique(self.cartes_analyse, "Lignes")
        self.ana_c_indic = self._carte_metrique(self.cartes_analyse, "Indicateurs")
        self.ana_c_lignes[0].pack(side="left", padx=(0, 10))
        self.ana_c_indic[0].pack(side="left", padx=(0, 10))

        cadre, self.tab_analyse = self._creer_tableau(corps2, hauteur=8)
        cadre.pack(fill="x", pady=(14, 0))

        self.rafraichisseurs["Analyse"] = lambda: self._maj_menu(
            self.ana_menu, self.ana_fichier, self.lister_brutes())

    def _action_analyse(self):
        cle = self.ana_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucun fichier brut sélectionné.")
            return
        symbole, intervalle = self._separer(cle)
        inclure_fng = self.ana_fng.get() == 1

        def tache():
            loader = CryptoDataLoader()
            fng = get_fear_and_greed_history() if inclure_fng else None
            brut = loader.load_crypto_data(symbole, intervalle)
            if brut is None or brut.empty:
                raise ValueError("Fichier brut illisible.")
            analyse = CryptoFeatureEngineer(brut).process_all(fng_data=fng)
            loader.save_to_excel(analyse, symbole, intervalle)
            return analyse

        def apres(df):
            self.ana_c_lignes[1].configure(text=f"{len(df):,}")
            self.ana_c_indic[1].configure(text=str(df.shape[1]))
            self._remplir_tableau(self.tab_analyse, df.tail(10).reset_index())

        self.executer(f"Analyse {symbole} ({intervalle})", tache, on_success=apres)

    def _action_analyse_batch(self):
        def tache():
            loader = CryptoDataLoader()
            fng = get_fear_and_greed_history() if self.ana_fng.get() == 1 else None
            for cle in self.lister_brutes():
                symbole, intervalle = self._separer(cle)
                brut = loader.load_crypto_data(symbole, intervalle)
                if brut is not None and not brut.empty:
                    analyse = CryptoFeatureEngineer(brut).process_all(fng_data=fng)
                    loader.save_to_excel(analyse, symbole, intervalle)
            return None

        self.executer("Analyse de tout le dossier", tache)

    def _action_portefeuille_dataset(self):
        """Concatène les cryptos analysées en un dataset 'MULTI' (Portefeuille)."""
        intervalle = None
        cle = self.ana_fichier.get()
        if cle not in ("(aucun)", ""):
            _, intervalle = self._separer(cle)
        if not intervalle:
            analyses = self.lister_analyses()
            if analyses:
                _, intervalle = self._separer(analyses[0])
        if not intervalle:
            self.log("❌ Analyse d'abord au moins 2 cryptos (même intervalle).")
            return

        def tache():
            return PortefeuilleDatasetBuilder(intervalle).construire()

        self.executer(
            f"Dataset Portefeuille ({intervalle})", tache,
            on_success=lambda pool: self.log(
                "🧺 Dataset 'MULTI' prêt → sélectionne-le dans l'onglet Modèle pour entraîner."
                if pool is not None else
                "⚠️ Pas assez de cryptos analysées (≥ 2 du même intervalle requises)."))

    # ======================================================================
    # PAGE 3 — MODÈLE (PRÉDICTION)
    # ======================================================================
    def _page_modele(self):
        page = self._nouvelle_page("Modèle")
        self._titre_page(page, "🧠  Modèle & Prédiction",
                         "Entraîne un XGBoost calibré et génère les signaux de trading.")

        # Sélection + config
        corps = self._section(page, "Configuration", aide=AIDES["sec_config"])
        ligne = ctk.CTkFrame(corps, fg_color="transparent")
        ligne.pack(fill="x")
        col, self.mod_fichier, self.mod_menu = self._menu(
            ligne, "Crypto analysée", ["(aucun)"], 200, aide=AIDES["crypto_analysee"])
        col.pack(side="left", padx=(0, 16))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Modèle")).pack(side="left", pady=(18, 0))

        ligne2 = ctk.CTkFrame(corps, fg_color="transparent")
        ligne2.pack(fill="x", pady=(12, 0))
        d_h = ModelConfig.TARGET_HORIZON if IMPORT_OK else 24
        d_m = ModelConfig.TARGET_THRESHOLD_MULTIPLIER if IMPORT_OK else 0.5
        d_t = ModelConfig.TEST_SIZE if IMPORT_OK else 0.2
        d_w = ModelConfig.WALK_FORWARD_SPLITS if IMPORT_OK else 6
        c1, self.mod_horizon = self._champ(ligne2, "Horizon (périodes)", d_h, 130, aide=AIDES["horizon"])
        c2, self.mod_mult = self._champ(ligne2, "Seuil ATR ×", d_m, 130, aide=AIDES["seuil_atr"])
        c3, self.mod_test = self._champ(ligne2, "Taille test", d_t, 130, aide=AIDES["taille_test"])
        c4, self.mod_wf = self._champ(ligne2, "Folds Walk-Fwd", d_w, 130, aide=AIDES["folds"])
        for c in (c1, c2, c3, c4):
            c.pack(side="left", padx=(0, 12))

        # --- Ligne 3 : choix du modèle, de la cible, embargo ---
        ligne3 = ctk.CTkFrame(corps, fg_color="transparent")
        ligne3.pack(fill="x", pady=(12, 0))
        modeles = modeles_disponibles() if IMPORT_OK else ["XGBoost"]
        cm, self.mod_type, self.mod_type_menu = self._menu(
            ligne3, "Modèle", modeles, 170, aide=AIDES["model_type"])
        self.mod_type_menu.configure(command=lambda _v: self._maj_info_modele())
        cm.pack(side="left", padx=(0, 12))
        cc, self.mod_cible, self.mod_cible_menu = self._menu(
            ligne3, "Type de cible", ["Seuil ATR", "Directionnel", "Triple-barrier"], 160,
            aide=AIDES["target_type"])
        self.mod_cible_menu.configure(command=lambda _v: self._maj_info_cible())
        cc.pack(side="left", padx=(0, 12))
        ce, self.mod_embargo = self._champ(ligne3, "Embargo (lignes)", "", 120, aide=AIDES["embargo"])
        ce.pack(side="left", padx=(0, 12))

        # Bandeaux dynamiques forces / faiblesses
        self.mod_info_modele = ctk.CTkLabel(
            corps, text="", font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], wraplength=900, anchor="w")
        self.mod_info_modele.pack(fill="x", padx=2, pady=(8, 0))
        self.mod_info_cible = ctk.CTkLabel(
            corps, text="", font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], wraplength=900, anchor="w")
        self.mod_info_cible.pack(fill="x", padx=2, pady=(2, 0))

        # --- Ligne 4 : triple-barrier + élagage corrélation ---
        ligne4 = ctk.CTkFrame(corps, fg_color="transparent")
        ligne4.pack(fill="x", pady=(12, 0))
        ctp, self.mod_tb_tp = self._champ(ligne4, "TB Take-Profit ×ATR", "1.5", 130, aide=AIDES["tb_tp"])
        csl, self.mod_tb_sl = self._champ(ligne4, "TB Stop-Loss ×ATR", "1.0", 130, aide=AIDES["tb_sl"])
        ctp.pack(side="left", padx=(0, 12))
        csl.pack(side="left", padx=(0, 12))
        self.mod_elagage = ctk.CTkCheckBox(ligne4, text="Élaguer features corrélées")
        self.mod_elagage.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(ligne4, AIDES["elagage"]).pack(side="left", padx=(0, 10), pady=(18, 0))
        cseuil, self.mod_corr = self._champ(ligne4, "Seuil corrélation", "0.95", 110, aide=AIDES["elagage"])
        cseuil.pack(side="left", padx=(0, 12))
        self.mod_classweights = ctk.CTkCheckBox(ligne4, text="⚖️ Équilibrer les classes")
        self.mod_classweights.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(ligne4, AIDES["class_weights"]).pack(side="left", padx=(0, 10), pady=(18, 0))

        self._maj_info_modele()
        self._maj_info_cible()

        # Actions
        corps2 = self._section(page, "Actions", aide=AIDES["sec_actions"])
        opts = ctk.CTkFrame(corps2, fg_color="transparent")
        opts.pack(fill="x")
        self.mod_force = ctk.CTkCheckBox(opts, text="Forcer le ré-entraînement")
        self.mod_force.select()
        self.mod_force.pack(side="left", padx=(0, 6))
        self._badge_info(opts, AIDES["force_retrain"]).pack(side="left", padx=(0, 16))
        self.mod_fs = ctk.CTkCheckBox(opts, text="Sélection features (SHAP)")
        self.mod_fs.pack(side="left", padx=(0, 6))
        self._badge_info(opts, AIDES["feature_selection"]).pack(side="left", padx=(0, 16))
        c5, self.mod_seuil = self._champ(opts, "Seuil signal", "0.60", 100, aide=AIDES["seuil_signal"])
        c5.pack(side="left", padx=(0, 16))
        self.mod_use_multi = ctk.CTkCheckBox(opts, text="🧺 Prédire avec le modèle Portefeuille (MULTI)")
        self.mod_use_multi.pack(side="left", padx=(0, 6))
        self._badge_info(opts, AIDES["use_multi"]).pack(side="left")

        boutons = ctk.CTkFrame(corps2, fg_color="transparent")
        boutons.pack(fill="x", pady=(14, 0))
        b1 = ctk.CTkButton(boutons, text="📏 Walk-Forward", command=self._action_walkforward,
                           fg_color=COULEURS["carte"], hover_color=COULEURS["accent"])
        b1.pack(side="left")
        b2 = ctk.CTkButton(boutons, text="🎯 Features SHAP", command=self._action_features,
                           fg_color=COULEURS["carte"], hover_color=COULEURS["accent"])
        b2.pack(side="left", padx=8)
        b3 = ctk.CTkButton(boutons, text="🔥 Entraîner", height=40, command=self._action_train)
        b3.pack(side="left", padx=8)
        b4 = ctk.CTkButton(boutons, text="🔮 Prédire", height=40, fg_color=COULEURS["vert"],
                           hover_color="#27ae60", command=self._action_predict)
        b4.pack(side="left", padx=8)
        InfoBulle(b1, AIDES["btn_walkforward"])
        InfoBulle(b2, AIDES["btn_features"])
        InfoBulle(b3, AIDES["btn_train"])
        InfoBulle(b4, AIDES["btn_predict"])

        # Signal courant + métriques
        corps3 = self._section(page, "Dernier signal", aide=AIDES["sec_signal"])
        cartes = ctk.CTkFrame(corps3, fg_color="transparent")
        cartes.pack(fill="x")
        self.mod_c_signal = self._carte_metrique(cartes, "Signal")
        self.mod_c_proba = self._carte_metrique(cartes, "Proba calibrée")
        self.mod_c_prix = self._carte_metrique(cartes, "Prix actuel")
        self.mod_c_lignes = self._carte_metrique(cartes, "Lignes prédites")
        for carte in (self.mod_c_signal, self.mod_c_proba, self.mod_c_prix, self.mod_c_lignes):
            carte[0].pack(side="left", padx=(0, 10), fill="x", expand=True)

        self.rafraichisseurs["Modèle"] = lambda: self._maj_menu(
            self.mod_menu, self.mod_fichier, self.lister_analyses())

    def _maj_info_modele(self):
        """Met à jour le bandeau forces/faiblesses du modèle sélectionné."""
        modele = self.mod_type.get()
        self.mod_info_modele.configure(text=f"🤖 {modele} —\n{MODELES_INFO.get(modele, '')}")

    def _maj_info_cible(self):
        """Met à jour le bandeau forces/faiblesses de la cible sélectionnée."""
        cible = self.mod_cible.get()
        self.mod_info_cible.configure(text=f"🎯 {cible} —\n{CIBLES_INFO.get(cible, '')}")

    def _config_modele(self):
        """Applique les paramètres de l'UI à ModelConfig."""
        try:
            ModelConfig.TARGET_HORIZON = int(self.mod_horizon.get())
            ModelConfig.TARGET_THRESHOLD_MULTIPLIER = float(self.mod_mult.get())
            ModelConfig.TEST_SIZE = float(self.mod_test.get())
            ModelConfig.WALK_FORWARD_SPLITS = int(self.mod_wf.get())
            ModelConfig.MODEL_TYPE = self.mod_type.get()
            ModelConfig.TARGET_TYPE = self.mod_cible.get()
            embargo_txt = self.mod_embargo.get().strip()
            ModelConfig.EMBARGO = int(embargo_txt) if embargo_txt else None
            ModelConfig.PRUNE_CORRELATION = (self.mod_elagage.get() == 1)
            ModelConfig.CORRELATION_THRESHOLD = float(self.mod_corr.get())
            ModelConfig.USE_CLASS_WEIGHTS = (self.mod_classweights.get() == 1)
            ModelConfig.TB_TP_MULT = float(self.mod_tb_tp.get())
            ModelConfig.TB_SL_MULT = float(self.mod_tb_sl.get())
        except ValueError:
            self.log("⚠️ Paramètres modèle invalides, valeurs par défaut conservées.")

    def _crypto_modele(self):
        cle = self.mod_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune crypto analysée sélectionnée.")
            return None, None
        return self._separer(cle)

    def _action_walkforward(self):
        symbole, intervalle = self._crypto_modele()
        if not symbole:
            return
        self._config_modele()
        self.executer(f"Walk-Forward {symbole}",
                      lambda: WalkForwardValidator(symbole, intervalle).run())

    def _action_features(self):
        symbole, intervalle = self._crypto_modele()
        if not symbole:
            return
        self._config_modele()
        self.executer(f"Sélection SHAP {symbole}",
                      lambda: FeatureSelector(symbole, intervalle, top_n=20).compute_and_save())

    def _action_train(self):
        symbole, intervalle = self._crypto_modele()
        if not symbole:
            return
        self._config_modele()
        force = self.mod_force.get() == 1
        fs = self.mod_fs.get() == 1
        self.executer(
            f"Entraînement {symbole} ({intervalle})",
            lambda: CryptoModelTrainer(symbole, intervalle).train(
                force_retrain=force, use_feature_selection=fs))

    def _action_predict(self):
        symbole, intervalle = self._crypto_modele()
        if not symbole:
            return
        self._config_modele()
        try:
            seuil = float(self.mod_seuil.get())
        except ValueError:
            seuil = 0.60

        # Modèle à appliquer : MULTI (Portefeuille) ou le modèle propre à la crypto
        modele = "MULTI" if self.mod_use_multi.get() == 1 else None
        if modele == "MULTI" and symbole == "MULTI":
            modele = None  # on prédit déjà sur le dataset poolé

        def tache():
            return CryptoPredictor(symbole, intervalle,
                                   modele_symbole=modele).run_inference(threshold=seuil)

        def apres(res):
            if res is None or res.empty:
                return
            derniere = res.iloc[-1]
            proba = float(derniere["Proba_Calibree"])
            prix = float(derniere["Price"])
            if proba >= seuil:
                txt, couleur = "ACHAT ▲", COULEURS["vert"]
            elif proba >= 0.5:
                txt, couleur = "Faible ↗", COULEURS["orange"]
            else:
                txt, couleur = "ATTENTE ▼", COULEURS["rouge"]
            self.mod_c_signal[1].configure(text=txt, text_color=couleur)
            self.mod_c_proba[1].configure(text=f"{proba:.1%}")
            self.mod_c_prix[1].configure(text=f"{prix:,.2f} $")
            self.mod_c_lignes[1].configure(text=f"{len(res):,}")

        self.executer(f"Prédiction {symbole} ({intervalle})", tache, on_success=apres)

    # ======================================================================
    # PAGE 4 — ÉVALUATION DU MODÈLE
    # ======================================================================
    def _page_evaluation(self):
        page = self._nouvelle_page("Évaluation")
        self._titre_page(page, "📋  Évaluation du modèle",
                         "Toutes les stats d'un modèle entraîné, expliquées et commentées.")

        corps = self._section(page, "Modèle à analyser")
        ligne = ctk.CTkFrame(corps, fg_color="transparent")
        ligne.pack(fill="x")
        col, self.eval_fichier, self.eval_menu = self._menu(
            ligne, "Modèle entraîné", ["(aucun)"], 200)
        col.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Évaluation")).pack(side="left", pady=(18, 0))
        ctk.CTkButton(ligne, text="📋 Analyser ce modèle", height=40,
                      command=self._action_evaluer).pack(side="left", padx=10, pady=(8, 0))

        # Conteneur dont le contenu est régénéré à chaque analyse
        self.eval_contenu = ctk.CTkFrame(page, fg_color="transparent")
        self.eval_contenu.pack(fill="both", expand=True, pady=(4, 0))
        ctk.CTkLabel(self.eval_contenu,
                     text="Sélectionne un modèle puis clique sur « Analyser ce modèle ».",
                     text_color=COULEURS["texte_doux"]).pack(pady=30)

        self.rafraichisseurs["Évaluation"] = lambda: self._maj_menu(
            self.eval_menu, self.eval_fichier, self.lister_modeles())

    def _action_evaluer(self):
        cle = self.eval_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucun modèle sélectionné.")
            return
        symbole, intervalle = self._separer(cle)
        self.executer(f"Analyse du modèle {cle}",
                      lambda: self._collecter_stats(symbole, intervalle),
                      on_success=self._afficher_evaluation)

    def _collecter_stats(self, symbole, intervalle):
        """Charge les métadonnées JSON si elles existent, sinon recalcule à la volée."""
        meta_path = os.path.join("models", f"META_{symbole}_{intervalle}.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["_source"] = "meta"
            return meta
        return self._recalculer_stats(symbole, intervalle)

    def _recalculer_stats(self, symbole, intervalle):
        """
        Recalcule les statistiques d'un modèle SANS métadonnées :
        on recharge le modèle + les données et on réévalue sur le jeu de test.
        Utilise la configuration ACTUELLE (horizon, seuil ATR, test size).
        """
        import joblib
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

        model_path = os.path.join("models", f"XGB_{symbole}_{intervalle}.joblib")
        cal_path = os.path.join("models", f"CAL_{symbole}_{intervalle}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        model = joblib.load(model_path)
        calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None

        manager = MLDataManager()
        df = manager.load_crypto_data(symbole, intervalle)
        if df is None:
            raise ValueError("Fichier analysé introuvable (lance d'abord l'étape Analyse).")

        X = manager.build_features(df)
        y = manager.build_target(df)
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

        # Le modèle connaît les features sur lesquelles il a été entraîné
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None:
            X = X[[c for c in expected if c in X.columns]]

        split_idx = int(len(X) * (1 - ModelConfig.TEST_SIZE))
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        raw = model.predict_proba(X_test)[:, 1]
        probs = calibrator.predict(raw) if calibrator is not None else raw
        y_pred = (probs >= 0.5).astype(int)

        rapport = classification_report(
            y_test, y_pred, target_names=["Baisse", "Hausse"],
            output_dict=True, zero_division=0,
        )
        auc = roc_auc_score(y_test, probs) if y_test.nunique() > 1 else 0.0
        params = model.get_params()
        balance = y.astype(int).value_counts(normalize=True)
        taux_hausse = float(np.mean(y_test))
        majorite = float(max(taux_hausse, 1 - taux_hausse))

        # Hyperparamètres pertinents selon le modèle (clés présentes seulement)
        cles_hp = ["max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "gamma", "min_child_weight", "num_leaves", "depth", "l2_leaf_reg",
                   "n_estimators", "min_samples_leaf", "max_features", "C"]
        hp = {k: params[k] for k in cles_hp if k in params and params[k] is not None}

        return {
            "_source": "recompute",
            "symbole": symbole,
            "intervalle": intervalle,
            "date_entrainement": "inconnue (modèle sans métadonnées)",
            "modele": type(model).__name__,
            "type_cible": getattr(ModelConfig, "TARGET_TYPE", "Seuil ATR"),
            "periode_debut": str(X.index.min()),
            "periode_fin": str(X.index.max()),
            "n_total": int(len(X)),
            "n_train": int(split_idx),
            "n_val": 0,
            "n_test": int(len(X_test)),
            "balance_hausse": float(balance.get(1, 0.0)),
            "balance_baisse": float(balance.get(0, 0.0)),
            "horizon": ModelConfig.TARGET_HORIZON,
            "seuil_atr": ModelConfig.TARGET_THRESHOLD_MULTIPLIER,
            "test_size": ModelConfig.TEST_SIZE,
            "feature_selection": bool(expected is not None and len(expected) < 30),
            "n_features": int(X.shape[1]),
            "features": [str(c) for c in X.columns],
            "hyperparametres": hp,
            "n_estimators_retenus": int(getattr(model, "best_iteration", 0) or 0),
            "metrics": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "auc": float(auc),
                "rapport": rapport,
                "baselines": {
                    "toujours_hausse_acc": taux_hausse,
                    "majorite_acc": majorite,
                },
            },
        }

    def _afficher_evaluation(self, meta):
        for w in self.eval_contenu.winfo_children():
            w.destroy()
        if not meta:
            ctk.CTkLabel(self.eval_contenu, text="Aucune donnée.",
                         text_color=COULEURS["texte_doux"]).pack(pady=30)
            return

        metrics = meta.get("metrics", {}) or {}
        rapport = metrics.get("rapport", {}) or {}
        hp = meta.get("hyperparametres", {}) or {}
        h = rapport.get("Hausse", {}) or {}
        b = rapport.get("Baisse", {}) or {}

        if meta.get("_source") == "recompute":
            avert = ctk.CTkFrame(self.eval_contenu, fg_color="#3a2e12", corner_radius=8)
            avert.pack(fill="x", pady=(0, 8))
            ctk.CTkLabel(
                avert,
                text="⚠️  Modèle sans métadonnées : métriques recalculées avec la configuration "
                     "ACTUELLE (horizon, seuil ATR, test size). Réentraîne ce modèle via l'onglet "
                     "Prédiction pour des statistiques exactes et figées.",
                text_color=COULEURS["orange"], wraplength=900, justify="left",
            ).pack(anchor="w", padx=12, pady=8)

        baselines = metrics.get("baselines", {}) or {}

        # --- Identité ---
        sec = self._section(self.eval_contenu, "🪪  Identité du modèle")
        self._ligne_stat(sec, "Crypto", meta.get("symbole", "?"))
        self._ligne_stat(sec, "Intervalle", meta.get("intervalle", "?"))
        self._ligne_stat(sec, "Type de modèle", meta.get("modele", "XGBoost"))
        self._ligne_stat(sec, "Type de cible", meta.get("type_cible") or "Seuil ATR (legacy)")
        self._ligne_stat(sec, "Date d'entraînement", meta.get("date_entrainement", "?"))
        self._ligne_stat(sec, "Période des données",
                         f"{str(meta.get('periode_debut', '?'))[:10]} → "
                         f"{str(meta.get('periode_fin', '?'))[:10]}")
        self._ligne_stat(sec, "Lignes (total / train / val / test)",
                         f"{meta.get('n_total', 0):,} / {meta.get('n_train', 0):,} / "
                         f"{meta.get('n_val', 0):,} / {meta.get('n_test', 0):,}")
        self._ligne_stat(sec, "Sélection de features (SHAP)",
                         "Oui" if meta.get("feature_selection") else "Non")

        # --- Configuration ---
        sec2 = self._section(self.eval_contenu, "⚙️  Configuration d'entraînement")
        self._stat_avec_note(sec2, "horizon", meta.get("horizon"))
        self._stat_avec_note(sec2, "seuil_atr", meta.get("seuil_atr"))
        self._stat_avec_note(sec2, "test_size", meta.get("test_size"))
        if "embargo" in meta:
            self._stat_avec_note(sec2, "embargo", meta.get("embargo"))
        if meta.get("elagage_correlation"):
            self._ligne_stat(sec2, "Élagage des features corrélées", "Oui")
        if meta.get("class_weights"):
            self._ligne_stat(sec2, "Rééquilibrage des classes", "Oui")
        self._stat_avec_note(sec2, "n_features", meta.get("n_features"))
        self._stat_avec_note(sec2, "n_estimators_retenus", meta.get("n_estimators_retenus"))
        # Hyperparamètres : dynamiques (varient selon le modèle)
        for cle_hp, val_hp in hp.items():
            self._stat_avec_note(sec2, cle_hp, val_hp)

        # --- Performance ---
        sec3 = self._section(self.eval_contenu, "🎯  Performance (sur le jeu de test)")
        self._stat_avec_note(sec3, "accuracy", metrics.get("accuracy"))
        self._stat_avec_note(sec3, "auc", metrics.get("auc"))
        self._stat_avec_note(sec3, "balance_hausse", meta.get("balance_hausse"))
        self._stat_avec_note(sec3, "precision_hausse", h.get("precision"))
        self._stat_avec_note(sec3, "recall_hausse", h.get("recall"))
        self._stat_avec_note(sec3, "f1_hausse", h.get("f1-score"))
        self._stat_avec_note(sec3, "precision_baisse", b.get("precision"))
        self._stat_avec_note(sec3, "recall_baisse", b.get("recall"))
        self._stat_avec_note(sec3, "f1_baisse", b.get("f1-score"))

        # --- Comparaison aux baselines (le modèle a-t-il un vrai avantage ?) ---
        if baselines:
            sec4 = self._section(self.eval_contenu, "⚖️  Comparaison aux baselines")
            acc = metrics.get("accuracy")
            maj = baselines.get("majorite_acc")
            self._ligne_stat(sec4, "Baseline « classe majoritaire »",
                             self._fmt_stat("accuracy", maj) if maj is not None else "—")
            self._ligne_stat(sec4, "Baseline « toujours acheter »",
                             self._fmt_stat("accuracy", baselines.get("toujours_hausse_acc"))
                             if baselines.get("toujours_hausse_acc") is not None else "—")
            if acc is not None and maj is not None:
                if acc > maj:
                    verdict, niveau = "Le modèle BAT la classe majoritaire ✔", "bon"
                else:
                    verdict, niveau = ("Le modèle NE BAT PAS la classe majoritaire → "
                                       "aucun avantage réel.", "mauvais")
                self._ligne_stat(sec4, "Verdict", f"{acc:.3f} vs {maj:.3f}",
                                 commentaire=verdict, niveau=niveau)

        self.log(f"📋 Stats affichées pour {meta.get('symbole')} ({meta.get('intervalle')}).")

    # ======================================================================
    # PAGE 5 — VISUALISATION
    # ======================================================================
    def _page_visualisation(self):
        page = self._nouvelle_page("Visualisation", scrollable=False)
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        entete = ctk.CTkFrame(page, fg_color="transparent")
        entete.grid(row=0, column=0, sticky="ew")
        self._titre_page(entete, "📊  Visualisation",
                         "Graphique prix + signaux et explications SHAP du modèle.")

        barre = ctk.CTkFrame(page, fg_color=COULEURS["panneau"], corner_radius=12)
        barre.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        inner = ctk.CTkFrame(barre, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=14)

        col, self.viz_fichier, self.viz_menu = self._menu(inner, "Crypto", ["(aucun)"], 180)
        col.pack(side="left", padx=(0, 16))
        ctk.CTkButton(inner, text="📈 Prix + Signaux", command=self._action_graphe_prix
                      ).pack(side="left", padx=4, pady=(18, 0))
        ctk.CTkButton(inner, text="✨ Générer SHAP", command=self._action_shap,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"]
                      ).pack(side="left", padx=4, pady=(18, 0))
        ctk.CTkButton(inner, text="🖼️ Afficher SHAP", command=self._action_afficher_shap,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"]
                      ).pack(side="left", padx=4, pady=(18, 0))
        ctk.CTkButton(inner, text="🔄", width=40,
                      command=lambda: self.afficher_page("Visualisation")
                      ).pack(side="left", padx=4, pady=(18, 0))

        zone = self._zone_graphe(page, "viz")
        zone.grid(row=2, column=0, sticky="nsew")

        self.rafraichisseurs["Visualisation"] = lambda: self._maj_menu(
            self.viz_menu, self.viz_fichier, self.lister_predictions() or self.lister_analyses())

    def _action_graphe_prix(self):
        cle = self.viz_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune crypto sélectionnée.")
            return
        symbole, intervalle = self._separer(cle)
        chemin = os.path.join("prediction_crypto", f"{symbole}_{intervalle}_prediction.xlsx")
        if not os.path.exists(chemin):
            self.log(f"❌ Pas de prédiction pour {symbole}. Lance d'abord 'Prédire'.")
            return

        def tache():
            return pd.read_excel(chemin, index_col=0, parse_dates=True)

        def apres(df):
            fig = self._nouvelle_figure((10, 5))
            ax = fig.add_subplot(111)
            self._styliser(ax)
            ax.plot(df.index, df["Price"], color=COULEURS["accent_clair"], lw=1.0,
                    label="Prix", zorder=1)

            # Vérité terrain : direction réelle du prix après l'horizon du modèle.
            # On la recalcule depuis le prix car la colonne Real_Target du fichier
            # peut être vide si l'horizon ne correspond pas aux colonnes Target_.
            horizon = ModelConfig.TARGET_HORIZON if IMPORT_OK else 24
            reel = None
            if "Real_Target" in df.columns and df["Real_Target"].notna().any():
                reel = df["Real_Target"]
            elif "Price" in df.columns:
                futur = df["Price"].shift(-horizon)
                reel = (futur > df["Price"]).astype(float)
                reel[futur.isna()] = np.nan

            if reel is not None and "Signal_Standard" in df.columns:
                # La prédiction de direction était-elle correcte ?
                masque = reel.notna()
                ok = (df["Signal_Standard"] == reel)
                corrects = df[masque & ok]
                faux = df[masque & ~ok]
                n = int(masque.sum())
                taux = len(corrects) / n * 100 if n else 0
                ax.scatter(corrects.index, corrects["Price"], color=COULEURS["vert"],
                           s=9, alpha=0.55, zorder=3,
                           label=f"Prédiction correcte ({len(corrects)})")
                ax.scatter(faux.index, faux["Price"], color=COULEURS["rouge"],
                           s=9, alpha=0.55, zorder=3,
                           label=f"Prédiction erronée ({len(faux)})")
                ax.set_title(f"{symbole} ({intervalle}) — Correctes vs erronées "
                             f"(justesse {taux:.1f} % · horizon {horizon})")
            else:
                ax.set_title(f"{symbole} ({intervalle}) — Prix")

            ax.legend(facecolor=COULEURS["carte"], labelcolor="white",
                      edgecolor="#555", loc="upper left")
            fig.tight_layout()
            self._afficher_figure("viz", fig, toolbar=True)

        self.executer(f"Graphe {symbole}", tache, on_success=apres)

    def _action_shap(self):
        cle = self.viz_fichier.get()
        if cle in ("(aucun)", ""):
            return
        symbole, intervalle = self._separer(cle)
        self.executer(f"SHAP {symbole}",
                      lambda: CryptoVisualizer(symbole, intervalle).plot_shap_summary())

    def _action_afficher_shap(self):
        cle = self.viz_fichier.get()
        if cle in ("(aucun)", ""):
            return
        symbole, intervalle = self._separer(cle)
        chemin = os.path.join("visualizations", f"SHAP_Bar_{symbole}_{intervalle}.png")
        if not os.path.exists(chemin):
            self.log("❌ Image SHAP introuvable. Clique d'abord sur 'Générer SHAP'.")
            return
        import matplotlib.image as mpimg
        fig = self._nouvelle_figure((10, 6))
        ax = fig.add_subplot(111)
        ax.imshow(mpimg.imread(chemin))
        ax.axis("off")
        fig.tight_layout()
        self._afficher_figure("viz", fig)
        self.log(f"🖼️ SHAP affiché : {chemin}")

    # ======================================================================
    # PAGE 6 — BACKTEST
    # ======================================================================
    def _page_backtest(self):
        page = self._nouvelle_page("Backtest", scrollable=True)

        self._titre_page(page, "💰  Simulateur de trades",
                         "Backtest long-only : Take Profit / Stop Loss / frais, vs Buy & Hold.")

        # Paramètres
        barre = ctk.CTkFrame(page, fg_color=COULEURS["panneau"], corner_radius=12)
        barre.pack(fill="x", pady=(0, 10))
        p = ctk.CTkFrame(barre, fg_color="transparent")
        p.pack(fill="x", padx=18, pady=14)

        col, self.bt_fichier, self.bt_menu = self._menu(p, "Crypto prédite", ["(aucun)"], 160)
        col.pack(side="left", padx=(0, 12))
        c1, self.bt_capital = self._champ(p, "Capital ($)", "1000", 100, aide=AIDES["bt_capital"])
        c2, self.bt_seuil = self._champ(p, "Seuil entrée", "0.60", 100, aide=AIDES["bt_seuil"])
        c3, self.bt_tp = self._champ(p, "Take Profit %", "5", 100, aide=AIDES["bt_tp"])
        c4, self.bt_sl = self._champ(p, "Stop Loss %", "3", 100, aide=AIDES["bt_sl"])
        c5, self.bt_duree = self._champ(p, "Durée max", "24", 100, aide=AIDES["bt_duree"])
        c6, self.bt_frais = self._champ(p, "Frais %", "0.1", 100, aide=AIDES["bt_frais"])
        for c in (c1, c2, c3, c4, c5, c6):
            c.pack(side="left", padx=(0, 10))

        # Deuxième ligne : période de backtest (optionnelle) + bouton Simuler
        p2 = ctk.CTkFrame(barre, fg_color="transparent")
        p2.pack(fill="x", padx=18, pady=(0, 14))
        cd, self.bt_debut = self._champ(p2, "Début (AAAA-MM-JJ)", "", 140, aide=AIDES["bt_periode"])
        cf, self.bt_fin = self._champ(p2, "Fin (AAAA-MM-JJ)", "", 140, aide=AIDES["bt_periode"])
        csg, self.bt_slippage = self._champ(p2, "Slippage %", "0.05", 100, aide=AIDES["bt_slippage"])
        cd.pack(side="left", padx=(0, 10))
        cf.pack(side="left", padx=(0, 10))
        csg.pack(side="left", padx=(0, 10))
        ctk.CTkButton(p2, text="📅 Période complète", width=160,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._reset_periode_backtest).pack(side="left", padx=(0, 10), pady=(16, 0))
        ctk.CTkButton(p2, text="▶️ Simuler", height=40, command=self._action_backtest
                      ).pack(side="left", padx=(6, 0), pady=(16, 0))

        # Cartes métriques
        cartes = ctk.CTkFrame(page, fg_color="transparent")
        cartes.pack(fill="x", pady=(0, 10))
        for i in range(4):
            cartes.grid_columnconfigure(i, weight=1)
        self.bt_cartes = {}
        noms = [("Rendement", 0, 0), ("Capital final", 1, 0), ("Buy & Hold", 2, 0),
                ("vs Marché", 3, 0), ("Nb trades", 0, 1), ("Win Rate", 1, 1),
                ("Profit Factor", 2, 1), ("Max Drawdown", 3, 1),
                ("Sharpe", 0, 2), ("Sortino", 1, 2), ("Calmar", 2, 2)]
        for nom, c, r in noms:
            carte, val = self._carte_metrique(cartes, nom)
            carte.grid(row=r, column=c, sticky="ew", padx=5, pady=5)
            self.bt_cartes[nom] = val

        # Graphe grand format : hauteur fixe → on scrolle la page pour le voir en entier
        zone = self._zone_graphe(page, "bt", hauteur=820)
        zone.pack(fill="x", pady=(0, 10))
        zone.pack_propagate(False)

        self.rafraichisseurs["Backtest"] = lambda: self._maj_menu(
            self.bt_menu, self.bt_fichier, self.lister_predictions())

    def _reset_periode_backtest(self):
        """Vide les champs de date → backtest sur toute la période disponible."""
        self.bt_debut.delete(0, "end")
        self.bt_fin.delete(0, "end")
        self.log("📅 Backtest réglé sur la période complète.")

    def _action_backtest(self):
        cle = self.bt_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune crypto prédite. Lance d'abord une prédiction.")
            return
        symbole, intervalle = self._separer(cle)
        chemin = os.path.join("prediction_crypto", f"{symbole}_{intervalle}_prediction.xlsx")
        if not os.path.exists(chemin):
            self.log("❌ Fichier de prédiction introuvable.")
            return

        # Période de backtest (champs vides = toute la période)
        debut_txt = self.bt_debut.get().strip()
        fin_txt = self.bt_fin.get().strip()
        for t in (debut_txt, fin_txt):
            if t and self._valider_date(t) is None:
                self.log("❌ Période invalide. Format attendu : AAAA-MM-JJ (ou laisse vide).")
                return

        try:
            params = dict(
                capital_initial=float(self.bt_capital.get()),
                seuil_entree=float(self.bt_seuil.get()),
                take_profit=float(self.bt_tp.get()) / 100,
                stop_loss=float(self.bt_sl.get()) / 100,
                duree_max=int(self.bt_duree.get()),
                frais=float(self.bt_frais.get()) / 100,
                slippage=float(self.bt_slippage.get()) / 100,
            )
        except ValueError:
            self.log("❌ Paramètres de backtest invalides.")
            return

        def tache():
            df = pd.read_excel(chemin, index_col=0, parse_dates=True)
            if "Proba_Calibree" not in df.columns or "Price" not in df.columns:
                raise ValueError("Colonnes Price / Proba_Calibree manquantes.")
            if debut_txt:
                df = df[df.index >= pd.Timestamp(debut_txt)]
            if fin_txt:
                df = df[df.index <= pd.Timestamp(fin_txt)]
            if len(df) < 5:
                raise ValueError("Période trop courte (moins de 5 bougies).")
            periode = f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}"
            print(f"📅 Période simulée : {periode} ({len(df)} bougies)")
            return SimulateurTrades(df, **params).simuler()

        self.executer(f"Backtest {symbole} ({intervalle})", tache,
                      on_success=self._afficher_backtest)

    def _afficher_backtest(self, res):
        c = self.bt_cartes
        rendement = res["rendement_total"]
        bh = res["rendement_bh"]
        couleur_r = COULEURS["vert"] if rendement >= 0 else COULEURS["rouge"]
        surperf = rendement - bh

        c["Rendement"].configure(text=f"{rendement:+.1f}%", text_color=couleur_r)
        c["Capital final"].configure(text=f"{res['capital_final']:,.0f} $")
        c["Buy & Hold"].configure(text=f"{bh:+.1f}%")
        c["Nb trades"].configure(text=str(res["nb_trades"]))
        c["Win Rate"].configure(text=f"{res['win_rate']:.0f}%")
        pf = res["profit_factor"]
        c["Profit Factor"].configure(text="∞" if pf == float("inf") else f"{pf:.2f}")
        c["Max Drawdown"].configure(text=f"{res['max_drawdown']:.1f}%",
                                    text_color=COULEURS["rouge"])
        c["vs Marché"].configure(
            text=f"{surperf:+.1f}%",
            text_color=COULEURS["vert"] if surperf >= 0 else COULEURS["rouge"])

        def _coul(v, bon=1.0, moyen=0.0):
            return (COULEURS["vert"] if v >= bon else
                    COULEURS["orange"] if v >= moyen else COULEURS["rouge"])
        sharpe, sortino, calmar = res["sharpe"], res["sortino"], res["calmar"]
        c["Sharpe"].configure(text=f"{sharpe:.2f}", text_color=_coul(sharpe, 1.0, 0.0))
        c["Sortino"].configure(text=f"{sortino:.2f}", text_color=_coul(sortino, 1.5, 0.0))
        c["Calmar"].configure(text=f"{calmar:.2f}", text_color=_coul(calmar, 1.0, 0.0))

        # Deux sous-graphes synchronisés : prix + trades (haut) / capital (bas)
        fig = self._nouvelle_figure((11, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        self._styliser(ax1)
        self._styliser(ax2)

        # --- Haut : prix + marqueurs de trades ---
        prix = res["prix"]
        ax1.plot(prix.index, prix.values, color=COULEURS["accent_clair"], lw=0.9,
                 label="Prix", zorder=1)
        trades = res["trades"]
        if trades is not None and not trades.empty:
            ent_x = pd.to_datetime(trades["Entrée"])
            gains = trades[trades["P&L"] > 0]
            pertes = trades[trades["P&L"] <= 0]
            ax1.scatter(ent_x, trades["Prix entrée"], color=COULEURS["bleu"],
                        s=26, marker="^", edgecolors="none", zorder=5,
                        label=f"Entrée ({len(trades)})")
            ax1.scatter(pd.to_datetime(gains["Sortie"]), gains["Prix sortie"],
                        color=COULEURS["vert"], s=26, marker="v", edgecolors="none",
                        zorder=5, label=f"Sortie gagnante ({len(gains)})")
            ax1.scatter(pd.to_datetime(pertes["Sortie"]), pertes["Prix sortie"],
                        color=COULEURS["rouge"], s=26, marker="v", edgecolors="none",
                        zorder=5, label=f"Sortie perdante ({len(pertes)})")
        ax1.set_title("Prix & trades (▲ entrée · ▼ sortie)")
        ax1.set_ylabel("Prix ($)")
        ax1.legend(facecolor=COULEURS["carte"], labelcolor="white",
                   edgecolor="#555", loc="upper left", fontsize=8)

        # --- Bas : capital vs Buy & Hold ---
        eq, bhc = res["equity"], res["buy_hold"]
        ax2.plot(eq.index, eq.values, color=COULEURS["vert"], lw=1.5, label="Stratégie IA")
        ax2.plot(bhc.index, bhc.values, color=COULEURS["texte_doux"], lw=1.0,
                 ls="--", label="Buy & Hold")
        ax2.fill_between(eq.index, eq.values, eq.iloc[0],
                         where=(eq.values >= eq.iloc[0]), alpha=0.08, color=COULEURS["vert"])
        ax2.set_title("Évolution du capital")
        ax2.set_ylabel("Capital ($)")
        ax2.legend(facecolor=COULEURS["carte"], labelcolor="white", edgecolor="#555")

        fig.tight_layout()
        self._afficher_figure("bt", fig)

        self.log(f"💰 {res['nb_trades']} trades | Rendement {rendement:+.1f}% "
                 f"| Win {res['win_rate']:.0f}% | DD {res['max_drawdown']:.1f}%")

    # ----------------------------------------------------------------------
    def _fermer(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


# ===========================================================================
if __name__ == "__main__":
    app = CryptoDashboard()
    app.mainloop()
