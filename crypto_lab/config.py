"""
Configuration centrale du projet Crypto Lab.

Ce module est la SOURCE UNIQUE DE VÉRITÉ pour :
  - les chemins des dossiers de travail,
  - la liste des 8 indicateurs (= les seules features vues par le modèle),
  - la liste des colonnes `variation_x` (= les cibles, valeurs FUTURES),
  - les réglages par défaut de l'entraînement.

Philosophie de la refonte : « trop de data tue la data ».
On ne garde que 8 indicateurs, tous stationnaires (donc exploitables par un
modèle de machine learning), au lieu des ~120 colonnes de l'ancienne version.
"""

from __future__ import annotations

import os

# ===========================================================================
# CHEMINS
# ===========================================================================
# Racine du projet = dossier parent du package crypto_lab
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOSSIER_DONNEES     = os.path.join(RACINE, "data_crypto")        # OHLCV brut
DOSSIER_ANALYSES    = os.path.join(RACINE, "analysis_crypto")    # OHLCV + indicateurs + variations
DOSSIER_MODELES     = os.path.join(RACINE, "models")             # modèles entraînés
DOSSIER_PREDICTIONS = os.path.join(RACINE, "prediction_crypto")  # sorties de prédiction
DOSSIER_GRAPHIQUES  = os.path.join(RACINE, "visualizations")     # exports PNG

DOSSIERS = (DOSSIER_DONNEES, DOSSIER_ANALYSES, DOSSIER_MODELES,
            DOSSIER_PREDICTIONS, DOSSIER_GRAPHIQUES)


def preparer_dossiers() -> None:
    """Crée les dossiers de travail s'ils n'existent pas encore."""
    for dossier in DOSSIERS:
        os.makedirs(dossier, exist_ok=True)


# ===========================================================================
# COLONNES DE PRIX (jamais des features)
# ===========================================================================
# Conservées dans le fichier analysé pour les graphiques et le backtest,
# mais JAMAIS données au modèle : ce sont des niveaux de prix non stationnaires.
# Un modèle qui les voit mémorise « BTC vaut 60 000 » et s'effondre dès que le
# prix sort de la plage apprise.
COLONNES_PRIX = ["Open", "High", "Low", "Close", "Volume"]


# ===========================================================================
# LES 8 INDICATEURS (= LES FEATURES DU MODÈLE)
# ===========================================================================
# Les 8 grands classiques de l'analyse technique, chacun ramené à une forme
# STATIONNAIRE (bornée ou exprimée en % du prix) donc utilisable en machine
# learning. Chaque indicateur couvre un axe d'information différent :
#
#   RSI_14         Momentum borné      -> sur-achat / sur-vente        (0-100)
#   Stoch_K        Position dans range -> où est le prix dans le canal (0-100)
#   MACD_Hist_Norm Accélération        -> la tendance s'essouffle ?    (% du prix)
#   Dist_SMA_50    Écart à la tendance -> au-dessus/en-dessous ?       (%)
#   BB_Position    Bandes de Bollinger -> extension vs volatilité      (0-1)
#   ATR_Pct        Volatilité          -> régime calme ou agité        (%)
#   ADX_14         Force de tendance   -> tendance nette ou range      (0-100)
#   OBV_Pct        Flux de volume      -> le volume accompagne-t-il ?  (%)
INDICATEURS = [
    "RSI_14",
    "Stoch_K",
    "MACD_Hist_Norm",
    "Dist_SMA_50",
    "BB_Position",
    "ATR_Pct",
    "ADX_14",
    "OBV_Pct",
]

# Libellés lisibles, réutilisés par l'interface et les graphiques.
LIBELLES_INDICATEURS = {
    "RSI_14":         "RSI (14) — momentum, sur-achat / sur-vente",
    "Stoch_K":        "Stochastique %K (14) — position dans le range",
    "MACD_Hist_Norm": "MACD histogramme / prix — accélération de tendance",
    "Dist_SMA_50":    "Écart à la SMA 50 (%) — position vs tendance",
    "BB_Position":    "Bollinger %B — position dans les bandes",
    "ATR_Pct":        "ATR (14) / prix — niveau de volatilité",
    "ADX_14":         "ADX (14) — force de la tendance",
    "OBV_Pct":        "OBV variation 20p (%) — flux de volume",
}


# ===========================================================================
# CONTEXTE MULTI-TIMEFRAME (les 8 mêmes indicateurs, un cran au-dessus)
# ===========================================================================
# Un modèle en 1h ne voit que l'agitation horaire. Les MÊMES 8 indicateurs
# recalculés sur l'intervalle supérieur lui disent dans quelle tendance de fond
# il se trouve — 8 colonnes de plus, pas 100 : le principe « trop de data tue
# la data » reste valable.
#
# Ces colonnes sont TOUJOURS décalées d'une bougie supérieure complète avant
# d'être rapprochées de l'index de base : à l'instant t, le modèle ne voit que
# des bougies 4h intégralement clôturées. Sans ce décalage, la bougie 4h en
# cours contiendrait déjà une partie du futur qu'on cherche à prédire.
SUFFIXE_MTF = "_MTF"
INDICATEURS_MTF = [nom + SUFFIXE_MTF for nom in INDICATEURS]

# Intervalle « du dessus » pour chaque intervalle de travail, et sa règle de
# rééchantillonnage pandas.
INTERVALLE_SUPERIEUR = {
    "1h":  ("4h",  "4h"),
    "2h":  ("8h",  "8h"),
    "4h":  ("1d",  "1D"),
    "6h":  ("1d",  "1D"),
    "12h": ("3d",  "3D"),
    "1d":  ("1w",  "7D"),
}


# ===========================================================================
# DONNÉES EXOGÈNES (Binance Futures — information NON dérivée du prix)
# ===========================================================================
# Tout le reste du fichier est calculé à partir de l'OHLCV : ce sont des
# transformations de la même information. Le funding rate et l'open interest
# viennent du marché des dérivés — c'est le seul vrai apport d'information
# nouvelle disponible gratuitement.
#
#   Funding_Rate     coût de portage payé par les longs aux shorts (ou l'inverse).
#                    Positif = les acheteurs paient, donc positionnement haussier
#                    tendu. Déjà stationnaire (de l'ordre de 0.0001).
#   Funding_Cumul    somme du funding sur 24 périodes — mesure la persistance
#                    du déséquilibre, plus informative que le point isolé.
#   OI_Variation     variation de l'open interest sur 24 périodes, en %.
#                    L'open interest brut est un niveau (donc inutilisable) ;
#                    sa variation dit si des positions s'ouvrent ou se soldent.
COLONNES_EXOGENES = ["Funding_Rate", "Funding_Cumul", "OI_Variation"]

LIBELLES_EXOGENES = {
    "Funding_Rate":  "Funding rate — coût de portage des positions longues",
    "Funding_Cumul": "Funding cumulé 24p — persistance du déséquilibre",
    "OI_Variation":  "Open interest, variation 24p (%) — flux de positions",
}

# Toutes les colonnes de contexte optionnelles, dans l'ordre.
COLONNES_CONTEXTE = INDICATEURS_MTF + COLONNES_EXOGENES

# Part minimale de lignes renseignées pour qu'une colonne de contexte soit
# retenue comme feature. L'open interest public de Binance ne remonte qu'à
# 30 jours : sur un historique de plusieurs années, la colonne est presque vide
# et serait plus nuisible qu'utile tant qu'elle n'a pas été accumulée.
COUVERTURE_MINIMALE = 0.60


def features_disponibles(colonnes, contexte: bool = True) -> list[str]:
    """
    Liste blanche des features, en fonction de ce que contient le fichier.

    Les 8 indicateurs de base sont obligatoires ; les colonnes de contexte
    (multi-timeframe, exogènes) ne sont ajoutées que si elles sont présentes.
    Aucun prix ni aucune colonne `variation_*` ne peut entrer ici.
    """
    presentes = set(colonnes)
    features = [nom for nom in INDICATEURS if nom in presentes]
    if contexte:
        features += [nom for nom in COLONNES_CONTEXTE if nom in presentes]
    return features


# ===========================================================================
# COLONNES CIBLES : variation_1 .. variation_24
# ===========================================================================
# variation_x = (Close[t+x] - Close[t]) / Close[t], en POURCENTAGE.
# Ce sont des valeurs FUTURES : elles servent à construire la cible et ne
# doivent jamais entrer dans les features (d'où le principe de liste blanche
# ci-dessous : seules les colonnes de INDICATEURS sont des features).
HORIZON_MAX = 24
HORIZONS = list(range(1, HORIZON_MAX + 1))
COLONNES_VARIATION = [f"variation_{h}" for h in HORIZONS]


def colonne_variation(horizon: int) -> str:
    """Nom de la colonne de variation future pour un horizon donné."""
    return f"variation_{int(horizon)}"


# Ordre des colonnes dans le fichier analysé (prix, puis features, puis cibles).
COLONNES_ANALYSE = COLONNES_PRIX + INDICATEURS + COLONNES_VARIATION


def colonnes_analyse(contexte: list[str] | None = None) -> list[str]:
    """Ordre des colonnes du fichier analysé, contexte optionnel intercalé."""
    return (COLONNES_PRIX + INDICATEURS + list(contexte or []) + COLONNES_VARIATION)


# ===========================================================================
# RÉGLAGES PAR DÉFAUT DE L'ENTRAÎNEMENT
# ===========================================================================
# Tout ce qui suit est géré AUTOMATIQUEMENT : l'interface n'expose que la
# crypto, le modèle, l'horizon et le seuil de confiance.

MODELE_DEFAUT   = "XGBoost"
HORIZON_DEFAUT  = 12      # périodes (12 h en 1h, 12 jours en 1d…)
# Confiance minimale pour retenir une prédiction. 0.55 plutôt que 0.60 : sur un
# marché très bruité, un modèle honnête dépasse rarement 0.60 de confiance, et
# un seuil trop haut ne laisse simplement passer aucun signal. Après chaque
# entraînement, l'interface propose le seuil réellement optimal pour le modèle.
SEUIL_DEFAUT    = 0.55

# Découpage chronologique (jamais aléatoire : ce sont des séries temporelles).
PART_TRAIN = 0.70   # apprentissage
PART_VAL   = 0.15   # calibration des probabilités + early stopping
PART_TEST  = 0.15   # évaluation finale, jamais vue pendant l'entraînement

# Écart (en lignes) purgé entre chaque bloc. La cible regarde `horizon`
# périodes dans le futur : sans cet embargo, la fin du train « connaît » déjà
# le début de la validation -> fuite de données et métriques trop belles.
# L'embargo réel vaut toujours l'horizon choisi.

# Rééquilibrage automatique : appliqué seulement si la classe majoritaire
# dépasse ce seuil (au-delà, le modèle a intérêt à répondre toujours pareil).
SEUIL_DESEQUILIBRE = 0.55

# Early stopping (modèles à boosting) : arrêt dès que la validation ne
# progresse plus pendant N itérations.
ARBRES_MAX          = 3000
PATIENCE_EARLY_STOP = 100

# Walk-forward : nombre de fenêtres successives de réentraînement. Le modèle
# est réentraîné sur tout le passé disponible, puis évalué sur la fenêtre
# suivante — jamais vue. 12 fenêtres offrent un bon compromis entre finesse de
# la courbe et temps de calcul (une douzaine d'entraînements complets).
FENETRES_WALKFORWARD = 12

# Intervalles proposés dans l'interface.
INTERVALLES = ["1h", "2h", "4h", "6h", "12h", "1d"]

# Seed global : deux entraînements identiques donnent le même résultat.
SEED = 42


# ===========================================================================
# COÛT DE TRANSACTION — la barre à franchir pour qu'un signal serve à quelque chose
# ===========================================================================
# Aller-retour complet : 2 × (frais + slippage). Avec 0.1 % de frais et 0.05 %
# de slippage chez Binance, un mouvement doit dépasser 0.30 % pour être
# exploitable. Cette constante sert à deux choses :
#   * la cible « direction nette », qui n'apprend QUE les mouvements rentables ;
#   * le calcul de l'espérance de gain, qui retranche le coût avant de trancher.
COUT_ALLER_RETOUR = 0.003          # 0.30 %
COUT_ALLER_RETOUR_PCT = COUT_ALLER_RETOUR * 100


# ===========================================================================
# AMPLITUDE : PRÉDIRE COMBIEN, PAS SEULEMENT DANS QUEL SENS
# ===========================================================================
# Quantiles estimés par la régression quantile. 10 / 50 / 90 donne directement
# un intervalle « 8 fois sur 10 le prix sera entre X et Y », c'est-à-dire la
# volatilité du mouvement — et pas seulement son sens probable.
QUANTILES = (0.10, 0.50, 0.90)
COLONNES_QUANTILES = {q: f"Q{int(q * 100):02d}" for q in QUANTILES}

# Cibles de régression disponibles.
#   amplitude   -> variation_h signée : donne l'intervalle de prix attendu.
#   volatilite  -> |variation_h| : de loin la plus prévisible des deux, grâce
#                  au clustering de volatilité (une bougie agitée en annonce
#                  d'autres). C'est là qu'il y a du vrai signal.
CIBLES_REGRESSION = {
    "amplitude":  "Amplitude signée — intervalle de variation attendu (quantiles)",
    "volatilite": "Volatilité — ampleur du mouvement, sans son sens",
}

# Bornes des 5 classes d'amplitude, exprimées en QUANTILES de la variation
# normalisée par l'ATR attendu.
#
# Pourquoi des quantiles plutôt que des multiples fixes de l'ATR (±0.5, ±1.5) :
# avec des bornes fixes, la classe « neutre » rassemblait 60 à 68 % des bougies.
# La calibration des probabilités — qui fait correctement son travail en
# ramenant chaque probabilité à la fréquence réelle de sa classe — rendait alors
# « neutre » systématiquement majoritaire, et le modèle répondait « neutre »
# 100 % du temps. Score obtenu : exactement celui de la réponse constante.
#
# Avec des quintiles, chaque classe pèse 20 % par construction. L'argmax
# redevient une vraie décision, et la confiance part bien de 20 % comme prévu.
# La normalisation par l'ATR est conservée : une classe dit toujours « ce
# mouvement est grand PAR RAPPORT au régime de volatilité du moment », et non
# « ce mouvement fait plus de 2 % ».
QUANTILES_CLASSES = (0.20, 0.40, 0.60, 0.80)
CLASSES_AMPLITUDE = ["Forte baisse", "Baisse", "Neutre", "Hausse", "Forte hausse"]

# Triple barrière : take-profit et stop-loss exprimés en multiples de l'ATR de
# la bougie d'entrée. Symétriques par défaut — une barrière asymétrique fabrique
# un déséquilibre de classes qui n'a rien à voir avec le pouvoir prédictif.
BARRIERE_TP_ATR = 1.0
BARRIERE_SL_ATR = 1.0


# ===========================================================================
# TAILLE DE POSITION (backtest)
# ===========================================================================
# En mode proportionnel, la fraction du capital engagée suit l'avantage estimé
# et recule quand la volatilité prévue monte. Bornée pour rester réaliste : ni
# position dérisoire, ni levier.
FRACTION_MINIMALE = 0.10
FRACTION_MAXIMALE = 1.00
