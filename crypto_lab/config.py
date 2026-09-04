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


# ===========================================================================
# RÉGLAGES PAR DÉFAUT DE L'ENTRAÎNEMENT
# ===========================================================================
# Tout ce qui suit est géré AUTOMATIQUEMENT : l'interface n'expose que la
# crypto, le modèle, l'horizon et le seuil de confiance.

MODELE_DEFAUT   = "XGBoost"
HORIZON_DEFAUT  = 12      # périodes (12 h en 1h, 12 jours en 1d…)
SEUIL_DEFAUT    = 0.60    # confiance minimale pour retenir une prédiction

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

# Intervalles proposés dans l'interface.
INTERVALLES = ["1h", "2h", "4h", "6h", "12h", "1d"]

# Seed global : deux entraînements identiques donnent le même résultat.
SEED = 42
