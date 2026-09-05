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

# Colonnes d'ORDER FLOW conservées telles quelles depuis les chandeliers Binance.
# Ce ne sont pas des features (ce sont des niveaux), mais la matière première de
# `COLONNES_FLUX` ci-dessous. Binance les renvoie dans chaque kline sans coût
# supplémentaire : les jeter revenait à se priver de la seule information du
# fichier qui ne soit pas dérivée du prix.
#
#   Trades     nombre de transactions dans la bougie
#   TakerBase  volume exécuté par des ACHETEURS agressifs (ordres au marché)
COLONNES_FLUX_BRUT = ["Trades", "TakerBase"]

# Colonnes du fichier brut : OHLCV + order flow quand la source le fournit.
COLONNES_BRUTES = COLONNES_PRIX + COLONNES_FLUX_BRUT


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
#   Basis            écart entre le perpétuel et le spot, en %. Positif = les
#                    acheteurs paient une prime pour rester exposés à effet de
#                    levier. Historique COMPLET dès le lancement du contrat,
#                    récupérable en une fois — contrairement à l'open interest.
#   Basis_Moyenne    moyenne du basis sur 24 périodes : la tension de fond.
COLONNES_EXOGENES = ["Funding_Rate", "Funding_Cumul", "OI_Variation",
                     "Basis", "Basis_Moyenne"]

LIBELLES_EXOGENES = {
    "Funding_Rate":  "Funding rate — coût de portage des positions longues",
    "Funding_Cumul": "Funding cumulé 24p — persistance du déséquilibre",
    "OI_Variation":  "Open interest, variation 24p (%) — flux de positions",
    "Basis":         "Basis perp/spot (%) — prime payée pour le levier",
    "Basis_Moyenne": "Basis moyen 24p — tension de fond du marché à terme",
}


# ===========================================================================
# ORDER FLOW (qui achète, qui vend — déjà dans les chandeliers téléchargés)
# ===========================================================================
# Le prix dit ce qui s'est passé ; l'order flow dit QUI l'a provoqué. Un même
# chandelier haussier n'a pas le même sens selon qu'il vient d'acheteurs
# agressifs (ordres au marché) ou de vendeurs passifs qui se retirent.
#
#   Flux_Desequilibre  2 × TakerBase / Volume − 1, donc entre −1 et +1.
#                      +0.2 = 60 % du volume est parti à l'achat agressif.
#   Flux_Cumul         moyenne du déséquilibre sur 12 périodes — la pression
#                      soutenue, bien plus informative que le point isolé.
#   Taille_Trade_Norm  log(taille moyenne d'un trade / sa médiane récente).
#                      Positif = de gros intervenants, négatif = poussière de
#                      détail. Le log-ratio rend la mesure stationnaire et
#                      comparable d'une crypto à l'autre.
COLONNES_FLUX = ["Flux_Desequilibre", "Flux_Cumul", "Taille_Trade_Norm"]

LIBELLES_FLUX = {
    "Flux_Desequilibre": "Déséquilibre acheteurs/vendeurs agressifs (−1 à +1)",
    "Flux_Cumul":        "Déséquilibre moyen 12p — pression soutenue",
    "Taille_Trade_Norm": "Taille de trade vs sa médiane — gros ou petits acteurs",
}

FENETRE_FLUX = 12        # périodes cumulées pour Flux_Cumul
FENETRE_TAILLE = 100     # médiane de référence de la taille de trade


# ===========================================================================
# TEMPS : SAISONNALITÉ ET CYCLE DE FUNDING
# ===========================================================================
# Le crypto se négocie 24 h/24, mais pas de façon uniforme : les volumes et la
# volatilité suivent les séances asiatique, européenne et américaine, et le
# funding tombe toutes les 8 h (00:00, 08:00, 16:00 UTC), ce qui déplace
# mécaniquement des positions juste avant.
#
# L'heure est CIRCULAIRE : 23 h est aussi proche de 0 h que de 22 h. La coder
# comme un nombre de 0 à 23 apprendrait au modèle une frontière absurde entre
# 23 h et minuit. Le couple sinus/cosinus supprime le problème.
COLONNES_TEMPS = ["Heure_Sin", "Heure_Cos", "Jour_Semaine", "Avant_Funding"]

LIBELLES_TEMPS = {
    "Heure_Sin":     "Heure du jour (sinus) — séance asiatique / US",
    "Heure_Cos":     "Heure du jour (cosinus) — séance asiatique / US",
    "Jour_Semaine":  "Jour de la semaine (0 = lundi) — week-end moins liquide",
    "Avant_Funding": "Part du cycle de 8 h écoulée avant le prochain funding",
}

# Le funding est versé toutes les 8 heures sur Binance.
CYCLE_FUNDING_HEURES = 8


# ===========================================================================
# RÉGIME DE MARCHÉ
# ===========================================================================
# Un modèle qui apprend en même temps le bull 2021 et le range 2023 apprend la
# moyenne des deux, qui ne s'est jamais produite. Plutôt que d'entraîner un
# modèle par régime — ce qui divise les données au moment où elles manquent le
# plus — on donne au modèle de quoi RECONNAÎTRE le régime, à charge pour les
# arbres de conditionner leurs règles dessus.
#
#   Regime_Volatilite  rang de percentile de l'ATR sur les 500 dernières
#                      périodes (0 = le plus calme jamais vu, 1 = le plus
#                      agité). L'ATR brut ne dit pas si 1.2 % est beaucoup :
#                      ça l'est pour du BTC, pas pour un altcoin récent.
#   Regime_Tendance    écart à la moyenne 200 périodes, exprimé en ATR.
#                      +3 = « trois bougies moyennes au-dessus de la tendance
#                      longue » — comparable entre cryptos et entre époques.
COLONNES_REGIME = ["Regime_Volatilite", "Regime_Tendance"]

LIBELLES_REGIME = {
    "Regime_Volatilite": "Volatilité actuelle vs ses 500 dernières périodes (0-1)",
    "Regime_Tendance":   "Écart à la moyenne 200p, mesuré en ATR",
}

FENETRE_REGIME = 500     # profondeur du rang de percentile
FENETRE_TENDANCE = 200   # moyenne longue de référence

# Toutes les colonnes de contexte optionnelles, dans l'ordre.
COLONNES_CONTEXTE = (INDICATEURS_MTF + COLONNES_FLUX + COLONNES_TEMPS
                     + COLONNES_REGIME + COLONNES_EXOGENES)

# Libellés de toutes les colonnes, base et contexte confondus.
LIBELLES_COLONNES = {**LIBELLES_INDICATEURS, **LIBELLES_EXOGENES,
                     **LIBELLES_FLUX, **LIBELLES_TEMPS, **LIBELLES_REGIME,
                     **{nom + SUFFIXE_MTF: libelle + " (intervalle supérieur)"
                        for nom, libelle in LIBELLES_INDICATEURS.items()}}

# Seules les colonnes multi-timeframe sont exigées ligne à ligne : ce sont des
# indicateurs, un trou y est un vrai trou. Toutes les autres colonnes de
# contexte ont une valeur NEUTRE bien définie (zéro : flux équilibré, funding
# nul, basis nul, taille de trade médiane), ce qui permet de conserver
# l'historique antérieur à leur existence au lieu de le sacrifier.
COLONNES_NEUTRALISABLES = [c for c in COLONNES_CONTEXTE if c not in INDICATEURS_MTF]

# Valeur qui signifie « aucune information » pour chaque colonne. Zéro convient
# presque partout (flux équilibré, funding nul, écart nul à la tendance) ; le
# régime de volatilité est un rang de percentile, dont le point neutre est 0.5.
VALEURS_NEUTRES = {"Regime_Volatilite": 0.5}


def valeur_neutre(colonne: str) -> float:
    """Valeur de remplissage d'une colonne de contexte absente ou trouée."""
    return VALEURS_NEUTRES.get(colonne, 0.0)

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


# ===========================================================================
# DURÉE D'UNE PÉRIODE
# ===========================================================================
# Nécessaire dès qu'on raisonne en DATES plutôt qu'en numéros de ligne : le
# découpage train/validation/test et l'embargo anti-fuite le font désormais,
# afin qu'un panier de plusieurs cryptos soit coupé aux mêmes dates pour toutes.
HEURES_INTERVALLE = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}


def heures(intervalle: str) -> float:
    """Durée d'une période, en heures. 1 par défaut si l'intervalle est inconnu."""
    return float(HEURES_INTERVALLE.get(intervalle, 1))


# ===========================================================================
# PANIER : PLUSIEURS CRYPTOS DANS UN SEUL MODÈLE
# ===========================================================================
# Chercher un avantage de quelques points sur 50 000 lignes d'une seule crypto,
# c'est chercher un signal faible dans très peu de données. Les 8 indicateurs
# sont conçus pour être comparables d'un actif à l'autre : rien n'empêche donc
# d'empiler 20 cryptos et d'entraîner UN modèle sur 1 000 000 de lignes.
#
# Le nom du panier tient lieu de symbole partout ailleurs (fichiers de modèle,
# métadonnées, prédictions). Il utilise le tiret comme séparateur, jamais le
# souligné, qui sert déjà à séparer symbole et intervalle dans les noms de
# fichiers.
PREFIXE_PANIER = "PANIER-"

# Fenêtre du rang de percentile appliqué à chaque feature, crypto par crypto,
# quand on mélange plusieurs actifs. 720 périodes = 30 jours en 1h.
FENETRE_RANG_PANIER = 720

# Features déjà comparables d'une crypto à l'autre : bornées par construction
# (RSI et stochastique entre 0 et 100, %B autour de [0,1], ADX entre 0 et 100,
# déséquilibre de flux entre −1 et +1, rang de volatilité entre 0 et 1, heure
# et jour de la semaine). Un RSI de 80 veut dire la même chose sur BTC et sur
# un altcoin : les normaliser leur ferait perdre ce sens absolu.
#
# Toutes les autres sont des grandeurs dont le NIVEAU dépend de l'actif : un
# ATR de 0.5 % est une tempête pour BTC et un jour ordinaire pour un altcoin.
# Empilées telles quelles dans un panier, elles apprendraient au modèle à
# reconnaître la crypto plutôt que la situation. On les remplace donc par leur
# rang de percentile glissant, calculé crypto par crypto et uniquement sur le
# passé — « où en est cet actif par rapport à SES 30 derniers jours ».
FEATURES_BORNEES = frozenset(
    ["RSI_14", "Stoch_K", "BB_Position", "ADX_14", "OBV_Pct"]
    + [nom + SUFFIXE_MTF for nom in ("RSI_14", "Stoch_K", "BB_Position",
                                     "ADX_14", "OBV_Pct")]
    + COLONNES_TEMPS
    + ["Regime_Volatilite", "Flux_Desequilibre", "Flux_Cumul"])


def colonnes_a_normaliser(colonnes) -> list[str]:
    """Features dont le niveau dépend de l'actif, donc à convertir en rang."""
    return [str(nom) for nom in colonnes if str(nom) not in FEATURES_BORNEES]

# Au-delà, les noms de fichiers deviennent illisibles et le modèle mélange des
# actifs qui n'ont plus grand-chose en commun.
CRYPTOS_MAX_PANIER = 25


def nom_panier(symboles) -> str:
    """['BTC', 'ETH'] -> 'PANIER-BTC-ETH' (ordre alphabétique, sans doublon)."""
    propres = sorted({str(s).upper().replace("_", "").replace("-", "")
                      for s in symboles if str(s).strip()})
    return PREFIXE_PANIER + "-".join(propres)


def est_panier(symbole: str) -> bool:
    """Vrai si ce « symbole » désigne en réalité un panier de cryptos."""
    return str(symbole).startswith(PREFIXE_PANIER)


def symboles_panier(nom: str) -> list[str]:
    """'PANIER-BTC-ETH' -> ['BTC', 'ETH']. Liste vide si ce n'est pas un panier."""
    if not est_panier(nom):
        return []
    return [s for s in nom[len(PREFIXE_PANIER):].split("-") if s]


# ===========================================================================
# VALIDATION CROISÉE PURGÉE
# ===========================================================================
# Choisir entre trois configurations sur un seul bloc de validation revient à
# trancher sur du bruit : les AUC obtenues diffèrent de quelques millièmes
# quand la marge d'erreur est de l'ordre du centième. En découpant la période
# d'apprentissage en blocs chronologiques successifs, on obtient N mesures au
# lieu d'une, donc un écart-type réel — et la règle « à un écart-type » cesse
# d'être une approximation.
#
# « Purgée » : autour de chaque bloc d'évaluation, les lignes dont la cible
# empiète sur ce bloc sont retirées de l'apprentissage. Sans cela, la fin d'un
# bloc d'entraînement connaît déjà le début du bloc évalué.
BLOCS_CV = 4

# Un bloc de validation croisée doit rester assez gros pour que son AUC
# signifie quelque chose.
LIGNES_MIN_BLOC_CV = 2000

# ===========================================================================
# PROFONDEUR DE LA RECHERCHE D'HYPERPARAMÈTRES
# ===========================================================================
# Trois configurations écrites à la main suffisent quand le signal est franc.
# Sur des données aussi bruitées que le marché, elles ne couvrent qu'un coin
# minuscule de l'espace des réglages : rien ne dit que la bonne profondeur
# d'arbre est 3, 4 ou 6 plutôt que 5, ni que le bon taux d'apprentissage est
# 0.02 plutôt que 0.012.
#
# Les modes ci-dessous ajoutent une recherche ALÉATOIRE autour de ces trois
# points de départ, évaluée avec la même validation croisée purgée et la même
# règle du 1 écart-type. Deux garde-fous :
#
#   * les trois configurations d'origine font toujours partie du tirage, donc
#     une recherche longue ne peut jamais rendre un résultat PIRE qu'une
#     recherche courte ;
#   * les candidates sont classées de la plus prudente à la plus souple, si
#     bien qu'à égalité statistique c'est toujours la plus simple qui gagne.
#
# Ce qu'il ne faut pas en attendre : un modèle deux fois plus long à entraîner
# n'est pas deux fois plus précis. Sur un jeu où la marge d'erreur de l'AUC est
# de l'ordre de 0.01, explorer 40 configurations au lieu de 3 déplace le
# résultat de quelques millièmes — c'est-à-dire de moins que le bruit. La
# recherche approfondie sert surtout à VÉRIFIER que le réglage par défaut
# n'était pas franchement mauvais.
RECHERCHE_RAPIDE = "rapide"
RECHERCHE_APPROFONDIE = "approfondie"
RECHERCHE_EXHAUSTIVE = "exhaustive"
RECHERCHE_DEFAUT = RECHERCHE_RAPIDE

MODES_RECHERCHE = {
    RECHERCHE_RAPIDE: {
        "libelle": "Rapide — 3 configurations",
        "n_configurations": 3,
        "description": ("Les trois réglages de référence (prudent, équilibré, "
                        "souple). Quelques secondes à quelques minutes."),
    },
    RECHERCHE_APPROFONDIE: {
        "libelle": "Approfondie — 18 configurations",
        "n_configurations": 18,
        "description": ("Les trois références plus quinze tirages aléatoires "
                        "autour d'elles. Compte environ six fois le temps du "
                        "mode rapide."),
    },
    RECHERCHE_EXHAUSTIVE: {
        "libelle": "Exhaustive — 40 configurations",
        "n_configurations": 40,
        "description": ("Balayage large de l'espace des réglages. Long, et le "
                        "gain attendu reste inférieur à la marge d'erreur de "
                        "l'AUC — à réserver à une vérification ponctuelle."),
    },
}


def mode_recherche(cle: str | None) -> dict:
    """Descriptif d'un mode de recherche, mode rapide par défaut."""
    return MODES_RECHERCHE.get(str(cle or RECHERCHE_DEFAUT),
                               MODES_RECHERCHE[RECHERCHE_DEFAUT])


def libelles_recherche() -> list[str]:
    """Libellés des modes, dans l'ordre croissant de durée."""
    return [MODES_RECHERCHE[cle]["libelle"]
            for cle in (RECHERCHE_RAPIDE, RECHERCHE_APPROFONDIE,
                        RECHERCHE_EXHAUSTIVE)]


def recherche_par_libelle(libelle: str) -> str:
    """Clé du mode correspondant à un libellé affiché."""
    for cle, mode in MODES_RECHERCHE.items():
        if mode["libelle"] == libelle:
            return cle
    return RECHERCHE_DEFAUT


# ===========================================================================
# FILTRAGE DES FEATURES PAR UTILITÉ MESURÉE
# ===========================================================================
# À la fin de chaque entraînement, l'utilité de chaque feature est mesurée par
# permutation : on mélange sa colonne au hasard et on regarde de combien l'AUC
# tombe. Une feature dont l'AUC ne bouge pas ne sert à rien ; une feature dont
# l'AUC MONTE quand on la détruit nuit activement.
#
# Le seuil ci-dessous permet de réentraîner en ne gardant que les features
# au-dessus d'une utilité donnée. Il s'exprime directement en PERTE D'AUC, la
# même unité que le graphique d'importance. L'ordre de grandeur réel sur ces
# données : la meilleure feature fait perdre 0.010 à 0.015 d'AUC, la médiane
# 0.0005, et la moitié du classement tourne autour de zéro. D'où une borne de
# curseur à 0.003, au-delà de laquelle il ne resterait presque rien.
#
# Point de méthode important : l'utilité qui sert à FILTRER est mesurée sur le
# bloc de validation, jamais sur le test. Choisir ses features d'après le test
# puis annoncer une performance sur ce même test, c'est se noter soi-même —
# le chiffre obtenu serait flatteur et faux. Le graphique de la page
# Visualisation, lui, reste mesuré sur le test : il sert à diagnostiquer, pas
# à décider.
SEUIL_UTILITE_MAX = 0.003      # perte d'AUC, borne haute du curseur
SEUIL_UTILITE_PAS = 60         # nombre de crans du curseur
FEATURES_MINIMUM = 3           # jamais moins, quel que soit le seuil

# Répétitions de la permutation pendant l'entraînement. Cinq suffisent pour
# classer ; en demander plus rallongerait l'entraînement sans changer l'ordre.
REPETITIONS_UTILITE = 5


