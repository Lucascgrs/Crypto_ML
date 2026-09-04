"""
Tous les textes explicatifs de l'interface.

Ils sont regroupés ici pour deux raisons : les écrans restent lisibles (pas de
pavés de texte au milieu du code d'affichage), et les explications peuvent être
relues et corrigées d'un seul endroit.
"""

from __future__ import annotations

# ===========================================================================
# INFO-BULLES (pastilles « ⓘ » au survol)
# ===========================================================================
AIDES = {
    # --- Page Données ---
    "top_n": "Nombre de cryptos à récupérer dans le classement CoinGecko,\n"
             "de la plus grosse capitalisation à la plus petite.",
    "symbole": "Code de la crypto, sans la paire : BTC, ETH, SOL…\n"
               "La paire USDT est ajoutée automatiquement pour Binance.",
    "source": "Binance : historique profond et fiable, à privilégier.\n"
              "Yahoo : dépannage pour les cryptos absentes de Binance.",
    "intervalle": "Durée d'une bougie. C'est aussi l'unité de l'horizon :\n"
                  "en 1h, un horizon de 12 signifie 12 heures ; en 1d, 12 jours.",
    "periode": "Plage d'historique à télécharger, au format AAAA-MM-JJ.\n"
               "Plus l'historique est long, plus le modèle a de régimes de\n"
               "marché à apprendre (hausse, baisse, stagnation).",

    # --- Page Analyse ---
    "analyse": "Calcule les 8 indicateurs et les 24 colonnes variation_x,\n"
               "puis enregistre le tout dans analysis_crypto/.\n"
               "À relancer après chaque nouveau téléchargement de données.",

    # --- Page Modèle ---
    "crypto_modele": "Fichier issu de l'étape Analyse. C'est sur ces données\n"
                     "que le modèle apprend, puis est évalué.",
    "type_modele": "Algorithme d'apprentissage. Ses forces et ses limites\n"
                   "s'affichent juste en dessous du menu.",
    "horizon": "Nombre de périodes à prédire : « le prix sera-t-il plus haut\n"
               "ou plus bas dans X périodes ? »\n"
               "En intervalle 1h, un horizon de 12 = 12 heures.\n"
               "Court (1-3) = très bruité. Moyen (6-12) = bon compromis.\n"
               "Long (24) = tendance plus lisible, signaux plus rares.",
    "seuil": "Confiance minimale pour retenir une prédiction.\n"
             "0.60 = on ne garde que les bougies où le modèle est sûr à 60 %,\n"
             "à la hausse COMME à la baisse.\n"
             "Plus haut = moins de signaux, mais plus fiables.",
    "modele_utilise": "Applique le modèle d'une autre crypto à celle-ci.\n"
                      "Les 8 indicateurs étant les mêmes partout, un modèle\n"
                      "entraîné sur BTC peut être testé sur ETH.",

    # --- Page Backtest ---
    "bt_capital": "Capital de départ de la simulation.",
    "bt_seuil": "Confiance minimale pour ouvrir une position.",
    "bt_tp": "Take Profit : sortie dès ce gain (en %). 0 = désactivé.",
    "bt_sl": "Stop Loss : sortie dès cette perte (en %). 0 = désactivé.",
    "bt_duree": "Durée de détention maximale, en périodes.\n"
                "La valeur naturelle est l'horizon du modèle : c'est\n"
                "exactement ce qu'il prétend savoir prédire.",
    "bt_frais": "Frais par transaction, en %. Binance prélève environ 0.1 %.\n"
                "Ils sont comptés à l'entrée ET à la sortie.",
    "bt_slippage": "Écart entre le prix affiché et le prix réellement obtenu,\n"
                   "en %. S'ajoute aux frais.",
    "bt_short": "Coché : un signal de BAISSE ouvre une vente à découvert.\n"
                "Décoché : seuls les signaux de hausse sont joués.",
    "bt_bloc": "Restreint la simulation aux données que le modèle n'a jamais\n"
               "vues pendant l'entraînement (bloc « test »).\n"
               "C'est le SEUL résultat représentatif : sur les données\n"
               "d'entraînement, le modèle rejoue ce qu'il a mémorisé.",
    "bt_periode": "Restreint la simulation à une plage de dates (AAAA-MM-JJ).\n"
                  "Laisser vide pour tout l'historique disponible.",
}


# ===========================================================================
# LE GRAND TEXTE : CE QUE L'ENTRAÎNEMENT FAIT TOUT SEUL
# ===========================================================================
# Affiché dans une fenêtre dédiée depuis la page Modèle. C'est la contrepartie
# de l'interface simplifiée : rien n'est caché, tout est expliqué ici.
EXPLICATION_ENTRAINEMENT = """\
L'écran ne demande que quatre choses — la crypto, le modèle, l'horizon et le
seuil de confiance. Tout le reste est décidé automatiquement, selon les règles
ci-dessous.


1 · CE QU'ON DEMANDE AU MODÈLE

   « Dans X périodes, le prix sera-t-il plus haut ou plus bas qu'aujourd'hui ? »

   La réponse est le signe de la colonne variation_X du fichier analysé.
   Le modèle rend une probabilité de hausse p, dont on tire :

        Sens      = HAUSSE si p > 0.5, BAISSE sinon
        Confiance = max(p, 1 - p)   →  toujours entre 50 % et 100 %

   La prédiction va donc dans LES DEUX SENS : une probabilité de 0.18 est un
   signal de baisse à 82 % de confiance, aussi exploitable qu'un 0.82.


2 · LES ENTRÉES DU MODÈLE : 8 INDICATEURS, PAS UN DE PLUS

   RSI (14)              momentum, sur-achat / sur-vente
   Stochastique %K       position du prix dans son range récent
   MACD histogramme / prix   accélération de la tendance
   Écart à la SMA 50     position par rapport à la tendance
   Bollinger %B          extension par rapport à la volatilité
   ATR / prix            niveau de volatilité (marché calme ou agité)
   ADX (14)              force de la tendance
   OBV sur 20 périodes   le volume accompagne-t-il le mouvement ?

   Chacun est soit borné, soit exprimé en proportion du prix. Aucun n'est un
   niveau de prix : un modèle qui voit « BTC = 60 000 » mémorise ce chiffre,
   affiche des scores parfaits sur le passé et s'effondre sur le présent.

   Les colonnes de prix et les 24 colonnes variation_x existent dans le fichier
   mais ne sont JAMAIS données au modèle : les features sont prises sur une
   liste blanche stricte, il n'y a donc aucun risque d'oubli.


3 · LE DÉCOUPAGE DES DONNÉES

   70 % apprentissage  ·  15 % validation  ·  15 % test

   Toujours dans l'ordre chronologique : le test est la période la PLUS
   RÉCENTE, jamais un échantillon tiré au hasard. C'est la seule façon
   honnête de savoir ce que le modèle vaudra demain.

   Un embargo égal à l'horizon est purgé entre les blocs. Sans lui, les
   dernières lignes d'apprentissage ont une cible qui regarde déjà dans la
   période de validation : le modèle connaîtrait une partie de ce sur quoi on
   l'évalue.

   La validation est coupée en deux moitiés, aux rôles distincts :
   la première choisit le modèle et déclenche l'arrêt de l'apprentissage,
   la seconde sert à calibrer les probabilités.


4 · LE RÉGLAGE AUTOMATIQUE

   • Équilibrage des classes — appliqué seulement si l'une des deux dépasse
     55 % des cas. En dessous, un rééquilibrage forcé dégraderait la
     calibration sans rien apporter.

   • Hyperparamètres — trois configurations (prudente, équilibrée, souple)
     sont entraînées, celle qui obtient la meilleure AUC sur la validation est
     retenue. Avec 8 indicateurs seulement, une grille exhaustive coûterait
     beaucoup de temps pour un gain négligeable.

   • Early stopping — les modèles à boosting s'arrêtent dès que la validation
     ne progresse plus. Le critère d'arrêt est le logloss, plus stable que
     l'AUC qui, trop bruitée, ferait arrêter l'apprentissage au hasard.

   • Calibration — un modèle qui annonce « 70 % » n'a pas forcément raison
     70 % du temps. La correction est apprise sur des données jamais vues, par
     tranches d'au moins 250 observations : impossible d'afficher « 95 % de
     confiance » sur la foi de trois bougies.

   • Ressources — le nombre de cœurs et la RAM libre sont mesurés au
     démarrage. La mémoire disponible est convertie en histogrammes plus fins
     (max_bin) et en parallélisme, au lieu de rester inutilisée.


5 · CE QU'IL FAUT REGARDER DANS LES RÉSULTATS

   L'AUC sur le TEST, et la comparaison à la baseline « toujours la même
   réponse ». En crypto, une AUC de 0.52 à 0.56 est déjà un vrai signal ; une
   AUC supérieure à 0.75 doit faire chercher une fuite de données.

   Et surtout la table des seuils de confiance : elle indique, pour chaque
   seuil, combien de bougies sont retenues et quelle proportion est correcte.
   Si la précision monte quand le seuil monte, le score de confiance est
   utile — c'est exactement ce qu'on veut pour filtrer les signaux.
"""


# ===========================================================================
# EXPLICATION DES 8 INDICATEURS (page Analyse)
# ===========================================================================
EXPLICATION_INDICATEURS = """\
L'analyse produit un fichier par crypto contenant trois familles de colonnes.


1 · LES PRIX (Open, High, Low, Close, Volume)

   Conservés pour les graphiques et le backtest. Ils ne sont jamais donnés au
   modèle : ce sont des niveaux, ils grandissent avec le temps, et un modèle
   qui les voit apprend le niveau de prix plutôt que le comportement.


2 · LES 8 INDICATEURS — les seules entrées du modèle

   RSI_14           Compare la force moyenne des hausses à celle des baisses.
                    Sous 30 = sur-vendu, au-dessus de 70 = sur-acheté. (0-100)

   Stoch_K          Position de la clôture dans le range des 14 dernières
                    bougies. 0 = sur le plus bas, 100 = sur le plus haut.

   MACD_Hist_Norm   Écart entre le MACD et sa ligne de signal, divisé par le
                    prix. Mesure si la tendance accélère ou s'essouffle.
                    La division par le prix est ce qui le rend comparable
                    dans le temps et d'une crypto à l'autre.

   Dist_SMA_50      Écart relatif entre le prix et sa moyenne 50 périodes.
                    +0.05 = le prix est 5 % au-dessus de sa moyenne.

   BB_Position      Position dans les bandes de Bollinger. 0 = bande basse,
                    1 = bande haute. Peut sortir de [0, 1] lors des
                    mouvements extrêmes — c'est justement l'information utile.

   ATR_Pct          Amplitude moyenne d'une bougie, en % du prix. Dit au
                    modèle s'il évolue en marché calme ou agité, ce qui change
                    le sens de tous les autres indicateurs.

   ADX_14           Force de la tendance, sans son sens. Sous 20 = marché sans
                    direction, au-dessus de 25 = tendance nette.

   OBV_Pct          Volume net directionnel des 20 dernières périodes, rapporté
                    au volume total. +0.4 = 40 % du volume est allé, net, du
                    côté acheteur. Répond à « le mouvement est-il soutenu ? »


3 · LES 24 COLONNES variation_x

   variation_x = (prix dans x périodes - prix actuel) / prix actuel, en %.

   Ce sont des valeurs FUTURES : elles servent à construire la cible du modèle
   et ne sont jamais des entrées. variation_12 > 0 signifie « 12 périodes plus
   tard, le prix était plus haut ».

   Les dernières lignes du fichier ont des variations vides, tout simplement
   parce que le futur n'est pas encore arrivé. Elles sont conservées : ce sont
   précisément les bougies sur lesquelles on veut une prédiction.
"""


# ===========================================================================
# EXPLICATIONS DES STATISTIQUES (page Évaluation)
# ===========================================================================
STATS = {
    "accuracy": {
        "label": "Justesse (accuracy)",
        "aide": "Part des prédictions correctes, tous sens confondus.\n"
                "50 % = niveau du hasard. En crypto, 52 à 56 % est déjà bon.\n"
                "Au-delà de 70 %, chercher une fuite de données.",
    },
    "auc": {
        "label": "AUC-ROC (pouvoir de séparation)",
        "aide": "Capacité à classer une hausse au-dessus d'une baisse,\n"
                "indépendamment du seuil de décision. C'est la métrique reine.\n"
                "0.5 = hasard, 1.0 = parfait. 0.52-0.56 = signal réel mais faible.",
    },
    "baseline_majoritaire": {
        "label": "Baseline « toujours la même réponse »",
        "aide": "Score obtenu en répondant systématiquement la classe la plus\n"
                "fréquente. Si le modèle ne la bat pas, il n'apporte rien.",
    },
    "part_hausse": {
        "label": "Part de hausses dans les données",
        "aide": "Proportion de bougies suivies d'une hausse à cet horizon.\n"
                "Proche de 50 % = classes équilibrées, situation idéale.\n"
                "Un fort déséquilibre déclenche le rééquilibrage automatique.",
    },
    "precision_hausse": {
        "label": "Précision — signaux de HAUSSE",
        "aide": "Parmi les bougies prédites en hausse, part réellement montée.",
    },
    "rappel_hausse": {
        "label": "Rappel — signaux de HAUSSE",
        "aide": "Parmi les vraies hausses, part détectée par le modèle.",
    },
    "precision_baisse": {
        "label": "Précision — signaux de BAISSE",
        "aide": "Parmi les bougies prédites en baisse, part réellement descendue.",
    },
    "rappel_baisse": {
        "label": "Rappel — signaux de BAISSE",
        "aide": "Parmi les vraies baisses, part détectée par le modèle.",
    },
    "auc_validation": {
        "label": "AUC sur la validation",
        "aide": "Score qui a servi à choisir la configuration.\n"
                "Un grand écart avec l'AUC de test signale du surapprentissage.",
    },
    "n_arbres": {
        "label": "Arbres retenus (early stopping)",
        "aide": "Nombre d'arbres gardés avant que la validation cesse de\n"
                "progresser. Très peu d'arbres = signal vite épuisé.",
    },
    "embargo": {
        "label": "Embargo (anti-fuite)",
        "aide": "Lignes purgées entre les blocs pour empêcher la cible, qui\n"
                "regarde vers le futur, de fuiter d'un bloc à l'autre.\n"
                "Vaut toujours l'horizon choisi.",
    },
}


def commenter(cle: str, valeur) -> tuple[str, str]:
    """
    Commentaire et niveau de qualité associés à la valeur d'une statistique.

    Retourne (commentaire, niveau) où niveau ∈ {bon, moyen, faible, mauvais}.
    Un commentaire vide signifie « pas d'appréciation à porter ».
    """
    try:
        v = float(valeur)
    except (TypeError, ValueError):
        return "", "moyen"

    if cle == "accuracy":
        if v < 0.50:
            return "Sous le hasard : le modèle se trompe plus qu'il ne réussit.", "mauvais"
        if v < 0.52:
            return "À peine mieux que pile ou face.", "faible"
        if v < 0.56:
            return "Correct pour de la crypto, marché très bruité.", "moyen"
        if v <= 0.70:
            return "Bonne justesse pour ce type de données.", "bon"
        return "Très élevé — vérifier l'absence de fuite de données.", "moyen"

    if cle in ("auc", "auc_validation"):
        if v < 0.50:
            return "Sous le hasard : signal inversé ou surapprentissage.", "mauvais"
        if v < 0.52:
            return "Pouvoir de séparation quasi nul.", "faible"
        if v < 0.56:
            return "Signal faible mais réel, exploitable avec un bon seuil.", "moyen"
        if v <= 0.75:
            return "Bon pouvoir prédictif pour de la crypto.", "bon"
        return "Excellent — se méfier d'une fuite de données.", "moyen"

    if cle in ("precision_hausse", "precision_baisse"):
        if v < 0.50:
            return "Plus de faux signaux que de bons.", "faible"
        if v < 0.55:
            return "Fiabilité modérée.", "moyen"
        return "Signaux globalement fiables.", "bon"

    if cle in ("rappel_hausse", "rappel_baisse"):
        if v < 0.30:
            return "Le modèle rate l'essentiel des mouvements de ce sens.", "faible"
        if v < 0.60:
            return "Couverture moyenne.", "moyen"
        return "Bonne couverture des mouvements réels.", "bon"

    if cle == "part_hausse":
        ecart = abs(v - 0.5)
        if ecart < 0.03:
            return "Classes bien équilibrées.", "bon"
        if ecart < 0.10:
            return "Léger déséquilibre, sans conséquence.", "moyen"
        return "Fort déséquilibre : le rééquilibrage automatique s'active.", "faible"

    if cle == "n_arbres":
        if v <= 5:
            return "Très peu d'arbres : le signal s'épuise immédiatement.", "faible"
        if v <= 50:
            return "Apprentissage court, typique d'un signal faible.", "moyen"
        return "Le modèle a continué à progresser longtemps.", "bon"

    return "", "moyen"
