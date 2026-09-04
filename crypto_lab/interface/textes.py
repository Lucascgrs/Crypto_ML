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

    "exogene": "Télécharge le funding rate et l'open interest depuis l'API\n"
               "publique Binance Futures. C'est la seule information du projet\n"
               "qui ne soit PAS dérivée du prix.\n\n"
               "Le funding remonte au lancement du contrat (2019-2020) et sert\n"
               "immédiatement. L'open interest, lui, n'est public que sur\n"
               "30 jours : chaque téléchargement complète le précédent, la\n"
               "couverture s'étend donc au fil des mises à jour.\n\n"
               "À relancer avant l'analyse pour que les colonnes soient prises\n"
               "en compte.",

    # --- Page Analyse ---
    "analyse": "Calcule les 8 indicateurs, le contexte multi-timeframe et les\n"
               "24 colonnes variation_x, puis enregistre le tout dans\n"
               "analysis_crypto/.\n"
               "À relancer après chaque nouveau téléchargement de données.",
    "contexte": "Ajoute les 8 mêmes indicateurs calculés sur l'intervalle\n"
                "supérieur (4h pour du 1h), plus le funding rate s'il a été\n"
                "téléchargé.\n\n"
                "Une colonne n'est retenue que si elle couvre au moins 60 %\n"
                "des lignes : inutile de garder une colonne vide qui\n"
                "obligerait à jeter des années d'historique.\n\n"
                "Décoche pour revenir aux 8 indicateurs seuls et comparer.",

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
    "seuil": "Confiance minimale pour retenir une prédiction, à la hausse\n"
             "COMME à la baisse. Plus haut = moins de signaux, mais plus sûrs.\n\n"
             "Attention : chaque modèle a un PLAFOND de confiance. Sur un marché\n"
             "bruité il dépasse rarement 0.60 — au-delà, aucun signal ne passe.\n"
             "Après un entraînement, ce curseur est placé automatiquement sur le\n"
             "seuil qui donne la meilleure justesse pour le modèle obtenu.",
    "walk_forward": "Réentraîne le modèle une douzaine de fois en avançant dans\n"
                    "le temps : à chaque étape il apprend sur tout le passé\n"
                    "disponible, puis prédit uniquement la période suivante,\n"
                    "jamais vue.\n\n"
                    "On obtient un fichier « …_wf » dont CHAQUE point est hors\n"
                    "échantillon. C'est le seul moyen de backtester tout\n"
                    "l'historique sans que le résultat soit faussé par les\n"
                    "données d'entraînement.\n\n"
                    "Plus long qu'un entraînement simple (une douzaine\n"
                    "d'entraînements complets), mais c'est le chiffre honnête.",
    "modele_utilise": "Applique le modèle d'une autre crypto à celle-ci.\n"
                      "Les 8 indicateurs étant les mêmes partout, un modèle\n"
                      "entraîné sur BTC peut être testé sur ETH.",
    "objectif": "Ce que le modèle doit apprendre. La direction simple reste\n"
                "le choix par défaut ; les trois autres répondent à une\n"
                "question un peu différente.\n\n"
                "• Direction — le prix monte ou descend, toutes bougies\n"
                "  confondues, y compris les micro-mouvements.\n"
                "• Direction nette — même chose, mais les mouvements qui ne\n"
                "  couvrent pas les frais sont ignorés à l'apprentissage.\n"
                "• Triple barrière — touche-t-on le take-profit avant le\n"
                "  stop-loss ? Bien plus proche d'un vrai trade.\n"
                "• Amplitude — 5 classes graduées en multiples de l'ATR :\n"
                "  chaque signal porte enfin une ampleur, pas juste un sens.\n\n"
                "Chaque objectif a ses propres fichiers : ils coexistent sans\n"
                "s'écraser, et se comparent dans l'onglet Évaluation.",

    # --- Page Amplitude ---
    "modele_regression": "Algorithme de régression. Seuls les modèles sachant\n"
                         "faire de la régression QUANTILE figurent ici — ni la\n"
                         "forêt aléatoire ni la régression logistique n'en\n"
                         "sont capables.\n"
                         "GradientBoosting est le repli scikit-learn, toujours\n"
                         "disponible.",
    "amplitude_entrainer": "Entraîne les deux modèles de régression :\n\n"
                           "• VOLATILITÉ — l'ampleur attendue du mouvement,\n"
                           "  sans son sens. C'est là qu'il y a du vrai signal :\n"
                           "  après une bougie agitée, la suivante l'est aussi.\n\n"
                           "• AMPLITUDE — les quantiles 10 / 50 / 90, qui\n"
                           "  donnent un INTERVALLE plutôt qu'un point :\n"
                           "  « entre −0.8 % et +1.6 %, 8 fois sur 10 ».",
    "esperance": "Combine le modèle de direction et celui d'amplitude :\n\n"
                 "    espérance = (2 × P(hausse) − 1) × amplitude − frais\n\n"
                 "Une position longue rapporte +A avec la probabilité p et −A\n"
                 "avec 1−p : son espérance vaut donc (2p−1)·A.\n\n"
                 "C'est le score qui départage enfin un signal à 55 % sur un\n"
                 "mouvement de 3 % et un signal à 60 % sur 0.2 %.\n\n"
                 "Nécessite un modèle de direction ET une régression de\n"
                 "volatilité au même horizon.",

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
    "bt_sizing": "Combien de capital engager sur chaque signal.\n\n"
                 "• Fixe — tout, à chaque fois. Un signal à 51 % est traité\n"
                 "  exactement comme un signal à 65 %.\n\n"
                 "• Proportionnel — la mise suit l'avantage estimé et RECULE\n"
                 "  quand la volatilité prévue monte. On n'égalise plus la\n"
                 "  mise entre deux trades mais le RISQUE pris.\n\n"
                 "Mesuré sur BTC 1h horizon 24 : rendement −21.7 % en fixe\n"
                 "contre −4.7 % en proportionnel, et surtout un drawdown qui\n"
                 "passe de −35.6 % à −13.9 %.\n\n"
                 "L'ajustement par la volatilité n'agit que sur un fichier\n"
                 "d'espérance, seul à contenir la colonne correspondante.",
    "bt_retenu": "Respecte la colonne « Retenu » du fichier plutôt que le seul\n"
                 "seuil de confiance.\n\n"
                 "N'a d'intérêt que sur un fichier « …_esperance », où Retenu\n"
                 "encode aussi « le mouvement attendu couvre-t-il les frais ? ».\n"
                 "Sur un fichier de prédiction ordinaire, Retenu ne reflète\n"
                 "que le seuil utilisé au moment de la prédiction : laisser\n"
                 "décoché.",
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
     confiance » sur la foi de trois bougies. Et si le bloc de calibration est
     lui-même trop petit, les écarts mesurés sont ramenés vers 50 % — sur
     230 bougies, une « fréquence de hausse de 67 % » est du hasard.

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


6 · POURQUOI LA CONFIANCE NE MONTE PAS PLUS HAUT

   Chaque modèle a un PLAFOND de confiance, affiché après l'entraînement. Sur
   BTC en 1h il se situe entre 0.54 et 0.60 selon l'horizon. Ce n'est pas un
   défaut de réglage : c'est la mesure de ce que le modèle sait vraiment.

   La confiance est une PROBABILITÉ, pas une note de qualité. « 56 % » veut
   dire « j'ai raison 56 fois sur 100 » — et c'est vérifié : au seuil de 0.56
   sur l'horizon 3, la justesse réelle constatée est bien de 56 %. Pour
   afficher 0.80, il faudrait avoir raison 80 % du temps sur la direction du
   Bitcoin ; aucun modèle honnête n'y parvient.

   Ce qui déplace vraiment le plafond, c'est la qualité du signal :

     • Horizon court (3 à 6) — nettement mieux que 12 ou 24 en intervalle 1h.
       Mesuré sur BTC : plafond 0.60 et 56 % de justesse à l'horizon 3, contre
       plafond 0.54 à l'horizon 12.

     • Assez de données — en dessous de ~20 000 bougies, le bloc de calibration
       devient trop petit pour établir une confiance fiable. C'est le piège des
       intervalles longs : en 1d on ne dispose que de quelques milliers de
       lignes, et une confiance élevée y serait purement fortuite.

   Un plafond bas accompagné d'un gain de précision réel reste parfaitement
   exploitable : c'est le cas normal, pas un échec.


7 · LE BOUTON WALK-FORWARD

   Un backtest lancé sur tout l'historique inclut la période d'apprentissage :
   le modèle y rejoue ce qu'il a mémorisé, le rendement explose et ne veut rien
   dire. Mesuré sur BTC 1h horizon 3, sans frais : +1 273 % sur la période
   d'entraînement contre +18 % en walk-forward.

   Le walk-forward supprime ce biais. L'historique est coupé en douze tranches
   successives ; pour chacune le modèle est entièrement réentraîné sur tout le
   passé disponible à cette date, puis prédit uniquement la tranche suivante,
   jamais vue. En recollant les tranches on obtient une courbe couvrant la
   moitié de l'historique dont CHAQUE point est hors échantillon.

   Deux avantages en plus du chiffre honnête :

     • On voit la STABILITÉ. Le modèle est évalué sur douze périodes
       différentes, pas une seule. Douze fenêtres au-dessus du hasard valent
       bien mieux qu'une bonne moyenne masquant deux effondrements.

     • Le modèle reste à jour. Chaque fenêtre est entraînée sur les données
       les plus récentes, ce qui correspond à l'usage réel.

   Le fichier produit se termine par « _wf ». Dans l'onglet Backtest, il peut
   être simulé sur toute sa durée sans fausser le résultat.


8 · LES QUATRE OBJECTIFS

   Le menu « Objectif » change la QUESTION posée, pas la machinerie. Découpage,
   embargo, calibration, seuil de confiance : tout reste identique.

   • Direction — le signe de variation_X. Toutes les bougies comptent, même
     celles qui bougent de 0.02 %. C'est le choix par défaut et la référence.

   • Direction nette — mêmes données, mais les bougies dont la variation ne
     couvre pas les frais (0.30 %) sont ÉCARTÉES de l'apprentissage et de
     l'évaluation. Le modèle cesse d'user sa capacité sur du bruit qu'on ne
     pourrait de toute façon pas exploiter. Environ 40 % des bougies
     disparaissent en 1h à l'horizon 3 : c'est voulu.

   • Triple barrière — on simule un vrai trade. Le prix touche-t-il d'abord
     le take-profit (+1 ATR) ou le stop-loss (−1 ATR) ? À défaut, on prend le
     signe à l'échéance. Un mouvement qui monte de 3 % avant de revenir à son
     point de départ n'est plus compté comme une stagnation.

   • Amplitude — cinq classes : forte baisse, baisse, neutre, hausse, forte
     hausse. Les bornes sont des multiples de l'amplitude attendue (ATR dilaté
     en √h), donc valables en marché calme comme agité. La classe « neutre »
     domine largement aux horizons courts (68 % à l'horizon 3) : c'est un fait,
     pas un défaut — la plupart des bougies ne font rien d'exploitable.

   La confiance ne part plus de 50 % mais de 1/nombre de classes : 20 % pour
   les cinq classes d'amplitude. Le curseur de seuil s'adapte tout seul.

   Chaque objectif écrit ses propres fichiers, avec son nom en suffixe. Ils
   coexistent sans s'écraser et se comparent dans l'onglet Évaluation.


9 · LE CONTEXTE : MULTI-TIMEFRAME ET DONNÉES EXOGÈNES

   Deux familles de colonnes s'ajoutent automatiquement aux 8 indicateurs,
   quand elles sont disponibles dans le fichier analysé.

   • Multi-timeframe (suffixe _MTF) — les 8 MÊMES indicateurs calculés sur
     l'intervalle supérieur. Un modèle en 1h ne voit que l'agitation horaire :
     il ignore s'il est dans une tendance 4h haussière ou dans un retournement.

     Le point délicat est l'anti-fuite. Une bougie 4h étiquetée 00:00 couvre
     00:00→04:00 : sa clôture n'est connue qu'à 04:00. Elle est donc décalée
     d'une bougie entière avant d'être rapprochée de l'index horaire. Sans ce
     décalage, le modèle verrait dès 00:00 une information contenant les quatre
     heures suivantes — exactement ce qu'on lui demande de prédire.

   • Exogènes — funding rate et open interest (Binance Futures). Tout le reste
     du fichier est calculé à partir du même OHLCV : ce sont des
     transformations d'une seule information. Le funding et l'open interest
     viennent du positionnement réel des intervenants sur les dérivés. C'est le
     seul apport d'information vraiment neuve disponible gratuitement.

   Une colonne n'est retenue que si elle couvre au moins 60 % des lignes.
   L'open interest public de Binance ne remontant qu'à 30 jours, il est écarté
   au début — chaque téléchargement complète le précédent, et la couverture
   s'étend au fil des mises à jour.

   CE QUE ÇA APPORTE, MESURÉ. Sur BTC 1h, à découpage et données identiques,
   avec contexte (18 features) contre sans (8 features) :

     horizon 3   : plafond 0.601 → 0.610, précision au seuil conseillé
                   56.56 % → 57.58 % (907 → 1 016 signaux)
     horizon 6   : 54.65 % → 58.14 %
     horizon 12  : 52.59 % → 54.74 %

   L'AUC, elle, ne bouge pas (0.5363 → 0.5345 à l'horizon 3, soit moins que sa
   marge d'erreur de ±0.0076). Autrement dit : le contexte n'améliore pas le
   classement d'ensemble, il rend le modèle plus TRANCHÉ là où il a raison —
   ce qui est exactement ce qu'on veut d'un outil de filtrage.

   À relativiser : chaque écart pris isolément tient dans la marge d'erreur
   (±1.6 point sur un millier de signaux). Ce sont les TROIS horizons allant
   dans le même sens qui rendent le résultat crédible, pas un chiffre seul.

   L'importance par permutation (onglet Visualisation) reste le juge de paix :
   sur BTC 1h, Stoch_K écrase tout (+0.030) et les colonnes de contexte
   plafonnent à +0.0006. Elles aident, mais très marginalement.
"""


# ===========================================================================
# EXPLICATION DE LA PAGE AMPLITUDE
# ===========================================================================
EXPLICATION_AMPLITUDE = """\
Le modèle de direction répond à « ça monte ou ça descend ? ». Il ne dit rien de
l'AMPLEUR du mouvement — or un signal juste à 55 % sur un mouvement de 3 % ne
vaut pas du tout un signal juste à 60 % sur 0.2 %.

Cette page ajoute les deux briques qui manquaient, puis les combine. Bonne
nouvelle : les 24 colonnes variation_x contenaient déjà l'amplitude, il n'y a
aucune donnée nouvelle à télécharger.


1 · LA VOLATILITÉ — l'ampleur attendue, sans le sens

   Un modèle sur |variation_X|. C'est ici qu'il y a du vrai signal : la
   direction est quasi imprévisible, l'amplitude beaucoup moins. Le clustering
   de volatilité — une bougie agitée en annonce d'autres — est le fait stylisé
   le plus robuste de la finance de marché.

   LA PERTE UTILISÉE COMPTE. L'erreur absolue estimerait la MÉDIANE, or
   |variation| est très asymétrique : sur BTC 1h à l'horizon 3, médiane 0.44 %
   contre moyenne 0.79 %. Un modèle médian sous-estimerait donc
   systématiquement l'amplitude, et l'espérance de gain serait faussée vers le
   bas. L'erreur au carré viserait bien la moyenne mais serait dominée par
   trois krachs. On utilise donc une perte de Tweedie, faite pour les cibles
   positives et asymétriques.

   COMMENT LIRE LE R². Il paraît petit (0.09 sur BTC 1h à l'horizon 3) et
   pourtant il est bon. On ne prédit pas une quantité déterministe mais
   l'ÉCHELLE d'un tirage aléatoire : même en connaissant exactement la
   volatilité de chaque bougie, |variation| resterait un tirage unique autour
   de cette échelle. Cette part-là est irréductible.

   L'écran affiche donc le PLAFOND théorique — 0.16 dans ce cas — et la part
   qu'en atteint le modèle : 55 %. À comparer au R² de la direction, qui
   tourne autour de zéro.

   Deux références sont données : la moyenne constante, et surtout l'ATR dilaté
   en √h. Cette seconde est la vraie barre à battre — l'ATR est déjà un
   excellent prédicteur de volatilité, et le battre sans le dire ne prouverait
   rien. Mesuré sur BTC 1h : le modèle gagne aux horizons 3 et 6, fait jeu égal
   à 12, et PERD à 24. Aux horizons longs, l'ATR suffit.


2 · LES QUANTILES — un intervalle plutôt qu'un point

   Trois modèles estiment les quantiles 10 / 50 / 90 de variation_X. La sortie
   n'est plus une valeur mais un INTERVALLE : « dans 3 h, entre −0.73 % et
   +0.68 %, 8 fois sur 10 ». C'est littéralement la volatilité du mouvement à
   venir, exprimée en prix.

   LA MÉTRIQUE QUI COMPTE EST LA COUVERTURE. Si 80 % des issues réelles tombent
   bien entre Q10 et Q90, l'intervalle veut dire ce qu'il annonce. Mesuré sur
   BTC 1h horizon 3 : 81.3 %. À l'horizon 24 en revanche : 85.1 %, l'intervalle
   est trop large — il annonce plus d'incertitude qu'il n'y en a.

   La perte pinball est comparée à celle du quantile constant. Gain mesuré :
   +4.9 % sur Q10, +4.2 % sur Q90, mais seulement +0.2 % sur Q50 — la médiane
   est proche de zéro et pratiquement imprévisible, ce qui est une autre façon
   de retrouver le problème de la direction.

   L'écart-type de la largeur (0.50 % pour une largeur moyenne de 1.60 %) est
   une bonne nouvelle : l'intervalle s'élargit vraiment en marché agité au lieu
   de rester figé.

   Détail technique : les trois quantiles étant estimés séparément, rien ne
   garantit qu'ils sortent ordonnés. Les valeurs sont donc triées ligne par
   ligne (« croisement de quantiles », défaut connu de la méthode).


3 · L'ESPÉRANCE DE GAIN — le score qui décide

       espérance = (2 × P(hausse) − 1) × amplitude attendue − frais

   Une position longue rapporte +A avec la probabilité p et −A avec 1−p : son
   espérance vaut (2p−1)·A. Le signe donne le sens à prendre, la valeur absolue
   l'intérêt du trade, et les frais tranchent.

   CE QUE ÇA DONNE, MESURÉ. Sur BTC 1h à l'horizon 24, bloc test uniquement,
   classé par espérance nette annoncée :

     espérance ≥        signaux   justesse   gain réel moyen
     −0.30 % (tout)      11 343     51.02 %      −0.336 %
     −0.10 %              1 483     55.23 %      −0.152 %
      0.00 %                721     55.76 %      −0.089 %
     +0.10 %                223     57.85 %      +0.065 %
     +0.20 %                 46     69.57 %      +0.930 %

   L'ordonnancement fonctionne : plus l'espérance annoncée est élevée, plus la
   justesse ET le gain réel montent, sur six niveaux consécutifs. C'est le
   résultat solide de ce tableau.

   En revanche, la RENTABILITÉ n'est pas démontrée. Sur 223 trades dont les
   rendements individuels ont un écart-type de l'ordre de 3 %, la marge
   d'erreur sur la moyenne est d'environ 0.2 % — soit trois fois le gain
   affiché. Ce +0.065 % est compatible avec zéro.

   À l'horizon 3, aucun signal ne dépasse les frais : la meilleure espérance
   nette atteinte est de −0.20 %. Ce n'est pas un défaut du modèle mais un
   constat économique. L'avantage existe, il est simplement plus petit que le
   coût d'un aller-retour. L'amplitude croît en √h, les frais non : c'est
   pourquoi les horizons longs s'en sortent mieux sur ce critère précis.


4 · CE QUE LE FICHIER PRODUIT PERMET

   Le fichier « …_esperance » contient tout ce qu'attend le Backtest, plus la
   volatilité prévue. Deux réglages du Backtest ne servent qu'avec lui :

   • « Respecter Retenu » — ajoute le filtre économique au seuil de confiance.
   • « Taille proportionnelle » — la mise suit l'avantage estimé et recule
     quand la volatilité prévue monte. Sans la colonne de volatilité, seul le
     premier facteur joue.

   Mesuré sur BTC 1h horizon 24, bloc test, frais réels de 0.30 % :

     fixe, sans filtre           −21.7 %   drawdown −35.6 %
     fixe + filtre espérance     −12.1 %   drawdown −29.0 %
     proportionnel               −4.7 %    drawdown −13.9 %
     proportionnel + filtre      −3.7 %    drawdown −11.6 %

   (Buy & Hold sur la même période : −23.2 %.)

   Chaque mécanisme réduit la perte et le risque, et les quatre variantes
   battent le marché sur cette période baissière. Aucune n'est rentable pour
   autant : l'avantage brut ne couvre toujours pas 0.30 % d'aller-retour.
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
    # --- Régression (page Amplitude) ---
    "r2": {
        "label": "R² — part de la variance expliquée",
        "aide": "Ne se compare pas à 1 mais au PLAFOND théorique affiché juste\n"
                "en dessous : |variation| reste un tirage unique autour de la\n"
                "volatilité, et cette part est irréductible.\n"
                "À comparer surtout au R² de la direction, proche de zéro.",
    },
    "r2_naif": {
        "label": "R² de la référence naïve (ATR dilaté)",
        "aide": "Score obtenu en prédisant simplement « ATR × √horizon ».\n"
                "C'est la vraie barre à battre : l'ATR est déjà un excellent\n"
                "prédicteur de volatilité. Ne pas le battre signifie que le\n"
                "modèle n'apporte rien de plus que l'indicateur seul.",
    },
    "r2_plafond": {
        "label": "Plafond théorique du R²",
        "aide": "R² maximal atteignable même avec un modèle parfait, puisqu'on\n"
                "prédit l'échelle d'un tirage aléatoire et non une quantité\n"
                "déterministe. Estimé sous hypothèse gaussienne, donc plutôt\n"
                "généreux : le vrai plafond est un peu plus bas.",
    },
    "correlation_rang": {
        "label": "Corrélation de rang",
        "aide": "Le modèle classe-t-il correctement les bougies de la plus\n"
                "calme à la plus agitée ? Insensible à l'échelle, c'est la\n"
                "métrique la plus robuste pour un usage de dimensionnement\n"
                "de position.",
    },
    "couverture": {
        "label": "Couverture de l'intervalle Q10–Q90",
        "aide": "Part des issues réelles tombées dans l'intervalle annoncé.\n"
                "Doit valoir 80 %. En dessous, l'intervalle est trop optimiste ;\n"
                "au-dessus, il annonce plus d'incertitude qu'il n'y en a.\n"
                "C'est LA métrique d'honnêteté d'une régression quantile.",
    },
    "largeur_moyenne": {
        "label": "Largeur moyenne de l'intervalle",
        "aide": "Amplitude, en %, de la fourchette Q10–Q90 annoncée.\n"
                "Son écart-type compte autant : un intervalle qui ne varie\n"
                "jamais n'apporte rien, il doit s'élargir en marché agité.",
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

    if cle == "couverture":
        ecart = abs(v - 0.80)
        if ecart < 0.03:
            return "Intervalle honnête : il annonce ce qu'il tient.", "bon"
        if v < 0.80:
            return "Intervalle trop étroit : la réalité en sort trop souvent.", "faible"
        return "Intervalle trop large : il annonce plus d'incertitude qu'il n'y en a.", "moyen"

    if cle == "correlation_rang":
        if v < 0.10:
            return "Le classement calme / agité n'est pas meilleur que le hasard.", "faible"
        if v < 0.25:
            return "Classement correct, exploitable pour dimensionner une position.", "moyen"
        return "Bon classement des régimes de volatilité.", "bon"

    if cle == "r2":
        if v <= 0:
            return "Sous la moyenne constante : le modèle n'explique rien.", "mauvais"
        if v < 0.05:
            return "Explique peu, mais toujours plus que la direction.", "faible"
        return "Signal réel — à rapporter au plafond théorique affiché.", "bon"

    return "", "moyen"
