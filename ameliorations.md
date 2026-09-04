# Notes de refonte — Crypto Lab v2.1

> Refonte complète de la partie analyse et de la partie prédiction (v2),
> puis ajout de la prévision d'amplitude et du contexte (v2.1).
> Fil conducteur : **trop de data tue la data**, et **une mesure qu'on ne peut
> pas rejouer en backtest ne vaut rien**.

---

## 1. Le diagnostic

La version précédente produisait ~120 colonnes d'indicateurs et des résultats
« brouillons » : bons sur l'historique, mauvais sur la période récente, et un
paramétrage si dense qu'il était impossible de savoir ce qui influençait quoi.

Trois causes, toutes traitées :

1. **Trop de features.** Momentum multi-horizons, z-scores, rangs glissants,
   lags, Ichimoku, order-flow, Fear & Greed… La plupart redondantes entre
   elles. Un modèle noyé sous des variables corrélées apprend du bruit.
2. **Une liste noire au lieu d'une liste blanche.** Les features étaient
   obtenues en *retirant* les colonnes interdites — il suffisait d'en oublier
   une pour laisser fuiter le niveau de prix.
3. **Trop de réglages exposés.** Type de cible, seuil ATR, triple-barrière,
   embargo, élagage de corrélation, équilibrage, sélection SHAP… autant de
   combinaisons qui rendaient les résultats incomparables d'un essai à l'autre.

---

## 2. Ce qui a changé

### Analyse — repartie de zéro

- **8 indicateurs**, un par axe d'information : RSI, Stochastique %K, MACD
  histogramme normalisé, écart à la SMA 50, Bollinger %B, ATR %, ADX, flux OBV.
  Tous stationnaires (bornés, ou rapportés au prix / au volume).
- **24 colonnes `variation_x`** : variation en % observée `x` périodes après la
  ligne, pour `x` de 1 à 24. Ce sont les cibles.
- **Liste blanche** : les features sont exactement `config.INDICATEURS`. Le
  prix ne peut plus fuiter, même par oubli.
- Les dernières lignes (variations encore inconnues) sont **conservées** — ce
  sont les bougies sur lesquelles on veut une prédiction.
- Fichiers analysés : 79 Mo → 30 Mo, et un cache Parquet les recharge en 0.1 s
  au lieu de ~20 s.

### Prédiction — bidirectionnelle et simplifiée

- Cible = signe de `variation_X`. Le modèle rend une probabilité de hausse,
  d'où `Sens` et `Confiance = max(p, 1−p)`.
- La confiance vaut **dans les deux sens** : `p = 0.18` est un signal de baisse
  à 82 %. Le backtest ouvre des positions courtes en conséquence.
- **Quatre réglages** dans l'interface : crypto, modèle, horizon (curseur 1–24),
  seuil de confiance. Le choix du modèle reste entre les mains de l'utilisateur.
- Tout le reste est automatique et expliqué dans un bouton dédié : découpage
  70/15/15, embargo, équilibrage conditionnel, trois configurations
  d'hyperparamètres départagées sur la validation, early stopping, calibration.

### Fiabilité des résultats

- **Early stopping sur le logloss** et non l'AUC : sur un signal faible, l'AUC
  est trop bruitée et faisait arrêter l'apprentissage au bout de 2 arbres.
- **Calibration par tranches** d'au moins 250 observations : plus de « 100 % de
  confiance » appuyé sur 4 bougies, ce qui rendait le seuil inutilisable.
- **Colonne `Bloc`** dans les prédictions (train / validation / test) et
  backtest limité au test par défaut : impossible de confondre une performance
  mémorisée avec une performance réelle.
- **Table des seuils de confiance** dans l'évaluation : pour chaque seuil, le
  nombre de signaux retenus, la couverture et la précision obtenue.

### Ressources

`crypto_lab/ressources.py` mesure cœurs et RAM libre au démarrage de chaque
entraînement, puis convertit la mémoire disponible en histogrammes plus fins
(`max_bin` jusqu'à 1024) et en parallélisme, au lieu de la laisser inutilisée.
Un cache mémoire évite aussi de relire les mêmes fichiers d'un écran à l'autre.

### Organisation

Quatre fichiers de 500 à 2300 lignes → un paquet `crypto_lab/` de modules
courts, avec une page d'interface par fichier. Les scripts historiques
(`GatherData.py`, `CryptoAnalysis.py`, `Predict.py`, `Dashboard.py`) sont
conservés comme points d'entrée en ligne de commande.

---

## 3. Ce qui a été retiré

| Retiré | Pourquoi | Remplacé par |
|---|---|---|
| ~110 indicateurs | redondants, sources de bruit | les 8 retenus |
| Fear & Greed | granularité journalière, apport nul en 1h | — |
| Order-flow Binance | aucune feature ne l'utilisait | — |
| Cibles « Seuil ATR » / triple-barrière | trois définitions incomparables | direction simple + seuil de confiance |
| Sélection SHAP | inutile avec 8 features ; dépendance lourde | importance par permutation |
| Dataset poolé « MULTI » | complexité pour un gain non démontré | « Prédire avec le modèle de … » (un modèle s'applique à toute crypto) |
| Élagage de corrélation, embargo manuel, équilibrage manuel | réglages que personne ne peut arbitrer à l'œil | décidés automatiquement |

---

## 4. Résultats mesurés (BTC 1h, bloc test)

| Horizon | AUC test | Justesse | Précision au seuil 60 % |
|---|---|---|---|
| 1  | 0.536 | 51.9 % | 55.0 % (471 signaux) |
| 6  | 0.531 | 52.4 % | — |
| 12 | 0.511 | 51.6 % | 55.2 % (573 signaux) |
| 24 | 0.516 | 51.6 % | — |

Lecture honnête : le signal brut est faible (AUC 0.51–0.54, conforme à ce qu'on
peut attendre sur de la direction crypto), **mais le filtrage par confiance
apporte un vrai gain** — de 51.6 % à 55.2 % de précision à l'horizon 12.

En revanche, le backtest reste négatif après frais : environ 0.3 % de coût par
aller-retour contre un avantage brut de l'ordre de 0.04 % par trade. Autrement
dit, l'avantage existe mais ne couvre pas les frais à cet horizon.

---

## 4 bis. Le plafond de confiance (correctif suivant)

Constat à l'usage : sur BTC 1h à l'horizon 12, la confiance ne dépassait jamais
0.55 et le seuil par défaut (0.60) ne laissait donc passer aucun signal, sans
que rien ne l'explique.

**Diagnostic.** Ce n'était pas un bug de calibration. En classant les
prédictions du test par extrémité de la probabilité brute, la justesse réelle
plafonne à ~56 % (et *retombe* à 46 % sur le 1 % le plus extrême). La
calibration faisait donc exactement son travail : refuser d'annoncer une
certitude que le modèle n'a pas.

**Ce qui a été corrigé.**

- Le plafond de confiance atteint est mesuré, stocké et **affiché** après
  chaque entraînement.
- Un **seuil conseillé** est calculé (meilleure justesse parmi les seuils
  laissant passer au moins 200 signaux) et le curseur s'y place tout seul.
  Quand aucun seuil n'apporte rien, l'interface le dit au lieu de laisser
  chercher.
- La grille de seuils évaluée est resserrée entre 0.50 et 0.60, là où tout se
  joue sur des marchés bruités.
- Seuil par défaut ramené de 0.60 à 0.55.
- **Fausse confiance sur petits échantillons** : la sortie du calibrateur est
  bornée aux fréquences réellement observées, elles-mêmes ramenées vers 50 %
  quand le bloc de calibration est trop petit. Mesuré avant correction sur BTC
  1d (232 bougies de calibration) : 0.68 de confiance annoncée pour 46.8 % de
  justesse réelle. Après : plafond 0.53, aucun signal trompeur, et
  « aucun seuil conseillé » affiché.

**Ce qui aide vraiment.** Comparaison mesurée intervalle × horizon :

| Intervalle | Horizon | AUC test | Plafond | Précision @0.55 |
|---|---|---|---|---|
| 1h | **3** | 0.534 | 0.595 | **55.9 %** |
| 1h | **6** | 0.535 | 0.574 | 55.2 % |
| 1h | 12 | 0.519 | 0.541 | — |
| 4h | 12 | 0.500 | 0.561 | 50.6 % |
| 1d | 24 | 0.497 | 0.681 → 0.534 | 46.8 % ⚠️ |

Conclusion contre-intuitive : **les intervalles longs n'aident pas**. Moins de
données, estimations plus bruitées, et une confiance élevée mais fausse. Le
levier réel est l'**horizon court** (3 à 6) en 1h.

---

## 4 ter. Surapprentissage : mesure honnête (correctif suivant)

Constat à l'usage : un backtest sur toute la période affiche un bénéfice
gigantesque. Le modèle mémorise-t-il au lieu d'apprendre ?

**Diagnostic.** Oui, et c'était déjà visible dans l'écart de justesse entre les
blocs (72 % en apprentissage contre 56 % en test). Mais le montant du backtest
n'était pas la bonne mesure : il est gonflé par le compounding sur des milliers
de trades, et par le fait que l'apprentissage occupe 85 % de la timeline.

Mesuré sur BTC 1h horizon 3, frais mis à zéro pour isoler la mémorisation :

| Période simulée | Rendement | Buy & Hold |
|---|---|---|
| apprentissage | **+1 273 %** | +183 % |
| test | +28 % | −24 % |
| **walk-forward** | **+18 %** | +118 % |

**Ce qui a été ajouté.**

- **Walk-forward** (`modele.walk_forward`) : douze réentraînements complets en
  avançant dans le temps, chaque tranche prédite n'ayant jamais été vue.
  Produit un fichier `…_wf` backtestable sur toute sa durée. Bonus : on mesure
  la stabilité — AUC 0.5435 ± 0.0141, **12/12 fenêtres au-dessus du hasard**,
  et un écart validation → réel de seulement +0.0105.
- **Règle du « un écart-type »** pour choisir la configuration. Les trois
  candidates étaient séparées par ~0.003 d'AUC alors que la marge d'erreur est
  de 0.0076 : on retenait donc systématiquement le modèle le plus complexe sur
  du bruit. À égalité statistique, la plus simple gagne désormais. Effet de
  bord agréable : la précision au seuil conseillé passe de 55.97 % à 56.62 %.
- **Zones train / validation / test grisées** sur les graphiques du backtest,
  plus un avertissement chiffré (« 85 % de la période simulée a servi à
  l'entraînement »).

---

## 5. Amplitude et contexte (v2.1)

Jusqu'ici le modèle savait dire dans quel SENS, jamais de COMBIEN. Or un signal
juste à 55 % sur un mouvement de 3 % ne vaut pas un signal juste à 60 % sur
0.2 %. Neuf ajouts, tous branchés sur des données déjà présentes ou sur une API
publique.

### 5.1 Ce qui a été ajouté

| # | Ajout | Où |
|---|---|---|
| 1 | Régression quantile 10/50/90 → un INTERVALLE de prix | `amplitude.py` |
| 2 | Régression de volatilité sur `\|variation_h\|` | `amplitude.py` |
| 3 | Classification en 5 classes d'amplitude | `cibles.py` |
| 4 | Espérance de gain = (2p−1) × amplitude − frais | `amplitude.esperance` |
| 5 | Cible « le mouvement dépasse-t-il les frais » | `cibles.cible_direction_nette` |
| 6 | Triple barrière TP / SL / temps | `cibles.triple_barriere` |
| 7 | Taille de position proportionnelle | `backtest.Simulateur._fractions` |
| 8 | Contexte multi-timeframe (8 colonnes `_MTF`) | `indicateurs.py` |
| 9 | Funding rate et open interest | `exogene.py` |

Nouvelle page « 4 · Amplitude » dans l'interface, nouveau point d'entrée
`Amplitude.py` en ligne de commande, menu « Objectif » sur la page Prédiction.

### 5.2 Trois pièges rencontrés — et ce qu'ils ont appris

**Le R² négatif de la volatilité.** Premier essai avec une perte d'erreur
absolue : R² de −0.018 alors que la MAE battait toutes les références. La perte
absolue estime la MÉDIANE, or `|variation|` est très asymétrique (médiane
0.44 %, moyenne 0.79 %). Le modèle prédisait donc correctement la médiane, ce
qui est un mauvais estimateur de la moyenne — et l'espérance de gain, qui a
besoin de E[amplitude], en aurait été faussée vers le bas. Passage à une perte
de **Tweedie** (puissance 1.5), faite pour les cibles positives et asymétriques :
R² de **−0.018 → +0.087**, au-dessus de la référence ATR (0.074).

Corollaire utile : le R² d'une prévision de volatilité ne se compare pas à 1. On
prédit l'échelle d'un tirage aléatoire, pas une quantité déterministe. Un
**plafond théorique** est désormais calculé et affiché (0.16 ici) — le modèle en
atteint 55 %.

**La classe « neutre » qui mange tout.** Les 5 classes d'amplitude étaient
d'abord bornées à ±0.5 et ±1.5 ATR. Résultat : 60 à 68 % de bougies « neutres ».
La calibration, qui fait correctement son travail en ramenant chaque probabilité
à la fréquence réelle de sa classe, rendait alors « neutre » systématiquement
majoritaire — le modèle répondait « neutre » 100 % du temps, pour exactement le
score de la réponse constante. Deux corrections :

- rééquilibrage réellement appliqué en multi-classe (XGBoost n'a pas de
  `class_weight` : il fallait passer par des poids d'échantillon explicites) ;
- bornes redéfinies comme les **quintiles** de la variation normalisée par
  l'ATR. Chaque classe pèse 20 %, l'argmax redevient une décision.

Après correction : 22.32 % de justesse brute contre 20 % de hasard, et
**31.79 % au seuil conseillé** — soit 1.6 × le hasard.

**Les 63.5 % qui n'existaient pas.** L'objectif « direction nette » annonçait
63.54 % de précision à l'horizon 3, de loin le meilleur chiffre du projet. Le
backtest des mêmes signaux donnait 34 % de trades gagnants. L'explication :
cette précision n'était mesurée que sur les bougies dont le mouvement dépasse
les frais — un événement qu'on ne connaît pas au moment de décider. Biais de
sélection pur.

Correction : `Tache.construire_evaluation` sépare la cible APPRISE de la cible
ÉVALUÉE. Le modèle continue de n'apprendre que sur les gros mouvements, mais il
est jugé sur toutes les bougies. Chiffre corrigé : **60.91 % sur 330 signaux** —
et le backtest sans frais donne 62 % de trades gagnants. Les deux mesures
concordent enfin.

### 5.3 Résultats mesurés (BTC 1h, bloc test, 75 639 bougies)

| Horizon | Objectif | AUC | Plafond | Seuil | Signaux | Précision | Gain |
|---|---|---|---|---|---|---|---|
| 3 | direction | 0.5345 | 0.610 | 0.60 | 1 016 | 57.58 % | +4.80 |
| 3 | **direction_nette** | 0.5349 | 0.617 | 0.60 | 330 | **60.91 %** | **+9.08** |
| 3 | barriere | 0.5167 | 0.559 | 0.55 | 1 653 | 53.48 % | +2.39 |
| 3 | amplitude | 0.5407 | 0.292 | 0.28 | 532 | 28.01 % | +5.97 |
| 12 | direction | 0.5030 | 0.572 | 0.55 | 886 | 54.74 % | +4.22 |
| 12 | **amplitude** | 0.5293 | 0.356 | 0.30 | 582 | **31.79 %** | **+9.47** |

Contexte (18 features contre 8), même découpage :

| Horizon | AUC | Précision au seuil conseillé |
|---|---|---|
| 3 | 0.5363 → 0.5345 | 56.56 % → **57.58 %** |
| 6 | 0.5345 → 0.5344 | 54.65 % → **58.14 %** |
| 12 | 0.5179 → 0.5030 | 52.59 % → **54.74 %** |

L'AUC ne bouge pas (écarts inférieurs à sa marge d'erreur de ±0.0076) : le
contexte n'améliore pas le classement d'ensemble, il rend le modèle plus
TRANCHÉ là où il a raison. Chaque écart pris isolément tient dans la marge
d'erreur ; ce sont les trois horizons concordants qui rendent le résultat
crédible. L'importance par permutation le confirme sans complaisance :
`Stoch_K` écrase tout (+0.030), les colonnes de contexte plafonnent à +0.0006.

Espérance de gain (horizon 24, classée par espérance nette annoncée) :

| Espérance ≥ | Signaux | Justesse | Gain réel moyen |
|---|---|---|---|
| −0.30 % (tout) | 11 343 | 51.02 % | −0.336 % |
| 0.00 % | 721 | 55.76 % | −0.089 % |
| +0.10 % | 223 | 57.85 % | **+0.065 %** |
| +0.20 % | 46 | 69.57 % | **+0.930 %** |

L'ordonnancement est net sur six niveaux consécutifs. La rentabilité, elle,
n'est pas démontrée : sur 223 trades d'écart-type ~3 %, la marge d'erreur est
d'environ 0.2 %, soit trois fois le gain affiché.

Taille de position (horizon 24, frais réels de 0.30 %) :

| Mode | Rendement | Drawdown |
|---|---|---|
| fixe, sans filtre | −21.7 % | −35.6 % |
| fixe + filtre espérance | −12.1 % | −29.0 % |
| proportionnel | −4.7 % | −13.9 % |
| proportionnel + filtre | −3.7 % | −11.6 % |

Buy & Hold : −23.2 %. Chaque mécanisme réduit la perte et le risque ; aucun ne
rend la stratégie rentable.

### 5.4 La limite qui ne bouge pas

Après neuf ajouts, le constat central est inchangé et se lit maintenant en
euros : **l'avantage existe mais ne couvre pas 0.30 % d'aller-retour**. À
l'horizon 3, la meilleure espérance nette atteinte est de −0.20 %. À l'horizon
24, seuls 2 % des signaux passent au-dessus de zéro.

Ce n'est pas un défaut de modélisation, c'est un ordre de grandeur : un
avantage directionnel de 8 à 11 points sur des mouvements de 0.8 % rapporte de
l'ordre de 0.1 % par trade, contre 0.30 % de coût. Les deux seules directions
qui déplaceraient vraiment cette ligne sont un coût de transaction plus bas
(ordres makers, ~0.02 % chez Binance) et des horizons plus longs — l'amplitude
croît en √h, les frais non.

---

## 6. Pistes restantes

1. **Frais makers** : refaire les mesures à 0.04 % d'aller-retour plutôt que
   0.30 %. C'est le seul levier qui change l'ordre de grandeur du problème.
2. **Open interest accumulé** : la colonne est écartée tant qu'elle ne couvre
   pas 60 % de l'historique. Quelques mois de téléchargements réguliers
   suffiraient à la rendre exploitable — et c'est la seule feature du projet
   qui apporte une information vraiment neuve.
3. **Validation croisée purgée** (CPCV) pour une estimation hors échantillon
   plus fiable que le simple bloc final.
4. **Modèles par régime** : un modèle en marché calme, un en marché agité
   (via `ATR_Pct`), plutôt qu'un modèle unique moyennant les deux.
5. **Meta-labelling** : un second modèle qui apprend quand faire confiance au
   premier, plutôt que d'améliorer le premier.
