# Notes de refonte — Crypto Lab v2.3

> Refonte complète de la partie analyse et de la partie prédiction (v2),
> ajout de la prévision d'amplitude et du contexte (v2.1),
> puis de l'order flow, du panier de cryptos et des marges d'erreur honnêtes (v2.2).
> Fil conducteur : **trop de data tue la data**, **une mesure qu'on ne peut
> pas rejouer en backtest ne vaut rien**, et **un chiffre sans marge d'erreur
> n'est pas un résultat**.

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

## 6. Plus de données, moins d'illusions (v2.2)

Le constat de la v2.1 était économique : l'avantage ne couvre pas les frais.
Cette version s'attaque aux deux causes en amont — pas assez d'information
neuve, et pas assez de données pour la distinguer du bruit — et corrige au
passage une manière de mesurer qui flattait systématiquement les résultats.

### 6.1 Ce qui a été ajouté

| # | Ajout | Où |
|---|---|---|
| 5 | **Order flow** — nombre de trades et volume acheté à l'agressif, déjà présents dans chaque chandelier Binance et jetés jusqu'ici | `extraction.py`, `indicateurs.calculer_flux` |
| 6 | **Panier de cryptos** — un modèle entraîné sur plusieurs actifs empilés, coupés aux mêmes dates | `panier.py` |
| 8 | **Temps** — heure en sinus/cosinus, jour de la semaine, position dans le cycle de funding | `indicateurs.calculer_temps` |
| 9 | **Basis perp/spot** — la prime payée pour le levier, historique complet en une passe | `exogene.telecharger_perp` |
| 10 | **Taille effective** — toutes les marges d'erreur recalculées sur les observations réellement indépendantes | `modele.taille_effective` |
| 11 | **Validation croisée purgée** — 4 blocs chronologiques au lieu d'un seul | `modele.blocs_cv` |
| 14 | **Régimes** — justesse ventilée par quartile de volatilité, et le régime donné comme feature | `modele._table_regimes`, `indicateurs.calculer_regime` |

Le fichier analysé passe de 16 à 29 features : 8 indicateurs, 8 en
multi-timeframe, 3 d'order flow, 4 de temps, 2 de régime, 4 exogènes.

### 6.2 Deux choix de conception qui méritent d'être expliqués

**Le découpage par dates plutôt que par lignes.** C'était la condition
nécessaire au panier. Si chaque crypto était coupée à « 70 % de *ses* lignes »,
le bloc de test de l'une couvrirait 2024 et celui de l'autre 2019 : on
comparerait des marchés différents, et surtout l'entraînement de la première
contiendrait le test de la seconde — une fuite pure et simple.

Les frontières sont donc repérées au 70ᵉ et au 85ᵉ centile du nombre de lignes
empilées, puis converties en dates et appliquées comme telles. On obtient à la
fois un vrai 70/15/15 en volume et une date de bascule unique. L'embargo
anti-fuite est devenu temporel du même coup, ce qui est plus juste : purger
« 5 lignes » n'a pas de sens quand les lignes de plusieurs cryptos alternent.

Sur une seule crypto, le résultat est identique à l'ancien découpage.

**Le rang glissant plutôt que le z-score.** Les features dont le niveau dépend
de l'actif (ATR, écart à la moyenne, funding, basis…) sont converties en rang
de percentile sur 720 périodes, calculé crypto par crypto. Deux raisons de ne
pas prendre le z-score classique : il reste sensible aux queues épaisses, très
présentes en crypto ; et surtout un z-score calculé sur tout l'historique
donnerait au modèle, dès la première bougie, une information sur la volatilité
des années suivantes. Le rang est causal par construction.

Les features déjà bornées (RSI, stochastique, ADX, %B, déséquilibre de flux)
ne sont **pas** normalisées : un RSI de 80 veut dire la même chose sur BTC et
sur un altcoin, et le rang lui ferait perdre ce sens absolu.

### 6.3 Le correctif qui change la lecture de tout le reste

La taille effective est l'ajout le plus inconfortable de cette version.

À l'horizon 5, cinq lignes consécutives décrivent en grande partie le même
morceau de futur ; dans un panier, deux cryptos à la même heure aussi (RSI de
BTC et d'ETH : corrélation mesurée de 0,83). La formule retenue règle les deux
cas d'un coup :

```
taille effective = nombre d'horodatages DISTINCTS ÷ horizon
```

Conséquence sur BTC 1h, horizon 5, bloc test :

| Mesure | Annoncé avant | Annoncé maintenant |
|---|---|---|
| Justesse globale (11 851 bougies) | 52,44 % | **52,44 % ± 2,01 %** |
| Précision au seuil 0,58 (455 signaux) | 60,88 % | **60,88 % ± 10,03 %** |

Le second chiffre ne prouve plus rien. Le rapport le dit maintenant
explicitement : « ce seuil est le plus flatteur, pas le plus solide ».

Cette correction se propage partout : à la règle du 1 écart-type qui choisit la
configuration (elle devient plus stricte, donc retient des modèles plus
simples), au choix du seuil conseillé, et au verdict global « l'écart à la
réponse constante dépasse-t-il sa marge ? ».

### 6.4 Ce que la validation croisée a révélé

Sur BTC 1h, horizon 5, les trois configurations candidates :

```
configuration 1 : 0.5314 / 0.5642 / 0.5536 / 0.5422   →  0.5479 ± 0.0142
configuration 2 : 0.5210 / 0.5606 / 0.5514 / 0.5391   →  0.5430 ± 0.0171
configuration 3 : 0.5167 / 0.5548 / 0.5451 / 0.5360   →  0.5382 ± 0.0162
```

L'écart **entre blocs** (±0,014) est cinq fois plus grand que l'écart **entre
configurations** (0,0097 du premier au dernier). Le réglage automatique
choisissait donc sur du bruit — ce que la version précédente soupçonnait sans
pouvoir le montrer. C'est maintenant mesuré à chaque entraînement.

### 6.5 Résultats mesurés (BTC 1h, horizon 5, objectif `direction_nette`)

Toutes les lignes portent sur le même bloc de test de BTC, même découpage,
même modèle (XGBoost). L'AUC « cv » est la moyenne des 4 blocs de validation
croisée ; l'AUC « test » et la justesse portent sur des données jamais vues.

| Configuration | features | AUC cv | AUC test | Justesse test |
|---|---|---|---|---|
| 8 indicateurs seuls | 8 | 0.5422 | 0.5281 | 50,65 % ± 2,01 |
| contexte v2.1 (multi-timeframe + funding) | 16 | 0.5395 | 0.5269 | 51,17 % ± 2,01 |
| **contexte v2.2** (+ order flow, temps, régime, basis) | 29 | **0.5479** | **0.5323** | **52,44 % ± 2,01** |
| panier de 11 cryptos, contexte v2.2 | 29 | 0.5477 | 0.5169 | 51,95 % ± 2,15 |

Ce qu'il faut en retenir, sans enjoliver :

**Les nouvelles familles de features apportent quelque chose.** L'AUC croisée
passe de 0.5395 à 0.5479 et la justesse de 51,17 % à 52,44 %. Chaque écart pris
isolément tient dans la marge (±2 points), mais les deux mesures — croisée et
test — vont dans le même sens, ce qui est le minimum exigible.

L'importance par permutation (BTC 1h, h=5, 5 répétitions) est plus nette
encore : **trois des six premières features sont des nouveautés de la v2.2**.

```
Stoch_K             +0.01428
RSI_14              +0.00401
Heure_Cos           +0.00263   ← temps          (v2.2)
BB_Position_MTF     +0.00233
BB_Position         +0.00229
Flux_Desequilibre   +0.00185   ← order flow     (v2.2)
Regime_Volatilite   +0.00101   ← régime         (v2.2)
Stoch_K_MTF         +0.00098
   …
OBV_Pct             -0.00034
ADX_14_MTF          -0.00089
Funding_Rate        -0.00104   ← nuisible
```

L'heure du jour se place 3ᵉ, devant tout le contexte multi-timeframe. À
l'inverse le funding rate, mis en avant en v2.1 comme « la seule information
vraiment neuve », arrive dernier et dégrade légèrement le modèle.

**Le panier n'améliore pas le résultat — et il explique pourquoi.** Quatre
tailles de panier ont été entraînées, mêmes features, même objectif, même
horizon :

| Panier | cryptos | lignes | AUC cv | écart entre blocs | **taille effective du test** |
|---|---|---|---|---|---|
| BTC seul | 1 | 79 006 | 0.5479 | 0.0328 | **2 370** |
| BTC + ETH | 2 | 158 012 | 0.5512 | 0.0215 | **2 370** |
| 7 cryptos | 7 | 496 648 | 0.5514 | 0.0420 | **2 129** |
| 11 cryptos | 11 | 691 440 | 0.5477 | 0.0382 | **2 074** |

Multiplier les lignes par **8,75** fait **baisser** la taille effective du test.
C'est contre-intuitif et parfaitement logique : onze cryptos à la même heure
bougent ensemble, ce sont onze lignes mais une seule observation. Le panier
ajoute du volume, pas de l'information — ce que la formule de la section 6.3
annonçait, et que la mesure confirme.

L'AUC croisée le confirme aussi : 0.5479 pour BTC seul, 0.5477 pour onze
cryptos. Sur une fenêtre de test strictement commune (à partir du 2025-06-29,
10 372 bougies de BTC), le modèle de panier fait 51,95 % ± 2,15 contre 52,27 %
± 2,15 pour le modèle propre à BTC. Il est nettement plus sélectif — 224
signaux à 60,71 % contre 4 745 à 53,13 % — mais avec une marge de ±14,30 points
sur ces 224 signaux, ce beau chiffre ne prouve rien.

Le panier n'est donc pas la solution telle quelle. Il devient en revanche la
condition nécessaire de la **cible cross-sectionnelle** (piste 4) : prédire
l'écart d'une crypto à la médiane du panier retire précisément le mouvement
commun, c'est-à-dire la part imprévisible ET la source de cette corrélation qui
annule le gain de données. Toute la plomberie — alignement des dates,
normalisation entre actifs, features communes — est désormais en place.

### 6.6 Ce que le correctif de marge fait aux résultats de la v2.1

Les chiffres annoncés en 5.3 n'étaient pas faux, ils étaient mal encadrés.
Relus avec la taille effective, à comparer à la baseline de 51,02 % :

| Résultat v2.1 | Signaux | Relecture | Verdict |
|---|---|---|---|
| h=3 `direction` — 57,58 % | 1 016 | 57,58 % ± 5,26 | **tient** |
| h=3 `direction_nette` — 60,91 % | 330 | 60,91 % ± 9,12 | **tient de justesse** |
| h=3 `barriere` — 53,48 % | 1 653 | 53,48 % ± 4,16 | dans le bruit |
| h=12 `amplitude` — 31,79 % (hasard 20 %) | 582 | 31,79 % ± 13,11 | **dans le bruit** |

Deux des quatre résultats mis en avant en v2.1 ne survivent pas — dont celui
qui était présenté comme le meilleur du projet. C'est désagréable et c'est
précisément l'intérêt de la correction : elle coûte deux résultats et elle
évite d'engager de l'argent derrière.

---

## 7. Lisibilité et contrôle (v2.3)

Trois demandes, qui se sont révélées être trois facettes du même problème :
**on ne voyait pas ce que le modèle faisait**. Ni sur quoi il s'appuyait, ni ce
qu'il essayait comme réglages, ni où il en était pendant qu'il tournait.

### 7.1 Le curseur d'utilité minimale

À la fin de chaque entraînement, l'utilité de chaque feature est mesurée par
permutation et enregistrée dans les métadonnées. Un curseur sur la page Modèle
permet de réentraîner en ne gardant que les features au-dessus d'un seuil.

**Le vrai sujet n'était pas le curseur, c'était le bloc de mesure.** L'importance
par permutation existait déjà, mais calculée sur le TEST. S'en servir pour
choisir ses features, puis annoncer une performance sur ce même test, c'est du
surapprentissage de sélection : le chiffre obtenu serait flatteur et faux. La
mesure qui pilote le curseur est donc faite sur la seconde moitié de la
validation — celle que le modèle brut n'a vue ni pour apprendre, ni pour décider
quand s'arrêter. Le graphique de la page Visualisation reste sur le test : il
diagnostique, il ne décide pas.

Trois détails qui font la différence à l'usage :

* le nombre de features conservées s'affiche **avant** de lancer l'entraînement.
  Un curseur dont on ne découvre l'effet qu'après vingt minutes de calcul n'est
  pas un réglage, c'est une loterie ;
* une feature écartée **garde sa dernière note connue**. Sans cela, baisser le
  curseur ne pourrait jamais faire revenir ce qu'il a enlevé, et le filtre
  serait un aller sans retour ;
* une feature jamais mesurée est **conservée** — le doute profite à la donnée.

Mesuré sur BTC 1h, horizon 5, objectif `direction_nette` :

| features | AUC test | justesse test |
|---|---|---|
| 29 (toutes) | 0.5323 | 51,82 % ± 2,06 |
| 9 (utilité ≥ 0.0005) | 0.5335 | 52,66 % ± 2,05 |

Vingt features en moins, 0,84 point de justesse en plus — moins de la moitié de
la marge d'erreur. **Le filtre ne rend pas le modèle meilleur, il le rend
lisible**, et c'est déjà beaucoup : un modèle à 9 entrées se diagnostique, un
modèle à 29 se subit.

Le classement lui-même est plus instructif que le gain. Sur BTC, `RSI_14`
(+0.0146) et `Stoch_K` (+0.0141) écrasent tout, et **14 features sur 29 ont une
utilité nulle ou négative** — dont l'intégralité du contexte multi-timeframe
sauf `BB_Position_MTF`, et les trois features de calendrier. Sur BNB au même
horizon, mesuré sur le test, le classement s'inverse : le calendrier domine et
`BB_Position_MTF` devient la plus nuisible. Deux actifs, deux classements
incompatibles — ce qui est en soi le résultat le plus important de cette
section : **ces notes ne se transportent pas d'un actif à l'autre.**

### 7.2 La profondeur de recherche

Les trois configurations écrites à la main couvrent un coin minuscule de
l'espace des réglages. Deux modes plus longs (18 et 40 configurations) y
ajoutent des tirages aléatoires — pas un balayage, qui ferait 1,7 million de
combinaisons. Les trois références restent toujours dans le lot, et les
candidates sont triées de la plus prudente à la plus souple pour que la règle du
1 écart-type continue de retenir la plus sobre.

| mode | configurations | durée | AUC cv | AUC test | justesse test |
|---|---|---|---|---|---|
| Rapide | 3 | 34 s | 0.5587 | 0.5323 | 51,82 % ± 2,06 |
| Approfondie | 18 | 140 s | 0.5526 | 0.5342 | 51,99 % ± 2,06 |

**Quatre fois plus de temps pour 0,17 point de justesse**, soit vingt fois moins
que la marge d'erreur. Le résultat était prévisible et il a été mesuré quand
même, parce que « l'entraînement me semble trop court » est une inquiétude
légitime qui méritait un chiffre plutôt qu'une opinion.

Ce que ce chiffre dit vraiment : le facteur limitant de ce projet n'est pas la
puissance de calcul, c'est la **quantité d'information dans les données**. Un
boosting sur 75 000 lignes et 29 colonnes est un problème petit pour une machine
moderne, et aucun réglage d'hyperparamètre ne crée de l'information absente.
C'est la même conclusion que le panier (section 6.5), par un autre chemin.

### 7.3 Le suivi en direct et l'arrêt coopératif

Un module `suivi.py` sert de boîte aux lettres entre le thread de calcul et
l'interface : le code métier publie des événements (progression, courbe
d'apprentissage, configuration évaluée, utilité mesurée), quelqu'un les consomme
ou personne ne les consomme. `modele.py` ne connaît ni Tkinter ni matplotlib —
et en ligne de commande, où personne n'écoute, la publication ne coûte rien.

Trois traitements longs y sont branchés : l'entraînement de classification, le
walk-forward (une douzaine d'entraînements complets) et la régression
d'amplitude (un modèle par quantile). `amplitude.py` réutilise directement les
callbacks de `modele.py` plutôt que de redupliquer la gestion des trois
librairies de boosting.

Trois choix de conception :

* **l'affichage est découplé du flux, et il a fallu trois niveaux.** Le premier
  jet — redessiner au plus huit fois par seconde, et seulement si quelque chose
  a changé — ne suffisait pas : mesuré, il coûtait **407 ms par image** pour un
  budget de 125 ms. Le coupable était le réflexe `axe.clear()` puis retracé
  complet, légende comprise, à chaque image.

  Le tracé est donc devenu **incrémentiel** : les objets matplotlib sont créés
  une fois et alimentés par `set_data`, la légende et les graduations ne sont
  refaites qu'au changement d'ajustement, et le panneau des configurations n'est
  retracé qu'à l'arrivée d'une configuration. Résultat : **1,8 ms** par mise à
  jour.

  Restaient 160 ms de rendu matplotlib pur, et jusqu'à 250 ms quand les seize
  cœurs sont pris par XGBoost. Le test d'intégration a montré ce que cela
  produisait vraiment : **un `update()` du thread graphique bloquant 29
  secondes** au milieu d'un entraînement. La boucle de service de l'application
  tourne toutes les 80 ms ; quand une image coûte 250 ms, le thread graphique ne
  fait plus que dessiner et ne rend jamais la main. Le symptôme était exactement
  celui qu'on voulait supprimer, et il avait été introduit par la fonctionnalité
  censée le supprimer.

  La réponse est le **blitting** : le fond de la figure (axes, graduations,
  grille, légende, panneau de droite) est capturé une fois, et chaque image se
  contente de le repeindre puis de tracer les deux courbes par-dessus. Le
  redessin complet ne subsiste que pour les changements de structure — nouvel
  ajustement, nouvelle configuration, données sorties du cadre — et les limites
  d'axes sont élargies par paliers avec 8 % de marge pour que ce cas reste rare.
  Résultat mesuré : **11,7 ms par image, zéro redessin complet** sur soixante
  mises à jour consécutives.

  S'y ajoute un garde-fou : la boucle mesure ce que son dernier rendu a coûté et
  s'espace d'autant, de façon à ne jamais prendre plus d'un cinquième du thread
  graphique. La cadence devient une propriété de la machine au lieu d'une
  constante optimiste. En pratique, 6 images/s pour 7 % du thread.

  Le test d'intégration, avant et après, sur le même entraînement de 18
  configurations :

  | | tours de boucle | latence médiane | pire latence | durée du calcul |
  |---|---|---|---|---|
  | avant | 707 | 11 ms | **167 s** | 190 s |
  | après | 3 745 | 0,3 ms | 1,4 s | **118 s** |

  Le calcul lui-même a gagné 38 % — le thread graphique ne lui vole plus ses
  cœurs. C'est le genre de gain qu'on ne va pas chercher en optimisant le
  modèle.

  Trois leçons qui valent au-delà de ce cas : une cadence d'affichage écrite en
  dur est un pari sur la machine de celui qui l'a écrite ; un affichage qui
  redessine tout à chaque image ralentit le calcul qu'il observe ; et un test
  d'intégration qui mesure la latence du thread graphique trouve ce qu'aucun
  test unitaire de la fenêtre ne pouvait trouver ;
* **l'arrêt est coopératif, jamais forcé.** Un `threading.Event` est consulté
  entre chaque étape ET dans les callbacks de XGBoost / LightGBM / CatBoost,
  donc à chaque arbre construit. Mesuré : une demande d'arrêt pendant une
  recherche de 40 configurations est honorée en moins d'une seconde, et le
  modèle précédemment enregistré reste intact puisque l'exception remonte avant
  la sauvegarde ;
* **les callbacks sont détachés avant `joblib.dump`.** Un callback garde une
  référence au moniteur, donc à des objets `threading.Event` et `queue.Queue`
  non sérialisables. Sans ce nettoyage, la sauvegarde échouerait — et uniquement
  quand l'interface est ouverte, ce qui est le pire type de bug.

La courbe d'apprentissage est la partie la plus utile : voir l'écart se creuser
entre apprentissage et validation, et le trait vert de l'arrêt anticipé tomber
exactement au minimum, rend concret ce que « surapprentissage » veut dire.

### 7.4 Deux bugs trouvés en vérifiant

**Le basis était faux sur toute la période de test.** La série du contrat
perpétuel s'arrêtait au 2025-07-03 alors que le spot allait jusqu'au 2026-08-31,
et l'alignement reportait en avant sa dernière valeur pendant quatorze mois. Le
`Basis` cessait alors de mesurer la prime du perpétuel pour mesurer « de combien
le prix a bougé depuis juillet 2025 » — une valeur qui dérive avec le marché,
saturée à la borne du clip sur **11 % des lignes**, et parfaitement corrélée au
niveau du prix. C'est-à-dire exactement le type de fuite que la liste blanche
des features est censée interdire.

Le report en avant est désormais **borné des deux côtés** et limité à trois
périodes : une série tronquée produit du vide, jamais une valeur inventée. Le
99ᵉ centile de `|Basis|` passe de 5,00 (la borne du clip) à 0,18 % — un ordre de
grandeur économiquement crédible. Le téléchargement réessaie aussi trois fois
avant d'abandonner, et prévient quand la série s'arrête trop tôt.

La leçon est générale : **une série tronquée est plus dangereuse qu'une série
absente**, parce qu'elle a l'air complète. Un `ffill()` sans borne est un
générateur silencieux de fausses features.

**Le résumé d'un panier figeait l'interface.** Décrire un panier de vingt-cinq
cryptos chargeait vingt-cinq fichiers complets dans le thread graphique. Un
`stockage.apercu_tableau` ne lit plus que la colonne d'index du cache Parquet et
mémorise le résultat : 174 ms pour six cryptos, contre plusieurs secondes.

---

## 8. Pistes restantes

1. **Frais makers** : refaire les mesures à 0,04 % d'aller-retour plutôt que
   0,30 %. Reste le seul levier qui change l'ordre de grandeur du problème —
   l'avantage brut mesuré (~0,11 % par trade) couvre 0,04 % mais pas 0,30 %.
2. **Sortie sur barrière plutôt qu'à horizon fixe** : la triple barrière est
   déjà la cible d'apprentissage, mais le backtest clôture toujours à
   l'horizon. Encaisser au take-profit augmente l'amplitude par trade sans
   augmenter les frais — donc agit sur le bon terme de l'équation.
3. **Open interest accumulé** : toujours écarté sous 60 % de couverture.
   Quelques mois de téléchargements réguliers suffiraient.
4. **Cible cross-sectionnelle** : prédire la performance d'une crypto
   *relative à la médiane du panier* plutôt qu'en absolu. Le mouvement commun
   du marché est essentiellement imprévisible ; le retirer de la cible devrait
   dégager un signal plus net. Le panier fournit désormais toute la
   plomberie nécessaire.
5. **Meta-labelling** : un second modèle qui apprend quand faire confiance au
   premier, plutôt que d'améliorer le premier.
