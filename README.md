# Crypto Lab

Pipeline complet d'analyse et de prédiction de cryptomonnaies.

Le modèle répond à **deux questions** :

- **dans quel sens ?** — « dans X périodes (X entre 1 et 24), le prix sera-t-il
  plus haut ou plus bas ? », avec un **niveau de confiance** exploitable dans
  les deux sens ;
- **de combien ?** — l'**amplitude** attendue du mouvement et son intervalle à
  80 %, puis l'**espérance de gain** qui combine les deux et retranche les frais.

```
extraction  →  indicateurs  →  direction  →  amplitude  →  backtest
  OHLCV        8 features +    hausse /     combien +     simulation
  + funding     contexte 4h     baisse      espérance     de trades
               + 24 cibles
```

---

## Installation

```bash
cd Crypto
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

### Interface graphique (recommandé)

```bash
python Dashboard.py
```

Sept onglets, dans l'ordre : Extraction → Analyse → Prédiction → Amplitude →
Évaluation → Visualisation → Backtest. Chaque écran comporte des pastilles ⓘ
et, pour les étapes importantes, un bouton « ❔ » qui ouvre l'explication
détaillée.

### Ligne de commande

```bash
python GatherData.py BTC ETH --intervalle 1h --debut 2022-01-01
python GatherData.py --top 5 --intervalle 4h        # Top N CoinGecko d'un coup
python GatherData.py BTC --exogene                  # funding rate + open interest

python CryptoAnalysis.py                             # analyse tout le dossier
python CryptoAnalysis.py BTC_1h                      # ou un fichier précis
python CryptoAnalysis.py BTC_1h --sans-contexte      # les 8 indicateurs seuls

python Predict.py BTC_1h --horizon 3 --modele XGBoost --seuil 0.60
python Predict.py BTC_1h --objectif direction_nette  # n'apprend que le rentable
python Predict.py BTC_1h --walk-forward              # évaluation hors échantillon
python Predict.py BTC_1h --predire-seulement         # réutilise le modèle existant

python Amplitude.py BTC_1h --horizon 3 --esperance   # amplitude + score de décision
```

---

## Les 8 indicateurs

Un seul principe : **aucune feature n'est un niveau de prix**. Chacune est soit
bornée, soit exprimée en proportion du prix ou du volume — sans quoi le modèle
mémoriserait « BTC vaut 60 000 », obtiendrait des scores parfaits sur le passé
et s'effondrerait sur le présent.

| Colonne | Ce qu'elle mesure | Échelle |
|---|---|---|
| `RSI_14` | Momentum, sur-achat / sur-vente | 0 – 100 |
| `Stoch_K` | Position du prix dans son range récent | 0 – 100 |
| `MACD_Hist_Norm` | Accélération de la tendance (÷ prix) | ~0 centré |
| `Dist_SMA_50` | Écart relatif à la moyenne 50 périodes | % |
| `BB_Position` | Position dans les bandes de Bollinger | 0 – 1 |
| `ATR_Pct` | Niveau de volatilité (÷ prix) | % |
| `ADX_14` | Force de la tendance, sans son sens | 0 – 100 |
| `OBV_Pct` | Flux de volume net sur 20 périodes | −1 – +1 |

Chacun couvre un axe d'information distinct : momentum, position, tendance,
volatilité, volume. C'est volontairement peu — la version précédente en
comptait environ 120 et le modèle apprenait surtout du bruit.

## Le contexte : 10 colonnes de plus, pas 100

Deux familles s'ajoutent automatiquement quand elles sont disponibles.

**Multi-timeframe** (suffixe `_MTF`) — les 8 mêmes indicateurs calculés sur
l'intervalle supérieur (4h pour du 1h). Un modèle horaire ne voit que
l'agitation horaire ; il ignore s'il se trouve dans une tendance 4h haussière
ou dans un retournement.

> **Anti-fuite.** Une bougie 4h étiquetée 00:00 couvre 00:00→04:00 : sa clôture
> n'est connue qu'à 04:00. Elle est donc décalée d'une bougie entière avant
> d'être rapprochée de l'index horaire. Sans ce décalage, le modèle verrait dès
> 00:00 une information contenant les quatre heures qu'on lui demande de prédire.

**Exogènes** (Binance Futures) — `Funding_Rate`, `Funding_Cumul`, `OI_Variation`.
Tout le reste du fichier dérive du même OHLCV : ce sont des transformations
d'une seule information. Le funding et l'open interest viennent du
positionnement réel sur les dérivés — le seul apport d'information vraiment
neuve disponible gratuitement.

Une colonne n'est retenue que si elle couvre au moins 60 % des lignes.
L'historique de funding remonte à 2019 et sert immédiatement ; **l'open
interest public de Binance ne remonte qu'à 30 jours** et est donc écarté au
début. Chaque `--exogene` complète le précédent, la couverture s'étend d'elle-même.

### Ce que le contexte apporte, mesuré

Sur BTC 1h, à données et découpage identiques, 18 features contre 8 :

| Horizon | AUC (8 → 18) | Plafond | Précision au seuil conseillé |
|---|---|---|---|
| 3  | 0.5363 → 0.5345 | 0.601 → 0.610 | 56.56 % → **57.58 %** |
| 6  | 0.5345 → 0.5344 | 0.565 → 0.551 | 54.65 % → **58.14 %** |
| 12 | 0.5179 → 0.5030 | 0.536 → 0.572 | 52.59 % → **54.74 %** |

L'AUC ne bouge pas (les écarts sont inférieurs à sa marge d'erreur de ±0.0076).
Autrement dit : le contexte n'améliore pas le classement d'ensemble, il rend le
modèle **plus tranché là où il a raison** — exactement ce qu'on attend d'un
outil de filtrage.

À relativiser : chaque écart pris isolément tient dans la marge d'erreur
(±1.6 point sur un millier de signaux). Ce sont les **trois horizons allant
dans le même sens** qui rendent le résultat crédible, pas un chiffre seul.
L'importance par permutation reste le juge de paix : `Stoch_K` écrase tout
(+0.030), les colonnes de contexte plafonnent à +0.0006.

## Les 24 colonnes `variation_x`

```
variation_x = (Close[t+x] − Close[t]) / Close[t] × 100
```

Valeurs **futures**, en pourcentage : elles servent à construire la cible et ne
sont jamais des entrées du modèle. `variation_12 > 0` signifie « douze périodes
plus tard, le prix était plus haut ».

Les dernières lignes du fichier ont des variations vides — le futur n'est pas
encore arrivé. Elles sont conservées : ce sont précisément les bougies sur
lesquelles on veut une prédiction.

---

## Les quatre objectifs

Le menu « Objectif » change la **question posée**, pas la machinerie. Découpage,
embargo, calibration, seuil de confiance restent identiques.

| Objectif | Ce qu'il apprend |
|---|---|
| `direction` | Le signe de `variation_X`. Le choix par défaut et la référence. |
| `direction_nette` | Idem, mais n'apprend **que** sur les mouvements dépassant les frais (0.30 %). |
| `barriere` | Touche-t-on le take-profit (+1 ATR) avant le stop-loss (−1 ATR) ? |
| `amplitude` | 5 classes d'égale fréquence, graduées en variation rapportée à l'ATR. |

> **Le piège de `direction_nette`.** Il n'apprend que sur les gros mouvements
> mais doit être **évalué sur toutes les bougies** : en situation réelle, on ne
> sait pas d'avance si le mouvement sera grand. Mesuré sur le seul
> sous-ensemble filtré, il annonçait 63.5 % de précision alors que les mêmes
> signaux rejoués en backtest donnaient 34 % de trades gagnants. L'écart venait
> entièrement de ce biais de sélection, corrigé depuis (`Tache.construire_evaluation`).

> **Le piège de `amplitude`.** Avec des bornes fixes en multiples d'ATR, la
> classe « neutre » pesait 60 à 68 % des bougies. La calibration — qui fait
> correctement son travail en ramenant chaque probabilité à la fréquence réelle
> de sa classe — rendait alors « neutre » systématiquement majoritaire : le
> modèle répondait « neutre » 100 % du temps, pour exactement le score de la
> réponse constante. Les bornes sont désormais les **quintiles** de la variation
> normalisée : chaque classe pèse 20 %, et l'argmax redevient une décision.

### Comparaison mesurée (BTC 1h, bloc test, 75 639 bougies)

| Horizon | Objectif | AUC | Justesse | Plafond | Seuil | Signaux | Précision | Gain |
|---|---|---|---|---|---|---|---|---|
| 3 | direction | 0.5345 | 52.78 % | 0.610 | 0.60 | 1 016 | 57.58 % | +4.80 |
| 3 | **direction_nette** | 0.5349 | 51.82 % | 0.617 | 0.60 | 330 | **60.91 %** | **+9.08** |
| 3 | barriere | 0.5167 | 51.08 % | 0.559 | 0.55 | 1 653 | 53.48 % | +2.39 |
| 3 | amplitude | 0.5407 | 22.03 % | 0.292 | 0.28 | 532 | 28.01 % | +5.97 |
| 12 | direction | 0.5030 | 50.52 % | 0.572 | 0.55 | 886 | 54.74 % | +4.22 |
| 12 | direction_nette | 0.4996 | 51.22 % | 0.548 | — | 0 | — | — |
| 12 | barriere | 0.5026 | 49.13 % | 0.555 | 0.52 | 889 | 51.41 % | +2.28 |
| 12 | **amplitude** | 0.5293 | 22.32 % | 0.356 | 0.30 | 582 | **31.79 %** | **+9.47** |

`direction_nette` à l'horizon 3 est le meilleur résultat du projet. Pour
`amplitude`, comparer 31.79 % non à 50 % mais au hasard à **20 %** : c'est 1.6 ×
le hasard.

## Ce que l'entraînement décide tout seul

L'interface n'expose que **cinq réglages** : crypto, modèle, objectif, horizon,
seuil de confiance. Tout le reste est automatique et documenté dans le bouton
« ❔ Ce qui est fait automatiquement » :

- **Découpage** 70 / 15 / 15, strictement chronologique, avec un embargo égal à
  l'horizon entre les blocs (sinon la cible, qui regarde vers le futur, fuite
  d'un bloc à l'autre).
- **Équilibrage des classes** — uniquement si l'une dépasse 55 %.
- **Hyperparamètres** — trois configurations testées, la meilleure sur la
  validation est retenue.
- **Early stopping** sur le logloss (plus stable que l'AUC, trop bruitée).
- **Calibration** des probabilités par tranches d'au moins 250 observations :
  impossible d'annoncer « 95 % de confiance » sur la foi de trois bougies.
- **Ressources** — la RAM libre est convertie en histogrammes plus fins et en
  parallélisme au lieu de rester inutilisée (voir `crypto_lab/ressources.py`).

## Lire les résultats

Deux choses comptent, dans cet ordre :

1. **L'AUC sur le test**, comparée à la baseline « toujours la même réponse ».
   En crypto, 0.52 à 0.56 est un signal réel ; au-delà de 0.75, chercher une
   fuite de données.
2. **La table des seuils de confiance** (onglet Évaluation). Si la précision
   monte quand le seuil monte, le score de confiance est fiable — c'est
   exactement ce qui permet de filtrer les signaux.

## Le plafond de confiance

Chaque modèle a un **plafond** : la confiance la plus élevée qu'il ait produite.
Il est affiché après chaque entraînement, et le curseur de seuil est
automatiquement placé sur la valeur qui donne la meilleure justesse. Régler le
seuil au-dessus du plafond ne retient simplement aucun signal.

Mesuré sur BTC 1h :

| Horizon | AUC test | Plafond | Seuil conseillé | Justesse au seuil |
|---|---|---|---|---|
| **3** | 0.534 | 0.595 | 0.56 | **55.97 %** (vs 52.00 %) |
| **6** | 0.535 | 0.574 | 0.55 | 55.17 % |
| 12 | 0.519 | 0.541 | 0.54 | 53.16 % (vs 51.24 %) |

**Les horizons courts (3 à 6) sont nettement meilleurs** que 12 ou 24 en 1h.

La confiance est une *probabilité*, pas une note de qualité : « 56 % » signifie
« correct 56 fois sur 100 », et c'est vérifié sur des données jamais vues. Pour
afficher 0.80 il faudrait avoir raison 80 % du temps sur la direction du
Bitcoin — aucun modèle honnête n'y parvient.

Attention aux intervalles longs : en 1d on ne dispose que de quelques milliers
de bougies, le bloc de calibration devient minuscule et la confiance mesurée
n'est plus que du bruit. Le calibrateur ramène alors les écarts vers 50 %
plutôt que d'afficher un chiffre flatteur mais faux (mesuré : 0.68 de confiance
annoncée pour 47 % de justesse réelle, avant correction).

Le backtest se lance par défaut sur le **bloc test uniquement** : sur les
données d'entraînement, le modèle rejoue ce qu'il a mémorisé et le résultat
n'a aucune valeur.

## Walk-forward : backtester tout l'historique honnêtement

Un backtest lancé sur toute la période inclut les données d'apprentissage. Le
rendement y est spectaculaire et ne veut rien dire — mesuré sur BTC 1h horizon
3, sans frais :

| Période simulée | Rendement | Buy & Hold |
|---|---|---|
| entraînement | **+1 273 %** | +183 % |
| test | +28 % | −24 % |
| **walk-forward** | **+18 %** | +118 % |

Le bouton **📏 Walk-forward** (onglet Prédiction) supprime ce biais : l'historique
est coupé en douze tranches, le modèle est entièrement réentraîné sur tout le
passé disponible avant chaque tranche, et ne prédit que la suivante — jamais
vue. Le fichier produit se termine par `_wf` et peut être simulé **sur toute sa
durée** sans fausser le résultat.

Bénéfice supplémentaire : on voit la **stabilité**. Sur BTC 1h horizon 3, AUC
hors échantillon de 0.5435 ± 0.0141 avec **12 fenêtres sur 12 au-dessus du
hasard** — bien plus informatif qu'une moyenne unique.

Quand une simulation déborde sur les données d'entraînement, la page Backtest
grise les zones concernées et affiche un avertissement chiffré.

---

## Amplitude : prédire combien, pas seulement dans quel sens

Les 24 colonnes `variation_x` contenaient déjà l'amplitude — il n'y a aucune
donnée nouvelle à télécharger.

### La volatilité, seul endroit où il y a du signal solide

Un modèle sur `|variation_X|`. La direction est quasi imprévisible, l'amplitude
beaucoup moins : le clustering de volatilité est le fait stylisé le plus robuste
de la finance de marché.

> **La perte compte.** L'erreur absolue estimerait la **médiane**, or
> `|variation|` est très asymétrique (médiane 0.44 %, moyenne 0.79 % sur BTC 1h
> à l'horizon 3). Un modèle médian sous-estimerait l'amplitude et fausserait
> l'espérance de gain vers le bas. L'erreur au carré viserait la moyenne mais
> serait dominée par trois krachs. On utilise donc une **perte de Tweedie**,
> faite pour les cibles positives et asymétriques. Le passage de l'une à l'autre
> a fait passer le R² de **−0.018 à +0.087**.

**Comment lire le R².** Il paraît petit et il est pourtant bon : on ne prédit
pas une quantité déterministe mais l'**échelle d'un tirage aléatoire**. Même en
connaissant exactement la volatilité, `|variation|` resterait un tirage unique
autour d'elle. L'écran affiche donc le plafond théorique atteignable.

| Horizon | R² modèle | R² ATR naïf | Plafond | Part atteinte |
|---|---|---|---|---|
| 3  | **0.0874** | 0.0742 | 0.160 | 55 % |
| 6  | **0.0679** | 0.0528 | — | — |
| 12 | 0.0664 | 0.0645 | — | jeu égal |
| 24 | 0.0104 | 0.0442 | 0.104 | 10 % ⚠️ |

À comparer au R² de la **direction**, qui tourne autour de zéro. Aux horizons
longs en revanche, l'ATR dilaté suffit : inutile de complexifier.

### L'intervalle Q10–Q90

Trois modèles estiment les quantiles 10 / 50 / 90. La sortie n'est plus une
valeur mais un intervalle : « dans 3 h, entre −0.73 % et +0.68 %, 8 fois sur 10 ».

La métrique qui compte est la **couverture** — mesurée à **81.3 %** pour 80 %
attendus sur BTC 1h horizon 3, donc un intervalle honnête. (À l'horizon 24 :
85.1 %, trop large.) Gain de perte pinball face au quantile constant : +4.9 % sur
Q10, +4.2 % sur Q90, mais seulement +0.2 % sur Q50 — la médiane est proche de
zéro et pratiquement imprévisible, ce qui est une autre façon de retrouver le
problème de la direction.

### L'espérance de gain

```
espérance = (2 × P(hausse) − 1) × amplitude attendue − frais
```

Une position longue rapporte +A avec la probabilité p et −A avec 1−p : son
espérance vaut (2p−1)·A. C'est le score qui départage enfin un signal à 55 % sur
un mouvement de 3 % et un signal à 60 % sur 0.2 %.

Mesuré sur BTC 1h horizon 24, bloc test, classé par espérance nette annoncée :

| Espérance nette ≥ | Signaux | Justesse | Gain réel moyen |
|---|---|---|---|
| −0.30 % (tout) | 11 343 | 51.02 % | −0.336 % |
| −0.10 % | 1 483 | 55.23 % | −0.152 % |
| 0.00 % | 721 | 55.76 % | −0.089 % |
| +0.10 % | 223 | 57.85 % | **+0.065 %** |
| +0.20 % | 46 | 69.57 % | **+0.930 %** |

**L'ordonnancement fonctionne** : sur six niveaux consécutifs, plus l'espérance
annoncée est élevée, plus la justesse et le gain réel montent. C'est le résultat
solide de ce tableau.

**La rentabilité n'est pas démontrée pour autant.** Sur 223 trades dont les
rendements ont un écart-type de l'ordre de 3 %, la marge d'erreur sur la moyenne
est d'environ 0.2 % — trois fois le gain affiché. Ce +0.065 % est compatible
avec zéro.

À l'horizon 3, aucun signal ne dépasse les frais (meilleure espérance nette :
−0.20 %). Ce n'est pas un défaut du modèle mais un constat économique :
l'amplitude croît en √h, les frais non.

## Taille de position

Le Backtest propose deux modes. En **fixe**, tout le capital part sur chaque
signal. En **proportionnel**, la mise suit l'avantage estimé et recule quand la
volatilité prévue monte : on n'égalise plus la mise entre deux trades mais le
**risque**.

Mesuré sur BTC 1h horizon 24, bloc test, frais réels de 0.30 % :

| Mode | Rendement | Drawdown |
|---|---|---|
| fixe, sans filtre | −21.7 % | −35.6 % |
| fixe + filtre espérance | −12.1 % | −29.0 % |
| proportionnel | −4.7 % | −13.9 % |
| **proportionnel + filtre** | **−3.7 %** | **−11.6 %** |

(Buy & Hold sur la même période : −23.2 %.) Chaque mécanisme réduit la perte et
le risque, et les quatre variantes battent le marché sur cette période
baissière. Aucune n'est rentable pour autant.

---

## Organisation du code

```
Crypto/
├── Dashboard.py            lance l'interface graphique
├── GatherData.py           extraction, en ligne de commande
├── CryptoAnalysis.py       analyse, en ligne de commande
├── Predict.py              entraînement + prédiction, en ligne de commande
├── Amplitude.py            régressions + espérance, en ligne de commande
│
├── crypto_lab/
│   ├── config.py           chemins, indicateurs, cibles, réglages
│   ├── ressources.py       dimensionnement RAM / CPU
│   ├── stockage.py         lecture/écriture Excel + cache Parquet et mémoire
│   ├── extraction.py       Binance, Yahoo, CoinGecko
│   ├── exogene.py          funding rate et open interest (Binance Futures)
│   ├── indicateurs.py      les 8 indicateurs, le contexte 4h, les variations
│   ├── cibles.py           les 4 objectifs d'apprentissage
│   ├── modele.py           entraînement, calibration, prédiction (classification)
│   ├── amplitude.py        régression quantile, volatilité, espérance de gain
│   ├── backtest.py         simulateur long / short, sizing proportionnel
│   └── interface/
│       ├── app.py          fenêtre principale, navigation, tâches de fond
│       ├── composants.py   briques de widgets
│       ├── theme.py        palette et styles
│       ├── textes.py       aides et explications
│       └── pages/          une page par étape
│
├── data_crypto/            OHLCV brut + EXO_*.xlsx (funding, open interest)
├── analysis_crypto/        prix + indicateurs + contexte + 24 variations
├── models/                 MODELE_* / REGRESSION_* (.joblib) + META_* (.json)
├── prediction_crypto/      signaux générés (…_wf, …_esperance, …_<objectif>)
└── visualizations/         exports PNG
```

### Nommage des fichiers produits

```
BTC_1h_h3_prediction.xlsx                direction (objectif par défaut)
BTC_1h_h3_barriere_prediction.xlsx       autre objectif
BTC_1h_h3_wf_prediction.xlsx             walk-forward
BTC_1h_h3_esperance_prediction.xlsx      direction × amplitude, backtestable
```

Les dossiers de données sont ignorés par git et régénérables à tout moment.

## Attentes réalistes

L'objectif n'est pas un rendement spectaculaire mais un **avantage faible et
stable**. Un modèle honnête sur de la direction crypto tourne autour de 0.52 à
0.56 d'AUC. Ce qui se juge vraiment, c'est :

- le gain de précision apporté par le filtrage sur la confiance ;
- la stabilité entre validation et test ;
- le résultat du backtest **après frais** — 0.1 % de frais par transaction
  suffit à effacer un avantage réel mais mince.
