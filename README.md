# Crypto Lab

Pipeline complet d'analyse et de prédiction de cryptomonnaies.

Le modèle répond à **une seule question** : « dans X périodes (X entre 1 et 24),
le prix sera-t-il plus haut ou plus bas qu'aujourd'hui ? » — accompagnée d'un
**niveau de confiance** exploitable dans les deux sens, pour ne retenir que les
signaux dont le modèle est sûr.

```
extraction  →  indicateurs  →  modèle  →  backtest
  OHLCV        8 features      hausse /   simulation
               + 24 cibles      baisse     de trades
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

Six onglets, dans l'ordre : Extraction → Analyse → Prédiction → Évaluation →
Visualisation → Backtest. Chaque écran comporte des pastilles ⓘ et, pour les
étapes importantes, un bouton « ❔ » qui ouvre l'explication détaillée.

### Ligne de commande

```bash
python GatherData.py BTC ETH --intervalle 1h --debut 2022-01-01
python GatherData.py --top 5 --intervalle 4h        # Top N CoinGecko d'un coup

python CryptoAnalysis.py                             # analyse tout le dossier
python CryptoAnalysis.py BTC_1h                      # ou un fichier précis

python Predict.py BTC_1h --horizon 12 --modele XGBoost --seuil 0.60
python Predict.py BTC_1h --predire-seulement         # réutilise le modèle existant
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

## Ce que l'entraînement décide tout seul

L'interface n'expose que **quatre réglages** : crypto, modèle, horizon, seuil de
confiance. Tout le reste est automatique et documenté dans le bouton
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

Le backtest se lance par défaut sur le **bloc test uniquement** : sur les
données d'entraînement, le modèle rejoue ce qu'il a mémorisé et le résultat
n'a aucune valeur.

---

## Organisation du code

```
Crypto/
├── Dashboard.py            lance l'interface graphique
├── GatherData.py           extraction, en ligne de commande
├── CryptoAnalysis.py       analyse, en ligne de commande
├── Predict.py              entraînement + prédiction, en ligne de commande
│
├── crypto_lab/
│   ├── config.py           chemins, 8 indicateurs, 24 cibles, réglages
│   ├── ressources.py       dimensionnement RAM / CPU
│   ├── stockage.py         lecture/écriture Excel + cache Parquet et mémoire
│   ├── extraction.py       Binance, Yahoo, CoinGecko
│   ├── indicateurs.py      les 8 indicateurs et les variations futures
│   ├── modele.py           entraînement, calibration, prédiction
│   ├── backtest.py         simulateur long / short
│   └── interface/
│       ├── app.py          fenêtre principale, navigation, tâches de fond
│       ├── composants.py   briques de widgets
│       ├── theme.py        palette et styles
│       ├── textes.py       aides et explications
│       └── pages/          une page par étape
│
├── data_crypto/            OHLCV brut
├── analysis_crypto/        prix + 8 indicateurs + 24 variations
├── models/                 modèles (.joblib) et métadonnées (.json)
├── prediction_crypto/      signaux générés
└── visualizations/         exports PNG
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
