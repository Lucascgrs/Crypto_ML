# Améliorations des modèles — refonte qualité

> Mise à jour : 2026-06-20. Document de suivi de la refonte visant à rendre les
> modèles **honnêtes et robustes** plutôt que faussement performants.

---

## 0. Le vrai problème (et pourquoi les rendements étaient trompeurs)

Le symptôme — « bon rendement sur tout l'historique, mauvais sur la fin, < 50 %
quel que soit le modèle » — venait d'une **fuite du prix dans les features**.

`build_features` ne retirait que l'OHLCV brut, mais **laissait passer des colonnes
qui SONT des niveaux de prix** : `SMA_20/50/200`, `MACD`, `MACD_Signal`, `MACD_Hist`,
et les lignes Ichimoku (`Ichi_Tenkan/Kijun/SpanA/SpanB`).

Conséquence : le modèle mémorisait des règles du type « si SMA_200 ≈ 60000 → … ».
Ça marche parfaitement sur les données vues (in-sample) et s'effondre dès que le prix
sort de la plage connue (out-of-sample = la période récente). Les « bons » rendements
historiques étaient donc une **illusion d'overfitting**, pas une vraie capacité prédictive.

**Vérification après correction** (BTC 1h, données OHLCV seules) :
AUC validation ≈ 0.516 vs AUC test ≈ 0.496 → l'écart massif a disparu. Le modèle ne bat
plus la baseline « classe majoritaire » : c'est la **réalité honnête** d'une prédiction
de direction horaire sur une seule crypto sans order-flow. Le travail ci-dessous vise à
donner au modèle les meilleures chances d'extraire un *vrai* petit edge.

---

## 1. ✅ Implémenté

### Anti-fuite du prix (priorité absolue)
- **`COLONNES_NON_FEATURES`** (constante unique dans `CryptoAnalysis.py`, importée par
  `Predict.build_features`) : liste canonique des colonnes interdites en feature
  (prix bruts, niveaux de prix, order-flow brut, artefacts).
- **MACD normalisé** (`MACD_Norm`, `MACD_Signal_Norm`, `MACD_Hist_Norm` = MACD / Close)
  et **distances Ichimoku** (`Dist_Tenkan/SpanA/SpanB`) → versions stationnaires.
  Les lignes brutes restent dans le fichier (pour les graphes) mais ne sont jamais des features.
- **Garde-fou automatique** : toute feature dont |corr avec Close| > 0.99 est retirée et
  signalée (`🛡️ Garde-fou anti-fuite`). Filet de sécurité contre tout oubli futur.

### Nouveaux indicateurs (tous stationnaires)
- **Momentum multi-horizons** : `Mom_{3,6,12,24,72,168}` + z-scores.
- **Oscillateurs** : Stochastique %K/%D, Williams %R, CCI, ROC, ADX/+DI/-DI.
- **Régime de volatilité** : vol réalisée multi-fenêtres, rang-percentile de l'ATR et de
  la largeur de Bollinger.
- **Volume / order-flow** : OBV normalisé, z-score du volume, **taker buy ratio**
  (pression acheteuse agressive), z-score du nombre de trades.
- **Position dans le range** (Donchian) : `Range_Pos_{24,72,168}`.

### Order-flow à l'extraction
- `GatherData.fetch_data_binance` conserve `Quote_Asset_Volume`, `Trades`,
  `Taker_Buy_Base`, `Taker_Buy_Quote` (microstructure). Compat Yahoo : si absentes, les
  features order-flow se désactivent proprement.

### Mode Portefeuille (multi-crypto)
- `GatherData.telecharger_top_n` : télécharge l'historique du Top N CoinGecko d'un coup
  (stablecoins/paires invalides ignorés).
- `PortefeuilleDatasetBuilder` : concatène les cryptos analysées en `MULTI_{int}_analyzed.xlsx`
  (index temporel, doublons d'horodatage tolérés via opérations positionnelles).
- Cible calculée **par crypto** (jamais à cheval entre deux actifs).
- Jeu de features **identique à une crypto seule** → le modèle `MULTI` est **portable** :
  on l'entraîne sur le pool, puis on l'applique à n'importe quelle crypto
  (case « 🧺 Prédire avec le modèle Portefeuille » → fichier de prédiction normal,
  backtestable comme d'habitude). Plus de données, plus de régimes, meilleure généralisation.
  Les modèles par crypto restent intacts.

### Entraînement
- **Rééquilibrage des classes** optionnel (`scale_pos_weight` / `class_weight` /
  `auto_class_weights`) selon le modèle.
- Sélection SHAP élargie (top 30) — calculée sur le **train uniquement** (déjà corrigé).

### Interface
- Page **Données** : « 📦 Télécharger l'historique du Top N » (Portefeuille).
- Page **Analyse** : « 🧺 Construire le dataset Portefeuille ».
- Page **Modèle** : case « ⚖️ Équilibrer les classes » ; `MULTI_*` apparaît dans la liste
  des cryptos analysées.
- Bulles d'aide (ⓘ) ajoutées pour chaque nouveauté.

---

## 2. 🔁 Procédure recommandée (à relancer)

Comme le jeu de features a changé, **il faut régénérer données, analyses et modèles** :

1. **Données → 📦 Télécharger l'historique du Top N** (ex. N = 5) — récupère l'order-flow.
2. **Analyse → 📦 Analyser TOUT le dossier** — recalcule tous les indicateurs.
3. (Multi) **Analyse → 🧺 Construire le dataset Portefeuille** — crée `MULTI`.
4. **Modèle** — pour le modèle MULTI : sélectionner `MULTI_1h`, **Entraîner** (l'évaluation
   se fait sur le pool). Pour une crypto précise : la sélectionner, **Entraîner** → **Prédire**.
5. **Utiliser le modèle MULTI sur une crypto** : sélectionner la crypto (ex. BTC), cocher
   « 🧺 Prédire avec le modèle Portefeuille », **Prédire** → puis **Backtest** normalement.
6. **Évaluation** — comparer **AUC test** vs **baseline majoritaire** et l'écart-type du
   **Walk-Forward** (un modèle honnête a un faible écart in/out-of-sample).

> Le bouton « 🚀 Pipeline complet » enchaîne extraction → analyse → train → prédiction
> pour une crypto donnée.

---

## 3. ⏳ Pistes restantes (par impact estimé)

1. **Horizon / timeframe** : la direction à 1h est quasi-aléatoire. Tester des horizons
   plus longs (24–72) et des intervalles 4h/1d, souvent plus prédictibles.
2. **Cible Triple-barrier** comme défaut (déjà dispo) : plus alignée sur le trading réel.
3. **Données externes** : funding rate & open interest (API publiques Binance futures),
   dominance BTC, corrélations macro. Fort potentiel d'edge.
4. **Multi-timeframe en features** : injecter le contexte 4h/1d dans le modèle 1h.
5. **Méta-modèle de position sizing** (López de Prado) : prédire la *taille* du pari plutôt
   que juste la direction.
6. **Validation purgée K-fold combinatoire** (CPCV) pour une estimation OOS plus fiable
   que le simple split 80/20.
7. **Coûts réalistes** dans la cible (frais + slippage) pour n'apprendre que les
   mouvements réellement exploitables.

---

## 4. ⚠️ Attentes réalistes

L'objectif n'est pas « > 50 % de rendement » mais un **edge faible mais stable** :
- AUC out-of-sample ≈ 0.52–0.56 = déjà exploitable.
- Un bon modèle se juge sur le **Sharpe/Sortino après frais** et la stabilité Walk-Forward,
  pas sur un rendement brut spectaculaire (souvent un signe d'overfitting ou de chance).
- Mieux vaut un modèle honnête à edge modeste qu'un modèle « magnifique » qui fuite.
