# Notes de refonte — Crypto Lab v2

> Refonte complète de la partie analyse et de la partie prédiction.
> Fil conducteur : **trop de data tue la data**.

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

## 5. Pistes restantes

1. **Intervalles plus longs** (4h, 1d) : la direction horaire est presque du
   bruit ; les données sont déjà téléchargeables dans ces intervalles.
2. **Frais dans la cible** : n'apprendre que les mouvements dépassant le coût
   de transaction, plutôt que le simple signe.
3. **Contexte multi-timeframe** : injecter les 8 mêmes indicateurs calculés en
   4h dans le modèle 1h — 8 colonnes de plus, pas 100.
4. **Validation croisée purgée** (CPCV) pour une estimation hors échantillon
   plus fiable que le simple bloc final.
5. **Taille de position** plutôt que direction seule : parier proportionnellement
   à la confiance.
