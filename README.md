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
  order flow    21 de contexte  baisse      espérance     de trades
  + funding     + 24 cibles     (1 ou N cryptos)
  + basis
```

Chaque chiffre annoncé vient avec sa **marge d'erreur**, calculée sur le nombre
d'observations réellement indépendantes — pas sur le nombre de lignes.

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
python GatherData.py BTC --exogene                  # funding, basis perp/spot, OI

python CryptoAnalysis.py                             # analyse tout le dossier
python CryptoAnalysis.py BTC_1h                      # ou un fichier précis
python CryptoAnalysis.py BTC_1h --sans-contexte      # les 8 indicateurs seuls

python Predict.py BTC_1h --horizon 3 --modele XGBoost --seuil 0.60
python Predict.py BTC_1h --objectif direction_nette  # n'apprend que le rentable
python Predict.py BTC_1h --walk-forward              # évaluation hors échantillon
python Predict.py BTC_1h --predire-seulement         # réutilise le modèle existant
python Predict.py BTC_1h --panier BTC,ETH,SOL,BNB    # un seul modèle sur 4 cryptos
python Predict.py BTC_1h --recherche approfondie     # 18 configurations au lieu de 3
python Predict.py BTC_1h --utilite 0.0005            # ne garder que les features utiles

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

## Le contexte : 21 colonnes de plus, pas 100

Cinq familles s'ajoutent automatiquement quand elles sont disponibles.

**Multi-timeframe** (suffixe `_MTF`) — les 8 mêmes indicateurs calculés sur
l'intervalle supérieur (4h pour du 1h). Un modèle horaire ne voit que
l'agitation horaire ; il ignore s'il se trouve dans une tendance 4h haussière
ou dans un retournement.

> **Anti-fuite.** Une bougie 4h étiquetée 00:00 couvre 00:00→04:00 : sa clôture
> n'est connue qu'à 04:00. Elle est donc décalée d'une bougie entière avant
> d'être rapprochée de l'index horaire. Sans ce décalage, le modèle verrait dès
> 00:00 une information contenant les quatre heures qu'on lui demande de prédire.

**Order flow** — `Flux_Desequilibre`, `Flux_Cumul`, `Taille_Trade_Norm`.
Binance renvoie dans **chaque chandelier** le nombre de trades et le volume
acheté à l'agressif ; ces colonnes étaient jetées au téléchargement. Le prix
dit ce qui s'est passé, l'order flow dit **qui** l'a provoqué : une bougie
verte produite par des acheteurs au marché n'annonce pas la même suite qu'une
bougie verte produite par des vendeurs qui se retirent. Chandelier identique,
information différente — et c'est gratuit.

**Temps** — `Heure_Sin`, `Heure_Cos`, `Jour_Semaine`, `Avant_Funding`. Le
marché ne ferme jamais mais ne respire pas uniformément : séances asiatique et
américaine, week-end peu liquide, échéance de funding toutes les 8 heures.
L'heure est codée en sinus/cosinus parce qu'elle est circulaire — codée de 0 à
23, elle inventerait une frontière entre 23 h et minuit.

**Régime** — `Regime_Volatilite` (rang de percentile de l'ATR sur 500
périodes), `Regime_Tendance` (écart à la moyenne 200, mesuré en ATR). Un ATR de
1,2 % est une tempête pour BTC et un jour ordinaire pour un altcoin : seul le
rang répond à « est-ce agité *par rapport à d'habitude* ? ». C'est aussi ce qui
rend les cryptos comparables entre elles dans un panier.

**Exogènes** (Binance Futures) — `Funding_Rate`, `Funding_Cumul`, `Basis`,
`Basis_Moyenne`, `OI_Variation`. Tout le reste du fichier dérive du même
OHLCV : ce sont des transformations d'une seule information. Ceux-là viennent
du positionnement réel sur les dérivés. Le **basis perp/spot** mesure la prime
que les acheteurs à levier acceptent de payer ; comme le funding, son
historique est complet dès le lancement du contrat et se récupère en une passe.

Une colonne n'est retenue que si elle couvre au moins 60 % des lignes, et les
colonnes constantes sont écartées (l'heure n'a pas de sens en journalier).
Funding et basis remontent à 2019 et servent immédiatement ; **l'open interest
public de Binance ne remonte qu'à 30 jours** et est donc écarté au début.
Chaque `--exogene` complète le précédent, la couverture s'étend d'elle-même.

### Ce que le contexte apporte, mesuré

BTC 1h, horizon 5, objectif `direction_nette`, même découpage, même modèle.
L'AUC « cv » est la moyenne des 4 blocs de validation croisée ; l'AUC « test »
et la justesse portent sur des données jamais vues.

| Configuration | features | AUC cv | AUC test | Justesse test |
|---|---|---|---|---|
| 8 indicateurs seuls | 8 | 0.5422 | 0.5281 | 50,65 % ± 2,01 |
| multi-timeframe + funding | 16 | 0.5395 | 0.5269 | 51,17 % ± 2,01 |
| **+ order flow, temps, régime, basis** | 29 | **0.5479** | **0.5323** | **52,44 % ± 2,01** |
| panier de 11 cryptos | 29 | 0.5477 | 0.5169 | 51,95 % ± 2,15 |

Les nouvelles familles apportent quelque chose : l'AUC croisée monte de 0.5395
à 0.5479 et la justesse de 51,17 % à 52,44 %. Chaque écart pris isolément tient
dans la marge de ±2 points, mais les deux mesures vont dans le même sens.

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

Le panier, lui, n'améliore pas le résultat — et il explique pourquoi. Quatre
tailles ont été entraînées, mêmes features et même objectif :

| Panier | cryptos | lignes | AUC cv | écart entre blocs | **taille effective du test** |
|---|---|---|---|---|---|
| BTC seul | 1 | 79 006 | 0.5479 | 0.0328 | **2 370** |
| BTC + ETH | 2 | 158 012 | 0.5512 | 0.0215 | **2 370** |
| 7 cryptos | 7 | 496 648 | 0.5514 | 0.0420 | **2 129** |
| 11 cryptos | 11 | 691 440 | 0.5477 | 0.0382 | **2 074** |

Multiplier les lignes par **8,75 fait baisser la taille effective**. C'est
contre-intuitif et parfaitement logique : onze cryptos à la même heure bougent
ensemble, ce sont onze lignes mais une seule observation. Le panier ajoute du
volume, pas de l'information.

Sur une fenêtre de test strictement commune, le modèle de panier fait 51,95 %
± 2,15 contre 52,27 % ± 2,15 pour le modèle propre à BTC. Il est bien plus
sélectif (224 signaux à 60,71 % contre 4 745 à 53,13 %), mais avec ±14,30
points de marge sur 224 signaux ce chiffre ne prouve rien.

Le panier devient en revanche la condition nécessaire de la **cible
cross-sectionnelle** : prédire l'écart d'une crypto à la médiane du panier
retire le mouvement commun, c'est-à-dire à la fois la part imprévisible et la
source de cette corrélation qui annule le gain de données.

## Le panier : un modèle pour plusieurs cryptos

Le vrai problème n'est peut-être pas le modèle mais la quantité de données. Sur
BTC en 1h, l'apprentissage porte sur ~50 000 lignes dont les cibles se
chevauchent : à l'horizon 5, cela ne fait qu'une dizaine de milliers
d'observations réellement indépendantes. Chercher là-dedans un avantage de deux
points, c'est chercher une aiguille dans un tas de bruit.

Le bouton **🧺 Panier** (page Prédiction) entraîne **un** modèle sur plusieurs
cryptos empilées. En ligne de commande :

```bash
python Predict.py BTC_1h --horizon 5 --panier BTC,ETH,SOL,BNB,XRP
```

Le modèle est enregistré sous `PANIER-BTC-ETH-…` et s'applique ensuite à
n'importe quelle crypto.

### Les deux pièges, et comment ils sont traités

**Des périodes différentes.** Une crypto née en 2015 et une née en 2021 n'ont
pas le même historique. Si chacune était coupée à « 70 % de *ses* lignes », le
test de l'une porterait sur 2024 et celui de l'autre sur 2019 : on comparerait
des marchés différents, et surtout l'entraînement de la première contiendrait
le test de la seconde — une fuite pure et simple.

Les frontières sont donc des **dates**, communes à tout le panier :

```
                2015        2018              frontière           aujourd'hui
crypto A   |═══════════════════════════════════|  val  |   test   |
crypto B                |══════════════════════|  val  |   test   |
                        (elle apprend depuis sa naissance,
                         mais bascule à la même date)
```

Chaque crypto apprend sur tout ce dont elle dispose avant la bascule, et toutes
sont jugées sur exactement la même période de marché. Les frontières sont
placées au 70ᵉ et au 85ᵉ centile du **nombre de lignes empilées**, puis
converties en dates : on obtient à la fois un vrai 70/15/15 en volume et une
date unique pour tout le monde.

**Des échelles différentes.** Un ATR de 0,5 % est une tempête pour BTC et un
jour ordinaire pour un altcoin. Empilées telles quelles, ces colonnes
apprendraient au modèle à reconnaître la *crypto* au lieu de la *situation*.
Les features dont le niveau dépend de l'actif sont donc converties en **rang de
percentile glissant** (fenêtre 720 périodes), calculé crypto par crypto et
uniquement sur le passé. Les features déjà bornées — RSI, stochastique, ADX,
%B, déséquilibre de flux, rang de volatilité, heure — gardent leur valeur : un
RSI de 80 veut dire la même chose partout.

Seules les features présentes chez **toutes** les cryptos sont conservées : une
colonne absente ailleurs deviendrait un identifiant de la crypto.

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

L'interface n'expose que **cinq réglages** — crypto, modèle, objectif, horizon,
seuil de confiance — plus **deux réglages facultatifs** dont le défaut convient :
la profondeur de la recherche d'hyperparamètres et l'utilité minimale exigée
d'une feature (voir les deux sections suivantes). Tout le reste est automatique
et documenté dans le bouton « ❔ Ce qui est fait automatiquement » :

- **Découpage** 70 / 15 / 15, strictement chronologique, avec un embargo égal à
  l'horizon entre les blocs (sinon la cible, qui regarde vers le futur, fuite
  d'un bloc à l'autre).
- **Équilibrage des classes** — uniquement si l'une dépasse 55 %.
- **Hyperparamètres** — trois configurations testées par défaut (18 ou 40 sur
  demande), départagées en validation croisée purgée ; à égalité statistique
  c'est la plus simple qui gagne.
- **Utilité des features** — mesurée à chaque entraînement, sur la validation.
- **Early stopping** sur le logloss (plus stable que l'AUC, trop bruitée).
- **Calibration** des probabilités par tranches d'au moins 250 observations :
  impossible d'annoncer « 95 % de confiance » sur la foi de trois bougies.
- **Ressources** — la RAM libre est convertie en histogrammes plus fins et en
  parallélisme au lieu de rester inutilisée (voir `crypto_lab/ressources.py`).


## Le curseur d'utilité : ne garder que les features qui servent

À la fin de chaque entraînement, chaque feature reçoit une note : on mélange sa
colonne au hasard et on mesure de combien l'AUC tombe. Trois cas de figure :

| note | lecture |
|---|---|
| `+0.0100` | feature essentielle — le modèle s'écroule sans elle |
| `0.0000` | feature inutile — elle dilue sans détruire |
| `-0.0010` | feature **nuisible** — la détruire *améliore* le modèle |

Le troisième cas n'est pas une anomalie de calcul. Il signifie que le modèle
avait appris à s'appuyer sur cette colonne pendant l'apprentissage, et que cet
appui se retourne contre lui ailleurs.

Le curseur « Utilité minimale » de la page Modèle réentraîne en ne gardant que
les features au-dessus d'un seuil. Le nombre de features conservées et la liste
des écartées s'affichent **avant** de lancer quoi que ce soit — un curseur dont
on ne découvrirait l'effet qu'après vingt minutes de calcul ne servirait à rien.

**Le point de méthode qui rend le curseur honnête** : la note qui sert à filtrer
est mesurée sur la seconde moitié du bloc de **validation**, celle que le modèle
brut n'a vue ni pour apprendre ni pour décider quand s'arrêter. Choisir ses
features d'après le test puis annoncer une performance sur ce même test
reviendrait à se noter soi-même. Le graphique de la page Visualisation, lui,
reste mesuré sur le test : il sert à diagnostiquer, pas à décider — et l'écart
entre les deux mesures est en soi une information sur la stabilité du modèle.

Une feature écartée conserve sa dernière note connue : abaisser le curseur la
fait revenir au prochain entraînement, rien n'est perdu.

Ce qu'il faut en attendre, mesuré sur BTC 1h à l'horizon 5 :

| features | AUC test | justesse test |
|---|---|---|
| 29 (toutes) | 0.5323 | 51,82 % ± 2,06 |
| 9 (utilité ≥ 0.0005) | 0.5335 | 52,66 % ± 2,05 |

Vingt features en moins pour 0,84 point de justesse en plus — soit moins que la
moitié de la marge d'erreur. Le filtre rend le modèle beaucoup plus **lisible**
(on sait enfin sur quoi il s'appuie) sans qu'on puisse affirmer qu'il le rend
meilleur. C'est déjà utile : un modèle à 9 entrées se diagnostique, un modèle à
29 entrées se subit.

## La profondeur de recherche : pourquoi l'entraînement est si court

Trois configurations écrites à la main, c'est un coin minuscule de l'espace des
réglages. Deux modes plus longs y ajoutent des tirages **aléatoires** — et non
un balayage complet, qui ferait 1,7 million de combinaisons pour huit
paramètres à six valeurs. La recherche aléatoire trouve un réglage équivalent en
quelques dizaines d'essais parce que la performance ne dépend fortement que de
deux ou trois paramètres, et que le tirage les explore tous à la fois.

Deux garde-fous : les trois configurations de référence font toujours partie du
lot (une recherche longue ne peut donc jamais faire *pire*), et les candidates
sont classées de la plus prudente à la plus souple, si bien qu'à égalité
statistique la règle du 1 écart-type garde la plus sobre.

Le résultat mesuré, BTC 1h horizon 5, objectif `direction_nette` :

| mode | configurations | durée | AUC test | justesse test |
|---|---|---|---|---|
| Rapide | 3 | 34 s | 0.5323 | 51,82 % ± 2,06 |
| Approfondie | 18 | 140 s | 0.5342 | 51,99 % ± 2,06 |

**Quatre fois plus de temps pour 0,17 point** — vingt fois moins que la marge
d'erreur. Si l'entraînement paraît court, ce n'est pas qu'il bâcle : le boosting
sur 75 000 lignes et 29 colonnes est un problème *petit* pour une machine
moderne. Le facteur limitant de ce projet n'a jamais été la puissance de calcul,
c'est la quantité d'information contenue dans les données — et aucun réglage
d'hyperparamètre ne crée de l'information qui n'y est pas.

Le mode approfondi sert donc à **vérifier** que le défaut n'était pas mauvais.
C'est une utilité réelle, mais ce n'est pas celle qu'on espère.

## Le suivi en direct, et le fait que rien ne bloque

Les quatre traitements longs du projet — entraînement, walk-forward, régression
d'amplitude, backtest — tournent dans un thread de travail : l'application reste
utilisable pendant qu'ils calculent (changer de page, lire la console, consulter
un graphique), et les trois premiers publient leur progression et sont
interruptibles. Le bouton « ⏹ Arrêter » de la barre de statut
interrompt proprement : le thread n'est **jamais tué de force**, il consulte une
demande d'arrêt entre chaque étape *et à chaque arbre construit*, puis s'arrête
de lui-même. Mesuré : une demande d'arrêt pendant une recherche de 40
configurations est honorée en moins d'une seconde, et le modèle précédemment
enregistré reste intact.

La fenêtre « 📺 Suivi en direct » montre trois choses :

- **la courbe d'apprentissage**, arbre après arbre, sur l'apprentissage et sur
  la validation. Les deux descendent ensemble, puis celle de validation remonte
  pendant que l'autre continue : c'est le surapprentissage, en direct, et le
  trait vert marque l'endroit où l'arrêt anticipé coupe ;
- **les configurations évaluées**, chacune avec sa dispersion entre blocs de
  validation croisée. Quand toutes les barres d'erreur se recouvrent, le réglage
  ne change rien — et c'est une information, pas une déception ;
- **l'utilité des features**, dès qu'elle est mesurée.

L'affichage est découplé du calcul sur trois niveaux, parce qu'un seul ne
suffisait pas :

1. **on ne redessine que si quelque chose a changé**, et jamais plus vite que la
   cadence nominale ;
2. **on ne redessine que ce qui a changé.** Les courbes ne sont pas retracées,
   elles sont repeintes par-dessus un fond mémorisé (*blitting*) : axes,
   graduations, grille, légende et panneau de droite ne sont recalculés qu'au
   changement d'ajustement ou quand les données sortent du cadre. **11,7 ms par
   image au lieu de 250** ;
3. **la cadence s'adapte à la machine.** La boucle mesure ce que son dernier
   rendu a coûté et s'espace d'autant, pour ne jamais prendre plus d'un
   cinquième du thread graphique. En pratique : 6 images par seconde pour 7 %
   du thread.

Le deuxième point n'est pas du raffinement. Mesuré sur le même entraînement de
18 configurations, avant et après :

| | tours de boucle | latence médiane | pire latence | durée du calcul |
|---|---|---|---|---|
| sans blitting | 707 | 11 ms | **167 s** | 190 s |
| avec blitting | 3 745 | 0,3 ms | 1,4 s | **118 s** |

Sans lui, le rendu complet de la figure coûte 250 ms quand les seize cœurs sont
pris par XGBoost, alors que la boucle de service de l'application tourne toutes
les 80 ms : l'interface passait tout son temps à dessiner. Et comme elle
dessinait, elle volait des cœurs au calcul — d'où les 38 % de temps
d'entraînement gagnés en réparant l'affichage.

Fermer la fenêtre n'arrête rien — et tant qu'elle est masquée, plus rien n'est
tracé du tout.

## La taille effective : pourquoi les chiffres sont moins solides qu'ils n'en ont l'air

C'est le correctif le plus inconfortable du projet, et le plus utile.

À l'horizon 5, la cible de la bougie de 10 h regarde jusqu'à 15 h, celle de
11 h jusqu'à 16 h : cinq lignes consécutives décrivent en grande partie le
**même** morceau de futur. Elles ne valent pas cinq observations, elles en
valent une. Dans un panier, deux cryptos à la même heure bougent ensemble à
plus de 80 % (RSI de BTC et d'ETH : corrélation mesurée de 0,83) — même
problème. D'où la définition retenue :

```
taille effective = nombre d'horodatages DISTINCTS ÷ horizon
```

Une seule formule qui règle les deux cas. Toutes les marges d'erreur affichées
en découlent :

| Mesure | Sans correction | Avec correction |
|---|---|---|
| Justesse sur 11 851 bougies (h=5) | 52,44 % ± 0,90 % | **52,44 % ± 2,01 %** |
| Précision sur 455 signaux filtrés | 60,88 % ± 4,49 % | **60,88 % ± 10,03 %** |

Le beau 60,9 % ne prouve plus rien du tout. Ce n'est pas une coquetterie
statistique : c'est exactement ce qui sépare un avantage réel d'un mirage, et
donc un backtest tenable d'une perte réelle.

## La validation croisée purgée

Auparavant, les trois configurations candidates étaient départagées sur un seul
bloc de validation. Or leurs AUC diffèrent de quelques millièmes quand la marge
d'erreur est de l'ordre du centième : un tirage au sort déguisé en optimisation.

La période d'apprentissage est maintenant découpée en blocs chronologiques
successifs. Chaque bloc est évalué par un modèle entraîné uniquement sur ce qui
le **précède** — la seule validation croisée honnête sur une série temporelle,
un K-fold ordinaire ferait apprendre le futur. Les lignes dont la cible empiète
sur le bloc évalué sont retirées de l'apprentissage (« purge »).

L'écart entre blocs est révélateur. Sur BTC 1h à l'horizon 5 :

```
configuration 1 : 0.5314 / 0.5642 / 0.5536 / 0.5422   →  0.5479 ± 0.0142
configuration 2 : 0.5210 / 0.5606 / 0.5514 / 0.5391   →  0.5430 ± 0.0171
configuration 3 : 0.5167 / 0.5548 / 0.5451 / 0.5360   →  0.5382 ± 0.0162
```

L'écart **entre blocs** (±0,014) est cinq fois plus grand que l'écart **entre
configurations** (0,0097 du premier au dernier). Autrement dit : le choix qu'on
croyait faire sur la qualité du modèle se faisait en réalité sur le bruit. La
configuration retenue est désormais celle qui tient sur quatre époques.

## La justesse par régime

Une justesse moyenne de 53 % peut recouvrir 58 % en marché calme et 48 % en
marché agité. Le modèle n'est alors pas mauvais : il est **conditionnel**, et
la bonne décision n'est pas de le jeter mais de ne le suivre que dans le régime
où il fonctionne.

Le rapport ventile donc la justesse en quatre quartiles de volatilité récente,
chacun avec sa marge d'erreur. Quand l'écart entre le meilleur et le pire
régime dépasse la somme des deux marges, une ligne le signale.

C'est la version économe des « modèles par régime » : entraîner quatre modèles
séparés diviserait les données par quatre au moment précis où elles manquent le
plus. On ne coupe pas les données, on coupe la **lecture** — et le modèle
reçoit de quoi reconnaître le régime lui-même (`Regime_Volatilite`).

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
│   ├── suivi.py            progression, courbes en direct, arrêt coopératif
│   ├── stockage.py         lecture/écriture Excel + cache Parquet et mémoire
│   ├── extraction.py       Binance, Yahoo, CoinGecko
│   ├── exogene.py          funding, basis perp/spot, open interest (Futures)
│   ├── indicateurs.py      les 8 indicateurs, le contexte, les variations
│   ├── cibles.py           les 4 objectifs d'apprentissage
│   ├── panier.py           empilement de plusieurs cryptos en un seul modèle
│   ├── modele.py           entraînement, calibration, prédiction (classification)
│   ├── amplitude.py        régression quantile, volatilité, espérance de gain
│   ├── backtest.py         simulateur long / short, sizing proportionnel
│   └── interface/
│       ├── app.py          fenêtre principale, navigation, tâches de fond
│       ├── moniteur.py     fenêtre de suivi en direct de l'entraînement
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

MODELE_PANIER-BTC-ETH-SOL_1h_h3_direction_nette.joblib   modèle de panier
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
