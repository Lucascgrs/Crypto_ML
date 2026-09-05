"""
Entraîner UN modèle sur PLUSIEURS cryptos à la fois.

POURQUOI CE MODULE EXISTE
-------------------------
Le modèle de direction cherche un avantage de quelques points de pourcentage.
Sur une seule crypto en 1h, il dispose d'environ 50 000 lignes d'apprentissage,
dont le futur se chevauche largement : à l'horizon 5, cela ne fait qu'une
dizaine de milliers d'observations réellement indépendantes. C'est très peu
pour distinguer un signal faible du bruit — et c'est la raison la plus probable
pour laquelle les AUC plafonnent autour de 0.53.

Empiler vingt cryptos multiplie la matière par vingt sans rien changer d'autre :
mêmes features, même code, même objectif. Les 8 indicateurs ont d'ailleurs été
conçus dès le départ pour être comparables d'un actif à l'autre (tous bornés ou
exprimés en proportion du prix), ce qui rend l'empilement légitime.

LES DEUX PIÈGES, ET COMMENT ILS SONT TRAITÉS
--------------------------------------------
1. **Des périodes différentes.** Une crypto née en 2015 et une née en 2021
   n'ont pas le même historique. Si chacune était coupée à « 70 % de SES
   lignes », le bloc de test de l'une couvrirait 2024 et celui de l'autre 2019 :
   on comparerait des modèles évalués sur des marchés différents, et le bloc
   d'entraînement de l'une contiendrait le bloc de test de l'autre — une fuite
   pure et simple.

   Ici, les frontières sont des DATES, communes à tout le panier. Chaque crypto
   apprend sur tout ce dont elle dispose avant la date de bascule (2015→2025
   pour l'une, 2018→2025 pour l'autre), et toutes sont validées puis testées
   sur exactement la même période de marché. C'est `modele._decouper` qui s'en
   charge, sur l'index de dates du jeu empilé.

2. **Des échelles différentes.** Un ATR de 0.5 % est une tempête pour du BTC et
   un jour ordinaire pour un altcoin récent. Empilées telles quelles, ces
   colonnes apprendraient au modèle à reconnaître la CRYPTO au lieu de la
   SITUATION. Les features dont le niveau dépend de l'actif (voir
   `config.FEATURES_BORNEES`) sont donc remplacées par leur rang de percentile
   glissant, calculé crypto par crypto et uniquement sur le passé.

Le panier se manipule comme une crypto ordinaire : son nom, `PANIER-BTC-ETH`,
tient lieu de symbole partout — fichier de modèle, métadonnées, prédictions.
Un modèle de panier s'applique ensuite à n'importe quelle crypto du panier
(ou d'ailleurs) via `modele.predire(..., symbole_modele="PANIER-BTC-ETH")`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stockage

# Rang de percentile : identifiant de la normalisation, stocké dans les
# métadonnées du modèle pour être rejoué à l'identique en prédiction.
NORMALISATION_RANG = "rang"
NORMALISATION_AUCUNE = "aucune"


# ===========================================================================
# NORMALISATION CROISÉE
# ===========================================================================
def normaliser(X: pd.DataFrame,
               fenetre: int = config.FENETRE_RANG_PANIER) -> pd.DataFrame:
    """
    Convertit les features dépendantes de l'actif en rang de percentile glissant.

    Le rang répond à « où en est cet actif par rapport à SES `fenetre`
    dernières périodes ? » — question dont la réponse a le même sens pour BTC
    et pour un altcoin, contrairement au niveau brut.

    La fenêtre est CAUSALE : elle ne contient que du passé. Une normalisation
    par la moyenne et l'écart-type de tout l'historique — le réflexe habituel —
    donnerait au modèle, dès la première bougie, une information sur la
    volatilité des années suivantes. C'est le type de fuite qui produit des
    backtests magnifiques et des pertes réelles.
    """
    colonnes = [c for c in config.colonnes_a_normaliser(X.columns) if c in X.columns]
    if not colonnes:
        return X

    resultat = X.copy()
    minimum = max(30, fenetre // 10)
    for colonne in colonnes:
        rang = X[colonne].rolling(fenetre, min_periods=minimum).rank(pct=True)
        # Les premières bougies n'ont pas d'historique : 0.5 = « au milieu »,
        # c'est-à-dire aucune information, plutôt qu'une valeur inventée.
        resultat[colonne] = rang.fillna(0.5).astype("float32")
    return resultat


# ===========================================================================
# CONSTRUCTION DU JEU EMPILÉ
# ===========================================================================
def charger(symboles, intervalle: str, horizon: int,
            tache=None, contexte: bool = True,
            normalisation: str = NORMALISATION_RANG):
    """
    Empile les jeux de plusieurs cryptos en un seul, trié par date.

    Le tri chronologique est ce qui permet à tout le reste du projet — le
    découpage en blocs, l'embargo, la validation croisée purgée — de continuer
    à fonctionner sans rien savoir du panier : il lui suffit que les lignes
    soient dans l'ordre du temps.

    Les colonnes retenues sont l'INTERSECTION des features disponibles chez
    toutes les cryptos du panier. Si l'une d'elles n'a pas d'order flow (parce
    qu'elle vient de Yahoo) ou pas de funding (parce qu'elle n'a pas de contrat
    perpétuel), la colonne correspondante est retirée pour tout le monde :
    mieux vaut une feature de moins que des trous artificiels dont le modèle
    déduirait l'identité de la crypto.
    """
    from . import modele  # import tardif : évite une dépendance circulaire

    symboles = [s for s in dict.fromkeys(str(s).upper() for s in symboles) if s]
    if len(symboles) < 2:
        raise ValueError("Un panier demande au moins deux cryptos.")
    if len(symboles) > config.CRYPTOS_MAX_PANIER:
        raise ValueError(f"Panier limité à {config.CRYPTOS_MAX_PANIER} cryptos.")

    jeux, ignorees = {}, []
    for symbole in symboles:
        try:
            jeux[symbole] = modele.charger_jeu(symbole, intervalle, horizon,
                                               tache, contexte)
        except (FileNotFoundError, ValueError) as err:
            ignorees.append(f"{symbole} ({err})")

    if len(jeux) < 2:
        raise ValueError(
            "Moins de deux cryptos exploitables dans le panier. "
            "Lance l'étape Analyse sur chacune d'elles. " + " ; ".join(ignorees))
    if ignorees:
        print(f"⚠️  Écartées du panier : {', '.join(ignorees)}")

    communes = set.intersection(*(set(jeu.X.columns) for jeu in jeux.values()))
    features = [c for c in next(iter(jeux.values())).X.columns if c in communes]
    perdues = sorted(set().union(*(set(jeu.X.columns) for jeu in jeux.values()))
                     - communes)
    if perdues:
        print(f"⚠️  Features absentes chez au moins une crypto, retirées du "
              f"panier : {', '.join(perdues)}")

    morceaux = []
    for symbole, jeu in jeux.items():
        X = jeu.X[features]
        if normalisation == NORMALISATION_RANG:
            X = normaliser(X)
        morceaux.append({
            "symbole": symbole, "X": X, "y": jeu.y,
            "y_evaluation": jeu.y_evaluation, "apprenable": jeu.apprenable,
            "regime": jeu.regime,
        })
        print(f"   • {symbole:<6} {len(X):>7,} lignes  "
              f"{X.index.min():%Y-%m-%d} → {X.index.max():%Y-%m-%d}")

    X = pd.concat([m["X"] for m in morceaux])
    y = pd.concat([m["y"] for m in morceaux])
    y_evaluation = pd.concat([m["y_evaluation"] for m in morceaux])
    apprenable = pd.concat([m["apprenable"] for m in morceaux])
    regime = pd.concat([m["regime"] for m in morceaux]) \
        if all(m["regime"] is not None for m in morceaux) else None
    origine = pd.concat([pd.Series(m["symbole"], index=m["X"].index)
                         for m in morceaux])

    # Tri stable : à date égale, les cryptos gardent l'ordre du panier. Le tri
    # est ce qui rend le jeu utilisable par le découpage chronologique.
    ordre = np.argsort(X.index.to_numpy(), kind="stable")
    prendre = lambda serie: None if serie is None else serie.iloc[ordre]  # noqa: E731

    from . import cibles
    objectif = tache if hasattr(tache, "cle") else cibles.obtenir(tache)
    jeu = modele.Jeu(
        X=X.iloc[ordre], y=y.iloc[ordre], horizon=horizon, tache=objectif,
        y_evaluation=prendre(y_evaluation), apprenable=prendre(apprenable),
        regime=prendre(regime), origine=prendre(origine))

    print(f"🧺 Panier — {len(jeux)} cryptos, {len(jeu.X):,} lignes, "
          f"{len(features)} features, {jeu.X.index.min():%Y-%m-%d} → "
          f"{jeu.X.index.max():%Y-%m-%d}")
    if normalisation == NORMALISATION_RANG:
        normalisees = len([c for c in config.colonnes_a_normaliser(features)])
        print(f"   Normalisation par rang glissant sur {normalisees} features "
              f"dont le niveau dépend de l'actif "
              f"({len(features) - normalisees} déjà comparables).")
    return jeu


# ===========================================================================
# INVENTAIRE
# ===========================================================================
def cryptos_analysees(intervalle: str) -> list[str]:
    """Symboles disposant d'un fichier analysé pour cet intervalle."""
    symboles = []
    for cle in stockage.lister_analyses():
        symbole, inter = stockage.separer_cle(cle)
        if inter == intervalle and not config.est_panier(symbole):
            symboles.append(symbole)
    return sorted(symboles)


def resume(symboles, intervalle: str) -> str:
    """
    Une ligne décrivant la couverture temporelle du panier envisagé.

    Appelée depuis le thread graphique à chaque case cochée : elle ne doit donc
    jamais charger les fichiers. `stockage.apercu_tableau` ne lit que l'index,
    ce qui rend la fenêtre de sélection instantanée même à vingt-cinq cryptos.
    """
    lignes, debut, fin = 0, None, None
    for symbole in symboles:
        apercu = stockage.apercu_tableau(stockage.chemin_analyse(symbole, intervalle))
        if apercu is None:
            continue
        lignes += apercu["lignes"]
        debut = apercu["debut"] if debut is None else min(debut, apercu["debut"])
        fin = apercu["fin"] if fin is None else max(fin, apercu["fin"])
    if not lignes:
        return "Aucune des cryptos sélectionnées n'est analysée."
    return (f"{len(symboles)} cryptos · {lignes:,} lignes cumulées · "
            f"{debut:%Y-%m-%d} → {fin:%Y-%m-%d}")
