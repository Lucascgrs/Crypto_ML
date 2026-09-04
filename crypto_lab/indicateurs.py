"""
Calcul des indicateurs techniques et des variations futures.

REFONTE COMPLÈTE : on repart de zéro avec **8 indicateurs seulement**, contre
~120 colonnes auparavant. Le principe est « trop de data tue la data » — un
modèle noyé sous des dizaines de variables redondantes apprend surtout du bruit.

Deux règles absolues :

  1. **Aucune feature n'est un niveau de prix.** Chaque indicateur est soit
     borné (RSI, Stochastique, ADX, %B), soit exprimé en proportion du prix
     (MACD normalisé, écart à la SMA, ATR %), soit en proportion du volume
     (OBV). Une feature qui suit le niveau du prix ferait mémoriser « BTC vaut
     60 000 » au modèle, avec d'excellents scores en apprentissage et un
     effondrement total sur les données récentes.

  2. **Les colonnes `variation_x` sont des valeurs FUTURES.** Elles servent
     uniquement à construire la cible et ne sont jamais des features (le
     module `modele` travaille sur une liste blanche : `config.INDICATEURS`).

Fichier produit : `analysis_crypto/{SYMBOLE}_{INTERVALLE}_analyzed.xlsx`
avec 5 colonnes de prix + 8 indicateurs + 24 variations = 37 colonnes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stockage


# ===========================================================================
# BRIQUES DE CALCUL (fonctions pures, testables indépendamment)
# ===========================================================================
def _lissage_wilder(serie: pd.Series, periode: int) -> pd.Series:
    """
    Moyenne mobile exponentielle de Wilder, utilisée par le RSI, l'ATR et l'ADX.
    Équivalente à un ewm de coefficient 1/periode (soit com = periode - 1).
    """
    return serie.ewm(alpha=1 / periode, adjust=False, min_periods=periode).mean()


def rsi(close: pd.Series, periode: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index (0 à 100).

    Compare la force moyenne des hausses à celle des baisses sur `periode`.
    < 30 = sur-vendu (rebond possible), > 70 = sur-acheté (correction possible).
    Borné, donc directement exploitable par un modèle.
    """
    delta = close.diff()
    gains = _lissage_wilder(delta.clip(lower=0), periode)
    pertes = _lissage_wilder(-delta.clip(upper=0), periode)
    force = gains / pertes.replace(0, np.nan)
    return 100 - 100 / (1 + force)


def stochastique_k(high: pd.Series, low: pd.Series, close: pd.Series,
                   periode: int = 14) -> pd.Series:
    """
    Stochastique %K (0 à 100) — position du prix dans son range récent.

    0 = le prix clôture sur le plus bas des `periode` dernières bougies,
    100 = sur le plus haut. Complémentaire du RSI : mesure la position dans le
    canal plutôt que l'équilibre hausses/baisses.
    """
    plus_bas = low.rolling(periode).min()
    plus_haut = high.rolling(periode).max()
    amplitude = (plus_haut - plus_bas).replace(0, np.nan)
    return 100 * (close - plus_bas) / amplitude


def macd_histogramme_normalise(close: pd.Series, rapide: int = 12,
                               lent: int = 26, signal: int = 9) -> pd.Series:
    """
    Histogramme MACD divisé par le prix — accélération de la tendance.

    Le MACD brut est une différence de moyennes de prix : il grandit avec le
    prix, donc inexploitable tel quel. Divisé par le Close il devient
    comparable dans le temps ET entre cryptos.
    Positif = la tendance haussière s'accélère, négatif = elle s'essouffle.
    """
    ema_rapide = close.ewm(span=rapide, adjust=False).mean()
    ema_lente = close.ewm(span=lent, adjust=False).mean()
    macd = ema_rapide - ema_lente
    ligne_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd - ligne_signal) / close


def distance_sma(close: pd.Series, periode: int = 50) -> pd.Series:
    """
    Écart relatif entre le prix et sa moyenne mobile (en proportion).

    +0.05 = le prix est 5 % au-dessus de sa moyenne 50 périodes.
    Résume à lui seul « où en est-on dans la tendance », sans transporter le
    niveau de prix.
    """
    moyenne = close.rolling(periode).mean()
    return (close - moyenne) / moyenne


def bollinger_position(close: pd.Series, periode: int = 20,
                       ecarts: float = 2.0) -> pd.Series:
    """
    Bollinger %B — position du prix dans ses bandes de volatilité.

    0 = sur la bande basse, 1 = sur la bande haute, 0.5 = sur la moyenne.
    Peut sortir de [0, 1] lors des mouvements extrêmes, ce qui est justement
    l'information intéressante (extension anormale).
    """
    moyenne = close.rolling(periode).mean()
    ecart_type = close.rolling(periode).std()
    basse = moyenne - ecarts * ecart_type
    largeur = (2 * ecarts * ecart_type).replace(0, np.nan)
    return (close - basse) / largeur


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range : amplitude réelle d'une bougie, gaps d'ouverture inclus."""
    cloture_precedente = close.shift()
    return pd.concat([
        high - low,
        (high - cloture_precedente).abs(),
        (low - cloture_precedente).abs(),
    ], axis=1).max(axis=1)


def atr_pourcent(high: pd.Series, low: pd.Series, close: pd.Series,
                 periode: int = 14) -> pd.Series:
    """
    ATR rapporté au prix — niveau de volatilité (régime de marché).

    0.01 = la bougie moyenne fait 1 % du prix. Sert au modèle à savoir s'il
    évolue dans un marché calme ou agité, ce qui change complètement la
    signification des autres indicateurs.
    """
    return _lissage_wilder(_true_range(high, low, close), periode) / close


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        periode: int = 14) -> pd.Series:
    """
    ADX — force de la tendance, indépendamment de son sens (0 à 100).

    < 20 = marché sans direction (range), > 25 = tendance nette. C'est le
    complément indispensable des indicateurs directionnels : un signal de
    momentum n'a pas la même valeur en range et en tendance.
    """
    mouvement_haut = high.diff()
    mouvement_bas = -low.diff()

    dm_plus = pd.Series(np.where((mouvement_haut > mouvement_bas) & (mouvement_haut > 0),
                                 mouvement_haut, 0.0), index=high.index)
    dm_moins = pd.Series(np.where((mouvement_bas > mouvement_haut) & (mouvement_bas > 0),
                                  mouvement_bas, 0.0), index=high.index)

    atr_lisse = _lissage_wilder(_true_range(high, low, close), periode)
    di_plus = 100 * _lissage_wilder(dm_plus, periode) / atr_lisse
    di_moins = 100 * _lissage_wilder(dm_moins, periode) / atr_lisse

    somme = (di_plus + di_moins).replace(0, np.nan)
    dx = 100 * (di_plus - di_moins).abs() / somme
    return _lissage_wilder(dx, periode)


def obv_pourcent(close: pd.Series, volume: pd.Series, periode: int = 20) -> pd.Series:
    """
    Flux de volume net sur `periode`, rapporté au volume total (-1 à +1).

    L'OBV brut est un cumul sans borne (donc inexploitable en l'état). On mesure
    ici sa variation sur 20 périodes divisée par le volume échangé sur la même
    fenêtre : +0.4 signifie que 40 % du volume de la fenêtre est allé, net, du
    côté acheteur. Répond à la question « le mouvement de prix est-il soutenu
    par les volumes ? ».
    """
    sens = np.sign(close.diff()).fillna(0.0)
    obv = (sens * volume).cumsum()
    flux_net = obv - obv.shift(periode)
    volume_total = volume.rolling(periode).sum().replace(0, np.nan)
    return flux_net / volume_total


# ===========================================================================
# CONSTRUCTION DU FICHIER ANALYSÉ
# ===========================================================================
def calculer_indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les 8 indicateurs à partir d'un DataFrame OHLCV.

    Retourne un DataFrame contenant exactement les colonnes de
    `config.INDICATEURS`, aligné sur l'index d'entrée.
    """
    manquantes = [c for c in config.COLONNES_PRIX if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes OHLCV manquantes : {manquantes}")

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    return pd.DataFrame({
        "RSI_14":         rsi(close),
        "Stoch_K":        stochastique_k(high, low, close),
        "MACD_Hist_Norm": macd_histogramme_normalise(close),
        "Dist_SMA_50":    distance_sma(close),
        "BB_Position":    bollinger_position(close),
        "ATR_Pct":        atr_pourcent(high, low, close),
        "ADX_14":         adx(high, low, close),
        "OBV_Pct":        obv_pourcent(close, volume),
    }, index=df.index)


def agreger(df: pd.DataFrame, regle: str) -> pd.DataFrame:
    """
    Rééchantillonne un OHLCV vers un intervalle plus large.

    Convention identique à celle de Binance : la bougie porte son heure
    d'OUVERTURE et couvre [T, T + durée). Une bougie 4h étiquetée 00:00 agrège
    donc les bougies 1h de 00:00 à 03:00.
    """
    agrege = df.resample(regle).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"})
    return agrege.dropna(subset=["Close"])


def calculer_indicateurs_superieurs(df: pd.DataFrame,
                                    intervalle: str) -> pd.DataFrame | None:
    """
    Les 8 mêmes indicateurs, calculés un cran au-dessus (solution multi-timeframe).

    Un modèle en 1h ne voit que l'agitation horaire : il ignore complètement
    s'il se trouve dans une tendance 4h haussière ou dans un retournement. Ces
    8 colonnes supplémentaires lui donnent ce contexte — 8, pas 100 : le
    principe « trop de data tue la data » reste la règle.

    ANTI-FUITE — le point délicat. Une bougie 4h étiquetée 00:00 couvre
    00:00→04:00 : sa clôture n'est connue qu'à 04:00. La rapprocher telle
    quelle de l'index horaire donnerait au modèle, dès 00:00, une information
    contenant les quatre heures suivantes — exactement ce qu'on lui demande de
    prédire. Le décalage d'une bougie (`shift(1)`) garantit qu'à l'instant t le
    modèle ne voit que des bougies supérieures INTÉGRALEMENT clôturées.

    Retourne None si l'intervalle n'a pas de supérieur défini ou si
    l'historique agrégé est trop court pour que les indicateurs se stabilisent.
    """
    superieur = config.INTERVALLE_SUPERIEUR.get(intervalle)
    if superieur is None:
        return None
    libelle, regle = superieur

    agrege = agreger(df, regle)
    if len(agrege) < 100:                # 50 périodes de chauffe + marge
        print(f"⚠️ Contexte {libelle} ignoré : seulement {len(agrege)} bougies agrégées.")
        return None

    contexte = calculer_indicateurs(agrege).shift(1)
    contexte.columns = [nom + config.SUFFIXE_MTF for nom in contexte.columns]

    # Report en avant sur l'index de base : entre deux bougies supérieures, la
    # dernière valeur connue reste valable.
    aligne = contexte.reindex(df.index.union(contexte.index)).ffill().reindex(df.index)
    print(f"🔭 Contexte {libelle} : {len(contexte.columns)} colonnes ajoutées "
          f"({len(agrege):,} bougies agrégées).")
    return aligne


def calculer_variations(close: pd.Series) -> pd.DataFrame:
    """
    Construit les 24 colonnes `variation_x`.

    variation_x = (Close[t+x] - Close[t]) / Close[t] × 100, en POURCENTAGE.
    C'est la variation observée x périodes APRÈS la ligne concernée : une
    valeur future, donc une cible et jamais une entrée du modèle.

    Les x dernières lignes ont logiquement une variation_x inconnue (NaN) :
    elles sont conservées car ce sont précisément les bougies récentes sur
    lesquelles on veut prédire.
    """
    variations = {
        config.colonne_variation(h): close.pct_change(h).shift(-h) * 100
        for h in config.HORIZONS
    }
    return pd.DataFrame(variations, index=close.index)


def _filtrer_contexte(contexte: pd.DataFrame) -> pd.DataFrame:
    """
    Écarte les colonnes de contexte trop clairsemées pour servir.

    L'open interest public de Binance ne remonte qu'à 30 jours : sur plusieurs
    années d'historique, la colonne serait vide à 99 %. La garder obligerait
    soit à jeter tout l'historique, soit à inventer des valeurs. On la retire
    tant qu'elle n'a pas été suffisamment accumulée (voir `exogene.py`, qui
    fusionne chaque téléchargement avec le précédent).
    """
    gardees = {}
    for nom in contexte.columns:
        couverture = float(contexte[nom].notna().mean())
        if couverture >= config.COUVERTURE_MINIMALE:
            gardees[nom] = contexte[nom]
        else:
            print(f"   ⏭️  {nom} ignorée — {couverture:.0%} de couverture "
                  f"(minimum {config.COUVERTURE_MINIMALE:.0%}).")
    return pd.DataFrame(gardees, index=contexte.index)


def analyser(df_brut: pd.DataFrame, symbole: str | None = None,
             intervalle: str | None = None, contexte: bool = True) -> pd.DataFrame:
    """
    Pipeline complet : OHLCV -> prix + 8 indicateurs + contexte + 24 variations.

    Le contexte, optionnel, regroupe deux familles de colonnes qui n'existaient
    pas dans la version précédente :

      * **multi-timeframe** — les 8 mêmes indicateurs calculés sur l'intervalle
        supérieur (4h pour du 1h), décalés d'une bougie pour interdire toute
        fuite du futur ;
      * **exogènes** — funding rate et open interest, seules données du fichier
        qui ne soient pas dérivées du prix.

    Les lignes de « chauffe » (indicateurs encore incalculables faute
    d'historique) sont supprimées. En revanche les dernières lignes, dont les
    variations futures sont inconnues, sont conservées : ce sont les bougies
    sur lesquelles on produira un signal.
    """
    df = df_brut.copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for colonne in config.COLONNES_PRIX:
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    prix = df[config.COLONNES_PRIX]
    indicateurs = calculer_indicateurs(df)
    variations = calculer_variations(df["Close"].astype(float))

    morceaux = [prix, indicateurs]
    colonnes_contexte: list[str] = []
    if contexte and intervalle:
        supplements = _rassembler_contexte(df, symbole, intervalle)
        if supplements is not None and not supplements.empty:
            morceaux.append(supplements)
            colonnes_contexte = list(supplements.columns)

    resultat = pd.concat(morceaux + [variations], axis=1)
    resultat = resultat.replace([np.inf, -np.inf], np.nan)

    # Chauffe : on ne garde que les lignes où les indicateurs (base et
    # multi-timeframe) existent. Les colonnes exogènes, elles, comportent
    # légitimement des trous — on les neutralise plus bas plutôt que de
    # sacrifier des années d'historique.
    obligatoires = config.INDICATEURS + [c for c in colonnes_contexte
                                         if c in config.INDICATEURS_MTF]
    resultat = resultat.dropna(subset=obligatoires)

    # Zéro est la valeur neutre des trois colonnes exogènes : funding nul
    # (positionnement équilibré), cumul nul, open interest inchangé. C'est donc
    # « aucune information » et non une valeur inventée.
    exogenes = [c for c in colonnes_contexte if c in config.COLONNES_EXOGENES]
    if exogenes:
        resultat[exogenes] = resultat[exogenes].fillna(0.0)

    # Tout ce qui n'est pas un prix passe en float32 : la précision est
    # largement suffisante (7 chiffres significatifs) et le fichier pèse deux
    # fois moins. Les prix restent en float64 pour ne pas perdre les centimes.
    colonnes_legeres = (config.INDICATEURS + colonnes_contexte
                        + config.COLONNES_VARIATION)
    resultat[colonnes_legeres] = resultat[colonnes_legeres].astype("float32")

    resultat.index.name = "Date"
    return resultat[config.colonnes_analyse(colonnes_contexte)]


def _rassembler_contexte(df: pd.DataFrame, symbole: str | None,
                         intervalle: str) -> pd.DataFrame | None:
    """Assemble les colonnes multi-timeframe et exogènes réellement utilisables."""
    from . import exogene  # import tardif : dépend du réseau, pas du calcul

    morceaux = []
    superieur = calculer_indicateurs_superieurs(df, intervalle)
    if superieur is not None:
        morceaux.append(superieur)

    if symbole:
        try:
            externe = exogene.contexte_exogene(symbole, intervalle, df.index)
        except Exception as err:                       # noqa: BLE001
            print(f"⚠️ Contexte exogène ignoré : {err}")
            externe = None
        if externe is not None:
            morceaux.append(externe)

    if not morceaux:
        return None
    return _filtrer_contexte(pd.concat(morceaux, axis=1))


def analyser_fichier(symbole: str, intervalle: str,
                     contexte: bool = True) -> pd.DataFrame | None:
    """Charge un fichier OHLCV brut, l'analyse et sauvegarde le résultat."""
    brut = stockage.lire_tableau(stockage.chemin_brut(symbole, intervalle))
    if brut is None or brut.empty:
        print(f"❌ Données brutes introuvables : {symbole} ({intervalle})")
        return None

    analyse = analyser(brut, symbole, intervalle, contexte=contexte)
    chemin = stockage.chemin_analyse(symbole, intervalle)
    stockage.ecrire_tableau(analyse, chemin)

    supplements = [c for c in analyse.columns if c in config.COLONNES_CONTEXTE]
    exploitables = int(analyse[config.colonne_variation(config.HORIZON_MAX)].notna().sum())
    print(f"✅ {symbole} ({intervalle}) — {len(analyse):,} lignes, "
          f"{len(config.INDICATEURS)} indicateurs"
          + (f" + {len(supplements)} de contexte" if supplements else "")
          + f", {len(config.COLONNES_VARIATION)} variations "
          f"({exploitables:,} lignes entièrement étiquetées)")
    return analyse


def analyser_tout(contexte: bool = True) -> list[str]:
    """Analyse tous les fichiers bruts disponibles. Retourne les clés traitées."""
    traites = []
    for cle in stockage.lister_donnees_brutes():
        symbole, intervalle = stockage.separer_cle(cle)
        try:
            if analyser_fichier(symbole, intervalle, contexte) is not None:
                traites.append(cle)
        except Exception as err:                       # noqa: BLE001
            print(f"❌ {cle} : {err}")
    print(f"\n📦 Analyse terminée — {len(traites)} fichier(s) traité(s).")
    return traites
