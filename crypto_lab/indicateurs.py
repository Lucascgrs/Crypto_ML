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


def calculer_flux(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Trois features d'ORDER FLOW, à partir des colonnes déjà téléchargées.

    Le prix dit ce qui s'est passé, l'order flow dit qui l'a provoqué. Une
    bougie verte produite par des acheteurs au marché n'annonce pas la même
    suite qu'une bougie verte produite par des vendeurs qui se retirent : dans
    le premier cas quelqu'un a PAYÉ pour entrer, dans le second personne ne
    voulait vendre. Le chandelier est identique, l'information ne l'est pas.

    C'est la seule famille de features du projet qui ne soit pas une
    transformation du prix — les 8 indicateurs, leur version multi-timeframe et
    les 24 variations viennent tous du même OHLCV.

    Retourne None si le fichier brut date d'avant l'ajout de ces colonnes : il
    suffit alors de relancer le téléchargement.
    """
    manquantes = [c for c in config.COLONNES_FLUX_BRUT if c not in df.columns]
    if manquantes:
        return None

    volume = df["Volume"].astype(float)
    trades = df["Trades"].astype(float)
    taker = df["TakerBase"].astype(float)

    # Part du volume partie à l'achat agressif, recentrée sur zéro : +1 = tout
    # le volume à l'achat au marché, −1 = tout à la vente, 0 = équilibre.
    desequilibre = 2 * taker / volume.replace(0, np.nan) - 1
    desequilibre = desequilibre.clip(-1, 1)

    # Taille moyenne d'un trade, rapportée à sa médiane récente. Le niveau brut
    # est en unités de la crypto (donc incomparable d'un actif à l'autre et
    # dérivant avec le prix) ; le log du rapport est stationnaire et centré.
    taille = volume / trades.replace(0, np.nan)
    reference = taille.rolling(config.FENETRE_TAILLE, min_periods=20).median()
    taille_norm = np.log(taille / reference.replace(0, np.nan))

    return pd.DataFrame({
        "Flux_Desequilibre": desequilibre,
        "Flux_Cumul": desequilibre.rolling(config.FENETRE_FLUX, min_periods=3).mean(),
        "Taille_Trade_Norm": taille_norm,
    }, index=df.index)


def calculer_temps(index: pd.DatetimeIndex, intervalle: str) -> pd.DataFrame | None:
    """
    Saisonnalité horaire, hebdomadaire, et position dans le cycle de funding.

    Le marché crypto ne dort jamais mais ne respire pas de façon uniforme : la
    volatilité et les volumes suivent l'ouverture des séances, le week-end est
    nettement moins liquide, et le funding tombe toutes les 8 heures, ce qui
    déplace mécaniquement des positions juste avant l'échéance.

    L'heure est codée en sinus/cosinus parce qu'elle est CIRCULAIRE : donnée
    comme un entier de 0 à 23, elle apprendrait au modèle une frontière
    imaginaire entre 23 h et minuit alors que ce sont deux heures voisines.

    Retourne None pour les intervalles d'un jour ou plus : l'heure y est
    constante, et une colonne constante n'apporte rien.
    """
    if config.heures(intervalle) >= 24:
        return None

    heure = index.hour + index.minute / 60.0
    angle = 2 * np.pi * heure / 24.0
    cycle = config.CYCLE_FUNDING_HEURES

    return pd.DataFrame({
        "Heure_Sin": np.sin(angle),
        "Heure_Cos": np.cos(angle),
        "Jour_Semaine": index.dayofweek.to_numpy(dtype=float),
        # 0 juste après un versement, proche de 1 juste avant le suivant.
        "Avant_Funding": (heure % cycle) / cycle,
    }, index=index)


def calculer_regime(df: pd.DataFrame, indicateurs: pd.DataFrame) -> pd.DataFrame:
    """
    Deux colonnes qui situent la bougie dans son RÉGIME de marché.

    Un ATR de 1.2 % n'a pas le même sens partout : c'est une tempête pour du
    BTC et un jour ordinaire pour un altcoin récent. Le rang de percentile sur
    les 500 dernières périodes répond à la seule question qui compte — « est-ce
    agité PAR RAPPORT À D'HABITUDE, ici et maintenant ? » — et il est
    directement comparable d'une crypto à l'autre, ce qui le rend indispensable
    dès qu'on entraîne un modèle sur un panier.

    C'est l'alternative économe aux « modèles par régime » : plutôt que de
    couper les données en quatre au moment où elles manquent le plus, on donne
    au modèle de quoi reconnaître le régime et on le laisse conditionner ses
    règles dessus — ce que les arbres font naturellement.
    """
    close = df["Close"].astype(float)
    atr_pct = indicateurs["ATR_Pct"].astype(float)

    # Rang de percentile CAUSAL : la fenêtre ne contient que du passé.
    volatilite = atr_pct.rolling(config.FENETRE_REGIME, min_periods=50).rank(pct=True)

    # Écart à la tendance longue, mesuré en bougies moyennes plutôt qu'en
    # pourcentage : 5 % au-dessus de la moyenne, c'est énorme en marché calme
    # et anodin en marché agité.
    moyenne = close.rolling(config.FENETRE_TENDANCE, min_periods=50).mean()
    atr_absolu = (atr_pct * close).replace(0, np.nan)
    tendance = (close - moyenne) / atr_absolu

    return pd.DataFrame({
        "Regime_Volatilite": volatilite,
        "Regime_Tendance": tendance.clip(-20, 20),
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
        serie = contexte[nom]
        couverture = float(serie.notna().mean())
        if couverture < config.COUVERTURE_MINIMALE:
            print(f"   ⏭️  {nom} ignorée — {couverture:.0%} de couverture "
                  f"(minimum {config.COUVERTURE_MINIMALE:.0%}).")
            continue
        # Une colonne constante n'apporte rien et fausse les mesures
        # d'importance : c'est le cas de l'heure sur des bougies journalières.
        if float(serie.dropna().nunique()) < 2:
            print(f"   ⏭️  {nom} ignorée — valeur constante sur cet intervalle.")
            continue
        gardees[nom] = serie
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

    for colonne in config.COLONNES_BRUTES:
        if colonne in df.columns:
            df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    prix = df[config.COLONNES_PRIX]
    indicateurs = calculer_indicateurs(df)
    variations = calculer_variations(df["Close"].astype(float))

    morceaux = [prix, indicateurs]
    colonnes_contexte: list[str] = []
    if contexte and intervalle:
        supplements = _rassembler_contexte(df, indicateurs, symbole, intervalle)
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

    # Toutes les colonnes de contexte autres que le multi-timeframe ont une
    # valeur NEUTRE bien définie — funding nul, flux équilibré, taille de trade
    # médiane, écart nul à la tendance. Les trous sont remplis par cette valeur
    # plutôt que par une suppression de ligne : sacrifier six années
    # d'historique parce qu'un contrat perpétuel n'existait pas encore serait
    # un très mauvais échange.
    for colonne in colonnes_contexte:
        if colonne in config.COLONNES_NEUTRALISABLES:
            resultat[colonne] = resultat[colonne].fillna(config.valeur_neutre(colonne))

    # Tout ce qui n'est pas un prix passe en float32 : la précision est
    # largement suffisante (7 chiffres significatifs) et le fichier pèse deux
    # fois moins. Les prix restent en float64 pour ne pas perdre les centimes.
    colonnes_legeres = (config.INDICATEURS + colonnes_contexte
                        + config.COLONNES_VARIATION)
    resultat[colonnes_legeres] = resultat[colonnes_legeres].astype("float32")

    resultat.index.name = "Date"
    return resultat[config.colonnes_analyse(colonnes_contexte)]


def _rassembler_contexte(df: pd.DataFrame, indicateurs: pd.DataFrame,
                         symbole: str | None,
                         intervalle: str) -> pd.DataFrame | None:
    """
    Assemble les quatre familles de colonnes de contexte réellement utilisables.

      multi-timeframe  les 8 indicateurs un cran au-dessus (4h pour du 1h) ;
      order flow       qui achète, qui vend — déjà dans les chandeliers ;
      temps            saisonnalité horaire et cycle de funding ;
      régime           où l'on se situe dans le cycle de volatilité ;
      exogènes         funding, open interest, basis perp/spot.

    Chacune est indépendante : l'absence de l'une n'empêche pas les autres.
    """
    from . import exogene  # import tardif : dépend du réseau, pas du calcul

    morceaux = []
    superieur = calculer_indicateurs_superieurs(df, intervalle)
    if superieur is not None:
        morceaux.append(superieur)

    flux = calculer_flux(df)
    if flux is not None:
        morceaux.append(flux)
        print(f"🌊 Order flow : {len(flux.columns)} colonnes (déséquilibre "
              f"acheteurs/vendeurs).")
    else:
        print("ℹ️  Order flow indisponible : relance le téléchargement pour "
              "récupérer les colonnes Trades et TakerBase.")

    temps = calculer_temps(df.index, intervalle)
    if temps is not None:
        morceaux.append(temps)

    morceaux.append(calculer_regime(df, indicateurs))

    if symbole:
        try:
            externe = exogene.contexte_exogene(
                symbole, intervalle, df.index, spot=df["Close"].astype(float))
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
