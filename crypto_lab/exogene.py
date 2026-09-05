"""
Données exogènes du marché des dérivés (API publique Binance Futures).

POURQUOI CE MODULE EXISTE
-------------------------
Les 8 indicateurs, leur version multi-timeframe et les 24 variations sont tous
calculés à partir du MÊME OHLCV. Ce sont des transformations d'une seule et
même information : on peut en empiler cent, on n'apprend rien de nouveau. Le
funding rate et l'open interest viennent d'ailleurs — du positionnement réel
des intervenants sur les contrats à terme. C'est le seul gisement
d'information vraiment neuve disponible gratuitement.

  * **Funding rate** — toutes les 8 heures, les positions longues paient les
    courtes (ou l'inverse) selon l'écart entre le prix du perpétuel et le spot.
    Un funding durablement positif signale un positionnement acheteur tendu,
    souvent avant une purge. Déjà stationnaire, de l'ordre de 0.0001.

  * **Basis perp/spot** — écart entre le prix du contrat perpétuel et celui du
    spot, en %. Quand les acheteurs à levier dominent, le perpétuel se négocie
    au-dessus du spot ; le basis mesure directement combien ils sont prêts à
    payer pour rester exposés. Son historique est COMPLET dès le lancement du
    contrat et se récupère en une seule passe, sans rien accumuler.

  * **Open interest** — nombre de contrats ouverts. C'est un NIVEAU, donc
    inutilisable tel quel : on en prend la variation sur 24 périodes. Un prix
    qui monte avec l'open interest en hausse, ce sont de nouvelles positions
    acheteuses ; le même prix avec l'open interest en baisse, ce ne sont que
    des shorts qui se débouclent. Deux situations opposées, invisibles dans le
    prix seul.

LA LIMITE À CONNAÎTRE
---------------------
L'historique de funding remonte au lancement du contrat (2019-2020 selon les
paires) : il est immédiatement exploitable. **L'open interest public de
Binance ne remonte qu'à 30 jours.** Sur un historique de plusieurs années, la
colonne serait vide à 99 % — c'est pourquoi ce module tient un fichier
d'historique CUMULATIF : chaque téléchargement complète le précédent, et la
couverture s'étend d'elle-même au fil des mises à jour.

Tant qu'une colonne de contexte ne couvre pas au moins
`config.COUVERTURE_MINIMALE` des lignes, elle n'est pas retenue comme feature
(voir `indicateurs.analyser`). Mieux vaut pas de colonne du tout qu'une colonne
vide qui ferait jeter tout l'historique.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import config, stockage

URL_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
URL_OPEN_INTEREST = "https://fapi.binance.com/futures/data/openInterestHist"
URL_PERP = "https://fapi.binance.com/fapi/v1/klines"

# Plafonds imposés par l'API publique.
FUNDING_PAR_REQUETE = 1000
OI_PAR_REQUETE = 500
PERP_PAR_REQUETE = 1500

# Périodes acceptées par l'endpoint open interest.
PERIODES_OI = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}

# Fenêtre (en périodes) sur laquelle on cumule le funding et on mesure la
# variation d'open interest. 24 périodes = une journée en 1h : assez long pour
# lisser le bruit, assez court pour rester réactif.
FENETRE_CUMUL = 24


# ===========================================================================
# TÉLÉCHARGEMENT
# ===========================================================================
def _horodatage(date: str) -> int:
    """Date AAAA-MM-JJ -> millisecondes UTC."""
    return int(datetime.strptime(date, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def telecharger_funding(symbole: str, debut: str, fin: str) -> pd.Series | None:
    """
    Historique complet du funding rate d'une paire perpétuelle.

    Retourne une série indexée par l'instant d'application du funding (toutes
    les 8 heures), ou None si la paire n'existe pas en futures.
    """
    paire = f"{symbole}USDT"
    curseur = _horodatage(debut)
    limite = _horodatage(fin)

    print(f"📥 [Binance Futures] funding rate {paire}…")
    lignes: list[dict] = []

    while curseur < limite:
        try:
            reponse = requests.get(
                URL_FUNDING, timeout=20,
                params={"symbol": paire, "startTime": curseur,
                        "endTime": limite, "limit": FUNDING_PAR_REQUETE})
        except requests.RequestException as err:
            print(f"\n⚠️ Réseau : {err}")
            break

        if reponse.status_code != 200:
            print(f"⚠️ Funding indisponible pour {paire} "
                  f"(HTTP {reponse.status_code}) — colonne ignorée.")
            return None

        lot = reponse.json()
        if not lot:
            break
        lignes.extend(lot)
        curseur = int(lot[-1]["fundingTime"]) + 1
        print(f"\r   → {datetime.fromtimestamp(curseur / 1000):%Y-%m-%d}", end="")
        time.sleep(0.1)                                # respect du quota public

    print()
    if not lignes:
        print("⚠️ Aucun funding récupéré.")
        return None

    df = pd.DataFrame(lignes)
    serie = pd.Series(
        pd.to_numeric(df["fundingRate"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["fundingTime"], unit="ms"), name="Funding_Brut")
    serie = serie[~serie.index.duplicated(keep="last")].sort_index().dropna()
    print(f"✅ {len(serie):,} versements de funding "
          f"({serie.index.min():%Y-%m-%d} → {serie.index.max():%Y-%m-%d}).")
    return serie


def telecharger_open_interest(symbole: str, intervalle: str) -> pd.Series | None:
    """
    Open interest agrégé, à la granularité de l'intervalle de travail.

    ⚠️ L'API publique ne conserve que ~30 jours. Cette fonction rend donc
    toujours une fenêtre courte ; c'est `mettre_a_jour` qui l'accumule dans le
    temps pour construire un historique utilisable.
    """
    paire = f"{symbole}USDT"
    periode = intervalle if intervalle in PERIODES_OI else "1h"

    print(f"📥 [Binance Futures] open interest {paire} ({periode})…")
    try:
        reponse = requests.get(
            URL_OPEN_INTEREST, timeout=20,
            params={"symbol": paire, "period": periode, "limit": OI_PAR_REQUETE})
    except requests.RequestException as err:
        print(f"⚠️ Réseau : {err}")
        return None

    if reponse.status_code != 200:
        print(f"⚠️ Open interest indisponible pour {paire} "
              f"(HTTP {reponse.status_code}) — colonne ignorée.")
        return None

    lot = reponse.json()
    if not lot:
        print("⚠️ Aucun open interest récupéré.")
        return None

    df = pd.DataFrame(lot)
    serie = pd.Series(
        pd.to_numeric(df["sumOpenInterest"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["timestamp"], unit="ms"), name="Open_Interest")
    serie = serie[~serie.index.duplicated(keep="last")].sort_index().dropna()
    print(f"✅ {len(serie):,} points d'open interest "
          f"({serie.index.min():%Y-%m-%d} → {serie.index.max():%Y-%m-%d}).")
    return serie


# Nombre maximum de périodes pendant lesquelles une valeur exogène peut être
# reportée en avant. Un trou d'une ou deux bougies dans la série perpétuelle est
# banal et sans conséquence ; au-delà, reporter ne produit pas une donnée
# approximative, cela produit une donnée FAUSSE. Le cas réel qui a motivé cette
# borne : une série tronquée au milieu de l'historique, dont la dernière valeur
# se retrouvait reportée sur quatorze mois — soit exactement la période de test.
REPORT_MAX = 3

# Au-delà, on considère que la série s'arrête franchement trop tôt et on le dit.
TOLERANCE_FIN_JOURS = 7


def telecharger_perp(symbole: str, debut: str, fin: str,
                     intervalle: str = "1h") -> pd.Series | None:
    """
    Clôtures du contrat PERPÉTUEL, à la granularité de l'intervalle de travail.

    Sert uniquement à calculer le basis (écart perp/spot). Contrairement à
    l'open interest, l'historique est intégral dès le lancement du contrat :
    une seule passe suffit, il n'y a rien à accumuler dans le temps.
    """
    paire = f"{symbole}USDT"
    curseur = _horodatage(debut)
    limite = _horodatage(fin)

    print(f"📥 [Binance Futures] perpétuel {paire} ({intervalle})…")
    bougies: list[list] = []

    while curseur < limite:
        # Une coupure réseau passagère ne doit PAS tronquer silencieusement
        # l'historique : une série coupée en son milieu est plus dangereuse
        # qu'une série absente, parce qu'elle a l'air complète.
        reponse = None
        for tentative in range(3):
            try:
                reponse = requests.get(
                    URL_PERP, timeout=20,
                    params={"symbol": paire, "interval": intervalle,
                            "startTime": curseur, "endTime": limite,
                            "limit": PERP_PAR_REQUETE})
                break
            except requests.RequestException as err:
                print(f"\n⚠️ Réseau ({tentative + 1}/3) : {err}")
                time.sleep(1.5 * (tentative + 1))

        if reponse is None:
            print(f"\n⚠️ Perpétuel {paire} interrompu à "
                  f"{datetime.fromtimestamp(curseur / 1000):%Y-%m-%d} après "
                  f"trois tentatives — la série récupérée est INCOMPLÈTE. "
                  f"Le basis sera vide au-delà de cette date plutôt que "
                  f"reporté en avant (ce qui produirait une fausse feature).")
            break

        if reponse.status_code != 200:
            print(f"⚠️ Perpétuel indisponible pour {paire} "
                  f"(HTTP {reponse.status_code}) — basis ignoré.")
            return None

        lot = reponse.json()
        if not lot:
            break
        bougies.extend(lot)
        curseur = int(lot[-1][6]) + 1                  # index 6 = heure de clôture
        print(f"\r   → {datetime.fromtimestamp(curseur / 1000):%Y-%m-%d}", end="")
        time.sleep(0.1)                                # respect du quota public

    print()
    if not bougies:
        print("⚠️ Aucune bougie perpétuelle récupérée.")
        return None

    manque = (limite - int(bougies[-1][6])) / 86_400_000
    if manque > TOLERANCE_FIN_JOURS:
        print(f"⚠️ La série perpétuelle s'arrête {manque:,.0f} jours avant la "
              f"fin demandée ({datetime.fromtimestamp(int(bougies[-1][6]) / 1000):%Y-%m-%d}). "
              f"Le basis sera vide sur cette période — relance "
              f"« GatherData.py {symbole} --exogene » pour la compléter.")

    df = pd.DataFrame(bougies)
    serie = pd.Series(
        pd.to_numeric(df[4], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[0], unit="ms"), name="Perp_Close")
    serie = serie[~serie.index.duplicated(keep="last")].sort_index().dropna()
    print(f"✅ {len(serie):,} bougies perpétuelles "
          f"({serie.index.min():%Y-%m-%d} → {serie.index.max():%Y-%m-%d}).")
    return serie


# ===========================================================================
# HISTORIQUE CUMULATIF SUR DISQUE
# ===========================================================================
def mettre_a_jour(symbole: str, intervalle: str, debut: str = "2019-09-01",
                  fin: str | None = None) -> pd.DataFrame | None:
    """
    Télécharge les séries exogènes et les FUSIONNE avec l'historique déjà stocké.

    La fusion est ce qui rend l'open interest exploitable : chaque passage
    ajoute les 30 derniers jours à ce qui avait été collecté auparavant. En
    relançant l'extraction régulièrement, la couverture finit par atteindre le
    seuil au-delà duquel la colonne devient une vraie feature.

    Retourne le tableau cumulé (colonnes `Funding_Brut`, `Open_Interest`).
    """
    config.preparer_dossiers()
    fin = fin or datetime.now().strftime("%Y-%m-%d")

    funding = telecharger_funding(symbole, debut, fin)
    perp = telecharger_perp(symbole, debut, fin, intervalle)
    open_interest = telecharger_open_interest(symbole, intervalle)
    series = [s for s in (funding, perp, open_interest) if s is not None]
    if not series:
        return None

    nouveau = pd.concat(series, axis=1)
    nouveau.index.name = "Date"

    chemin = stockage.chemin_exogene(symbole, intervalle)
    ancien = stockage.lire_tableau(chemin, cache=False)
    if ancien is not None and not ancien.empty:
        avant = len(ancien)
        nouveau = pd.concat([ancien, nouveau])
        nouveau = nouveau.groupby(level=0).last().sort_index()
        print(f"🔄 Fusion avec l'historique exogène existant : "
              f"{avant:,} → {len(nouveau):,} lignes.")

    stockage.ecrire_tableau(nouveau, chemin)
    print(f"💾 Données exogènes sauvegardées : {chemin}")
    _resumer(nouveau)
    return nouveau


def _resumer(df: pd.DataFrame) -> None:
    """Rappelle ce qui est couvert, colonne par colonne."""
    for colonne in df.columns:
        serie = df[colonne].dropna()
        if serie.empty:
            continue
        print(f"   {colonne:<14} {len(serie):>7,} points | "
              f"{serie.index.min():%Y-%m-%d} → {serie.index.max():%Y-%m-%d}")


def charger(symbole: str, intervalle: str) -> pd.DataFrame | None:
    """Historique exogène stocké sur disque, ou None s'il n'y en a pas."""
    df = stockage.lire_tableau(stockage.chemin_exogene(symbole, intervalle))
    if df is None or df.empty:
        return None
    return df


# ===========================================================================
# MISE AU FORMAT « FEATURE »
# ===========================================================================
def aligner(brut: pd.DataFrame, index: pd.Index,
            fenetre: int = FENETRE_CUMUL,
            spot: pd.Series | None = None) -> pd.DataFrame:
    """
    Transforme les séries brutes en features stationnaires alignées sur `index`.

    Trois colonnes produites (voir `config.COLONNES_EXOGENES`) :

      Funding_Rate    dernier taux de funding en vigueur (report en avant).
      Funding_Cumul   somme des funding EFFECTIVEMENT versés sur la fenêtre.
                      Le report en avant compterait trois fois le même
                      versement de 8 h sur une fenêtre de 24 h ; on somme donc
                      les événements, pas la série reportée.
      OI_Variation    variation de l'open interest sur la fenêtre, en %.
      Basis           (perp / spot − 1) × 100 : la prime payée pour le levier.
      Basis_Moyenne   sa moyenne sur la fenêtre — la tension de fond.

    Toutes sont centrées sur zéro et sans unité de prix : elles se
    comparent dans le temps et d'une crypto à l'autre, comme les 8 indicateurs.
    """
    resultat = pd.DataFrame(index=index)

    if "Funding_Brut" in brut.columns:
        funding = brut["Funding_Brut"].dropna()
        if not funding.empty:
            # Niveau courant : le dernier taux appliqué, reporté en avant.
            resultat["Funding_Rate"] = funding.reindex(
                index.union(funding.index)).ffill().reindex(index)

            # Cumul : on place chaque versement sur la bougie qui le contient,
            # puis on somme sur la fenêtre. Un versement = une ligne, jamais trois.
            evenements = pd.Series(0.0, index=index)
            positions = index.searchsorted(funding.index, side="right") - 1
            garde = (positions >= 0) & (positions < len(index))
            if garde.any():
                evenements.iloc[:] = np.bincount(
                    positions[garde], weights=funding.to_numpy()[garde],
                    minlength=len(index))
            resultat["Funding_Cumul"] = evenements.rolling(fenetre, min_periods=1).sum()

    if "Perp_Close" in brut.columns and spot is not None:
        perp = brut["Perp_Close"].dropna()
        if not perp.empty:
            # Le report en avant est BORNÉ des deux côtés. Sans cette borne, une
            # série tronquée fait du basis une mesure de l'écart de prix depuis
            # la troncature : une valeur qui dérive avec le marché, saturée à la
            # limite du clip, et parfaitement corrélée au niveau du prix — soit
            # précisément la fuite que la liste blanche des features interdit.
            aligne = perp.reindex(index.union(perp.index)) \
                         .ffill(limit=REPORT_MAX).reindex(index)
            aligne[(index < perp.index.min()) | (index > perp.index.max())] = np.nan

            couvert = float(aligne.notna().mean())
            if couvert < 0.98:
                print(f"ℹ️  Basis disponible sur {couvert:.0%} des lignes "
                      f"(perpétuel jusqu'au {perp.index.max():%Y-%m-%d}). "
                      f"Les lignes sans perpétuel restent VIDES : elles seront "
                      f"neutralisées, pas inventées.")

            reference = pd.to_numeric(spot.reindex(index), errors="coerce")
            basis = (aligne / reference.replace(0, np.nan) - 1) * 100
            # Le basis dépasse rarement ±1 % ; au-delà c'est un décalage
            # d'horodatage ou une bougie manquante, pas une information.
            resultat["Basis"] = basis.clip(-5, 5)
            # `min_periods=1` sur une fenêtre entièrement vide rendrait NaN,
            # ce qui est le comportement voulu : pas de moyenne sans données.
            resultat["Basis_Moyenne"] = resultat["Basis"].rolling(
                fenetre, min_periods=1).mean()

    if "Open_Interest" in brut.columns:
        oi = brut["Open_Interest"].dropna()
        if not oi.empty:
            aligne = oi.reindex(index.union(oi.index)) \
                       .ffill(limit=REPORT_MAX).reindex(index)
            # Le report en avant ne doit inventer de données ni AVANT le premier
            # point connu, ni APRÈS le dernier : ces lignes restent vides, et
            # c'est voulu.
            aligne[(index < oi.index.min()) | (index > oi.index.max())] = np.nan
            resultat["OI_Variation"] = aligne.pct_change(fenetre) * 100

    return resultat.replace([np.inf, -np.inf], np.nan)


def contexte_exogene(symbole: str, intervalle: str, index: pd.Index,
                     spot: pd.Series | None = None) -> pd.DataFrame | None:
    """
    Colonnes exogènes prêtes à concaténer, ou None si rien n'est disponible.

    Appelée par `indicateurs.analyser` : l'absence de fichier exogène n'est pas
    une erreur, elle signifie simplement que ces colonnes ne seront pas créées.
    """
    brut = charger(symbole, intervalle)
    if brut is None:
        return None
    aligne = aligner(brut, index, spot=spot)
    return aligne if not aligne.empty and aligne.notna().any().any() else None


def couverture(df: pd.DataFrame) -> dict[str, float]:
    """Part de lignes renseignées, colonne par colonne — sert au diagnostic."""
    if df is None or df.empty:
        return {}
    return {str(colonne): float(df[colonne].notna().mean()) for colonne in df.columns}


def chemin_disponible(symbole: str, intervalle: str) -> bool:
    """Vrai si un historique exogène a déjà été téléchargé pour cette paire."""
    return os.path.exists(stockage.chemin_exogene(symbole, intervalle))
