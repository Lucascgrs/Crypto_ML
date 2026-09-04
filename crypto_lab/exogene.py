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

# Plafonds imposés par l'API publique.
FUNDING_PAR_REQUETE = 1000
OI_PAR_REQUETE = 500

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
    open_interest = telecharger_open_interest(symbole, intervalle)
    if funding is None and open_interest is None:
        return None

    nouveau = pd.concat([s for s in (funding, open_interest) if s is not None], axis=1)
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
            fenetre: int = FENETRE_CUMUL) -> pd.DataFrame:
    """
    Transforme les séries brutes en features stationnaires alignées sur `index`.

    Trois colonnes produites (voir `config.COLONNES_EXOGENES`) :

      Funding_Rate    dernier taux de funding en vigueur (report en avant).
      Funding_Cumul   somme des funding EFFECTIVEMENT versés sur la fenêtre.
                      Le report en avant compterait trois fois le même
                      versement de 8 h sur une fenêtre de 24 h ; on somme donc
                      les événements, pas la série reportée.
      OI_Variation    variation de l'open interest sur la fenêtre, en %.

    Toutes trois sont centrées sur zéro et sans unité de prix : elles se
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

    if "Open_Interest" in brut.columns:
        oi = brut["Open_Interest"].dropna()
        if not oi.empty:
            aligne = oi.reindex(index.union(oi.index)).ffill().reindex(index)
            # Le report en avant ne doit pas inventer de données AVANT le premier
            # point connu : ces lignes restent vides, et c'est voulu.
            aligne[index < oi.index.min()] = np.nan
            resultat["OI_Variation"] = aligne.pct_change(fenetre) * 100

    return resultat.replace([np.inf, -np.inf], np.nan)


def contexte_exogene(symbole: str, intervalle: str,
                     index: pd.Index) -> pd.DataFrame | None:
    """
    Colonnes exogènes prêtes à concaténer, ou None si rien n'est disponible.

    Appelée par `indicateurs.analyser` : l'absence de fichier exogène n'est pas
    une erreur, elle signifie simplement que ces colonnes ne seront pas créées.
    """
    brut = charger(symbole, intervalle)
    if brut is None:
        return None
    aligne = aligner(brut, index)
    return aligne if not aligne.empty and aligne.notna().any().any() else None


def couverture(df: pd.DataFrame) -> dict[str, float]:
    """Part de lignes renseignées, colonne par colonne — sert au diagnostic."""
    if df is None or df.empty:
        return {}
    return {str(colonne): float(df[colonne].notna().mean()) for colonne in df.columns}


def chemin_disponible(symbole: str, intervalle: str) -> bool:
    """Vrai si un historique exogène a déjà été téléchargé pour cette paire."""
    return os.path.exists(stockage.chemin_exogene(symbole, intervalle))
