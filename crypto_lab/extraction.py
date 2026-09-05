"""
Récupération des données de marché.

Trois sources :
  - **Binance** (par défaut) : chandeliers OHLCV, historique profond, gratuit.
  - **Yahoo Finance** : repli pour les cryptos absentes de Binance.
  - **CoinGecko** : classement des cryptos par capitalisation (Top N).

On conserve l'OHLCV **et deux colonnes d'order flow** que Binance renvoie déjà
dans chaque chandelier, sans requête supplémentaire : le nombre de trades et le
volume acheté à l'agressif. Elles étaient jetées jusqu'ici, alors qu'elles sont
la seule information du fichier qui ne soit pas dérivée du prix — voir
`config.COLONNES_FLUX`. Yahoo ne les fournit pas : les cryptos rapatriées par
ce repli n'auront simplement pas ces trois features.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import requests

from . import config, stockage

# Cryptos sans paire spot ...USDT exploitable : téléchargement voué à l'échec.
STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDE", "PYUSD"}

URL_BINANCE = "https://api.binance.com/api/v3/klines"
URL_COINGECKO = "https://api.coingecko.com/api/v3/coins/markets"

# Binance plafonne à 1000 bougies par requête : la pagination est obligatoire.
BOUGIES_PAR_REQUETE = 1000


# ===========================================================================
# BINANCE
# ===========================================================================
def telecharger_binance(symbole: str, debut: str, fin: str,
                        intervalle: str = "1h") -> pd.DataFrame | None:
    """
    Télécharge l'historique OHLCV d'une crypto depuis l'API publique Binance.

    `debut` / `fin` au format AAAA-MM-JJ. La pagination est gérée
    automatiquement (1000 bougies par appel).
    """
    paire = f"{symbole}USDT"
    horodatage_debut = int(datetime.strptime(debut, "%Y-%m-%d").timestamp() * 1000)
    horodatage_fin = int(datetime.strptime(fin, "%Y-%m-%d").timestamp() * 1000)

    print(f"📥 [Binance] {paire} ({intervalle})…")
    bougies: list[list] = []
    curseur = horodatage_debut

    while curseur < horodatage_fin:
        parametres = {
            "symbol": paire, "interval": intervalle,
            "startTime": curseur, "endTime": horodatage_fin,
            "limit": BOUGIES_PAR_REQUETE,
        }
        try:
            reponse = requests.get(URL_BINANCE, params=parametres, timeout=20)
            if reponse.status_code != 200:
                print(f"\n⚠️ Binance a répondu {reponse.status_code} — arrêt.")
                break
            lot = reponse.json()
            if not lot:
                break
            bougies.extend(lot)
            curseur = lot[-1][6] + 1          # index 6 = heure de clôture
            print(f"\r   → {datetime.fromtimestamp(curseur / 1000):%Y-%m-%d}", end="")
            time.sleep(0.1)                    # respect du quota public
        except requests.RequestException as err:
            print(f"\n⚠️ Réseau : {err}")
            break

    print()
    if not bougies:
        return None

    colonnes = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote", "Trades", "TakerBase", "TakerQuote", "Ignore"]
    df = pd.DataFrame(bougies, columns=colonnes)
    df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
    df = df.set_index("Date")

    for colonne in config.COLONNES_BRUTES:
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    # Les colonnes d'order flow ne conditionnent pas la validité d'une bougie :
    # on ne supprime une ligne que si son OHLCV est incomplet.
    df = df[config.COLONNES_BRUTES].dropna(subset=config.COLONNES_PRIX)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"✅ {len(df):,} bougies récupérées "
          f"(order flow inclus : {len(config.COLONNES_FLUX_BRUT)} colonnes).")
    return df


# ===========================================================================
# YAHOO FINANCE (repli)
# ===========================================================================
def telecharger_yahoo(symbole: str, debut: str, fin: str,
                      intervalle: str = "1d") -> pd.DataFrame | None:
    """Télécharge l'OHLCV depuis Yahoo Finance (ticker `{SYMBOLE}-USD`)."""
    import yfinance as yf  # import tardif : librairie lente à charger

    ticker = f"{symbole}-USD"
    print(f"📥 [Yahoo] {ticker} ({intervalle})…")

    df = yf.download(ticker, start=debut, end=fin, interval=intervalle,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        print("⚠️ Aucune donnée récupérée.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    manquantes = [c for c in config.COLONNES_PRIX if c not in df.columns]
    if manquantes:
        print(f"❌ Format Yahoo inattendu, colonnes manquantes : {manquantes}")
        return None

    df = df[config.COLONNES_PRIX].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    print(f"✅ {len(df):,} bougies récupérées.")
    return df


def telecharger(symbole: str, debut: str, fin: str, intervalle: str = "1h",
                source: str = "Binance") -> pd.DataFrame | None:
    """Point d'entrée unique : aiguille vers Binance ou Yahoo."""
    if source.lower().startswith("y"):
        return telecharger_yahoo(symbole, debut, fin, intervalle)
    return telecharger_binance(symbole, debut, fin, intervalle)


# ===========================================================================
# SAUVEGARDE
# ===========================================================================
def sauvegarder(df: pd.DataFrame | None, symbole: str, intervalle: str,
                ecraser: bool = True) -> str | None:
    """
    Écrit l'historique OHLCV dans `data_crypto/{SYMBOLE}_{INTERVALLE}.xlsx`.

    `ecraser=False` fusionne avec l'existant (mise à jour incrémentale) en
    conservant la version la plus récente de chaque bougie.
    """
    if df is None or df.empty:
        print("❌ Rien à sauvegarder.")
        return None

    chemin = stockage.chemin_brut(symbole, intervalle)
    if not ecraser:
        ancien = stockage.lire_tableau(chemin, cache=False)
        if ancien is not None and not ancien.empty:
            df = pd.concat([ancien, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            print(f"🔄 Fusion avec l'historique existant → {len(df):,} lignes.")

    stockage.ecrire_tableau(df, chemin)
    print(f"💾 Sauvegardé : {chemin}")
    return chemin


# ===========================================================================
# COINGECKO — CLASSEMENT DU MARCHÉ
# ===========================================================================
def top_cryptos(n: int = 10) -> pd.DataFrame | None:
    """
    Classement des `n` premières cryptos par capitalisation (CoinGecko).

    Retourne un tableau : rang, symbole, nom, prix, capitalisation, volume.
    """
    parametres = {"vs_currency": "usd", "order": "market_cap_desc",
                  "per_page": n + 5, "page": 1, "sparkline": "false"}
    entetes = {"User-Agent": "CryptoLab/2.0"}

    print(f"🌍 CoinGecko — Top {n}…")
    try:
        reponse = requests.get(URL_COINGECKO, params=parametres,
                               headers=entetes, timeout=20)
        if reponse.status_code == 429:
            print("⏳ Quota atteint, pause de 10 s…")
            time.sleep(10)
            reponse = requests.get(URL_COINGECKO, params=parametres,
                                   headers=entetes, timeout=20)
        if reponse.status_code != 200:
            print(f"❌ CoinGecko : erreur {reponse.status_code}")
            return None

        df = pd.DataFrame(reponse.json(), columns=[
            "market_cap_rank", "symbol", "name",
            "current_price", "market_cap", "total_volume"])
        df["symbol"] = df["symbol"].str.upper()
        return df.head(n).reset_index(drop=True)
    except requests.RequestException as err:
        print(f"❌ CoinGecko injoignable : {err}")
        return None


def sauvegarder_top(df: pd.DataFrame | None, n: int) -> None:
    """Écrit le classement dans `data_crypto/TOP_{n}.xlsx` (toujours écrasé)."""
    if df is None or df.empty:
        return
    import os
    chemin = os.path.join(config.DOSSIER_DONNEES, f"TOP_{n}.xlsx")
    df.reset_index(drop=True).to_excel(chemin)
    print(f"💾 Classement sauvegardé : {chemin}")


def telecharger_top_n(n: int, debut: str, fin: str, intervalle: str = "1h",
                      source: str = "Binance") -> list[str]:
    """
    Télécharge d'un coup l'historique des `n` premières cryptos du marché.

    Les stablecoins et les paires indisponibles sont ignorés sans bloquer le
    reste du lot. Retourne la liste des symboles effectivement récupérés.
    """
    classement = top_cryptos(n)
    if classement is None:
        return []
    sauvegarder_top(classement, n)

    symboles = classement["symbol"].tolist()
    print(f"📋 Top {n} : {symboles}")

    reussis: list[str] = []
    for symbole in symboles:
        if symbole in STABLECOINS:
            print(f"⏭️  {symbole} : stablecoin ignoré.")
            continue
        try:
            df = telecharger(symbole, debut, fin, intervalle, source)
            if df is not None and not df.empty:
                sauvegarder(df, symbole, intervalle, ecraser=True)
                reussis.append(symbole)
            else:
                print(f"⚠️  {symbole} : aucune donnée, ignoré.")
        except Exception as err:                       # noqa: BLE001
            print(f"❌ {symbole} : {err}")

    print(f"\n📦 {len(reussis)}/{len(symboles)} cryptos téléchargées : {reussis}")
    return reussis
