import os
import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime


class CryptoDataManager:
    def __init__(self, data_folder="data_crypto"):
        """
        Initialise le gestionnaire de données.
        Crée le dossier de stockage s'il n'existe pas.
        """
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            print(f"📁 Dossier '{self.data_folder}' créé.")
        else:
            print(f"📁 Dossier '{self.data_folder}' détecté.")
        self.COLONNES_STANDARD = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Colonnes d'order-flow Binance (microstructure du carnet) conservées EN PLUS
        # de l'OHLCV. Très utiles pour mesurer la pression acheteuse agressive.
        # Indisponibles via Yahoo : elles seront simplement absentes dans ce cas.
        self.COLONNES_ORDERFLOW = [
            'Quote_Asset_Volume', 'Trades', 'Taker_Buy_Base', 'Taker_Buy_Quote'
        ]

    def fetch_data_yahoo(self, symbole, debut, fin, intervalle="1d"):
        """
        Récupère les données depuis Yahoo Finance et les formate au standard OHLCV.
        Utile pour les cryptos non disponibles sur Binance.
        """
        ticker = f"{symbole}-USD"
        print(f"\n📥 [Yahoo] Téléchargement de {ticker} ({intervalle})...")

        data = yf.download(ticker, start=debut, end=fin, interval=intervalle, progress=False, auto_adjust=True)

        if data.empty:
            print("⚠️ Aucune donnée récupérée.")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if 'Adj Close' in data.columns and 'Close' not in data.columns:
            data = data.rename(columns={'Adj Close': 'Close'})

        try:
            data = data[self.COLONNES_STANDARD]
        except KeyError as e:
            print(f"❌ Erreur format Yahoo : Colonne manquante {e}")
            return None

        data = data.dropna()
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.index.name = 'Date'

        return data

    def fetch_data_binance(self, symbol, start_date, end_date, interval='1h'):
        """
        Récupère les chandeliers OHLCV depuis l'API publique Binance.
        Gère automatiquement la pagination (limite de 1000 bougies par requête).
        """
        symbol_pair = f"{symbol}USDT"
        base_url = "https://api.binance.com/api/v3/klines"

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        data_list = []
        print(f"📥 [Binance] Récupération de {symbol_pair} ({interval})...")

        current_start = start_ts

        while current_start < end_ts:
            params = {
                'symbol': symbol_pair, 'interval': interval,
                'startTime': current_start, 'endTime': end_ts, 'limit': 1000
            }
            try:
                r = requests.get(base_url, params=params)
                if r.status_code != 200:
                    break
                candles = r.json()
                if not candles:
                    break
                data_list.extend(candles)
                current_start = candles[-1][6] + 1
                time.sleep(0.1)
                print(f"\r   -> {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')}", end="")
            except Exception:
                break

        print("\n✅ Téléchargement terminé.")

        if not data_list:
            return None

        cols_binance = [
            'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close Time', 'Quote Asset Volume', 'Trades',
            'Taker Buy Base', 'Taker Buy Quote', 'Ignore'
        ]

        df = pd.DataFrame(data_list, columns=cols_binance)

        # Renommage des colonnes order-flow (underscore, cohérent avec les features)
        df = df.rename(columns={
            'Quote Asset Volume': 'Quote_Asset_Volume',
            'Taker Buy Base':     'Taker_Buy_Base',
            'Taker Buy Quote':    'Taker_Buy_Quote',
        })

        colonnes_a_garder = self.COLONNES_STANDARD + self.COLONNES_ORDERFLOW
        for c in colonnes_a_garder:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Date', inplace=True)
        df = df[colonnes_a_garder]

        return df

    def save_ohlcv_to_excel(self, df, symbole, intervalle, overwrite=False):
        """
        Sauvegarde un DataFrame OHLCV (index = dates) dans un fichier Excel.

        Modes :
        - overwrite=True  : écrase le fichier existant
        - overwrite=False : fusionne avec l'existant en évitant les doublons (mode update)

        Uniquement pour les fichiers de prix (index DatetimeIndex).
        Pour le fichier TOP, utiliser save_top_to_excel().
        """
        if df is None or df.empty:
            print("❌ Pas de données à sauvegarder.")
            return

        filename = f"{symbole}_{intervalle}.xlsx"
        filepath = os.path.join(self.data_folder, filename)

        if overwrite or not os.path.exists(filepath):
            df.to_excel(filepath)
            print(f"✅ Fichier CRÉÉ/ÉCRASÉ : {filepath}")
        else:
            print(f"🔄 Mise à jour du fichier existant : {filename}")

            old_df = pd.read_excel(filepath, index_col=0, parse_dates=True)

            combined_df = pd.concat([old_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()

            combined_df.to_excel(filepath)
            print(f"✅ Fichier MIS À JOUR : {filepath}")

    def save_top_to_excel(self, df, n):
        """
        Sauvegarde le classement Top N des cryptos dans un fichier Excel dédié.
        Ce fichier a un index numérique (pas de dates), il est donc toujours écrasé
        pour refléter le classement actuel au moment de l'appel.
        """
        if df is None or df.empty:
            print("❌ Pas de données TOP à sauvegarder.")
            return

        filename = f"TOP_{n}.xlsx"
        filepath = os.path.join(self.data_folder, filename)

        # Index numérique remis à zéro pour un fichier propre
        df = df.reset_index(drop=True)
        df.to_excel(filepath)
        print(f"✅ Classement TOP {n} sauvegardé : {filepath}")

    def get_top_cryptos(self, n=10, min_mcap=0):
        """
        Récupère le Top N des cryptos par capitalisation boursière via CoinGecko.
        Filtre optionnel par market cap minimum (min_mcap en USD).
        Retourne un DataFrame avec le rang, symbole, nom, prix, market cap et volume.
        """
        url = "https://api.coingecko.com/api/v3/coins/markets"
        headers = {'User-Agent': 'MonProjetEcole/1.0'}
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': n + 5,  # Légère marge pour le filtre min_mcap
            'page': 1,
            'sparkline': 'false'
        }

        print(f"🌍 Interrogation de CoinGecko (Top {n})...")

        try:
            response = requests.get(url, params=params, headers=headers)

            if response.status_code == 429:
                print("⏳ Trop de requêtes. Pause de 10s...")
                time.sleep(10)
                response = requests.get(url, params=params, headers=headers)

            if response.status_code != 200:
                print(f"❌ Erreur API CoinGecko : {response.status_code}")
                return None

            data = response.json()
            df = pd.DataFrame(data, columns=[
                'market_cap_rank', 'symbol', 'name',
                'current_price', 'market_cap', 'total_volume'
            ])

            if min_mcap > 0:
                df = df[df['market_cap'] >= min_mcap]

            df = df.head(n)
            df['symbol'] = df['symbol'].str.upper()

            return df

        except Exception as e:
            print(f"❌ Erreur critique CoinGecko : {e}")
            return None

    def extract_symbols_list(self, df_cryptos):
        """
        Extrait la liste des symboles à partir du DataFrame retourné par get_top_cryptos().
        Retourne une liste Python, ex : ['BTC', 'ETH', 'BNB', ...]
        """
        if df_cryptos is None or df_cryptos.empty:
            return []
        return df_cryptos['symbol'].tolist()

    # Symboles à ignorer pour le mode Portefeuille : stablecoins et actifs sans
    # paire spot ...USDT exploitable (téléchargement voué à l'échec).
    STABLECOINS = {'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FDUSD', 'USDE', 'PYUSD'}

    def telecharger_top_n(self, n, debut, fin, intervalle='1h', source='Binance'):
        """
        Pipeline 'Portefeuille' : récupère le Top N CoinGecko puis télécharge
        l'historique OHLCV + order-flow de chaque crypto et le sauvegarde
        (un fichier {SYMBOLE}_{intervalle}.xlsx par actif).

        Les stablecoins et les paires indisponibles sont automatiquement ignorés.
        Retourne la liste des symboles effectivement téléchargés.
        """
        df_top = self.get_top_cryptos(n)
        self.save_top_to_excel(df_top, n)
        symboles = self.extract_symbols_list(df_top)
        print(f"📋 Top {n} CoinGecko : {symboles}")

        reussis = []
        for sym in symboles:
            if sym in self.STABLECOINS:
                print(f"⏭️  {sym} : stablecoin ignoré.")
                continue
            try:
                if source == 'Binance':
                    df = self.fetch_data_binance(sym, debut, fin, intervalle)
                else:
                    df = self.fetch_data_yahoo(sym, debut, fin, intervalle)

                if df is not None and not df.empty:
                    self.save_ohlcv_to_excel(df, sym, intervalle, overwrite=True)
                    reussis.append(sym)
                else:
                    print(f"⚠️  {sym} : aucune donnée (paire indisponible ?), ignoré.")
            except Exception as e:
                print(f"❌ {sym} : échec ({e}), ignoré.")

        print(f"\n✅ Portefeuille téléchargé : {len(reussis)}/{len(symboles)} "
              f"cryptos → {reussis}")
        return reussis


# --- EXÉCUTION PRINCIPALE ---
if __name__ == "__main__":

    N_CRYPTOS = 1

    manager = CryptoDataManager()

    # Récupération et sauvegarde du Top N (fichier de référence, toujours écrasé)
    df_cryptos = manager.get_top_cryptos(N_CRYPTOS)
    manager.save_top_to_excel(df_cryptos, N_CRYPTOS)  # ← Fonction dédiée

    liste_cryptos = manager.extract_symbols_list(df_cryptos)
    print(f"📋 Cryptos sélectionnées : {liste_cryptos}")

    for crypto in liste_cryptos:
        start = "2020-01-01"
        end   = "2026-01-01"
        freq  = "1h"

        df_result = manager.fetch_data_binance(crypto, start, end, freq)
        manager.save_ohlcv_to_excel(df_result, crypto, freq, overwrite=True)  # ← Fonction dédiée