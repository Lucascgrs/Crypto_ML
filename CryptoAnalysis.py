import pandas as pd
import numpy as np
import os
import requests


# ==========================================
# COLONNES À NE JAMAIS DONNER AU MODÈLE
# ==========================================
# Source UNIQUE de vérité, importée par Predict.build_features.
# Toute colonne ici est soit un prix brut, soit une valeur d'ÉCHELLE DE PRIX
# (non stationnaire), soit de l'order-flow brut. On les garde dans le fichier
# analysé (utiles aux graphes / au calcul de la cible) mais elles ne doivent
# JAMAIS servir de feature : sinon le modèle mémorise le niveau de prix et
# s'effondre hors échantillon.
# Les colonnes cibles (préfixes 'Target_' et 'Ret_') sont gérées à part.
COLONNES_NON_FEATURES = {
    # Prix bruts (non stationnaires)
    'Open', 'High', 'Low', 'Close', 'Volume',
    # Order-flow brut (échelle absolue → on dérive des ratios stationnaires à la place)
    'Quote_Asset_Volume', 'Trades', 'Taker_Buy_Base', 'Taker_Buy_Quote',
    # Artefacts Yahoo Finance
    'Dividends', 'Stock Splits',
    # Indicateurs de NIVEAU de prix (gardés pour les graphes, jamais en features)
    'SMA_20', 'SMA_50', 'SMA_200',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB',
}


# ==========================================
# 1. CLASSE DE CHARGEMENT & SAUVEGARDE
# ==========================================
class CryptoDataLoader:
    def __init__(self, data_folder="data_crypto", write_folder="analysis_crypto"):
        """
        Initialise le chargeur de données.
        data_folder : Dossier source (Excel bruts)
        write_folder : Dossier destination (Excel analysés)
        """
        self.data_folder = data_folder
        self.write_folder = write_folder

        # Vérification dossier source
        if not os.path.exists(self.data_folder):
            print(f"⚠️ Attention : Le dossier source '{self.data_folder}' n'existe pas.")

        # Création dossier destination
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)
            print(f"📁 Dossier de sortie '{self.write_folder}' créé.")
        else:
            print(f"📁 Dossier de sortie '{self.write_folder}' détecté.")

    def load_crypto_data(self, symbole, intervalle):
        """ Lit un fichier Excel brut """
        filename = f"{symbole}_{intervalle}.xlsx"
        filepath = os.path.join(self.data_folder, filename)

        print(f"📂 Chargement du fichier : {filename} ...")

        if not os.path.exists(filepath):
            print(f"❌ Erreur : Le fichier '{filepath}' est introuvable.")
            return None

        try:
            df = pd.read_excel(filepath, index_col=0, parse_dates=True)
            df = df.sort_index()
            print(f"✅ Chargé avec succès : {len(df)} lignes.")
            return df
        except Exception as e:
            print(f"❌ Erreur lors de la lecture : {e}")
            return None

    def save_to_excel(self, df, symbole, intervalle):
        """ Sauvegarde le fichier analysé """
        if df is None or df.empty:
            print("❌ Pas de données à sauvegarder.")
            return

        filename = f"{symbole}_{intervalle}_analyzed.xlsx"
        filepath = os.path.join(self.write_folder, filename)

        try:
            df.to_excel(filepath)
            print(f"✅ Fichier analysé sauvegardé : {filepath}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde : {e}")


# ==========================================
# 2. CLASSE D'INGÉNIERIE (FEATURE ENGINEER)
# ==========================================
class CryptoFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def add_basic_indicators(self):
        """ Variation Prix, Volume et Log Returns """
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Vol_Change'] = self.df['Volume'].pct_change()
        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        return self.df

    def add_moving_averages(self):
        """
        MA20, MA50, MA200 + Distances + Croisements
        + PENTES (Nouveau !)
        """
        mas = [20, 50, 200]

        for ma in mas:
            col_name = f'SMA_{ma}'
            self.df[col_name] = self.df['Close'].rolling(window=ma).mean()

            # 1. Distance (%) : Où on est ?
            dist_col = f'Dist_SMA_{ma}'
            self.df[dist_col] = (self.df['Close'] - self.df[col_name]) / self.df[col_name]

            # 2. Pente de la SMA (Slope) : Où va la tendance ?
            # On regarde la variation de la moyenne par rapport à la bougie précédente
            # Si positif = La moyenne monte. Si négatif = La moyenne baisse.
            self.df[f'Slope_SMA_{ma}'] = np.arctan(self.df[col_name].diff() / self.df[col_name])
            # Note : np.arctan normalise un peu la valeur (en radians), c'est propre pour l'IA.

            # 3. Dynamique de la distance : On s'approche ou on s'éloigne ?
            # Si positif = On s'éloigne vers le haut (Momentum)
            # Si négatif = On revient vers la moyenne (Mean Reversion) ou on plonge
            self.df[f'Delta_Dist_{ma}'] = self.df[dist_col].diff()

        # Croisements classiques
        self.df['Cross_50_200'] = (self.df['SMA_50'] - self.df['SMA_200']) / self.df['SMA_200']
        self.df['Cross_20_50'] = (self.df['SMA_20'] - self.df['SMA_50']) / self.df['SMA_50']

        return self.df

    def add_bollinger_bands(self, window=20, num_std=2):
        """ Bollinger Width & Position """
        sma = self.df['Close'].rolling(window=window).mean()
        std = self.df['Close'].rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)

        self.df['BB_Width'] = (upper - lower) / sma
        self.df['BB_Position'] = (self.df['Close'] - lower) / (upper - lower)
        return self.df

    def add_rsi(self, window=14):
        """ RSI """
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.ewm(com=window - 1, adjust=True, min_periods=window).mean()
        ma_down = down.ewm(com=window - 1, adjust=True, min_periods=window).mean()
        rs = ma_up / ma_down
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df

    def add_macd(self, slow=26, fast=12, signal=9):
        """
        MACD + versions NORMALISÉES par le prix.

        Le MACD brut est une différence d'EMA de prix : il grandit avec le prix
        (non stationnaire, ~corrélé au niveau). On ajoute donc les versions en %
        du prix, comparables dans le temps ET entre cryptos. Le brut reste dans le
        fichier (graphes) mais figure dans COLONNES_NON_FEATURES.
        """
        exp1 = self.df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=slow, adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=signal, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']

        # Versions stationnaires (% du prix) → ce que l'IA va manger
        self.df['MACD_Norm'] = self.df['MACD'] / self.df['Close']
        self.df['MACD_Signal_Norm'] = self.df['MACD_Signal'] / self.df['Close']
        self.df['MACD_Hist_Norm'] = self.df['MACD_Hist'] / self.df['Close']
        return self.df

    def add_candle_streaks(self):
        """ Séries de bougies consécutives """
        direction = np.where(self.df['Close'] > self.df['Open'], 1, -1)
        series_dir = pd.Series(direction, index=self.df.index)
        groups = (series_dir != series_dir.shift()).cumsum()
        streak_size = series_dir.groupby(groups).cumcount() + 1
        self.df['Streak'] = streak_size * series_dir
        return self.df

    # --- NOUVEAUTÉ 1 : TIME FEATURES ---
    def add_time_features(self):
        """ Encode l'heure et le jour (Cyclique) """
        # Sin/Cos de l'heure (0-23)
        self.df['Hour_Sin'] = np.sin(2 * np.pi * self.df.index.hour / 24)
        self.df['Hour_Cos'] = np.cos(2 * np.pi * self.df.index.hour / 24)
        # Sin/Cos du jour (0-6)
        self.df['Day_Sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7)
        self.df['Day_Cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7)
        return self.df

    # --- NOUVEAUTÉ 2 : VOLATILITY & RVOL ---
    def add_volatility_indicators(self):
        """ ATR normalisé (%) et Relative Volume (RVOL) """
        # ATR Calculation
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()

        # ATR en % du prix (Stationnaire)
        self.df['ATR_Pct'] = atr / self.df['Close']

        # RVOL (Volume Relatif vs Moyenne 20 périodes)
        avg_vol = self.df['Volume'].rolling(20).mean()
        self.df['RVOL'] = self.df['Volume'] / avg_vol
        return self.df

    # --- NOUVEAUTÉ 3 : LAGS (MÉMOIRE) ---
    def add_lag_features(self, lags=[1, 2, 3, 5]):
        """ Ajoute les rendements passés sur la ligne actuelle """
        for lag in lags:
            self.df[f'Return_Lag_{lag}'] = self.df['Returns'].shift(lag)
        return self.df

    def add_fear_and_greed(self, fng_df=None):
        """ Ajoute l'index Fear & Greed """
        if fng_df is not None:
            if not isinstance(fng_df.index, pd.DatetimeIndex):
                fng_df.index = pd.to_datetime(fng_df.index)

            fng_df = fng_df[['value']].rename(columns={'value': 'FNG_Index'})
            self.df = self.df.join(fng_df, how='left')
            self.df['FNG_Index'] = self.df['FNG_Index'].ffill()
        return self.df

    def add_ichimoku(self):
        """
        Ajoute les composantes d'Ichimoku.
        Tenkan (9), Kijun (26), Senkou A & B (52), décalés de 26 vers le futur.
        Pour l'IA, on aligne le nuage 'futur' sur la bougie 'actuelle' pour voir si ça fait support.
        """
        # 1. Tenkan-sen (Conversion Line) : (Max + Min) / 2 sur 9 périodes
        high_9 = self.df['High'].rolling(window=9).max()
        low_9 = self.df['Low'].rolling(window=9).min()
        self.df['Ichi_Tenkan'] = (high_9 + low_9) / 2

        # 2. Kijun-sen (Base Line) : (Max + Min) / 2 sur 26 périodes
        high_26 = self.df['High'].rolling(window=26).max()
        low_26 = self.df['Low'].rolling(window=26).min()
        self.df['Ichi_Kijun'] = (high_26 + low_26) / 2

        # 3. Senkou Span A (Leading Span A) : (Tenkan + Kijun) / 2
        # Normalement projeté 26 périodes dans le futur.
        # Ici, on le calcule et on le shiftera après pour l'aligner.
        span_a = (self.df['Ichi_Tenkan'] + self.df['Ichi_Kijun']) / 2

        # 4. Senkou Span B (Leading Span B) : (Max + Min) / 2 sur 52 périodes
        high_52 = self.df['High'].rolling(window=52).max()
        low_52 = self.df['Low'].rolling(window=52).min()
        span_b = (high_52 + low_52) / 2

        # --- ALIGNEMENT POUR L'IA ---
        # Le nuage qui agit comme support AUJOURD'HUI a été calculé il y a 26 périodes.
        # Donc on décale les Spans de 26 vers l'avant (shift positif).
        self.df['Ichi_SpanA'] = span_a.shift(26)
        self.df['Ichi_SpanB'] = span_b.shift(26)

        # --- FEATURES DÉRIVÉES (Ce que l'IA va manger) ---

        # Est-ce qu'on est au-dessus du nuage ? (1 = Oui, -1 = Dessous, 0 = Dedans)
        # Le nuage est défini par la zone entre Span A et Span B
        cloud_top = self.df[['Ichi_SpanA', 'Ichi_SpanB']].max(axis=1)
        cloud_bottom = self.df[['Ichi_SpanA', 'Ichi_SpanB']].min(axis=1)

        self.df['Ichi_Above_Cloud'] = np.where(self.df['Close'] > cloud_top, 1,np.where(self.df['Close'] < cloud_bottom, -1, 0))

        # --- Distances stationnaires (% vs Close) : ce que l'IA va manger ---
        # Les lignes brutes Ichi_* sont des NIVEAUX de prix (non stationnaires),
        # on ne garde donc que les écarts relatifs.
        self.df['Dist_Kijun'] = (self.df['Close'] - self.df['Ichi_Kijun']) / self.df['Ichi_Kijun']
        self.df['Dist_Tenkan'] = (self.df['Close'] - self.df['Ichi_Tenkan']) / self.df['Ichi_Tenkan']
        self.df['Dist_SpanA'] = (self.df['Close'] - self.df['Ichi_SpanA']) / self.df['Ichi_SpanA']
        self.df['Dist_SpanB'] = (self.df['Close'] - self.df['Ichi_SpanB']) / self.df['Ichi_SpanB']

        return self.df

    # --- NOUVEAUTÉ 4 : MOMENTUM MULTI-HORIZONS ---
    def add_momentum_features(self, windows=(3, 6, 12, 24, 72, 168)):
        """
        Rendements cumulés sur plusieurs horizons + leur z-score (régime relatif).
        Capture le momentum court/moyen/long terme de façon stationnaire.
        """
        for w in windows:
            mom = self.df['Close'].pct_change(w)
            self.df[f'Mom_{w}'] = mom
            roll = mom.rolling(window=max(w * 3, 50))
            self.df[f'Mom_{w}_Z'] = (mom - roll.mean()) / (roll.std() + 1e-9)
        return self.df

    # --- NOUVEAUTÉ 5 : OSCILLATEURS ---
    def add_oscillators(self, window=14):
        """
        Stochastique %K/%D, Williams %R, CCI, ROC et ADX/+DI/-DI (force de
        tendance). Tous bornés ou normalisés → stationnaires.
        """
        high, low, close = self.df['High'], self.df['Low'], self.df['Close']

        # Stochastique
        low_n = low.rolling(window).min()
        high_n = high.rolling(window).max()
        denom = (high_n - low_n).replace(0, np.nan)
        self.df['Stoch_K'] = 100 * (close - low_n) / denom
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(3).mean()

        # Williams %R
        self.df['Williams_R'] = -100 * (high_n - close) / denom

        # CCI (Commodity Channel Index)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window).mean()
        mad = (tp - sma_tp).abs().rolling(window).mean()
        self.df['CCI'] = (tp - sma_tp) / (0.015 * (mad + 1e-9))

        # ROC (Rate of Change %)
        self.df['ROC'] = close.pct_change(window) * 100

        # ADX / +DI / -DI (Wilder approximé par moyenne mobile)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                            index=self.df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                             index=self.df.index)
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        plus_di = 100 * plus_dm.rolling(window).mean() / (atr + 1e-9)
        minus_di = 100 * minus_dm.rolling(window).mean() / (atr + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
        self.df['Plus_DI'] = plus_di
        self.df['Minus_DI'] = minus_di
        self.df['ADX'] = dx.rolling(window).mean()
        return self.df

    # --- NOUVEAUTÉ 6 : RÉGIME DE VOLATILITÉ ---
    @staticmethod
    def _rang_glissant(serie, fenetre):
        """Rang-percentile (0..1) de la valeur courante sur une fenêtre glissante."""
        mp = max(10, fenetre // 2)
        try:
            return serie.rolling(fenetre, min_periods=mp).rank(pct=True)
        except (AttributeError, TypeError):  # pandas < 1.4
            return serie.rolling(fenetre, min_periods=mp).apply(
                lambda x: (x <= x[-1]).mean(), raw=True)

    def add_volatility_regime(self, windows=(24, 72, 168), fenetre_rang=500):
        """
        Volatilité réalisée multi-fenêtres + rang-percentile de l'ATR et de la
        largeur de Bollinger → situe le marché dans son régime (calme / agité).
        """
        if 'Returns' not in self.df.columns:
            self.df['Returns'] = self.df['Close'].pct_change()
        for w in windows:
            self.df[f'RealVol_{w}'] = self.df['Returns'].rolling(w).std()
        if 'ATR_Pct' in self.df.columns:
            self.df['ATR_Pct_Rank'] = self._rang_glissant(self.df['ATR_Pct'], fenetre_rang)
        if 'BB_Width' in self.df.columns:
            self.df['BB_Width_Rank'] = self._rang_glissant(self.df['BB_Width'], fenetre_rang)
        return self.df

    # --- NOUVEAUTÉ 7 : VOLUME & ORDER-FLOW ---
    def add_volume_orderflow(self, window=20):
        """
        OBV normalisé, z-score du volume, et — si les colonnes order-flow Binance
        sont présentes — la pression acheteuse agressive (taker buy ratio) et le
        z-score du nombre de trades. Se désactive proprement sur données Yahoo.
        """
        direction = np.sign(self.df['Close'].diff()).fillna(0)
        obv = (direction * self.df['Volume']).cumsum()
        self.df['OBV_Norm'] = obv.pct_change(window)

        vmean = self.df['Volume'].rolling(window).mean()
        vstd = self.df['Volume'].rolling(window).std()
        self.df['Volume_Z'] = (self.df['Volume'] - vmean) / (vstd + 1e-9)

        if 'Taker_Buy_Base' in self.df.columns:
            ratio = self.df['Taker_Buy_Base'] / (self.df['Volume'] + 1e-12)
            ratio = ratio.clip(0, 1)
            self.df['Taker_Buy_Ratio'] = ratio
            self.df['Taker_Buy_Ratio_Z'] = (
                (ratio - ratio.rolling(window).mean()) / (ratio.rolling(window).std() + 1e-9))
        if 'Trades' in self.df.columns:
            tmean = self.df['Trades'].rolling(window).mean()
            tstd = self.df['Trades'].rolling(window).std()
            self.df['Trades_Z'] = (self.df['Trades'] - tmean) / (tstd + 1e-9)
        return self.df

    # --- NOUVEAUTÉ 8 : POSITION DANS LE RANGE (DONCHIAN) ---
    def add_range_position(self, windows=(24, 72, 168)):
        """ Position du prix dans le canal Donchian (0 = plus bas, 1 = plus haut). """
        for w in windows:
            bas = self.df['Low'].rolling(w).min()
            haut = self.df['High'].rolling(w).max()
            self.df[f'Range_Pos_{w}'] = (self.df['Close'] - bas) / ((haut - bas) + 1e-9)
        return self.df

    def add_targets(self, horizons=range(1, 12)):
        """
        Génère plusieurs cibles pour différents horizons de temps.
        Ex: Target_6h = 1 si le prix dans 6h est supérieur au prix actuel.

        Les colonnes sont ajoutées en une seule fois (concat) pour éviter la
        fragmentation du DataFrame (PerformanceWarning de pandas).
        """
        nouvelles = {}
        for h in horizons:
            # 1. Target Binaire (Classification) : Monte ou Baisse ?
            future_close = self.df['Close'].shift(-h)
            nouvelles[f'Target_{h}'] = (future_close > self.df['Close']).astype(int)
            # 2. Target Retour (Régression) : De combien ça monte ?
            nouvelles[f'Ret_{h}'] = self.df['Close'].pct_change(h).shift(-h)

        self.df = pd.concat([self.df, pd.DataFrame(nouvelles, index=self.df.index)], axis=1)
        return self.df

    def process_all(self, fng_data=None):
        """ Exécute tout le pipeline """
        self.add_basic_indicators()
        self.add_moving_averages()
        self.add_bollinger_bands()
        self.add_rsi()
        self.add_macd()
        self.add_candle_streaks()

        # --- Appels des nouvelles fonctions ---
        self.add_time_features()
        self.add_volatility_indicators()
        self.add_lag_features()

        self.add_ichimoku()

        # --- Groupes d'indicateurs stationnaires (refonte qualité modèle) ---
        self.add_momentum_features()
        self.add_oscillators()
        self.add_volatility_regime()
        self.add_volume_orderflow()
        self.add_range_position()
        # -------------------------------------

        if fng_data is not None:
            self.add_fear_and_greed(fng_data)

        self.add_targets(horizons=range(1, 12))

        self.df = self.df.dropna()
        return self.df


# ==========================================
# 3. UTILITAIRE FEAR & GREED
# ==========================================
def get_fear_and_greed_history():
    """ Récupère F&G via requests pour éviter les erreurs Pandas """
    try:
        url = "https://api.alternative.me/fng/?limit=0"
        response = requests.get(url)
        data = response.json()['data']

        df = pd.DataFrame(data)

        # Conversion explicite pour éviter les Warnings
        df['value'] = pd.to_numeric(df['value'])
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'

        return df[['value']].sort_index()
    except Exception as e:
        print(f"⚠️ Impossible de récupérer Fear & Greed : {e}")
        return None


# ==========================================
# 4. EXÉCUTION (MAIN) AUTOMATISÉE
# ==========================================
if __name__ == "__main__":

    # 1. Configuration
    input_folder = "data_crypto"
    loader = CryptoDataLoader(data_folder=input_folder, write_folder="analysis_crypto")

    print("\n🌍 Récupération de l'historique Fear & Greed global...")
    df_fng = get_fear_and_greed_history()

    if os.path.exists(input_folder):
        files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx') and not f.startswith('~$')]

        print(f"\n📂 {len(files)} fichiers trouvés dans '{input_folder}'. Traitement en cours...\n")

        for filename in files:
            try:
                # Format attendu : "SYMBOLE_FREQUENCE.xlsx" (ex: BTC_1h.xlsx)
                name_clean = filename.replace(".xlsx", "")
                parts = name_clean.split('_')

                if len(parts) < 2:
                    print(f"⚠️ Format de fichier ignoré (pas de '_') : {filename}")
                    continue

                freq = parts[-1]
                crypto = "_".join(parts[:-1])

                print(f"🔹 Traitement de {crypto} ({freq})...")

                df = loader.load_crypto_data(crypto, freq)

                if df is not None and not df.empty:
                    engineer = CryptoFeatureEngineer(df)
                    df_final = engineer.process_all(fng_data=df_fng)

                    loader.save_to_excel(df_final, crypto, freq)

                    print(f"   -> OK : {len(df_final)} lignes, {len(df_final.columns)} indicateurs.")
                else:
                    print(f"   -> ⚠️ Fichier vide ou illisible.")

                print("-" * 30)

            except Exception as e:
                print(f"❌ Erreur sur le fichier {filename} : {e}")

    else:
        print(f"❌ Le dossier '{input_folder}' n'existe pas.")

    print("\n✅ Traitement de lot terminé.")