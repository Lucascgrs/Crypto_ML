import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Denylist canonique des colonnes interdites en feature (source unique de vérité
# dans CryptoAnalysis). Fallback embarqué si l'import échoue.
try:
    from CryptoAnalysis import COLONNES_NON_FEATURES
except Exception:
    COLONNES_NON_FEATURES = {
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Quote_Asset_Volume', 'Trades', 'Taker_Buy_Base', 'Taker_Buy_Quote',
        'Dividends', 'Stock Splits',
        'SMA_20', 'SMA_50', 'SMA_200',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_SpanA', 'Ichi_SpanB',
    }


# ==========================================
# MODÈLES DISPONIBLES (factory)
# ==========================================
# Détection des librairies optionnelles
try:
    from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log
    LIGHTGBM_OK = True
except Exception:
    LIGHTGBM_OK = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except Exception:
    CATBOOST_OK = False


# Modèles qui supportent l'early stopping via eval_set
MODELES_EARLY_STOPPING = {"XGBoost", "LightGBM"}

# Grilles d'hyperparamètres par modèle (utilisées par le GridSearch)
GRILLES_PARAMS = {
    "XGBoost": {
        'max_depth':        [4, 6],
        'learning_rate':    [0.01, 0.05],
        'subsample':        [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'min_child_weight': [1, 5],
    },
    "LightGBM": {
        'max_depth':        [4, 6, -1],
        'learning_rate':    [0.01, 0.05],
        'num_leaves':       [31, 63],
        'subsample':        [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
    },
    "RandomForest": {
        'n_estimators':     [200, 400],
        'max_depth':        [6, 12, None],
        'min_samples_leaf': [1, 5, 20],
        'max_features':     ['sqrt', 0.5],
    },
    "LogisticRegression": {
        'logisticregression__C': [0.01, 0.1, 1.0],
    },
    "CatBoost": {
        'depth':         [4, 6],
        'learning_rate': [0.03, 0.1],
        'l2_leaf_reg':   [1, 5],
    },
}


def modeles_disponibles():
    """Liste des modèles utilisables selon les librairies installées."""
    dispo = ["XGBoost", "RandomForest", "LogisticRegression"]
    if LIGHTGBM_OK:
        dispo.insert(1, "LightGBM")
    if CATBOOST_OK:
        dispo.append("CatBoost")
    return dispo


def construire_modele(type_modele, params=None, n_estimators=300,
                      device="cpu", early_stopping_rounds=None):
    """
    Construit un classifieur non entraîné selon le type demandé.
    Tous exposent predict_proba (nécessaire pour la calibration).
    """
    params = dict(params or {})

    if type_modele == "XGBoost":
        kw = dict(objective='binary:logistic', tree_method='hist', device=device,
                  eval_metric='auc', verbosity=0, n_estimators=n_estimators, n_jobs=-1)
        if early_stopping_rounds:
            kw['early_stopping_rounds'] = early_stopping_rounds
        kw.update(params)
        return XGBClassifier(**kw)

    if type_modele == "LightGBM":
        if not LIGHTGBM_OK:
            raise ImportError("LightGBM non installé : pip install lightgbm")
        kw = dict(objective='binary', n_estimators=n_estimators, n_jobs=-1, verbose=-1)
        kw.update(params)
        return LGBMClassifier(**kw)

    if type_modele == "RandomForest":
        kw = dict(n_jobs=-1, random_state=42, n_estimators=n_estimators)
        kw.update(params)
        return RandomForestClassifier(**kw)

    if type_modele == "LogisticRegression":
        # Régression logistique : nécessite une normalisation (pipeline)
        lr_params = {k.replace('logisticregression__', ''): v for k, v in params.items()}
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, **lr_params))

    if type_modele == "CatBoost":
        if not CATBOOST_OK:
            raise ImportError("CatBoost non installé : pip install catboost")
        kw = dict(iterations=n_estimators, verbose=0, random_state=42)
        kw.update(params)
        return CatBoostClassifier(**kw)

    raise ValueError(f"Modèle inconnu : {type_modele}")


# ==========================================
# 1. CONFIGURATION CENTRALE
# ==========================================
class ModelConfig:
    """
    Classe de configuration unique.
    Tous les paramètres du pipeline sont centralisés ici.
    Modifier une valeur ici se répercute sur tout le code.
    """

    # --- Modèle à utiliser ---
    # "XGBoost" | "LightGBM" | "RandomForest" | "LogisticRegression" | "CatBoost"
    MODEL_TYPE = "XGBoost"

    # --- Cible à prédire ---
    # "Target_24" = est-ce que le prix sera plus haut dans 24 périodes ?
    # Horizons conseillés : 6h (court), 24h (moyen), 72h (long)
    TARGET_HORIZON = 24

    # Type de cible :
    # "Directionnel"   = monte / baisse (sans filtre)
    # "Seuil ATR"      = filtre le bruit via l'ATR (mouvements trop faibles ignorés)
    # "Triple-barrier" = TP / SL / temps (méthode López de Prado, alignée trading réel)
    TARGET_TYPE = "Seuil ATR"

    # Seuil de significativité de la target (multiplicateur de l'ATR%)
    # Ex: 0.5 = le mouvement doit dépasser 50% de l'ATR pour ne pas être considéré comme du bruit
    TARGET_THRESHOLD_MULTIPLIER = 0.5

    # --- Triple-barrier (utilisé si TARGET_TYPE == "Triple-barrier") ---
    # Barrières exprimées en multiples de l'ATR% courant
    TB_TP_MULT = 1.5   # take-profit
    TB_SL_MULT = 1.0   # stop-loss

    # --- Anti-fuite de données ---
    # Embargo : nombre de lignes purgées entre le train et le test pour neutraliser
    # le chevauchement des labels (la cible regarde TARGET_HORIZON périodes en avant).
    # None -> embargo automatique = TARGET_HORIZON.
    EMBARGO = None

    # Élagage des features trop corrélées entre elles (réduit le bruit/redondance)
    PRUNE_CORRELATION = False
    CORRELATION_THRESHOLD = 0.95

    # Rééquilibrage des classes (scale_pos_weight / class_weight / auto_class_weights)
    # Utile quand Hausse/Baisse sont déséquilibrées (fréquent avec le filtre ATR).
    USE_CLASS_WEIGHTS = False

    # --- Taille du jeu de test (fraction chronologique finale) ---
    TEST_SIZE = 0.2

    # --- Dossiers ---
    INPUT_FOLDER  = "analysis_crypto"
    MODEL_FOLDER  = "models"
    PRED_FOLDER   = "prediction_crypto"
    VIZ_FOLDER    = "visualizations"

    # --- Hyperparamètres testés par GridSearch ---
    # n_estimators absent intentionnellement : géré par l'early stopping
    PARAM_GRID = {
        'max_depth':        [4, 6, 8],
        'learning_rate':    [0.01, 0.05],
        'subsample':        [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma':            [0.0, 0.1],
        'min_child_weight': [1, 5],
    }

    # --- Walk-Forward Validation ---
    # Nombre de fenêtres temporelles utilisées pour évaluer la robustesse du modèle
    WALK_FORWARD_SPLITS = 6


# ==========================================
# 2. GESTIONNAIRE DE DONNÉES
# ==========================================
class MLDataManager:
    """
    Responsable du chargement, du nettoyage et de la préparation des données.
    Chaque modèle est propre à une crypto ET un intervalle.
    """

    def __init__(self):
        os.makedirs(ModelConfig.MODEL_FOLDER, exist_ok=True)
        os.makedirs(ModelConfig.PRED_FOLDER,  exist_ok=True)
        os.makedirs(ModelConfig.VIZ_FOLDER,   exist_ok=True)

    def load_crypto_data(self, symbole, intervalle):
        """
        Charge le fichier analysé d'UNE SEULE crypto pour UN SEUL intervalle.
        Ex : BTC + 1h → lit 'analysis_crypto/BTC_1h_analyzed.xlsx'
        Retourne None si le fichier est introuvable ou vide.
        """
        filepath = os.path.join(
            ModelConfig.INPUT_FOLDER,
            f"{symbole}_{intervalle}_analyzed.xlsx"
        )

        if not os.path.exists(filepath):
            print(f"❌ Fichier introuvable : {filepath}")
            return None

        df = pd.read_excel(filepath, index_col=0, parse_dates=True)
        df = df.sort_index()

        if df.empty:
            print(f"⚠️ Fichier vide : {filepath}")
            return None

        print(f"✅ Chargé : {symbole} ({intervalle}) — {len(df)} lignes.")
        return df

    def build_features(self, df):
        """
        Construit la matrice de features X à partir du DataFrame analysé.

        Colonnes supprimées :
        - Tout COLONNES_NON_FEATURES : prix bruts + indicateurs de NIVEAU de prix
          (SMA, MACD, lignes Ichimoku) + order-flow brut + artefacts. Non stationnaires :
          les laisser ferait mémoriser le niveau de prix au modèle (fuite → effondrement
          hors échantillon).
        - Colonnes cibles (Target_*, Ret_*) : ce qu'on prédit, jamais un input
        - Colonne de groupe 'symbole' (dataset Portefeuille)

        Un GARDE-FOU automatique retire en plus toute colonne dont la corrélation avec
        Close dépasse 0.99 (proxy de prix oublié).

        Retourne un DataFrame float32 propre (NaN et infinis remplacés).
        """
        cols_to_drop = set(COLONNES_NON_FEATURES)
        cols_to_drop |= {c for c in df.columns if c.startswith('Target_') or c.startswith('Ret_')}
        cols_to_drop |= {'symbole', 'Symbole', 'Date'}

        existing_drop = [c for c in cols_to_drop if c in df.columns]
        X = df.drop(columns=existing_drop)

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors='coerce').astype('float32')

        # --- Garde-fou anti-fuite : retire toute feature qui suit le NIVEAU du prix ---
        # Calcul POSITIONNEL (numpy) → robuste même si l'index a des doublons (Portefeuille).
        if 'Close' in df.columns:
            close_vals = pd.to_numeric(df['Close'], errors='coerce').to_numpy()
            suspectes = []
            for col in X.columns:
                s = X[col].to_numpy()
                m = np.isfinite(s) & np.isfinite(close_vals)
                if m.sum() < 50:
                    continue
                sv = s[m]
                if np.unique(sv).size < 5:
                    continue
                corr = np.corrcoef(sv, close_vals[m])[0, 1]
                if np.isfinite(corr) and abs(corr) > 0.99:
                    suspectes.append(col)
            if suspectes:
                print(f"🛡️  Garde-fou anti-fuite : {len(suspectes)} colonne(s) de niveau "
                      f"de prix retirée(s) → {suspectes}")
                X = X.drop(columns=suspectes)

        return X

    def build_target(self, df):
        """
        Construit la cible selon ModelConfig.TARGET_TYPE :
        - "Directionnel"   : 1 si le prix monte après l'horizon, 0 sinon (aucun filtre)
        - "Seuil ATR"      : ignore les mouvements trop faibles (bruit) via l'ATR
        - "Triple-barrier" : 1 si TP touché avant SL dans l'horizon, 0 sinon

        Si une colonne 'symbole' est présente (dataset Portefeuille), la cible est
        calculée SÉPARÉMENT par crypto pour ne jamais franchir une frontière d'actif
        (sinon le rendement futur d'une crypto fuiterait sur une autre).

        Retourne une Series 0/1 (NaN = lignes à exclure).
        """
        if 'symbole' in df.columns and df['symbole'].nunique() > 1:
            syms = df['symbole'].values
            y_full = np.full(len(df), np.nan)
            for sym in pd.unique(syms):
                pos = np.where(syms == sym)[0]
                y_full[pos] = self._build_target_single(df.iloc[pos]).values
            y = pd.Series(y_full, index=df.index)
        else:
            y = self._build_target_single(df)

        type_cible = getattr(ModelConfig, 'TARGET_TYPE', 'Seuil ATR')
        total = int(y.notna().sum())
        pct_kept = total / len(y) * 100 if len(y) else 0
        print(f"🎯 Cible '{type_cible}' horizon={ModelConfig.TARGET_HORIZON} | "
              f"Utilisables : {total} lignes ({pct_kept:.1f}%) | "
              f"Bruit exclu : {100 - pct_kept:.1f}%")
        return y

    def _build_target_single(self, df):
        """
        Construit la cible pour UNE série temporelle continue (un seul actif).
        Appelé directement en mono-crypto, ou par groupe en mode Portefeuille.
        """
        type_cible = getattr(ModelConfig, 'TARGET_TYPE', 'Seuil ATR')
        h = ModelConfig.TARGET_HORIZON

        if type_cible == "Triple-barrier":
            return self._target_triple_barriere(df)

        if type_cible == "Directionnel":
            future_ret = df['Close'].pct_change(h).shift(-h)
            raw = np.where(future_ret > 0, 1.0, 0.0)
            y = pd.Series(raw, index=df.index)
            y[future_ret.isna()] = np.nan
            return y

        # "Seuil ATR" (défaut)
        multiplier = ModelConfig.TARGET_THRESHOLD_MULTIPLIER
        future_ret = df['Close'].pct_change(h).shift(-h)
        if 'ATR_Pct' in df.columns:
            threshold = df['ATR_Pct'] * multiplier
        else:
            threshold = 0.005
        raw = np.where(future_ret > threshold, 1.0,
                       np.where(future_ret < -threshold, 0.0, np.nan))
        return pd.Series(raw, index=df.index)

    def _target_triple_barriere(self, df):
        """
        Méthode de la triple barrière (López de Prado).
        Pour chaque bougie : barrière haute (TP), basse (SL) et temporelle (horizon).
        Le label = première barrière touchée (1 = TP, 0 = SL). Si aucune touchée à temps,
        on tranche par le signe du rendement à l'horizon.
        """
        h = ModelConfig.TARGET_HORIZON
        tp_mult = ModelConfig.TB_TP_MULT
        sl_mult = ModelConfig.TB_SL_MULT

        close = df['Close'].values.astype(float)
        if 'ATR_Pct' in df.columns:
            atr = df['ATR_Pct'].values.astype(float)
            atr = np.where(np.isfinite(atr) & (atr > 0), atr, 0.01)
        else:
            atr = np.full(len(close), 0.01)

        n = len(close)
        y = np.full(n, np.nan)
        for i in range(n - 1):
            seuil_tp = close[i] * (1 + tp_mult * atr[i])
            seuil_sl = close[i] * (1 - sl_mult * atr[i])
            fin = min(i + h, n - 1)
            label = np.nan
            for j in range(i + 1, fin + 1):
                if close[j] >= seuil_tp:
                    label = 1.0
                    break
                if close[j] <= seuil_sl:
                    label = 0.0
                    break
            if np.isnan(label) and fin > i:
                label = 1.0 if close[fin] > close[i] else 0.0
            y[i] = label
        return pd.Series(y, index=df.index)

    @staticmethod
    def elaguer_correlation(X, seuil=0.95):
        """
        Supprime les features trop corrélées entre elles (|corr| > seuil).
        Sur chaque paire redondante, on garde la première colonne.
        Retourne (X_réduit, colonnes_supprimées).
        """
        corr = X.corr().abs()
        haut = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        a_supprimer = [c for c in haut.columns if (haut[c] > seuil).any()]
        if a_supprimer:
            X = X.drop(columns=a_supprimer)
        return X, a_supprimer


# ==========================================
# 2bis. DATASET PORTEFEUILLE (MULTI-CRYPTO)
# ==========================================
class PortefeuilleDatasetBuilder:
    """
    Construit un dataset 'Portefeuille' poolé à partir de plusieurs cryptos
    déjà analysées, pour entraîner un modèle MULTI qui généralise mieux
    (plus de données, plus de régimes, patterns transverses aux actifs).

    - Charge chaque {sym}_{interval}_analyzed.xlsx
    - Concatène le tout, trié par date, et sauvegarde MULTI_{interval}_analyzed.xlsx
      consommable tel quel par CryptoModelTrainer/Predictor avec symbole='MULTI'.

    Le jeu de features est IDENTIQUE à celui d'une crypto seule (pas de feature
    cross-sectionnelle) : le modèle MULTI peut donc être appliqué ensuite à n'importe
    quelle crypto pour la prédiction et le backtest (voir CryptoPredictor.modele_symbole).

    Choix anti-fuite : pas de z-score global par actif (qui utiliserait des stats du
    futur). Les features sont déjà stationnaires et comparables entre cryptos.
    """

    NOM = "MULTI"

    def __init__(self, intervalle, symboles=None):
        self.intervalle = intervalle
        self.symboles   = symboles
        self.manager    = MLDataManager()

    def _lister_symboles(self):
        """Symboles à pooler : liste fournie, sinon toutes les cryptos analysées."""
        if self.symboles:
            return [s for s in self.symboles if s != self.NOM]
        dossier = ModelConfig.INPUT_FOLDER
        suffixe = f"_{self.intervalle}_analyzed.xlsx"
        trouves = []
        if os.path.isdir(dossier):
            for f in os.listdir(dossier):
                if f.endswith(suffixe) and not f.startswith('~$'):
                    sym = f[:-len(suffixe)]
                    if sym != self.NOM:
                        trouves.append(sym)
        return sorted(trouves)

    def construire(self):
        """Construit et sauvegarde le dataset poolé. Retourne le DataFrame ou None."""
        symboles = self._lister_symboles()
        if len(symboles) < 2:
            print(f"⚠️ Portefeuille : il faut au moins 2 cryptos analysées "
                  f"(trouvé : {symboles}). Analyse d'abord plusieurs cryptos.")
            return None

        print(f"\n🧺 Construction du dataset Portefeuille ({self.intervalle}) : {symboles}")
        frames = []
        for sym in symboles:
            df = self.manager.load_crypto_data(sym, self.intervalle)
            if df is None or df.empty:
                continue
            df = df.copy()
            df['symbole'] = sym
            df['Date']    = df.index
            frames.append(df)

        if len(frames) < 2:
            print("⚠️ Pas assez de cryptos chargées pour le portefeuille.")
            return None

        pool = pd.concat(frames, ignore_index=True)

        # Tri chronologique global puis index temporel (doublons d'horodatage tolérés :
        # le pipeline est rendu robuste aux index dupliqués en aval, opérations positionnelles).
        pool = pool.sort_values(['Date', 'symbole']).set_index('Date')
        pool.index.name = 'Date'

        chemin = os.path.join(ModelConfig.INPUT_FOLDER,
                              f"{self.NOM}_{self.intervalle}_analyzed.xlsx")
        pool.to_excel(chemin)
        print(f"✅ Dataset Portefeuille sauvegardé : {chemin}  "
              f"({len(pool)} lignes, {pool['symbole'].nunique()} cryptos)")
        return pool


# ==========================================
# 3. WALK-FORWARD VALIDATION
# ==========================================
class WalkForwardValidator:
    """
    Évalue la robustesse du modèle sur plusieurs fenêtres temporelles glissantes.

    Contrairement au simple split 80/20 (qui ne teste qu'une seule période),
    le Walk-Forward teste le modèle sur N périodes différentes consécutives :
    bull market, bear market, range... et retourne une AUC moyenne + écart-type.

    Un modèle robuste aura une faible variance entre les folds.
    """

    def __init__(self, symbole, intervalle):
        self.symbole    = symbole
        self.intervalle = intervalle
        self.manager    = MLDataManager()

    def run(self):
        """
        Lance la Walk-Forward Validation complète.

        Pour chaque fold :
        - Entraîne un XGBoost léger (500 arbres) sur la partie train
        - Évalue l'AUC-ROC sur la partie test (toujours dans le futur)

        Affiche l'AUC par fold, la moyenne et l'écart-type.
        Un écart-type élevé signifie que le modèle est instable selon les périodes.
        """
        df = self.manager.load_crypto_data(self.symbole, self.intervalle)
        if df is None:
            return

        X = self.manager.build_features(df)
        y = self.manager.build_target(df)

        # Alignement strict : on garde uniquement les lignes valides pour X ET y
        mask = X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        X, y = X[mask], y[mask]

        print(f"\n🔄 Walk-Forward Validation ({ModelConfig.WALK_FORWARD_SPLITS} folds)...")
        print(f"   {self.symbole} ({self.intervalle}) — {len(X)} lignes utilisables\n")

        tscv = TimeSeriesSplit(n_splits=ModelConfig.WALK_FORWARD_SPLITS)
        aucs = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            # Modèle léger pour la validation (pas besoin de GridSearch ici)
            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='auc',
                device='cpu',
                tree_method='hist',
                verbosity=0,
            )
            model.fit(X_tr, y_tr)

            proba = model.predict_proba(X_te)[:, 1]
            auc   = roc_auc_score(y_te, proba)
            aucs.append(auc)

            # Période couverte par ce fold
            date_start = X_te.index[0].strftime('%Y-%m-%d')
            date_end   = X_te.index[-1].strftime('%Y-%m-%d')
            print(f"   Fold {fold+1}/{ModelConfig.WALK_FORWARD_SPLITS} | {date_start} → {date_end} | AUC = {auc:.4f}")

        print(f"\n   📊 AUC moyen  : {np.mean(aucs):.4f}")
        print(f"   📊 Écart-type : {np.std(aucs):.4f}  (plus bas = plus stable)")

        return aucs


# ==========================================
# 4. SÉLECTION DE FEATURES VIA SHAP
# ==========================================
class FeatureSelector:
    """
    Utilise les valeurs SHAP pour identifier et conserver uniquement
    les features les plus importantes pour le modèle.

    Supprimer les features peu utiles réduit le bruit et peut légèrement
    améliorer la généralisation.
    """

    def __init__(self, symbole, intervalle, top_n=20):
        self.symbole    = symbole
        self.intervalle = intervalle
        self.top_n      = top_n
        self.manager    = MLDataManager()
        self.top_features_path = os.path.join(
            ModelConfig.MODEL_FOLDER,
            f"TOP_FEATURES_{symbole}_{intervalle}.joblib"
        )

    def compute_and_save(self):
        """
        Entraîne un modèle rapide, calcule les SHAP values et sauvegarde
        la liste des top N features les plus importantes.

        Le fichier résultant est chargé automatiquement par le Trainer
        pour filtrer X avant l'entraînement final.
        """
        try:
            import shap
        except ImportError:
            print("❌ SHAP non installé. Lance : pip install shap")
            return None

        print(f"\n🔍 Calcul de l'importance des features via SHAP...")

        df = self.manager.load_crypto_data(self.symbole, self.intervalle)
        if df is None:
            return None

        X = self.manager.build_features(df)
        y = self.manager.build_target(df)

        mask = X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        X, y = X[mask], y[mask]

        # Entraînement d'un modèle rapide (pas besoin d'être optimal ici)
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            objective='binary:logistic',
            device='cpu',
            tree_method='hist',
            verbosity=0,
        )
        model.fit(X, y)

        # Calcul SHAP sur un échantillon (1000 lignes max pour la vitesse)
        sample = X.sample(min(1000, len(X)), random_state=42)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # Importance = moyenne des valeurs absolues SHAP par feature
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=X.columns
        ).sort_values(ascending=False)

        top_features = importance.head(self.top_n).index.tolist()

        print(f"\n   Top {self.top_n} features sélectionnées :")
        for i, (feat, score) in enumerate(importance.head(self.top_n).items()):
            print(f"   {i+1:2d}. {feat:<30} SHAP = {score:.5f}")

        joblib.dump(top_features, self.top_features_path)
        print(f"\n✅ Features sauvegardées : {self.top_features_path}")

        return top_features

    def load(self):
        """
        Charge la liste des top features précédemment calculées.
        Retourne None si le fichier n'existe pas (pas de filtrage appliqué).
        """
        if not os.path.exists(self.top_features_path):
            return None
        return joblib.load(self.top_features_path)


# ==========================================
# 5. ENTRAÎNEUR DU MODÈLE
# ==========================================
class CryptoModelTrainer:
    """
    Entraîne un modèle XGBoost dédié à UNE crypto et UN intervalle.
    Pipeline complet : GridSearch → Entraînement final → Calibration isotonique.

    Fichiers produits :
    - models/XGB_BTC_1h.joblib  : le modèle XGBoost entraîné
    - models/CAL_BTC_1h.joblib  : le calibrateur isotonique des probabilités
    """

    def __init__(self, symbole, intervalle):
        self.symbole         = symbole
        self.intervalle      = intervalle
        self.manager         = MLDataManager()
        self.model_path      = os.path.join(ModelConfig.MODEL_FOLDER, f"XGB_{symbole}_{intervalle}.joblib")
        self.calibrator_path = os.path.join(ModelConfig.MODEL_FOLDER, f"CAL_{symbole}_{intervalle}.joblib")
        self.device          = self._detect_device()

    def _detect_device(self):
        """
        Détecte si un GPU NVIDIA est disponible pour accélérer XGBoost.
        Retourne 'cuda' si GPU détecté, sinon 'cpu'.
        """
        try:
            import subprocess
            subprocess.check_output('nvidia-smi', stderr=subprocess.DEVNULL)
            print("🖥️  GPU détecté → entraînement sur CUDA.")
            return "cuda"
        except Exception:
            print("🖥️  Pas de GPU → entraînement sur CPU.")
            return "cpu"

    def _chronological_split(self, X, y):
        """
        Découpe X et y en train/test en respectant STRICTEMENT l'ordre chronologique.
        Le test est toujours la période la plus récente.

        Embargo : on purge `EMBARGO` lignes à la fin du train pour neutraliser le
        chevauchement des labels (la cible regarde TARGET_HORIZON périodes en avant).
        Sans cet embargo, les dernières lignes du train « connaissent » le début du test.

        ⚠️ Ne jamais utiliser train_test_split(shuffle=True) sur des séries temporelles.
        """
        embargo = ModelConfig.EMBARGO
        if embargo is None:
            embargo = ModelConfig.TARGET_HORIZON
        embargo = max(0, int(embargo))

        split_idx = int(len(X) * (1 - ModelConfig.TEST_SIZE))
        fin_train = max(1, split_idx - embargo)
        if embargo:
            print(f"🚧 Embargo : {embargo} lignes purgées entre train et test (anti-fuite).")

        return (
            X.iloc[:fin_train], y.iloc[:fin_train],
            X.iloc[split_idx:], y.iloc[split_idx:]
        )

    @staticmethod
    def _grille_params(type_modele):
        """Grille d'hyperparamètres selon le modèle (XGBoost garde ModelConfig.PARAM_GRID)."""
        if type_modele == "XGBoost":
            return ModelConfig.PARAM_GRID
        return GRILLES_PARAMS.get(type_modele, {})

    @staticmethod
    def _poids_classes(type_modele, y_tr):
        """
        Paramètres de rééquilibrage des classes selon le modèle (si activé).
        - XGBoost/LightGBM : scale_pos_weight = n_baisse / n_hausse
        - RandomForest / LogisticRegression : class_weight='balanced'
        - CatBoost : auto_class_weights='Balanced'
        """
        if not getattr(ModelConfig, 'USE_CLASS_WEIGHTS', False):
            return {}
        y_int = y_tr.astype(int)
        n_pos = int((y_int == 1).sum())
        n_neg = int((y_int == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return {}
        if type_modele in ("XGBoost", "LightGBM"):
            return {'scale_pos_weight': n_neg / n_pos}
        if type_modele == "RandomForest":
            return {'class_weight': 'balanced'}
        if type_modele == "LogisticRegression":
            return {'logisticregression__class_weight': 'balanced'}
        if type_modele == "CatBoost":
            return {'auto_class_weights': 'Balanced'}
        return {}

    def _selectionner_features_train(self, X, y, top_n=20):
        """
        Sélectionne les top features via SHAP, calculées UNIQUEMENT sur le train.
        (Corrige la fuite de données : l'ancienne version utilisait tout le dataset.)
        Utilise un XGBoost rapide comme « ranker », quel que soit le modèle final.
        """
        try:
            import shap
        except ImportError:
            print("⚠️ SHAP non installé → pas de sélection de features.")
            return None

        ranker = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               objective='binary:logistic', device='cpu',
                               tree_method='hist', verbosity=0)
        ranker.fit(X, y)
        echantillon = X.sample(min(1000, len(X)), random_state=42)
        vals = shap.TreeExplainer(ranker).shap_values(echantillon)
        importance = pd.Series(np.abs(vals).mean(axis=0), index=X.columns)
        return importance.sort_values(ascending=False).head(top_n).index.tolist()

    def _sauver_features(self, features):
        """Sauvegarde la liste des features utilisées (pour l'inférence et la viz)."""
        chemin = os.path.join(ModelConfig.MODEL_FOLDER,
                              f"TOP_FEATURES_{self.symbole}_{self.intervalle}.joblib")
        joblib.dump(list(features), chemin)

    def _run_grid_search(self, X_train, y_train, type_modele):
        """
        Recherche les meilleurs hyperparamètres via GridSearchCV (générique).

        TimeSeriesSplit pour respecter l'ordre chronologique pendant la recherche.
        Scoring : AUC-ROC (plus robuste que l'accuracy sur classes équilibrées).
        """
        grille = self._grille_params(type_modele)
        print(f"🔎 GridSearch {type_modele} (TimeSeriesSplit, AUC-ROC)...")
        if not grille:
            print("   (aucune grille définie → paramètres par défaut)")
            return {}

        base_model = construire_modele(type_modele, n_estimators=300, device="cpu")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=grille,
            cv=TimeSeriesSplit(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print(f"✅ Meilleurs paramètres : {grid.best_params_}")
        print(f"   AUC-ROC (validation croisée) : {grid.best_score_:.4f}")
        return grid.best_params_

    def _calibrate_probabilities(self, model, X_val, y_val):
        """
        Calibre les probabilités brutes du modèle via IsotonicRegression.

        XGBoost produit souvent des probabilités mal calibrées :
        "0.70" ne veut pas dire que l'événement arrive vraiment 70% du temps.

        La régression isotonique apprend à corriger ce biais :
        elle mappe les probabilités brutes vers des probabilités plus réalistes,
        en utilisant des données que le modèle n'a jamais vues (X_val).

        Retourne le calibrateur entraîné.
        """
        print("📐 Calibration des probabilités (isotonic)...")

        raw_probs  = model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(raw_probs, y_val)

        cal_probs = calibrator.predict(raw_probs)
        print(f"   Avant calibration — min: {raw_probs.min():.3f} | max: {raw_probs.max():.3f} | moy: {raw_probs.mean():.3f}")
        print(f"   Après calibration — min: {cal_probs.min():.3f}  | max: {cal_probs.max():.3f}  | moy: {cal_probs.mean():.3f}")

        return calibrator

    def _evaluate(self, model, calibrator, X_test, y_test):
        """
        Évalue le modèle final (XGBoost + calibrateur) sur le jeu de test.

        Métriques affichées :
        - Accuracy  : % de bonnes prédictions (utile mais insuffisant seul)
        - AUC-ROC   : capacité à séparer les hausses des baisses (0.5=hasard, 1.0=parfait)
        - Rapport   : précision, rappel, f1-score par classe (Hausse/Baisse)
        """
        raw_probs  = model.predict_proba(X_test)[:, 1]
        cal_probs  = calibrator.predict(raw_probs)
        y_pred     = (cal_probs >= 0.5).astype(int)

        accuracy   = accuracy_score(y_test, y_pred)
        auc        = roc_auc_score(y_test, cal_probs) if len(np.unique(y_test)) > 1 else 0.5

        # Baselines de comparaison (honnêteté) : si le modèle ne bat pas la classe
        # majoritaire, il n'a aucun avantage réel.
        taux_hausse = float(np.mean(y_test))
        majorite    = float(max(taux_hausse, 1 - taux_hausse))

        print(f"\n📊 ===== ÉVALUATION FINALE ({self.symbole} {self.intervalle}) =====")
        print(f"   Accuracy : {accuracy:.2%}")
        print(f"   AUC-ROC  : {auc:.4f}  (0.5 = aléatoire | 1.0 = parfait)")
        print(f"   Baseline classe majoritaire : {majorite:.2%}  "
              f"({'✅ battue' if accuracy > majorite else '❌ NON battue'})")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Baisse (0)', 'Hausse (1)'])}")

        rapport = classification_report(
            y_test, y_pred, target_names=['Baisse', 'Hausse'],
            output_dict=True, zero_division=0
        )
        return {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "rapport": rapport,
            "baselines": {
                "toujours_hausse_acc": taux_hausse,
                "majorite_acc": majorite,
            },
        }

    def _sauver_metadata(self, type_modele, X_tr, X_val, X_test, y,
                         best_params, metrics, feature_selection,
                         best_iteration, features):
        """
        Sauvegarde toutes les infos sur l'entraînement dans un fichier JSON.
        Permet à l'interface (onglet Évaluation) d'afficher des stats exactes
        sans avoir à réentraîner le modèle.
        """
        try:
            balance = y.astype(int).value_counts(normalize=True)
            embargo = ModelConfig.EMBARGO
            if embargo is None:
                embargo = ModelConfig.TARGET_HORIZON

            # Conversion JSON-safe des hyperparamètres
            hp = {str(k): (v if isinstance(v, (int, float, str, bool, type(None)))
                           else str(v))
                  for k, v in (best_params or {}).items()}

            meta = {
                "symbole":            self.symbole,
                "intervalle":         self.intervalle,
                "date_entrainement":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modele":             type_modele,
                "type_cible":         getattr(ModelConfig, 'TARGET_TYPE', 'Seuil ATR'),
                "periode_debut":      str(X_tr.index.min()),
                "periode_fin":        str(X_test.index.max()),
                "n_total":            int(len(X_tr) + len(X_val) + len(X_test)),
                "n_train":            int(len(X_tr)),
                "n_val":              int(len(X_val)),
                "n_test":             int(len(X_test)),
                "balance_hausse":     float(balance.get(1, 0.0)),
                "balance_baisse":     float(balance.get(0, 0.0)),
                "horizon":            ModelConfig.TARGET_HORIZON,
                "seuil_atr":          ModelConfig.TARGET_THRESHOLD_MULTIPLIER,
                "test_size":          ModelConfig.TEST_SIZE,
                "embargo":            int(max(0, embargo)),
                "elagage_correlation": bool(ModelConfig.PRUNE_CORRELATION),
                "class_weights":      bool(getattr(ModelConfig, 'USE_CLASS_WEIGHTS', False)),
                "feature_selection":  bool(feature_selection),
                "n_features":         int(len(features)),
                "features":           [str(c) for c in features],
                "hyperparametres":    hp,
                "n_estimators_retenus": int(best_iteration or 0),
                "metrics":            metrics,
            }
            chemin = os.path.join(ModelConfig.MODEL_FOLDER,
                                  f"META_{self.symbole}_{self.intervalle}.json")
            with open(chemin, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"💾 Métadonnées sauvegardées : {chemin}")
        except Exception as e:
            print(f"⚠️ Impossible de sauvegarder les métadonnées : {e}")

    def train(self, force_retrain=False, use_feature_selection=True):
        """
        Pipeline d'entraînement complet :

        1. Chargement des données
        2. Construction de la target améliorée (avec filtre de bruit)
        3. Feature selection via SHAP (optionnel)
        4. Split chronologique train / validation / test
        5. GridSearch pour les meilleurs hyperparamètres
        6. Entraînement final avec early stopping
        7. Calibration isotonique des probabilités
        8. Évaluation et sauvegarde

        Paramètres :
        - force_retrain (bool)        : si False, utilise le modèle existant sans réentraîner
        - use_feature_selection (bool): si True, filtre les features via SHAP avant l'entraînement
        """
        if os.path.exists(self.model_path) and not force_retrain:
            print(f"💾 Modèle existant trouvé : {self.model_path} (force_retrain=False)")
            return

        type_modele = getattr(ModelConfig, 'MODEL_TYPE', 'XGBoost')
        print(f"\n🤖 Modèle choisi : {type_modele} | Cible : "
              f"{getattr(ModelConfig, 'TARGET_TYPE', 'Seuil ATR')}")

        # --- 1. Chargement ---
        df = self.manager.load_crypto_data(self.symbole, self.intervalle)
        if df is None:
            return

        # --- 2. Features + cible ---
        X = self.manager.build_features(df)
        y = self.manager.build_target(df)
        mask = X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        X, y = X[mask], y[mask]

        class_balance = y.astype(int).value_counts(normalize=True)
        print(f"⚖️  Balance → Hausse: {class_balance.get(1, 0):.1%} | Baisse: {class_balance.get(0, 0):.1%}")
        print(f"📐 {len(X)} lignes | {X.shape[1]} features (avant réduction)")

        # --- 3. Split chronologique AVEC embargo (le test reste intouché) ---
        X_train, y_train, X_test, y_test = self._chronological_split(X, y)
        split_val  = int(len(X_train) * 0.8)
        X_tr,  y_tr  = X_train.iloc[:split_val],  y_train.iloc[:split_val]
        X_val, y_val = X_train.iloc[split_val:],  y_train.iloc[split_val:]
        print(f"✂️  Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # --- 4. Réduction des features — UNIQUEMENT sur le train (anti-fuite) ---
        features = list(X_tr.columns)

        if ModelConfig.PRUNE_CORRELATION:
            X_red, supprimees = self.manager.elaguer_correlation(
                X_tr, ModelConfig.CORRELATION_THRESHOLD)
            features = list(X_red.columns)
            print(f"🧹 Élagage corrélation (>{ModelConfig.CORRELATION_THRESHOLD}) : "
                  f"{len(supprimees)} supprimées → {len(features)} restantes.")

        if use_feature_selection:
            print("🔍 Sélection de features via SHAP (sur le train uniquement)...")
            top = self._selectionner_features_train(X_tr[features], y_tr, top_n=30)
            if top:
                features = top
                print(f"🎯 {len(features)} features retenues.")

        X_tr, X_val, X_test = X_tr[features], X_val[features], X_test[features]
        self._sauver_features(features)

        # --- 5. GridSearch ---
        best_params = self._run_grid_search(X_tr, y_tr, type_modele)

        # --- 6. Entraînement final (early stopping si supporté) ---
        print(f"\n🔥 Entraînement final {type_modele} ({self.device.upper()})...")
        best_iteration = 0

        # Rééquilibrage des classes (optionnel) fusionné aux meilleurs hyperparamètres
        poids = self._poids_classes(type_modele, y_tr)
        if poids:
            print(f"⚖️  Rééquilibrage des classes activé : {poids}")
        params_finaux = {**(best_params or {}), **poids}

        if type_modele in MODELES_EARLY_STOPPING:
            final_model = construire_modele(
                type_modele, params=params_finaux, n_estimators=3000,
                device=self.device, early_stopping_rounds=75)
            if type_modele == "XGBoost":
                final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                best_iteration = int(getattr(final_model, "best_iteration", 0) or 0)
            else:  # LightGBM
                final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                                callbacks=[lgb_early_stopping(75, verbose=False), lgb_log(0)])
                best_iteration = int(getattr(final_model, "best_iteration_", 0) or 0)
            print(f"⛔ Early stopping → {best_iteration} arbres retenus.")
        else:
            final_model = construire_modele(
                type_modele, params=params_finaux, n_estimators=400, device=self.device)
            final_model.fit(X_tr, y_tr)

        # --- 7. Calibration (sur val, jamais utilisé par le fit) ---
        calibrator = self._calibrate_probabilities(final_model, X_val, y_val)

        # --- 8. Évaluation (+ baselines) ---
        metrics = self._evaluate(final_model, calibrator, X_test, y_test)

        # --- 9. Sauvegarde ---
        joblib.dump(final_model,  self.model_path)
        joblib.dump(calibrator,   self.calibrator_path)
        print(f"\n💾 Modèle sauvegardé      : {self.model_path}")
        print(f"💾 Calibrateur sauvegardé : {self.calibrator_path}")

        # --- 10. Métadonnées (pour l'onglet Évaluation) ---
        self._sauver_metadata(type_modele, X_tr, X_val, X_test, y,
                              best_params, metrics, use_feature_selection,
                              best_iteration, features)


# ==========================================
# 6. PRÉDICTEUR (INFÉRENCE)
# ==========================================
class CryptoPredictor:
    """
    Charge un modèle entraîné + son calibrateur et produit des signaux de trading
    sur les données les plus récentes d'une crypto.

    Deux niveaux de signal sont générés :
    - Standard  (≥ 0.50) : signal classique basé sur la majorité
    - Sécurisé  (≥ threshold) : signal haute confiance uniquement
    """

    def __init__(self, symbole, intervalle, modele_symbole=None):
        """
        symbole         : crypto sur laquelle on génère les signaux (les DONNÉES)
        modele_symbole  : crypto dont on charge le MODÈLE (défaut = symbole).
                          Permet d'appliquer un modèle 'MULTI' (Portefeuille) à n'importe
                          quelle crypto : données = BTC, modèle = MULTI.
        """
        self.symbole         = symbole
        self.intervalle      = intervalle
        self.modele_symbole  = modele_symbole or symbole
        self.manager         = MLDataManager()
        self.model_path      = os.path.join(ModelConfig.MODEL_FOLDER, f"XGB_{self.modele_symbole}_{intervalle}.joblib")
        self.calibrator_path = os.path.join(ModelConfig.MODEL_FOLDER, f"CAL_{self.modele_symbole}_{intervalle}.joblib")

    def _load_model_and_calibrator(self):
        """
        Charge le modèle XGBoost et le calibrateur isotonique depuis le disque.
        Les deux fichiers sont nécessaires pour produire des probabilités calibrées.
        Retourne (None, None) si un fichier est manquant.
        """
        for path in [self.model_path, self.calibrator_path]:
            if not os.path.exists(path):
                print(f"❌ Fichier manquant : {path}")
                print("   → Lance d'abord CryptoModelTrainer.train()")
                return None, None

        return joblib.load(self.model_path), joblib.load(self.calibrator_path)

    def _apply_feature_selection(self, X):
        """
        Applique la même sélection de features que lors de l'entraînement.
        Si aucune sélection n'a été sauvegardée, X est retourné tel quel.
        Sans cette étape, le modèle recevrait des colonnes inconnues → erreur.
        """
        selector     = FeatureSelector(self.modele_symbole, self.intervalle)
        top_features = selector.load()

        if top_features is not None:
            top_features = [f for f in top_features if f in X.columns]
            return X[top_features]

        return X

    def _real_target(self, df):
        """
        Vérité terrain (1 si hausse après l'horizon) calculée PAR crypto si un dataset
        Portefeuille est passé ('symbole' présent), pour ne jamais franchir une frontière
        d'actif. Robuste à tout horizon (calculé depuis le prix, pas de colonne Target_*).
        """
        h = ModelConfig.TARGET_HORIZON

        def _single(sub):
            fr = sub['Close'].pct_change(h).shift(-h)
            rt = (fr > 0).astype(float)
            rt[fr.isna()] = np.nan
            return rt

        if 'symbole' in df.columns and df['symbole'].nunique() > 1:
            syms = df['symbole'].values
            out = np.full(len(df), np.nan)
            for s in pd.unique(syms):
                pos = np.where(syms == s)[0]
                out[pos] = _single(df.iloc[pos]).to_numpy()
            return pd.Series(out, index=df.index)
        return _single(df)

    def run_inference(self, threshold=0.60):
        """
        Génère les prédictions pour la crypto configurée.

        Processus :
        1. Chargement du modèle + calibrateur
        2. Construction des features (identique à l'entraînement)
        3. Application de la sélection de features SHAP
        4. Prédiction : probabilités brutes → probabilités calibrées → signaux
        5. Sauvegarde du tableau de résultats + rapport console

        Paramètre :
        - threshold (float) : seuil de confiance pour le signal sécurisé (ex: 0.60 = 60%)
        """
        etiquette = self.symbole if self.modele_symbole == self.symbole \
            else f"{self.symbole} ← modèle {self.modele_symbole}"
        print(f"\n🔮 Inférence : {etiquette} ({self.intervalle}) | Seuil : {threshold:.0%}")

        model, calibrator = self._load_model_and_calibrator()
        if model is None:
            return None

        df = self.manager.load_crypto_data(self.symbole, self.intervalle)
        if df is None:
            return None

        # Vérité terrain (par crypto si dataset Portefeuille) et prix, en numpy positionnel
        # → robuste aux index dupliqués (pas de .loc qui exploserait sur des doublons).
        real_target = self._real_target(df).to_numpy()
        prix        = pd.to_numeric(df['Close'], errors='coerce').to_numpy()

        # Construction et filtrage des features (identique au train)
        X = self.manager.build_features(df)
        X = self._apply_feature_selection(X)
        valid_mask = X.notna().all(axis=1).to_numpy()

        if not valid_mask.any():
            print("⚠️ Aucune ligne valide après nettoyage.")
            return None

        X_valid = X[valid_mask]

        # Probabilités brutes → calibrées
        raw_probs  = model.predict_proba(X_valid)[:, 1]
        cal_probs  = calibrator.predict(raw_probs)

        # Construction du tableau de résultats (alignement positionnel via valid_mask)
        results = pd.DataFrame(index=X_valid.index)
        results['Price']                           = prix[valid_mask]
        results['Real_Target']                     = real_target[valid_mask]
        results['Proba_Brute']                     = np.round(raw_probs, 4)
        results['Proba_Calibree']                  = np.round(cal_probs, 4)
        results['Signal_Standard']                 = (cal_probs >= 0.50).astype(int)
        results[f'Signal_{int(threshold*100)}pct'] = (cal_probs >= threshold).astype(int)

        save_path = os.path.join(ModelConfig.PRED_FOLDER, f"{self.symbole}_{self.intervalle}_prediction.xlsx")
        results.to_excel(save_path)
        print(f"✅ Prédictions sauvegardées : {save_path}")

        self._print_report(results, threshold)
        return results

    def _print_report(self, results, threshold):
        """
        Affiche un rapport de performance dans la console.

        Métriques :
        - AUC-ROC global : qualité de séparation sur toute la période
        - Nombre de trades pris au seuil de confiance demandé
        - Win Rate : % de signaux corrects parmi les signaux émis
        """
        col_secure = f'Signal_{int(threshold*100)}pct'
        valid      = results.dropna(subset=['Real_Target'])

        if valid.empty:
            print("⚠️ Pas de vraies valeurs disponibles pour évaluer.")
            return

        print(f"\n📊 ===== RAPPORT DE PERFORMANCE =====")

        if valid['Proba_Calibree'].nunique() > 1:
            auc = roc_auc_score(valid['Real_Target'], valid['Proba_Calibree'])
            print(f"   AUC-ROC global  : {auc:.4f}")

        secure = valid[valid[col_secure] == 1]
        if len(secure) > 0:
            win_rate = (secure['Real_Target'] == 1).mean()
            print(f"   Trades (≥{threshold:.0%}) : {len(secure)}")
            print(f"   Win Rate        : {win_rate:.2%}")
        else:
            print(f"   Aucun trade ne dépasse le seuil {threshold:.0%}.")

        print(f"=====================================\n")


# ==========================================
# 7. VISUALISATION
# ==========================================
class CryptoVisualizer:
    """
    Génère des graphiques pour comprendre et interpréter le modèle entraîné.

    Outils disponibles :
    - SHAP Beeswarm : impact individuel de chaque feature sur chaque prédiction
    - SHAP Bar      : importance globale moyenne de chaque feature
    - Texte d'arbre : règles brutes d'un arbre (pour debug)
    """

    def __init__(self, symbole, intervalle):
        self.symbole    = symbole
        self.intervalle = intervalle
        self.manager    = MLDataManager()
        self.model_path = os.path.join(ModelConfig.MODEL_FOLDER, f"XGB_{symbole}_{intervalle}.joblib")
        os.makedirs(ModelConfig.VIZ_FOLDER, exist_ok=True)

    def _load_model_and_sample(self, n_samples=1000):
        """
        Charge le modèle et prépare un échantillon aléatoire de données.
        Limiter à n_samples lignes évite des calculs SHAP trop longs.
        Applique aussi la même feature selection que pendant l'entraînement.
        Retourne (model, X_sample) ou (None, None) si erreur.
        """
        if not os.path.exists(self.model_path):
            print(f"❌ Modèle introuvable : {self.model_path}")
            return None, None

        model = joblib.load(self.model_path)
        df    = self.manager.load_crypto_data(self.symbole, self.intervalle)
        if df is None:
            return None, None

        X = self.manager.build_features(df)

        # Application de la feature selection pour rester cohérent avec l'entraînement
        selector     = FeatureSelector(self.symbole, self.intervalle)
        top_features = selector.load()
        if top_features is not None:
            top_features = [f for f in top_features if f in X.columns]
            X = X[top_features]

        X = X.dropna()
        if len(X) > n_samples:
            X = X.sample(n_samples, random_state=42)

        return model, X

    def plot_shap_summary(self):
        """
        Génère deux graphiques SHAP sauvegardés dans visualizations/ :

        1. Beeswarm plot — Pour chaque feature, montre comment ses valeurs
           influencent la prédiction. Couleur = valeur de la feature (rouge=haute,
           bleu=basse). Axe X = impact sur la prédiction (positif = pousse vers hausse).
           → "Quand le RSI est bas, ça pousse vers achat ou vente ?"

        2. Bar chart — Importance globale (valeur SHAP absolue moyenne par feature).
           → "Quelles sont les 10 features les plus déterminantes ?"
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            print("❌ SHAP non installé. Lance : pip install shap")
            return

        print(f"\n✨ Génération SHAP pour {self.symbole} ({self.intervalle})...")

        model, X_sample = self._load_model_and_sample()
        if model is None:
            return

        # SHAP TreeExplainer ne gère que les modèles à arbres. Pour LogisticRegression
        # (Pipeline) ou autre, on saute proprement au lieu de planter.
        ARBRES = {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"}
        nom_modele = type(model).__name__
        if nom_modele not in ARBRES:
            print(f"ℹ️  Graphes SHAP réservés aux modèles à arbres "
                  f"(modèle actuel : {nom_modele}). Étape ignorée.")
            return

        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # 1. Beeswarm
            plt.figure(figsize=(12, 9))
            shap.summary_plot(shap_values, X_sample, show=False)
            path1 = os.path.join(ModelConfig.VIZ_FOLDER, f"SHAP_Beeswarm_{self.symbole}_{self.intervalle}.png")
            plt.savefig(path1, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"✅ Beeswarm sauvegardé : {path1}")

            # 2. Bar chart
            plt.figure(figsize=(12, 9))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            path2 = os.path.join(ModelConfig.VIZ_FOLDER, f"SHAP_Bar_{self.symbole}_{self.intervalle}.png")
            plt.savefig(path2, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"✅ Bar chart sauvegardé : {path2}")

        except Exception as e:
            print(f"❌ Erreur SHAP : {e}")

    def save_tree_as_text(self, tree_index=0):
        """
        Exporte les règles brutes d'un arbre XGBoost en fichier texte lisible.
        Utile pour vérifier que le modèle apprend des règles financièrement sensées.

        Exemple de règle lisible :
        0:[RSI<30] yes=1, no=2
            1:[Dist_SMA_20<-0.05] yes=3, no=4
                3:leaf=0.15  → faible signal achat
                4:leaf=0.72  → fort signal achat
        """
        if not os.path.exists(self.model_path):
            print(f"❌ Modèle introuvable : {self.model_path}")
            return

        model = joblib.load(self.model_path)
        if not hasattr(model, "get_booster"):
            print(f"ℹ️  Export d'arbre réservé à XGBoost (modèle actuel : "
                  f"{type(model).__name__}). Étape ignorée.")
            return
        tree_dump = model.get_booster().get_dump()[tree_index]

        save_path = os.path.join(
            ModelConfig.VIZ_FOLDER,
            f"Tree_{self.symbole}_{self.intervalle}_idx{tree_index}.txt"
        )
        with open(save_path, "w") as f:
            f.write(tree_dump)

        print(f"✅ Arbre n°{tree_index} sauvegardé : {save_path}")


# ==========================================
# 8. EXÉCUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":

    # ┌─────────────────────────────────────┐
    # │         CONFIGURATION DU RUN        │
    # └─────────────────────────────────────┘
    SYMBOLE    = "BTC"
    INTERVALLE = "1h"
    THRESHOLD  = 0.60   # Seuil de confiance pour les signaux sécurisés

    # ── ÉTAPE 0 (optionnel) : WALK-FORWARD VALIDATION ─────────────────────
    # Évalue la robustesse du modèle sur plusieurs périodes avant d'entraîner.
    # Utile pour décider si le signal est stable sur bull/bear/range market.
    # Commenter si on veut aller directement à l'entraînement.
    # wfv = WalkForwardValidator(SYMBOLE, INTERVALLE)
    # wfv.run()

    # ── ÉTAPE 1 (optionnel) : FEATURE SELECTION ───────────────────────────
    # Calcule les features les plus importantes via SHAP et les sauvegarde.
    # À lancer une première fois, puis le Trainer les chargera automatiquement.
    # Commenter si déjà calculé (fichier TOP_FEATURES_BTC_1h.joblib existant).
    # selector = FeatureSelector(SYMBOLE, INTERVALLE, top_n=20)
    # selector.compute_and_save()

    # ── ÉTAPE 2 : ENTRAÎNEMENT ────────────────────────────────────────────
    # force_retrain=True  → réentraîne même si un modèle existe
    # force_retrain=False → utilise le modèle existant
    # use_feature_selection=True → filtre via SHAP si disponible
    trainer = CryptoModelTrainer(SYMBOLE, INTERVALLE)
    trainer.train(force_retrain=False, use_feature_selection=False)

    # ── ÉTAPE 3 : PRÉDICTION ──────────────────────────────────────────────
    predictor = CryptoPredictor(SYMBOLE, INTERVALLE)
    predictor.run_inference(threshold=THRESHOLD)

    # ── ÉTAPE 4 : VISUALISATION ───────────────────────────────────────────
    viz = CryptoVisualizer(SYMBOLE, INTERVALLE)
    viz.plot_shap_summary()
    viz.save_tree_as_text(tree_index=0)