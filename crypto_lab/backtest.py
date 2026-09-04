"""
Simulateur de trades (backtest) sur un fichier de prédictions.

Le modèle prédisant dans les DEUX SENS, le simulateur les exploite tous les
deux : on achète (position longue) sur un signal de hausse et on vend à
découvert (position courte) sur un signal de baisse — cette seconde partie
restant désactivable. Un signal NEUTRE (objectif « amplitude », classe centrale)
ne déclenche aucune position : c'est précisément ce qu'il annonce.

Une position est ouverte dès que la confiance atteint le seuil, puis fermée sur
take-profit, stop-loss, ou à l'échéance de l'horizon prédit (par défaut). Les
frais et le slippage sont facturés à l'entrée ET à la sortie.

TAILLE DE POSITION
------------------
Deux modes. En **fixe**, tout le capital part sur chaque signal — simple, mais
un signal à 51 % est traité comme un signal à 65 %. En **proportionnel**, la
fraction engagée suit l'avantage estimé et recule quand la volatilité prévue
monte (voir `_fraction`). Le second améliore généralement le ratio de Sharpe
sans toucher au modèle : on ne gagne pas plus souvent, on mise mieux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Nombre de secondes dans une année, pour annualiser les ratios de risque.
SECONDES_PAR_AN = 365.25 * 24 * 3600

# Confiance à partir de laquelle on engage la totalité de la fraction autorisée,
# en mode proportionnel. Au-delà de 65 % sur de la direction crypto, on est déjà
# dans le domaine des modèles trop beaux pour être vrais.
CONFIANCE_PLEINE = 0.65


@dataclass
class ParametresBacktest:
    """Réglages d'une simulation."""

    capital_initial: float = 1000.0
    seuil_confiance: float = 0.60
    # Take-profit et stop-loss sont DÉSACTIVÉS par défaut (0 = pas de barrière).
    # La sortie se fait alors à l'échéance de l'horizon, ce qui teste exactement
    # ce que le modèle prétend : « dans X périodes, le prix sera plus haut ».
    # Des barrières mal réglées (TP lointain, SL serré) suffisent à transformer
    # un signal correct en stratégie perdante, indépendamment du modèle.
    take_profit: float = 0.0       # en proportion (0.05 = +5 %), 0 = désactivé
    stop_loss: float = 0.0         # en proportion, 0 = désactivé
    duree_max: int = 12            # périodes de détention maximum
    frais: float = 0.001           # par transaction (0.001 = 0.1 %)
    slippage: float = 0.0005       # glissement de prix par transaction
    ventes_a_decouvert: bool = True
    # Dimensionnement : "fixe" (tout le capital) ou "proportionnel"
    # (fraction ∝ avantage estimé, ÷ volatilité prévue).
    sizing: str = "fixe"
    # Respecter la colonne `Retenu` du fichier plutôt que le seul seuil de
    # confiance. Utile pour les fichiers d'espérance de gain, où `Retenu`
    # encode aussi « le mouvement attendu couvre-t-il les frais ? ». Faux par
    # défaut : sur un fichier de prédiction ordinaire, `Retenu` reflète le
    # seuil choisi au moment de la prédiction, pas celui de la simulation.
    respecter_retenu: bool = False

    @property
    def cout_aller_retour(self) -> float:
        """Coût total d'un trade : entrée + sortie."""
        return 2 * (self.frais + self.slippage)

    @property
    def proportionnel(self) -> bool:
        return str(self.sizing).lower().startswith("prop")


class Simulateur:
    """
    Rejoue les signaux bougie par bougie et reconstruit la courbe de capital.

    Une seule position ouverte à la fois : le capital est engagé en entier, ce
    qui rend le résultat directement comparable au « buy & hold ».
    """

    def __init__(self, df: pd.DataFrame, parametres: ParametresBacktest):
        self.df = df
        self.p = parametres
        self.fractions = self._fractions()

    # -- boucle principale --------------------------------------------------
    def simuler(self) -> dict:
        prix = self.df["Prix"].to_numpy(dtype=float)
        confiance = self.df["Confiance"].to_numpy(dtype=float)
        sens_predit = self.df["Sens_Predit"].to_numpy()
        retenu = (self.df["Retenu"].to_numpy(dtype=float)
                  if self.p.respecter_retenu and "Retenu" in self.df.columns else None)
        dates = self.df.index

        capital = self.p.capital_initial
        courbe: list[float] = []
        trades: list[dict] = []

        position = 0          # 0 = hors marché, +1 = long, -1 = short
        prix_entree = 0.0
        indice_entree = 0
        date_entree = None
        fraction = 1.0

        for i, cours in enumerate(prix):
            if position != 0:
                variation = position * (cours - prix_entree) / prix_entree
                raison = self._raison_sortie(variation, i - indice_entree)

                if raison is None:
                    courbe.append(capital * (1 + fraction * variation))
                    continue

                capital, trade = self._cloturer(
                    capital, position, prix_entree, cours,
                    date_entree, dates[i], variation, raison, fraction)
                trades.append(trade)
                position = 0
                courbe.append(capital)
                continue

            courbe.append(capital)
            sens = self._signal(confiance[i], sens_predit[i],
                                None if retenu is None else retenu[i])
            if sens != 0:
                position, prix_entree, indice_entree, date_entree = sens, cours, i, dates[i]
                fraction = self.fractions[i]

        # Position encore ouverte en fin d'historique : on la solde au dernier cours.
        if position != 0:
            variation = position * (prix[-1] - prix_entree) / prix_entree
            capital, trade = self._cloturer(
                capital, position, prix_entree, prix[-1],
                date_entree, dates[-1], variation, "Fin de période", fraction)
            trades.append(trade)
            courbe[-1] = capital

        return self._resultats(pd.Series(courbe, index=dates),
                               pd.DataFrame(trades),
                               pd.Series(prix, index=dates),
                               capital)

    # -- règles d'entrée / sortie -------------------------------------------
    def _signal(self, confiance: float, sens: str, retenu: float | None) -> int:
        """Sens à prendre pour cette bougie : +1 long, -1 short, 0 rien."""
        if confiance < self.p.seuil_confiance:
            return 0
        # Les fichiers d'espérance de gain portent leur propre décision (le
        # mouvement couvre-t-il les frais ?) : quand on choisit de la respecter,
        # elle s'ajoute au seuil de confiance.
        if retenu is not None and retenu <= 0:
            return 0
        if sens == "HAUSSE":
            return 1
        if sens == "BAISSE":
            return -1 if self.p.ventes_a_decouvert else 0
        return 0                     # NEUTRE : le modèle dit « ne rien faire »

    # -- taille de position -------------------------------------------------
    def _fractions(self) -> np.ndarray:
        """
        Fraction du capital engagée sur chaque bougie.

        En mode fixe, toujours 1. En mode proportionnel, deux facteurs se
        multiplient :

          * l'**avantage** — (confiance − seuil) rapporté à l'écart qui reste
            jusqu'à `CONFIANCE_PLEINE`. Un signal tout juste au-dessus du seuil
            engage peu, un signal très tranché engage tout ;
          * la **volatilité prévue** — si le modèle d'amplitude annonce un
            mouvement deux fois plus agité que d'habitude, on réduit de moitié.
            C'est le principe du risque constant : ce qu'on égalise entre deux
            trades, ce n'est pas la mise mais le risque encouru.

        Le résultat est borné entre `config.FRACTION_MINIMALE` et
        `FRACTION_MAXIMALE` : ni position dérisoire, ni levier.
        """
        n = len(self.df)
        if not self.p.proportionnel:
            return np.ones(n)

        confiance = self.df["Confiance"].to_numpy(dtype=float)
        marge = max(1e-6, CONFIANCE_PLEINE - self.p.seuil_confiance)
        avantage = np.clip((confiance - self.p.seuil_confiance) / marge, 0.0, 1.0)

        # Un avantage nul ne doit pas donner une position nulle : le seuil a
        # déjà fait le tri, on part donc du plancher et on monte jusqu'à 1.
        fraction = config.FRACTION_MINIMALE + avantage * (1 - config.FRACTION_MINIMALE)

        if "Volatilite_Prevue" in self.df.columns:
            volatilite = self.df["Volatilite_Prevue"].to_numpy(dtype=float)
            reference = np.nanmedian(volatilite)
            if np.isfinite(reference) and reference > 0:
                ajustement = np.divide(reference, volatilite,
                                       out=np.ones_like(volatilite),
                                       where=volatilite > 0)
                fraction = fraction * np.clip(ajustement, 0.25, 2.0)

        return np.clip(fraction, config.FRACTION_MINIMALE, config.FRACTION_MAXIMALE)

    def _raison_sortie(self, variation: float, duree: int) -> str | None:
        """Motif de clôture, ou None si la position reste ouverte."""
        if self.p.take_profit > 0 and variation >= self.p.take_profit:
            return "Take Profit"
        if self.p.stop_loss > 0 and variation <= -self.p.stop_loss:
            return "Stop Loss"
        if duree >= self.p.duree_max:
            return "Horizon atteint"
        return None

    def _cloturer(self, capital, position, prix_entree, prix_sortie,
                  date_entree, date_sortie, variation, raison, fraction=1.0):
        """
        Applique le résultat net d'un trade au capital et journalise l'opération.

        Frais et slippage portent sur le montant RÉELLEMENT engagé : engager la
        moitié du capital ne coûte que la moitié des frais. Sans cela, le mode
        proportionnel serait pénalisé deux fois.
        """
        rendement_net = fraction * (variation - self.p.cout_aller_retour)
        nouveau_capital = capital * (1 + rendement_net)
        trade = {
            "Entrée": date_entree.strftime("%Y-%m-%d %H:%M"),
            "Sortie": date_sortie.strftime("%Y-%m-%d %H:%M"),
            "Sens": "Long" if position > 0 else "Short",
            "Mise %": round(fraction * 100, 1),
            "Prix entrée": round(prix_entree, 4),
            "Prix sortie": round(prix_sortie, 4),
            "Rendement %": round(rendement_net * 100, 3),
            "P&L": round(nouveau_capital - capital, 2),
            "Raison": raison,
        }
        return nouveau_capital, trade

    # -- métriques -----------------------------------------------------------
    def _resultats(self, courbe: pd.Series, trades: pd.DataFrame,
                   prix: pd.Series, capital_final: float) -> dict:
        buy_hold = self.p.capital_initial * prix / prix.iloc[0]

        nb_trades = len(trades)
        gagnants = int((trades["P&L"] > 0).sum()) if nb_trades else 0
        gains = float(trades.loc[trades["P&L"] > 0, "P&L"].sum()) if nb_trades else 0.0
        pertes = abs(float(trades.loc[trades["P&L"] < 0, "P&L"].sum())) if nb_trades else 0.0

        drawdown = (courbe - courbe.cummax()) / courbe.cummax()
        max_drawdown = float(drawdown.min() * 100) if len(drawdown) else 0.0
        rendement = (capital_final / self.p.capital_initial - 1) * 100

        sharpe, sortino = self._ratios_risque(courbe)

        return {
            "trades": trades,
            "equity": courbe,
            "buy_hold": buy_hold,
            "prix": prix,
            "capital_final": capital_final,
            "rendement_total": rendement,
            "rendement_bh": float(buy_hold.iloc[-1] / self.p.capital_initial - 1) * 100,
            "nb_trades": nb_trades,
            "nb_longs": int((trades["Sens"] == "Long").sum()) if nb_trades else 0,
            "nb_shorts": int((trades["Sens"] == "Short").sum()) if nb_trades else 0,
            "win_rate": (gagnants / nb_trades * 100) if nb_trades else 0.0,
            "profit_factor": (gains / pertes) if pertes > 0 else float("inf"),
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": (rendement / abs(max_drawdown)) if max_drawdown else 0.0,
            "sizing": self.p.sizing,
            "mise_moyenne": (float(trades["Mise %"].mean()) if nb_trades else 100.0),
        }

    @staticmethod
    def _ratios_risque(courbe: pd.Series) -> tuple[float, float]:
        """
        Sharpe et Sortino annualisés.

        Le facteur d'annualisation est déduit de l'écart réel entre deux points
        de la courbe : le même code fonctionne donc en 1 h comme en 1 jour.
        """
        rendements = courbe.pct_change().dropna()
        if len(rendements) < 3:
            return 0.0, 0.0

        duree = (courbe.index[-1] - courbe.index[0]).total_seconds()
        periodes_par_an = SECONDES_PAR_AN / (duree / max(1, len(courbe) - 1)) if duree > 0 else 252
        racine = float(np.sqrt(periodes_par_an))

        ecart = rendements.std()
        sharpe = float(rendements.mean() / ecart * racine) if ecart > 0 else 0.0

        negatifs = rendements[rendements < 0]
        sortino = (float(rendements.mean() / negatifs.std() * racine)
                   if len(negatifs) > 1 and negatifs.std() > 0 else 0.0)
        return sharpe, sortino


def simuler_fichier(df: pd.DataFrame, parametres: ParametresBacktest,
                    debut: str | None = None, fin: str | None = None,
                    bloc: str | None = None) -> dict:
    """
    Lance une simulation sur un tableau de prédictions, filtré au besoin.

    `bloc="test"` restreint la simulation aux données jamais vues par le modèle
    pendant l'entraînement — c'est le seul résultat vraiment représentatif.
    """
    colonnes = {"Prix", "Confiance", "Sens_Predit"}
    if not colonnes.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans le fichier de prédiction : "
                         f"{sorted(colonnes - set(df.columns))}")

    if bloc and "Bloc" in df.columns:
        df = df[df["Bloc"] == bloc]
    if debut:
        df = df[df.index >= pd.Timestamp(debut)]
    if fin:
        df = df[df.index <= pd.Timestamp(fin)]

    if len(df) < 5:
        raise ValueError("Période trop courte pour simuler (moins de 5 bougies).")

    print(f"📅 Simulation sur {len(df):,} bougies "
          f"({df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d})")

    resultat = Simulateur(df, parametres).simuler()
    resultat["blocs"] = _bornes_blocs(df)
    resultat["part_apprise"] = _part_apprise(df)
    if resultat["part_apprise"] > 0:
        print(f"⚠️  {resultat['part_apprise']:.0%} de la période simulée a servi à "
              f"l'entraînement : le rendement affiché n'est pas représentatif.")
    return resultat


def _bornes_blocs(df: pd.DataFrame) -> dict:
    """
    Début et fin de chaque bloc (train / validation / test) dans la période simulée.

    Sert à griser la zone d'apprentissage sur les graphiques : un backtest qui
    couvre les données d'entraînement affiche un résultat spectaculaire et faux,
    autant que ça se voie au premier coup d'œil.
    """
    if "Bloc" not in df.columns:
        return {}
    bornes = {}
    for nom, sous_ensemble in df.groupby("Bloc"):
        if len(sous_ensemble):
            bornes[str(nom)] = (sous_ensemble.index.min(), sous_ensemble.index.max())
    return bornes


def _part_apprise(df: pd.DataFrame) -> float:
    """Proportion de la période simulée que le modèle a vue pendant l'entraînement."""
    if "Bloc" not in df.columns or df.empty:
        return 0.0
    return float(df["Bloc"].isin(["train", "validation"]).mean())
