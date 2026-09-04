"""
Simulateur de trades (backtest) sur un fichier de prédictions.

Le modèle prédisant dans les DEUX SENS, le simulateur les exploite tous les
deux : on achète (position longue) sur un signal de hausse et on vend à
découvert (position courte) sur un signal de baisse — cette seconde partie
restant désactivable.

Une position est ouverte dès que la confiance atteint le seuil, puis fermée sur
take-profit, stop-loss, ou à l'échéance de l'horizon prédit (par défaut). Les
frais et le slippage sont facturés à l'entrée ET à la sortie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Nombre de secondes dans une année, pour annualiser les ratios de risque.
SECONDES_PAR_AN = 365.25 * 24 * 3600


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

    @property
    def cout_aller_retour(self) -> float:
        """Coût total d'un trade : entrée + sortie."""
        return 2 * (self.frais + self.slippage)


class Simulateur:
    """
    Rejoue les signaux bougie par bougie et reconstruit la courbe de capital.

    Une seule position ouverte à la fois : le capital est engagé en entier, ce
    qui rend le résultat directement comparable au « buy & hold ».
    """

    def __init__(self, df: pd.DataFrame, parametres: ParametresBacktest):
        self.df = df
        self.p = parametres

    # -- boucle principale --------------------------------------------------
    def simuler(self) -> dict:
        prix = self.df["Prix"].to_numpy(dtype=float)
        confiance = self.df["Confiance"].to_numpy(dtype=float)
        est_hausse = (self.df["Sens_Predit"].to_numpy() == "HAUSSE")
        dates = self.df.index

        capital = self.p.capital_initial
        courbe: list[float] = []
        trades: list[dict] = []

        position = 0          # 0 = hors marché, +1 = long, -1 = short
        prix_entree = 0.0
        indice_entree = 0
        date_entree = None

        for i, cours in enumerate(prix):
            if position != 0:
                variation = position * (cours - prix_entree) / prix_entree
                raison = self._raison_sortie(variation, i - indice_entree)

                if raison is None:
                    courbe.append(capital * (1 + variation))
                    continue

                capital, trade = self._cloturer(
                    capital, position, prix_entree, cours,
                    date_entree, dates[i], variation, raison)
                trades.append(trade)
                position = 0
                courbe.append(capital)
                continue

            courbe.append(capital)
            sens = self._signal(confiance[i], est_hausse[i])
            if sens != 0:
                position, prix_entree, indice_entree, date_entree = sens, cours, i, dates[i]

        # Position encore ouverte en fin d'historique : on la solde au dernier cours.
        if position != 0:
            variation = position * (prix[-1] - prix_entree) / prix_entree
            capital, trade = self._cloturer(
                capital, position, prix_entree, prix[-1],
                date_entree, dates[-1], variation, "Fin de période")
            trades.append(trade)
            courbe[-1] = capital

        return self._resultats(pd.Series(courbe, index=dates),
                               pd.DataFrame(trades),
                               pd.Series(prix, index=dates),
                               capital)

    # -- règles d'entrée / sortie -------------------------------------------
    def _signal(self, confiance: float, hausse: bool) -> int:
        """Sens à prendre pour cette bougie : +1 long, -1 short, 0 rien."""
        if confiance < self.p.seuil_confiance:
            return 0
        if hausse:
            return 1
        return -1 if self.p.ventes_a_decouvert else 0

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
                  date_entree, date_sortie, variation, raison):
        """Applique le résultat net d'un trade au capital et journalise l'opération."""
        rendement_net = variation - self.p.cout_aller_retour
        nouveau_capital = capital * (1 + rendement_net)
        trade = {
            "Entrée": date_entree.strftime("%Y-%m-%d %H:%M"),
            "Sortie": date_sortie.strftime("%Y-%m-%d %H:%M"),
            "Sens": "Long" if position > 0 else "Short",
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
    return Simulateur(df, parametres).simuler()
