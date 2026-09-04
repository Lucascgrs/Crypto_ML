"""Page 6 — Simulation de trading à partir des signaux."""

from __future__ import annotations

import customtkinter as ctk
import pandas as pd

from ... import backtest, stockage
from .. import theme
from ..textes import AIDES
from ..theme import COULEURS


class PageBacktest:
    """Rejoue les signaux du modèle et compare le résultat au « buy & hold »."""

    METRIQUES = [
        ("Rendement", 0, 0), ("Capital final", 1, 0), ("Buy & Hold", 2, 0),
        ("vs Marché", 3, 0), ("Trades", 0, 1), ("Win Rate", 1, 1),
        ("Profit Factor", 2, 1), ("Max Drawdown", 3, 1),
        ("Sharpe", 0, 2), ("Sortino", 1, 2), ("Calmar", 2, 2),
        ("Long / Short", 3, 2),
    ]

    def _page_backtest(self):
        page = self._nouvelle_page("Backtest")
        self._titre_page(page, "💰  Simulation de trading",
                         "Achat sur signal de hausse, vente à découvert sur signal "
                         "de baisse, frais compris.")

        barre = ctk.CTkFrame(page, fg_color=COULEURS["panneau"], corner_radius=12)
        barre.pack(fill="x", pady=(0, 10))

        # --- Première rangée : stratégie ----------------------------------
        rangee = ctk.CTkFrame(barre, fg_color="transparent")
        rangee.pack(fill="x", padx=18, pady=(14, 0))

        colonne, self.bt_fichier, self.bt_menu = self._menu(
            rangee, "Prédiction", ["(aucun)"], 200)
        colonne.pack(side="left", padx=(0, 12))

        champs = [
            ("Capital ($)", "1000", 100, "bt_capital", "bt_capital"),
            ("Seuil confiance", "0.60", 110, "bt_seuil", "bt_seuil"),
            ("Durée max", "12", 90, "bt_duree", "bt_duree"),
            ("Take Profit %", "0", 100, "bt_tp", "bt_tp"),
            ("Stop Loss %", "0", 100, "bt_sl", "bt_sl"),
            ("Frais %", "0.1", 90, "bt_frais", "bt_frais"),
            ("Slippage %", "0.05", 90, "bt_slippage", "bt_slippage"),
        ]
        for libelle, defaut, largeur, attribut, aide in champs:
            colonne, entree = self._champ(rangee, libelle, defaut, largeur,
                                          aide=AIDES[aide])
            colonne.pack(side="left", padx=(0, 10))
            setattr(self, attribut, entree)

        # --- Seconde rangée : périmètre et lancement ----------------------
        rangee2 = ctk.CTkFrame(barre, fg_color="transparent")
        rangee2.pack(fill="x", padx=18, pady=(10, 14))

        colonne_debut, self.bt_debut = self._champ(rangee2, "Début (AAAA-MM-JJ)", "",
                                                   140, aide=AIDES["bt_periode"])
        colonne_debut.pack(side="left", padx=(0, 10))
        colonne_fin, self.bt_fin = self._champ(rangee2, "Fin (AAAA-MM-JJ)", "",
                                               140, aide=AIDES["bt_periode"])
        colonne_fin.pack(side="left", padx=(0, 16))

        self.bt_test_seul = ctk.CTkCheckBox(rangee2, text="Uniquement le bloc test")
        self.bt_test_seul.select()
        self.bt_test_seul.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(rangee2, AIDES["bt_bloc"]).pack(side="left", padx=(0, 16),
                                                         pady=(18, 0))

        self.bt_short = ctk.CTkCheckBox(rangee2, text="Ventes à découvert")
        self.bt_short.select()
        self.bt_short.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(rangee2, AIDES["bt_short"]).pack(side="left", padx=(0, 16),
                                                          pady=(18, 0))

        self.bt_retenu = ctk.CTkCheckBox(rangee2, text="Respecter « Retenu »")
        self.bt_retenu.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(rangee2, AIDES["bt_retenu"]).pack(side="left", padx=(0, 16),
                                                           pady=(18, 0))

        colonne_sizing, self.bt_sizing, _ = self._menu(
            rangee2, "Taille de position", ["fixe", "proportionnel"], 150,
            aide=AIDES["bt_sizing"])
        colonne_sizing.pack(side="left", padx=(0, 16))

        ctk.CTkButton(rangee2, text="▶️ Simuler", height=40, width=140,
                      command=self._action_simuler).pack(side="left", pady=(14, 0))

        # --- Cartes de résultat -------------------------------------------
        cartes = ctk.CTkFrame(page, fg_color="transparent")
        cartes.pack(fill="x", pady=(0, 10))
        for colonne_grille in range(4):
            cartes.grid_columnconfigure(colonne_grille, weight=1)

        self.bt_cartes = {}
        for nom, colonne_grille, rangee_grille in self.METRIQUES:
            carte, valeur = self._carte_metrique(cartes, nom)
            carte.grid(row=rangee_grille, column=colonne_grille,
                       sticky="ew", padx=5, pady=5)
            self.bt_cartes[nom] = valeur

        # Bandeau d'avertissement : visible seulement quand la simulation
        # déborde sur les données d'entraînement.
        self.bt_avertissement = ctk.CTkLabel(
            page, text="", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COULEURS["rouge"], wraplength=1000, justify="left", anchor="w")
        self.bt_avertissement.pack(fill="x", pady=(0, 6))

        zone = self._zone_graphe(page, "backtest", hauteur=780)
        zone.pack(fill="x", pady=(0, 10))
        zone.pack_propagate(False)

        self.rafraichisseurs["Backtest"] = lambda: self._maj_menu(
            self.bt_menu, self.bt_fichier, stockage.lister_predictions())

    # ----------------------------------------------------------------------
    def _action_simuler(self):
        cle = self.bt_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune prédiction disponible. Lance d'abord « Prédire ».")
            return
        infos = stockage.analyser_cle(cle)
        symbole, intervalle, horizon = infos.symbole, infos.intervalle, infos.horizon

        debut = self.bt_debut.get().strip() or None
        fin = self.bt_fin.get().strip() or None
        for texte in (debut, fin):
            if texte and self._valider_date(texte) is None:
                self.log("❌ Période invalide — format AAAA-MM-JJ, ou champ vide.")
                return

        parametres = backtest.ParametresBacktest(
            capital_initial=self._lire_float(self.bt_capital, 1000.0),
            seuil_confiance=self._lire_float(self.bt_seuil, 0.60),
            take_profit=self._lire_float(self.bt_tp, 0.0) / 100,
            stop_loss=self._lire_float(self.bt_sl, 0.0) / 100,
            duree_max=max(1, self._lire_int(self.bt_duree, horizon)),
            frais=self._lire_float(self.bt_frais, 0.1) / 100,
            slippage=self._lire_float(self.bt_slippage, 0.05) / 100,
            ventes_a_decouvert=self.bt_short.get() == 1,
            sizing=self.bt_sizing.get(),
            respecter_retenu=self.bt_retenu.get() == 1,
        )
        bloc = "test" if self.bt_test_seul.get() == 1 else None

        def tache():
            df = stockage.lire_tableau(stockage.chemin_prediction(
                symbole, intervalle, horizon,
                walk_forward=infos.walk_forward, tache=infos.tache))
            if df is None or df.empty:
                raise FileNotFoundError("Fichier de prédiction introuvable.")
            return backtest.simuler_fichier(df, parametres, debut, fin, bloc)

        self.executer(f"Simulation {symbole} ({intervalle})", tache,
                      apres=self._afficher_resultats)

    # ----------------------------------------------------------------------
    def _afficher_resultats(self, resultat):
        self._avertir_donnees_apprises(resultat)
        self._remplir_cartes(resultat)
        self._tracer_courbes(resultat)
        mise = (f" · mise moyenne {resultat['mise_moyenne']:.0f}%"
                if resultat.get("sizing") == "proportionnel" else "")
        self.log(f"💰 {resultat['nb_trades']} trades · "
                 f"rendement {resultat['rendement_total']:+.1f}% · "
                 f"Buy & Hold {resultat['rendement_bh']:+.1f}% · "
                 f"drawdown {resultat['max_drawdown']:.1f}%{mise}")

    def _avertir_donnees_apprises(self, resultat):
        """
        Prévient quand la simulation couvre des données d'entraînement.

        Sur ces périodes le modèle rejoue ce qu'il a mémorisé : le rendement
        explose et ne dit rien de sa valeur réelle. Le message reste affiché
        tant que le cas se présente, pour qu'on ne lise pas le chiffre de
        travers.
        """
        part = resultat.get("part_apprise", 0.0)
        if part <= 0:
            self.bt_avertissement.configure(text="")
            return
        self.bt_avertissement.configure(
            text=(f"⚠️  {part:.0%} de la période simulée a servi à entraîner le modèle "
                  f"(zones rouge et orange sur les graphes). Sur ces périodes il rejoue "
                  f"ce qu'il a mémorisé : le rendement affiché est illusoire.\n"
                  f"Pour un résultat honnête sur tout l'historique, lance un "
                  f"« 📏 Walk-forward » depuis l'onglet Prédiction et simule le fichier "
                  f"« …_wf » qu'il produit."))

    def _remplir_cartes(self, resultat):
        cartes = self.bt_cartes
        rendement = resultat["rendement_total"]
        buy_hold = resultat["rendement_bh"]
        surperformance = rendement - buy_hold

        def teinte(valeur, bon=1.0, moyen=0.0):
            if valeur >= bon:
                return COULEURS["vert"]
            return COULEURS["orange"] if valeur >= moyen else COULEURS["rouge"]

        cartes["Rendement"].configure(
            text=f"{rendement:+.1f}%",
            text_color=COULEURS["vert"] if rendement >= 0 else COULEURS["rouge"])
        cartes["Capital final"].configure(text=f"{resultat['capital_final']:,.0f} $")
        cartes["Buy & Hold"].configure(text=f"{buy_hold:+.1f}%")
        cartes["vs Marché"].configure(
            text=f"{surperformance:+.1f}%",
            text_color=COULEURS["vert"] if surperformance >= 0 else COULEURS["rouge"])
        cartes["Trades"].configure(text=f"{resultat['nb_trades']:,}")
        cartes["Win Rate"].configure(text=f"{resultat['win_rate']:.0f}%")

        facteur = resultat["profit_factor"]
        cartes["Profit Factor"].configure(
            text="∞" if facteur == float("inf") else f"{facteur:.2f}")
        cartes["Max Drawdown"].configure(text=f"{resultat['max_drawdown']:.1f}%",
                                         text_color=COULEURS["rouge"])
        cartes["Sharpe"].configure(text=f"{resultat['sharpe']:.2f}",
                                   text_color=teinte(resultat["sharpe"]))
        cartes["Sortino"].configure(text=f"{resultat['sortino']:.2f}",
                                    text_color=teinte(resultat["sortino"], 1.5))
        cartes["Calmar"].configure(text=f"{resultat['calmar']:.2f}",
                                   text_color=teinte(resultat["calmar"]))
        # En mode proportionnel, la mise moyenne est la lecture qui manque :
        # un rendement plus faible avec 35 % de mise n'est pas un moins bon
        # résultat, c'est un risque trois fois moindre.
        mise = (f"  ·  mise {resultat['mise_moyenne']:.0f}%"
                if resultat.get("sizing") == "proportionnel" else "")
        cartes["Long / Short"].configure(
            text=f"{resultat['nb_longs']:,} / {resultat['nb_shorts']:,}{mise}",
            font=ctk.CTkFont(size=16, weight="bold"))

    # Zones colorées derrière les courbes : ce que le modèle a vu, et ce qu'il
    # n'a pas vu. Sans ce repère, un backtest lancé sur tout l'historique donne
    # un résultat magnifique qu'on peut prendre pour une performance réelle.
    TEINTES_BLOCS = {
        "train":      ("#e74c3c", 0.10, "Apprentissage — résultat non représentatif"),
        "validation": ("#f39c12", 0.10, "Validation — a servi au réglage"),
        "test":       ("#2ecc71", 0.07, "Jamais vu par le modèle"),
    }

    def _ombrer_blocs(self, axe, blocs, avec_legende=False):
        """Grise les périodes d'apprentissage et de validation sur un axe."""
        for nom, (debut, fin) in sorted(blocs.items(), key=lambda item: item[1][0]):
            teinte = self.TEINTES_BLOCS.get(nom)
            if teinte is None:
                continue
            couleur, opacite, libelle = teinte
            axe.axvspan(debut, fin, color=couleur, alpha=opacite, zorder=0,
                        label=libelle if avec_legende else None)

    def _tracer_courbes(self, resultat):
        """Deux graphes synchronisés : prix et trades en haut, capital en bas."""
        figure = self._nouvelle_figure((11, 8))
        axe_prix = figure.add_subplot(211)
        axe_capital = figure.add_subplot(212, sharex=axe_prix)
        theme.styliser_axes(axe_prix)
        theme.styliser_axes(axe_capital)

        blocs = resultat.get("blocs") or {}
        self._ombrer_blocs(axe_prix, blocs, avec_legende=True)
        self._ombrer_blocs(axe_capital, blocs)

        prix = resultat["prix"]
        axe_prix.plot(prix.index, prix.values, color=COULEURS["accent_clair"], lw=0.9,
                      label="Prix", zorder=1)

        trades = resultat["trades"]
        if not trades.empty:
            longs = trades[trades["Sens"] == "Long"]
            shorts = trades[trades["Sens"] == "Short"]
            if not longs.empty:
                axe_prix.scatter(pd.to_datetime(longs["Entrée"]), longs["Prix entrée"],
                                 color=COULEURS["vert"], s=26, marker="^",
                                 edgecolors="none", zorder=5,
                                 label=f"Achat ({len(longs)})")
            if not shorts.empty:
                axe_prix.scatter(pd.to_datetime(shorts["Entrée"]), shorts["Prix entrée"],
                                 color=COULEURS["rouge"], s=26, marker="v",
                                 edgecolors="none", zorder=5,
                                 label=f"Vente à découvert ({len(shorts)})")
        axe_prix.set_title("Prix et points d'entrée")
        axe_prix.set_ylabel("Prix ($)")
        theme.legende(axe_prix, loc="upper left", fontsize=8)

        capital = resultat["equity"]
        reference = resultat["buy_hold"]
        axe_capital.plot(capital.index, capital.values, color=COULEURS["vert"], lw=1.5,
                         label="Stratégie")
        axe_capital.plot(reference.index, reference.values, color=COULEURS["texte_doux"],
                         lw=1.0, ls="--", label="Buy & Hold")
        axe_capital.fill_between(capital.index, capital.values, capital.iloc[0],
                                 where=(capital.values >= capital.iloc[0]),
                                 alpha=0.08, color=COULEURS["vert"])
        axe_capital.set_title("Évolution du capital")
        axe_capital.set_ylabel("Capital ($)")
        theme.legende(axe_capital)

        figure.tight_layout()
        self._afficher_figure("backtest", figure)
