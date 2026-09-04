"""Page 5 — Graphiques : signaux, importance des indicateurs, fiabilité."""

from __future__ import annotations

import customtkinter as ctk
import numpy as np

from ... import modele, stockage
from .. import theme
from ..theme import COULEURS


class PageVisualisation:
    """Trois lectures complémentaires d'un modèle entraîné."""

    def _page_visualisation(self):
        page = self._nouvelle_page("Visualisation", defilante=False)
        page.grid_rowconfigure(2, weight=1)
        page.grid_columnconfigure(0, weight=1)

        entete = ctk.CTkFrame(page, fg_color="transparent")
        entete.grid(row=0, column=0, sticky="ew")
        self._titre_page(entete, "📊  Visualisation",
                         "Où le modèle voit juste, sur quoi il s'appuie, "
                         "et si sa confiance est honnête.")

        barre = ctk.CTkFrame(page, fg_color=COULEURS["panneau"], corner_radius=12)
        barre.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        interieur = ctk.CTkFrame(barre, fg_color="transparent")
        interieur.pack(fill="x", padx=18, pady=14)

        colonne, self.viz_fichier, self.viz_menu = self._menu(
            interieur, "Prédiction", ["(aucun)"], 220)
        colonne.pack(side="left", padx=(0, 16))

        for texte, commande in (
            ("📈 Prix & signaux", self._graphe_signaux),
            ("🏅 Importance des indicateurs", self._graphe_importance),
            ("🎯 Fiabilité de la confiance", self._graphe_fiabilite),
        ):
            ctk.CTkButton(interieur, text=texte, command=commande,
                          fg_color=COULEURS["carte"], hover_color=COULEURS["accent"]
                          ).pack(side="left", padx=4, pady=(18, 0))
        ctk.CTkButton(interieur, text="🔄", width=40,
                      command=lambda: self.afficher_page("Visualisation")
                      ).pack(side="left", padx=4, pady=(18, 0))

        zone = self._zone_graphe(page, "viz")
        zone.grid(row=2, column=0, sticky="nsew")

        self.rafraichisseurs["Visualisation"] = lambda: self._maj_menu(
            self.viz_menu, self.viz_fichier, stockage.lister_predictions())

    # ----------------------------------------------------------------------
    def _selection_viz(self):
        """(symbole, intervalle, horizon) de la prédiction sélectionnée."""
        cle = self.viz_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune prédiction. Lance d'abord « Prédire ».")
            return None
        return stockage.separer_cle_modele(cle)

    def _charger_prediction(self, symbole, intervalle, horizon):
        df = stockage.lire_tableau(stockage.chemin_prediction(symbole, intervalle, horizon))
        if df is None or df.empty:
            raise FileNotFoundError("Fichier de prédiction introuvable.")
        return df

    # ----------------------------------------------------------------------
    # 1. Prix et justesse des signaux
    # ----------------------------------------------------------------------
    def _graphe_signaux(self):
        selection = self._selection_viz()
        if selection is None:
            return
        symbole, intervalle, horizon = selection
        seuil = self.lire_seuil()

        def apres(df):
            figure = self._nouvelle_figure((11, 6))
            axe = figure.add_subplot(111)
            theme.styliser_axes(axe)

            axe.plot(df.index, df["Prix"], color=COULEURS["accent_clair"], lw=0.9,
                     label="Prix", zorder=1)

            # On ne montre que les signaux retenus, ceux qu'on aurait joués.
            retenues = df[(df["Confiance"] >= seuil) & df["Correct"].notna()]
            justes = retenues[retenues["Correct"] == 1]
            faux = retenues[retenues["Correct"] == 0]

            axe.scatter(justes.index, justes["Prix"], color=COULEURS["vert"],
                        s=11, alpha=0.6, zorder=3, label=f"Correct ({len(justes):,})")
            axe.scatter(faux.index, faux["Prix"], color=COULEURS["rouge"],
                        s=11, alpha=0.6, zorder=3, label=f"Erroné ({len(faux):,})")

            # Frontière du bloc test : à droite, données jamais vues à l'entraînement.
            test = df[df["Bloc"] == "test"]
            if not test.empty:
                axe.axvline(test.index[0], color=COULEURS["orange"], ls="--", lw=1.2,
                            label="Début du test (jamais vu)")

            justesse = retenues["Correct"].mean() if len(retenues) else float("nan")
            axe.set_title(f"{symbole} ({intervalle}) — signaux retenus à ≥ {seuil:.0%} "
                          f"de confiance · horizon {horizon} · justesse {justesse:.1%}")
            axe.set_ylabel("Prix ($)")
            theme.legende(axe, loc="upper left", fontsize=9)
            figure.tight_layout()
            self._afficher_figure("viz", figure)

        self.executer(f"Graphique {symbole}",
                      lambda: self._charger_prediction(symbole, intervalle, horizon),
                      apres=apres)

    # ----------------------------------------------------------------------
    # 2. Importance des indicateurs
    # ----------------------------------------------------------------------
    def _graphe_importance(self):
        selection = self._selection_viz()
        if selection is None:
            return
        symbole, intervalle, horizon = selection

        def apres(importance):
            figure = self._nouvelle_figure((10, 5))
            axe = figure.add_subplot(111)
            theme.styliser_axes(axe)

            valeurs = importance.sort_values()
            couleurs = [COULEURS["vert"] if v > 0 else COULEURS["rouge"]
                        for v in valeurs.values]
            axe.barh(valeurs.index, valeurs.values, color=couleurs)
            axe.axvline(0, color="#777777", lw=0.8)
            axe.set_title(f"{symbole} ({intervalle}) — perte d'AUC quand l'indicateur "
                          f"est mélangé (horizon {horizon})")
            axe.set_xlabel("Perte d'AUC (plus c'est grand, plus l'indicateur compte)")
            figure.tight_layout()
            self._afficher_figure("viz", figure)

            classement = " · ".join(f"{nom} {valeur:+.4f}"
                                    for nom, valeur in importance.items())
            self.log(f"🏅 Importance : {classement}")

        self.executer(
            f"Importance {symbole}",
            lambda: modele.importance_indicateurs(symbole, intervalle, horizon),
            apres=apres)

    # ----------------------------------------------------------------------
    # 3. Fiabilité de la confiance annoncée
    # ----------------------------------------------------------------------
    def _graphe_fiabilite(self):
        """
        Compare la confiance annoncée à la justesse réellement observée.

        Une courbe proche de la diagonale signifie que « 60 % de confiance »
        correspond bien à 60 % de réussite : le seuil est alors interprétable.
        """
        selection = self._selection_viz()
        if selection is None:
            return
        symbole, intervalle, horizon = selection

        def calculer():
            df = self._charger_prediction(symbole, intervalle, horizon)
            test = df[(df["Bloc"] == "test") & df["Correct"].notna()]
            base = test if len(test) > 200 else df.dropna(subset=["Correct"])
            if base.empty:
                raise ValueError("Pas assez de prédictions évaluables.")

            # Tranches de confiance de même effectif : chaque point du graphe
            # s'appuie sur le même nombre d'observations.
            rangs = np.argsort(base["Confiance"].to_numpy())
            groupes = np.array_split(rangs, min(10, max(2, len(base) // 200)))
            annonce, observe, effectifs = [], [], []
            for groupe in groupes:
                if len(groupe) == 0:
                    continue
                annonce.append(base["Confiance"].to_numpy()[groupe].mean())
                observe.append(base["Correct"].to_numpy()[groupe].mean())
                effectifs.append(len(groupe))
            return annonce, observe, effectifs, len(base), len(test) > 200

        def apres(resultat):
            annonce, observe, effectifs, total, sur_test = resultat
            figure = self._nouvelle_figure((9, 6))
            axe = figure.add_subplot(111)
            theme.styliser_axes(axe)

            axe.plot([0.5, 1.0], [0.5, 1.0], ls="--", color=COULEURS["texte_doux"],
                     lw=1.0, label="Confiance parfaitement honnête")
            axe.plot(annonce, observe, "o-", color=COULEURS["accent_clair"], lw=1.8,
                     markersize=7, label="Modèle")
            axe.axhline(0.5, color=COULEURS["rouge"], ls=":", lw=1.0, label="Hasard")

            for x, y, n in zip(annonce, observe, effectifs):
                axe.annotate(f"n={n:,}", (x, y), textcoords="offset points",
                             xytext=(0, 9), ha="center", fontsize=8,
                             color=COULEURS["texte_doux"])

            portee = "bloc test" if sur_test else "tout l'historique"
            axe.set_title(f"{symbole} ({intervalle}) — confiance annoncée vs justesse "
                          f"observée ({portee}, {total:,} bougies)")
            axe.set_xlabel("Confiance annoncée par le modèle")
            axe.set_ylabel("Part de prédictions correctes")
            theme.legende(axe, loc="upper left", fontsize=9)
            figure.tight_layout()
            self._afficher_figure("viz", figure)

        self.executer(f"Fiabilité {symbole}", calculer, apres=apres)
