"""
Page 4 — Amplitude : combien, et pas seulement dans quel sens.

Trois actions, dans l'ordre logique :

  📐 Entraîner   les deux régressions (volatilité + quantiles 10/50/90)
  🧮 Espérance   combine direction × amplitude en un score de décision
  📊 Intervalle  trace la fourchette Q10–Q90 autour du prix

L'horizon et le seuil de confiance sont ceux de la page Prédiction : ce sont
les mêmes notions, il n'y a aucune raison d'avoir deux réglages pour une seule
idée.
"""

from __future__ import annotations

import customtkinter as ctk
import numpy as np

from ... import amplitude, config, stockage
from .. import theme
from ..textes import AIDES, EXPLICATION_AMPLITUDE
from ..theme import COULEURS

# Nombre de bougies affichées sur le graphique d'intervalle. Au-delà, la bande
# Q10–Q90 devient un aplat illisible.
BOUGIES_AFFICHEES = 400


class PageAmplitude:
    """Régression quantile, volatilité attendue et espérance de gain."""

    def _page_amplitude(self):
        page = self._nouvelle_page("Amplitude")
        self._titre_page(
            page, "📐  Amplitude & espérance",
            "« De combien ça bouge ? » — puis « est-ce que ça paie les frais ? »")

        # ------------------------------------------------------------------
        # Réglages : deux menus, le reste vient de la page Prédiction
        # ------------------------------------------------------------------
        corps = self._section(page, "Réglages")
        ligne = self._ligne(corps, espace=(0, 0))

        colonne, self.amp_fichier, self.amp_menu = self._menu(
            ligne, "Crypto analysée", ["(aucun)"], 190, aide=AIDES["crypto_modele"])
        colonne.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Amplitude")
                      ).pack(side="left", padx=(0, 20), pady=(18, 0))

        colonne_modele, self.amp_type, _ = self._menu(
            ligne, "Modèle de régression", amplitude.modeles_disponibles(), 190,
            aide=AIDES["modele_regression"])
        colonne_modele.pack(side="left", padx=(0, 12))

        self.amp_rappel = ctk.CTkLabel(
            corps, text="", font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], anchor="w")
        self.amp_rappel.pack(fill="x", padx=2, pady=(14, 0))

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------
        corps2 = self._section(page, "Actions")
        boutons = self._ligne(corps2, espace=(0, 0))

        bouton_entrainer = ctk.CTkButton(
            boutons, text="📐 Entraîner l'amplitude", height=42, width=200,
            command=self._action_entrainer_amplitude)
        bouton_entrainer.pack(side="left")
        self._badge_info(boutons, AIDES["amplitude_entrainer"]).pack(
            side="left", padx=(6, 14), pady=(12, 0))

        bouton_esperance = ctk.CTkButton(
            boutons, text="🧮 Espérance de gain", height=42, width=190,
            fg_color=COULEURS["vert"], hover_color="#27ae60",
            command=self._action_esperance)
        bouton_esperance.pack(side="left")
        self._badge_info(boutons, AIDES["esperance"]).pack(
            side="left", padx=(6, 14), pady=(12, 0))

        ctk.CTkButton(boutons, text="📊 Intervalle prévu", height=42, width=170,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._action_graphe_intervalle).pack(side="left")

        self._bouton_explication(
            boutons, "❔ Comment ça marche",
            "Prédire l'amplitude, pas seulement le sens",
            EXPLICATION_AMPLITUDE).pack(side="left", padx=(14, 0), pady=(5, 0))

        # ------------------------------------------------------------------
        # Résultats
        # ------------------------------------------------------------------
        corps3 = self._section(page, "Dernière bougie")
        cartes = self._ligne(corps3, espace=(0, 0))
        self.amp_carte_sens = self._carte_metrique(cartes, "Sens & confiance")
        self.amp_carte_volatilite = self._carte_metrique(cartes, "Amplitude attendue")
        self.amp_carte_intervalle = self._carte_metrique(cartes, "Intervalle 80 %")
        self.amp_carte_esperance = self._carte_metrique(cartes, "Espérance nette")
        for carte in (self.amp_carte_sens, self.amp_carte_volatilite,
                      self.amp_carte_intervalle, self.amp_carte_esperance):
            carte[0].pack(side="left", padx=(0, 10), fill="x", expand=True)

        self.amp_resume = ctk.CTkLabel(
            corps3, text="Entraîne d'abord un modèle de direction (onglet Prédiction), "
                         "puis l'amplitude ici.",
            font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], wraplength=900, anchor="w")
        self.amp_resume.pack(fill="x", pady=(14, 0))

        zone = self._zone_graphe(page, "amplitude", hauteur=430)
        zone.pack(fill="x", pady=(4, 10))
        zone.pack_propagate(False)

        self.rafraichisseurs["Amplitude"] = self._rafraichir_amplitude

    # ----------------------------------------------------------------------
    def _rafraichir_amplitude(self):
        self._maj_menu(self.amp_menu, self.amp_fichier, stockage.lister_analyses())
        self.amp_rappel.configure(
            text=f"Horizon {self.lire_horizon()} période(s) · seuil de confiance "
                 f"{self.lire_seuil():.0%} · objectif « {self._objectif().libelle} » "
                 f"— réglés sur l'onglet « 3 · Prédiction », pour n'avoir qu'un seul "
                 f"endroit où les changer.")

    def _reglages_amplitude(self):
        """(symbole, intervalle, horizon, seuil, modèle) ou None."""
        cle = self.amp_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune crypto analysée. Lance d'abord l'étape Analyse.")
            return None
        symbole, intervalle = stockage.separer_cle(cle)
        return (symbole, intervalle, self.lire_horizon(), self.lire_seuil(),
                self.amp_type.get())

    # ----------------------------------------------------------------------
    # Actions
    # ----------------------------------------------------------------------
    def _action_entrainer_amplitude(self):
        """
        Entraîne les DEUX régressions d'un coup.

        Les séparer n'aurait aucun intérêt : la volatilité sert à l'espérance
        de gain, les quantiles à l'affichage de l'intervalle, et on veut
        toujours les deux.
        """
        reglages = self._reglages_amplitude()
        if reglages is None:
            return
        symbole, intervalle, horizon, _, nom_modele = reglages

        def tache():
            volatilite = amplitude.entrainer(symbole, intervalle, horizon,
                                             nom_modele, "volatilite")
            quantiles = amplitude.entrainer(symbole, intervalle, horizon,
                                            nom_modele, "amplitude")
            return volatilite, quantiles

        self.executer(f"Amplitude {symbole} ({intervalle}) h={horizon}", tache,
                      apres=self._afficher_bilan_amplitude, suivre=True)

    def _afficher_bilan_amplitude(self, resultat):
        """Résume les deux régressions en une phrase lisible."""
        if not resultat:
            return
        volatilite, quantiles = resultat
        mv = volatilite["metriques"]
        mq = quantiles["metriques"]

        couverture = mq["couverture"]
        attendue = mq["couverture_attendue"]
        honnete = abs(couverture - attendue) < 0.03

        bat_atr = mv["r2"] > mv["r2_naif"]
        verdict_volatilite = (
            f"R² {mv['r2']:.3f} sur un plafond théorique de {mv['r2_plafond']:.3f} "
            f"({mv['part_du_plafond']:.0%} de ce qui est atteignable) — "
            + ("mieux que l'ATR dilaté." if bat_atr else
               f"moins bien que l'ATR dilaté ({mv['r2_naif']:.3f}) : "
               f"à cet horizon, l'ATR seul suffit."))

        verdict_intervalle = (
            f"Intervalle Q10–Q90 : couverture {couverture:.1%} pour {attendue:.0%} "
            f"attendus — " + ("honnête." if honnete else
                              "trop étroit." if couverture < attendue else
                              "trop large, il surestime l'incertitude."))

        self.amp_resume.configure(
            text=(f"📐 Volatilité — {verdict_volatilite}\n{verdict_intervalle} "
                  f"Largeur moyenne {mq['largeur_moyenne']:.2f} % "
                  f"(écart-type {mq['largeur_ecart_type']:.2f} % : "
                  f"l'intervalle s'adapte bien au régime).\n"
                  f"👉 Lance maintenant « 🧮 Espérance de gain »."),
            text_color=COULEURS["vert"] if (bat_atr and honnete) else COULEURS["orange"])

    # ----------------------------------------------------------------------
    def _action_esperance(self):
        reglages = self._reglages_amplitude()
        if reglages is None:
            return
        symbole, intervalle, horizon, seuil, _ = reglages

        def apres(resultat):
            if resultat is None or resultat.empty:
                return
            self._afficher_esperance(resultat, symbole, intervalle, horizon, seuil)

        objectif = self._objectif()
        self.executer(
            f"Espérance {symbole} ({intervalle}) h={horizon}",
            lambda: amplitude.esperance(symbole, intervalle, horizon, seuil,
                                        tache=objectif.cle),
            apres=apres)

    def _afficher_esperance(self, resultat, symbole, intervalle, horizon, seuil):
        """Cartes de la dernière bougie + bilan économique sur le bloc test."""
        derniere = resultat.iloc[-1]
        sens = derniere["Sens_Predit"]
        couleur_sens = (COULEURS["vert"] if sens == "HAUSSE" else
                        COULEURS["rouge"] if sens == "BAISSE" else
                        COULEURS["texte_doux"])

        self.amp_carte_sens[1].configure(
            text=f"{sens} {float(derniere['Confiance']):.0%}", text_color=couleur_sens,
            font=ctk.CTkFont(size=18, weight="bold"))
        self.amp_carte_volatilite[1].configure(
            text=f"± {float(derniere['Volatilite_Prevue']):.2f} %")

        if {"Q10", "Q90"}.issubset(resultat.columns):
            self.amp_carte_intervalle[1].configure(
                text=f"{float(derniere['Q10']):+.2f} → {float(derniere['Q90']):+.2f} %",
                font=ctk.CTkFont(size=16, weight="bold"))

        esperance = float(derniere["Esperance_Nette"])
        self.amp_carte_esperance[1].configure(
            text=f"{esperance:+.3f} %",
            text_color=COULEURS["vert"] if esperance > 0 else COULEURS["rouge"])

        self.amp_resume.configure(text=self._bilan_economique(resultat, seuil),
                                  text_color=self._couleur_bilan(resultat))

    def _bilan_economique(self, resultat, seuil):
        """
        Ce que rapportent réellement les signaux retenus, sur le bloc test.

        On affiche le gain moyen par trade plutôt qu'un rendement cumulé :
        c'est la seule grandeur directement comparable au coût des frais, et
        elle ne se laisse pas gonfler par les intérêts composés.
        """
        test = resultat[(resultat["Bloc"] == "test") & resultat["Correct"].notna()]
        if test.empty:
            return "Aucune bougie du bloc test évaluable."

        retenues = test[test["Retenu"] == 1]
        cout = config.COUT_ALLER_RETOUR_PCT
        meilleure = float(test["Esperance_Nette"].max())

        if retenues.empty:
            return (f"⚠️ Aucun signal ne couvre les frais sur le bloc test. "
                    f"Meilleure espérance nette atteinte : {meilleure:+.3f} % "
                    f"(il en faudrait plus de 0).\n"
                    f"Ce n'est pas un défaut du modèle mais un constat économique : "
                    f"l'avantage existe, il est plus petit que le coût d'un "
                    f"aller-retour ({cout:.2f} %). L'amplitude croît en √horizon, "
                    f"les frais non — essaie un horizon plus long.")

        sens = np.where(retenues["Sens_Predit"] == "HAUSSE", 1.0, -1.0)
        gain = sens * retenues["Variation_Reelle"].to_numpy(dtype=float) - cout
        moyen = float(np.mean(gain))
        # Marge d'erreur de la moyenne : sans elle, un gain de +0.06 % sur
        # 200 trades passerait pour une preuve de rentabilité.
        marge = float(np.std(gain, ddof=1) / np.sqrt(len(gain))) if len(gain) > 1 else 0.0

        if moyen > 2 * marge:
            verdict, couleur = "rentable, et l'écart dépasse la marge d'erreur", "✅"
        elif moyen > 0:
            verdict, couleur = ("positif mais dans la marge d'erreur — "
                                "pas encore une preuve"), "⚠️"
        else:
            verdict, couleur = "perdant après frais", "❌"

        return (f"{couleur} Bloc test : {len(retenues):,} signaux retenus sur "
                f"{len(test):,} bougies ({len(retenues) / len(test):.1%} du temps), "
                f"justesse {retenues['Correct'].mean():.2%}.\n"
                f"Gain moyen par trade : {moyen:+.3f} % ± {marge:.3f} % "
                f"(frais de {cout:.2f} % déjà déduits) — {verdict}.\n"
                f"Le fichier « …_esperance » est prêt : va dans Backtest et coche "
                f"« Respecter Retenu » et « Taille proportionnelle ».")

    @staticmethod
    def _couleur_bilan(resultat):
        test = resultat[(resultat["Bloc"] == "test") & resultat["Correct"].notna()]
        if test.empty or test["Retenu"].sum() == 0:
            return COULEURS["orange"]
        return COULEURS["vert"]

    # ----------------------------------------------------------------------
    # Graphique de l'intervalle prévu
    # ----------------------------------------------------------------------
    def _action_graphe_intervalle(self):
        """
        Trace la fourchette Q10–Q90 convertie en PRIX autour du cours.

        Les quantiles sont des pourcentages de variation ; les reporter sur le
        prix est la seule façon de voir d'un coup d'œil si l'intervalle
        s'élargit vraiment quand le marché s'agite.
        """
        reglages = self._reglages_amplitude()
        if reglages is None:
            return
        symbole, intervalle, horizon, _, _ = reglages

        def calculer():
            quantiles = amplitude.predire(symbole, intervalle, horizon, "amplitude")
            analyse = stockage.lire_tableau(
                stockage.chemin_analyse(symbole, intervalle))
            recent = quantiles.tail(BOUGIES_AFFICHEES)
            prix = analyse.loc[recent.index, "Close"].astype(float)
            return recent, prix

        def apres(resultat):
            recent, prix = resultat
            bas = config.COLONNES_QUANTILES[config.QUANTILES[0]]
            milieu = config.COLONNES_QUANTILES[config.QUANTILES[1]]
            haut = config.COLONNES_QUANTILES[config.QUANTILES[-1]]

            figure = self._nouvelle_figure((11, 4.2))
            axe = figure.add_subplot(111)
            theme.styliser_axes(axe)

            # Chaque quantile est une variation en % : on la reporte sur le prix
            # de la bougie pour obtenir le niveau attendu dans `horizon` périodes.
            niveau_bas = prix * (1 + recent[bas] / 100)
            niveau_haut = prix * (1 + recent[haut] / 100)
            niveau_median = prix * (1 + recent[milieu] / 100)

            axe.fill_between(recent.index, niveau_bas, niveau_haut,
                             color=COULEURS["accent_clair"], alpha=0.18,
                             label="Intervalle 80 % (Q10–Q90)")
            axe.plot(prix.index, prix.values, color=COULEURS["texte"], lw=1.2,
                     label="Prix observé")
            axe.plot(recent.index, niveau_median, color=COULEURS["orange"], lw=1.0,
                     ls="--", label="Médiane prévue (Q50)")

            largeur = float((recent[haut] - recent[bas]).mean())
            axe.set_title(f"{symbole} ({intervalle}) — fourchette prévue à "
                          f"{horizon} période(s) · largeur moyenne {largeur:.2f} % "
                          f"· {len(recent)} dernières bougies")
            axe.set_ylabel("Prix ($)")
            theme.legende(axe, loc="upper left", fontsize=9)
            figure.tight_layout()
            self._afficher_figure("amplitude", figure)

        self.executer(f"Intervalle {symbole}", calculer, apres=apres)
