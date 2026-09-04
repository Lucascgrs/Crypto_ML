"""
Page 3 — Modèle et prédiction.

Volontairement dépouillée : quatre réglages seulement (crypto, modèle,
horizon, seuil de confiance). Tout le reste — découpage, embargo, équilibrage
des classes, hyperparamètres, calibration, exploitation de la RAM — est décidé
automatiquement et détaillé dans la fenêtre « Ce qui est fait automatiquement ».
"""

from __future__ import annotations

import customtkinter as ctk

from ... import config, modele, stockage
from ..textes import AIDES, EXPLICATION_ENTRAINEMENT
from ..theme import COULEURS


class PageModele:
    """Entraînement et génération des signaux, dans les deux sens."""

    # Écart de justesse en dessous duquel le filtrage par confiance ne peut pas
    # être déclaré utile : sur quelques centaines de signaux, un demi-point de
    # différence tient dans le bruit d'échantillonnage.
    GAIN_SIGNIFICATIF = 0.005

    def _page_modele(self):
        page = self._nouvelle_page("Modèle")
        self._titre_page(
            page, "🧠  Modèle & Prédiction",
            "« Dans X périodes, le prix sera-t-il plus haut ou plus bas ? » "
            "— avec un niveau de confiance.")

        # ------------------------------------------------------------------
        # Les quatre seuls réglages
        # ------------------------------------------------------------------
        corps = self._section(page, "Réglages")

        ligne = self._ligne(corps, espace=(0, 0))
        colonne, self.mod_fichier, self.mod_menu = self._menu(
            ligne, "Crypto analysée", ["(aucun)"], 190, aide=AIDES["crypto_modele"])
        colonne.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Modèle")
                      ).pack(side="left", padx=(0, 20), pady=(18, 0))

        modeles = modele.modeles_disponibles()
        colonne_modele, self.mod_type, self.mod_type_menu = self._menu(
            ligne, "Modèle", modeles, 190, aide=AIDES["type_modele"])
        self.mod_type_menu.configure(command=lambda _v: self._maj_description_modele())
        colonne_modele.pack(side="left", padx=(0, 12))

        ligne2 = self._ligne(corps, espace=(16, 0))
        colonne_horizon, self.mod_horizon = self._curseur(
            ligne2, "Horizon (périodes)", 1, config.HORIZON_MAX,
            config.HORIZON_DEFAUT, aide=AIDES["horizon"], largeur=280)
        colonne_horizon.pack(side="left", padx=(0, 30))

        colonne_seuil, self.mod_seuil = self._curseur(
            ligne2, "Seuil de confiance", 0.50, 0.95, config.SEUIL_DEFAUT,
            aide=AIDES["seuil"], largeur=240, format_valeur="{:.0%}", nb_pas=45)
        colonne_seuil.pack(side="left")

        # Description du modèle choisi, mise à jour à chaque changement.
        self.mod_description = ctk.CTkLabel(
            corps, text="", font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], wraplength=900, anchor="w")
        self.mod_description.pack(fill="x", padx=2, pady=(14, 0))
        self._maj_description_modele()

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------
        corps2 = self._section(page, "Actions")
        boutons = self._ligne(corps2, espace=(0, 0))

        ctk.CTkButton(boutons, text="🔥 Entraîner", height=42, width=150,
                      command=self._action_entrainer).pack(side="left")
        ctk.CTkButton(boutons, text="🔮 Prédire", height=42, width=150,
                      fg_color=COULEURS["vert"], hover_color="#27ae60",
                      command=self._action_predire).pack(side="left", padx=10)
        self._bouton_explication(
            boutons, "❔ Ce qui est fait automatiquement",
            "Ce que l'entraînement décide tout seul",
            EXPLICATION_ENTRAINEMENT).pack(side="left", padx=(10, 0), pady=(5, 0))

        # Réglage avancé, discret : appliquer le modèle d'une autre crypto.
        ligne3 = self._ligne(corps2, espace=(14, 0))
        colonne_source, self.mod_source, self.mod_source_menu = self._menu(
            ligne3, "Prédire avec le modèle de", ["(cette crypto)"], 220,
            aide=AIDES["modele_utilise"])
        colonne_source.pack(side="left")

        # ------------------------------------------------------------------
        # Résultats
        # ------------------------------------------------------------------
        corps3 = self._section(page, "Dernier signal")
        cartes = self._ligne(corps3, espace=(0, 0))
        self.mod_carte_sens = self._carte_metrique(cartes, "Sens prédit")
        self.mod_carte_confiance = self._carte_metrique(cartes, "Confiance")
        self.mod_carte_prix = self._carte_metrique(cartes, "Dernier prix")
        self.mod_carte_retenues = self._carte_metrique(cartes, "Signaux retenus (test)")
        for carte in (self.mod_carte_sens, self.mod_carte_confiance,
                      self.mod_carte_prix, self.mod_carte_retenues):
            carte[0].pack(side="left", padx=(0, 10), fill="x", expand=True)

        self.mod_resume = ctk.CTkLabel(
            corps3, text="Entraîne un modèle puis lance une prédiction.",
            font=ctk.CTkFont(size=12), justify="left",
            text_color=COULEURS["texte_doux"], wraplength=900, anchor="w")
        self.mod_resume.pack(fill="x", pady=(14, 0))

        self.rafraichisseurs["Modèle"] = self._rafraichir_modele

    # ----------------------------------------------------------------------
    # Rafraîchissement
    # ----------------------------------------------------------------------
    def _rafraichir_modele(self):
        """Recharge les listes de cryptos analysées et de modèles entraînés."""
        self._maj_menu(self.mod_menu, self.mod_fichier, stockage.lister_analyses())
        cryptos = sorted({stockage.separer_cle_modele(c)[0]
                          for c in stockage.lister_modeles()})
        self._maj_menu(self.mod_source_menu, self.mod_source,
                       ["(cette crypto)"] + cryptos)

    def _maj_description_modele(self):
        nom = self.mod_type.get()
        self.mod_description.configure(
            text=f"🤖 {nom} — {modele.DESCRIPTIONS_MODELES.get(nom, '')}")

    # ----------------------------------------------------------------------
    # Lecture des réglages
    # ----------------------------------------------------------------------
    def _reglages(self):
        """(symbole, intervalle, horizon, seuil) ou None si rien n'est sélectionné."""
        cle = self.mod_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucune crypto analysée. Lance d'abord l'étape Analyse.")
            return None
        symbole, intervalle = stockage.separer_cle(cle)
        return symbole, intervalle, self.lire_horizon(), self.lire_seuil()

    # ----------------------------------------------------------------------
    # Actions
    # ----------------------------------------------------------------------
    def _action_entrainer(self):
        reglages = self._reglages()
        if reglages is None:
            return
        symbole, intervalle, horizon, _ = reglages
        nom_modele = self.mod_type.get()

        def apres(meta):
            if not meta:
                return
            metriques = meta["metriques"]
            self.mod_resume.configure(
                text=(f"✅ {nom_modele} entraîné sur {symbole} ({intervalle}), "
                      f"horizon {horizon}.\n"
                      f"Test : justesse {metriques['accuracy']:.1%} · "
                      f"AUC {metriques['auc']:.3f} · "
                      f"baseline {metriques['baseline_majoritaire']:.1%}. "
                      f"Détail complet dans l'onglet Évaluation."),
                text_color=COULEURS["texte"])
            self._rafraichir_modele()

        self.executer(f"Entraînement {symbole} ({intervalle}) h={horizon}",
                      lambda: modele.entrainer(symbole, intervalle, horizon, nom_modele),
                      apres=apres)

    def _action_predire(self):
        reglages = self._reglages()
        if reglages is None:
            return
        symbole, intervalle, horizon, seuil = reglages

        source = self.mod_source.get()
        symbole_modele = None if source in ("(cette crypto)", "") else source

        def apres(resultats):
            if resultats is None or resultats.empty:
                return
            self._afficher_dernier_signal(resultats, seuil)

        self.executer(
            f"Prédiction {symbole} ({intervalle}) h={horizon}",
            lambda: modele.predire(symbole, intervalle, horizon, seuil, symbole_modele),
            apres=apres)

    # ----------------------------------------------------------------------
    def _afficher_dernier_signal(self, resultats, seuil):
        """Met à jour les cartes avec la dernière bougie et le bilan sur le test."""
        derniere = resultats.iloc[-1]
        sens = derniere["Sens_Predit"]
        confiance = float(derniere["Confiance"])

        if confiance < seuil:
            texte, couleur = f"{sens} (sous le seuil)", COULEURS["texte_doux"]
        elif sens == "HAUSSE":
            texte, couleur = "HAUSSE ▲", COULEURS["vert"]
        else:
            texte, couleur = "BAISSE ▼", COULEURS["rouge"]

        self.mod_carte_sens[1].configure(text=texte, text_color=couleur)
        self.mod_carte_confiance[1].configure(text=f"{confiance:.1%}")
        self.mod_carte_prix[1].configure(text=f"{float(derniere['Prix']):,.2f} $")

        # Bilan sur le bloc test : les seules données jamais vues à l'entraînement.
        test = resultats[(resultats["Bloc"] == "test") & resultats["Correct"].notna()]
        retenues = test[test["Retenu"] == 1]
        self.mod_carte_retenues[1].configure(text=f"{len(retenues):,}")

        if retenues.empty:
            self.mod_resume.configure(
                text=(f"Aucune bougie du bloc test n'atteint {seuil:.0%} de confiance. "
                      f"Baisse le seuil ou change d'horizon."),
                text_color=COULEURS["orange"])
            return

        hausses = int((retenues["Sens_Predit"] == "HAUSSE").sum())
        justesse = retenues["Correct"].mean()
        reference = test["Correct"].mean()

        # Un écart inférieur à un demi-point n'est pas un gain : sur quelques
        # centaines de signaux, il tient dans la marge d'erreur statistique.
        ecart = justesse - reference
        if ecart >= self.GAIN_SIGNIFICATIF:
            verdict, couleur_resume = "apporte un gain", COULEURS["vert"]
        elif ecart <= -self.GAIN_SIGNIFICATIF:
            verdict, couleur_resume = "dégrade le résultat", COULEURS["rouge"]
        else:
            verdict, couleur_resume = "ne change rien de net", COULEURS["orange"]

        self.mod_resume.configure(
            text=(f"Sur le bloc test ({len(test):,} bougies jamais vues) : "
                  f"{len(retenues):,} signaux retenus au seuil de {seuil:.0%} "
                  f"({len(retenues) / len(test):.1%} du temps), "
                  f"dont {hausses:,} ▲ et {len(retenues) - hausses:,} ▼.\n"
                  f"Justesse des signaux retenus : {justesse:.2%} "
                  f"— contre {reference:.2%} sans filtrage. "
                  f"Le seuil de confiance {verdict} ici."),
            text_color=couleur_resume)
