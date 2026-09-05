"""Page 2 — Calcul des 8 indicateurs et des 24 variations."""

from __future__ import annotations

import customtkinter as ctk

from ... import config, indicateurs, stockage
from ..textes import AIDES, EXPLICATION_INDICATEURS
from ..theme import COULEURS


class PageAnalyse:
    """Transforme un fichier OHLCV brut en jeu de données prêt pour le modèle."""

    def _page_analyse(self):
        page = self._nouvelle_page("Analyse")
        self._titre_page(
            page, "🔬  Analyse",
            f"{len(config.INDICATEURS)} indicateurs + {len(config.COLONNES_VARIATION)} "
            f"colonnes variation_x. Rien de plus : trop de données tue la donnée.")

        corps = self._section(page, "Fichier à analyser", aide=AIDES["analyse"])
        ligne = self._ligne(corps, espace=(0, 0))

        colonne, self.ana_fichier, self.ana_menu = self._menu(
            ligne, "Données brutes", ["(aucun)"], 200)
        colonne.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Analyse")
                      ).pack(side="left", pady=(18, 0))

        self.ana_contexte = ctk.CTkCheckBox(
            ligne, text="Contexte multi-timeframe et exogène")
        self.ana_contexte.select()
        self.ana_contexte.pack(side="left", padx=(0, 6), pady=(18, 0))
        self._badge_info(ligne, AIDES["contexte"]).pack(side="left", pady=(18, 0))

        boutons = self._ligne(corps, espace=(14, 0))
        ctk.CTkButton(boutons, text="🔬 Analyser ce fichier", height=40,
                      command=self._action_analyser).pack(side="left")
        ctk.CTkButton(boutons, text="📦 Analyser tout le dossier", height=40,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._action_analyser_tout).pack(side="left", padx=10)
        self._bouton_explication(
            boutons, "❔ Que contient le fichier analysé ?",
            "Les 8 indicateurs et les 24 variations",
            EXPLICATION_INDICATEURS).pack(side="left", pady=(4, 0))

        # --- Aperçu des indicateurs retenus -------------------------------
        corps2 = self._section(page, "Les 8 indicateurs retenus")
        for nom in config.INDICATEURS:
            ligne_indic = ctk.CTkFrame(corps2, fg_color=COULEURS["carte"], corner_radius=8)
            ligne_indic.pack(fill="x", pady=2)
            ctk.CTkLabel(ligne_indic, text=nom, width=140, anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(side="left", padx=(14, 8), pady=7)
            ctk.CTkLabel(ligne_indic, text=config.LIBELLES_INDICATEURS[nom],
                         text_color=COULEURS["texte_doux"], anchor="w",
                         font=ctk.CTkFont(size=12)).pack(side="left", pady=7)

        # --- Résultat ------------------------------------------------------
        corps3 = self._section(page, "Résultat")
        cartes = self._ligne(corps3, espace=(0, 0))
        self.ana_carte_lignes = self._carte_metrique(cartes, "Lignes")
        self.ana_carte_colonnes = self._carte_metrique(cartes, "Colonnes")
        self.ana_carte_periode = self._carte_metrique(cartes, "Période couverte")
        for carte in (self.ana_carte_lignes, self.ana_carte_colonnes, self.ana_carte_periode):
            carte[0].pack(side="left", padx=(0, 10), fill="x", expand=True)

        cadre, self.tab_analyse = self._creer_tableau(corps3, hauteur=8)
        cadre.pack(fill="x", pady=(14, 0))

        self.rafraichisseurs["Analyse"] = lambda: self._maj_menu(
            self.ana_menu, self.ana_fichier, stockage.lister_donnees_brutes())

    # ----------------------------------------------------------------------
    def _action_analyser(self):
        cle = self.ana_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucun fichier sélectionné.")
            return
        symbole, intervalle = stockage.separer_cle(cle)

        contexte = self.ana_contexte.get() == 1

        def apres(df):
            if df is None or df.empty:
                return
            self.ana_carte_lignes[1].configure(text=f"{len(df):,}")
            self.ana_carte_colonnes[1].configure(text=str(df.shape[1]))
            self.ana_carte_periode[1].configure(
                text=f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}",
                font=ctk.CTkFont(size=14, weight="bold"))
            apercu = df[config.COLONNES_PRIX[3:] + config.INDICATEURS].tail(10)
            self._remplir_tableau(self.tab_analyse, apercu.round(4).reset_index())
            self._resumer_contexte(df)

        self.executer(f"Analyse {symbole} ({intervalle})",
                      lambda: indicateurs.analyser_fichier(symbole, intervalle,
                                                           contexte),
                      apres=apres)

    # Les cinq familles de contexte, dans l'ordre d'apparition du fichier.
    FAMILLES_CONTEXTE = (
        ("multi-timeframe", "INDICATEURS_MTF"),
        ("order flow", "COLONNES_FLUX"),
        ("temps", "COLONNES_TEMPS"),
        ("régime", "COLONNES_REGIME"),
        ("exogènes", "COLONNES_EXOGENES"),
    )

    def _resumer_contexte(self, df):
        """Dit lesquelles des colonnes optionnelles ont réellement été créées."""
        presentes = [c for c in config.COLONNES_CONTEXTE if c in df.columns]
        if not presentes:
            self.log("ℹ️ Aucune colonne de contexte — ni intervalle supérieur "
                     "exploitable, ni données exogènes téléchargées.")
            return

        detail = []
        for libelle, attribut in self.FAMILLES_CONTEXTE:
            famille = getattr(config, attribut)
            trouvees = [c for c in presentes if c in famille]
            if trouvees:
                detail.append(f"{len(trouvees)} {libelle}")

        # Ce qui manque compte autant que ce qui est là : une colonne écartée
        # se rattrape (téléchargement exogène, ou order flow à re-télécharger),
        # encore faut-il savoir laquelle.
        absentes = [c for c in config.COLONNES_CONTEXTE
                    if c not in presentes and c not in config.INDICATEURS_MTF]

        message = f"🔭 Contexte ajouté : {len(presentes)} colonnes — {', '.join(detail)}."
        if absentes:
            message += (f" Absentes : {', '.join(absentes)}. L'order flow revient "
                        f"avec un nouveau téléchargement ; le funding et le basis "
                        f"avec le bouton 📡 ; l'open interest s'accumule sur "
                        f"plusieurs mois.")
        self.log(message)

    def _action_analyser_tout(self):
        contexte = self.ana_contexte.get() == 1
        self.executer("Analyse de tout le dossier",
                      lambda: indicateurs.analyser_tout(contexte),
                      apres=lambda cles: self.log(
                          f"📦 {len(cles or [])} fichier(s) analysé(s) — "
                          f"passe à l'onglet Prédiction."))
