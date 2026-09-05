"""Page 1 — Extraction des données de marché."""

from __future__ import annotations

import customtkinter as ctk

from ... import config, exogene, extraction, indicateurs, modele
from ..textes import AIDES
from ..theme import COULEURS


class PageDonnees:
    """Téléchargement de l'historique OHLCV (Binance / Yahoo) et du Top CoinGecko."""

    def _page_donnees(self):
        page = self._nouvelle_page("Données")
        self._titre_page(page, "📥  Extraction des données",
                         "Télécharge l'historique des cours depuis Binance ou Yahoo Finance.")

        # --- Classement du marché ----------------------------------------
        corps = self._section(page, "Classement du marché (CoinGecko)")
        ligne = self._ligne(corps, espace=(0, 0))
        colonne, self.don_top_n = self._champ(ligne, "Top N cryptos", "5", 100,
                                              aide=AIDES["top_n"])
        colonne.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🌍 Voir le classement", command=self._action_classement
                      ).pack(side="left", pady=(18, 0))
        ctk.CTkButton(ligne, text="📦 Télécharger tout le Top N",
                      fg_color=COULEURS["bleu"], hover_color="#0097a7",
                      command=self._action_telecharger_top
                      ).pack(side="left", padx=10, pady=(18, 0))

        cadre, self.tab_classement = self._creer_tableau(corps, hauteur=6)
        cadre.pack(fill="x", pady=(12, 0))
        self.tab_classement.bind("<Double-1>", self._choisir_depuis_classement)

        # --- Téléchargement ciblé ----------------------------------------
        corps2 = self._section(page, "Historique d'une crypto")
        ligne2 = self._ligne(corps2, espace=(0, 0))

        col_sym, self.don_symbole = self._champ(ligne2, "Symbole", "BTC", 110,
                                                aide=AIDES["symbole"])
        col_sym.pack(side="left", padx=(0, 12))

        col_src = ctk.CTkFrame(ligne2, fg_color="transparent")
        self._etiquette(col_src, "Source", AIDES["source"])
        self.don_source = ctk.StringVar(value="Binance")
        ctk.CTkSegmentedButton(col_src, values=["Binance", "Yahoo"],
                               variable=self.don_source).pack(anchor="w", pady=(2, 0))
        col_src.pack(side="left", padx=(0, 12))

        col_int, self.don_intervalle, _ = self._menu(
            ligne2, "Intervalle", config.INTERVALLES, 100, aide=AIDES["intervalle"])
        col_int.pack(side="left", padx=(0, 12))

        col_deb, self.don_debut = self._champ(ligne2, "Début", "2022-01-01", 120,
                                              aide=AIDES["periode"])
        col_deb.pack(side="left", padx=(0, 12))
        col_fin, self.don_fin = self._champ(ligne2, "Fin", "2026-01-01", 120,
                                            aide=AIDES["periode"])
        col_fin.pack(side="left", padx=(0, 12))

        boutons = self._ligne(corps2, espace=(14, 0))
        ctk.CTkButton(boutons, text="⬇️  Télécharger", height=40,
                      command=self._action_telecharger).pack(side="left")

        bouton_exo = ctk.CTkButton(
            boutons, text="📡 Funding, basis & open interest", height=40,
            fg_color=COULEURS["bleu"], hover_color="#0097a7",
            command=self._action_exogene)
        bouton_exo.pack(side="left", padx=10)
        self._badge_info(boutons, AIDES["exogene"]).pack(side="left", padx=(0, 10),
                                                         pady=(10, 0))

        ctk.CTkButton(boutons, text="🚀  Tout enchaîner", height=40,
                      fg_color=COULEURS["orange"], hover_color="#d68910",
                      command=self._action_pipeline).pack(side="left")

        cadre2, self.tab_apercu = self._creer_tableau(corps2, hauteur=8)
        cadre2.pack(fill="x", pady=(14, 0))

    # ----------------------------------------------------------------------
    # Actions
    # ----------------------------------------------------------------------
    def _action_classement(self):
        n = self._lire_int(self.don_top_n, 5)

        def tache():
            classement = extraction.top_cryptos(n)
            extraction.sauvegarder_top(classement, n)
            return classement

        self.executer(f"Classement Top {n}", tache,
                      apres=lambda df: self._remplir_tableau(self.tab_classement, df))

    def _action_telecharger_top(self):
        parametres = self._parametres_telechargement(avec_symbole=False)
        if parametres is None:
            return
        n = self._lire_int(self.don_top_n, 5)

        self.executer(
            f"Téléchargement du Top {n}",
            lambda: extraction.telecharger_top_n(
                n, parametres["debut"], parametres["fin"],
                parametres["intervalle"], parametres["source"]),
            apres=lambda syms: self.log(
                f"📦 {len(syms or [])} cryptos prêtes — passe à l'onglet Analyse."))

    def _action_telecharger(self):
        parametres = self._parametres_telechargement()
        if parametres is None:
            return

        def tache():
            df = extraction.telecharger(
                parametres["symbole"], parametres["debut"], parametres["fin"],
                parametres["intervalle"], parametres["source"])
            extraction.sauvegarder(df, parametres["symbole"], parametres["intervalle"])
            return df

        def apres(df):
            if df is not None and not df.empty:
                self._remplir_tableau(self.tab_apercu, df.tail(10).reset_index())

        self.executer(f"Téléchargement {parametres['symbole']}", tache, apres=apres)

    def _action_exogene(self):
        """
        Télécharge le funding rate et l'open interest de la paire perpétuelle.

        Ces deux séries sont les seules données du projet qui ne soient pas
        dérivées du prix. Elles sont FUSIONNÉES avec ce qui a déjà été
        collecté : c'est ce qui permet à l'open interest, public sur 30 jours
        seulement, de devenir exploitable au fil des mises à jour.
        """
        parametres = self._parametres_telechargement()
        if parametres is None:
            return
        symbole, intervalle = parametres["symbole"], parametres["intervalle"]

        def apres(df):
            if df is None or df.empty:
                self.log(f"⚠️ Aucune donnée exogène pour {symbole} — la paire n'a "
                         f"peut-être pas de contrat perpétuel sur Binance.")
                return
            couverture = exogene.couverture(df)
            detail = " · ".join(f"{nom} {part:.0%}" for nom, part in couverture.items())
            self.log(f"📡 {symbole} — {len(df):,} lignes exogènes ({detail}). "
                     f"Relance l'Analyse pour que les colonnes soient prises en compte.")

        self.executer(
            f"Données exogènes {symbole} ({intervalle})",
            lambda: exogene.mettre_a_jour(symbole, intervalle,
                                          debut=parametres["debut"],
                                          fin=parametres["fin"]),
            apres=apres)

    def _action_pipeline(self):
        """Enchaîne les cinq étapes du projet pour une crypto donnée."""
        parametres = self._parametres_telechargement()
        if parametres is None:
            return
        symbole, intervalle = parametres["symbole"], parametres["intervalle"]
        horizon = config.HORIZON_DEFAUT

        def tache():
            df = extraction.telecharger(symbole, parametres["debut"], parametres["fin"],
                                        intervalle, parametres["source"])
            extraction.sauvegarder(df, symbole, intervalle)

            # Les données exogènes sont un bonus : leur absence (paire sans
            # contrat perpétuel, réseau indisponible) ne doit pas casser la chaîne.
            try:
                exogene.mettre_a_jour(symbole, intervalle,
                                      debut=parametres["debut"], fin=parametres["fin"])
            except Exception as err:                   # noqa: BLE001
                print(f"⚠️ Données exogènes ignorées : {err}")

            indicateurs.analyser_fichier(symbole, intervalle)
            modele.entrainer(symbole, intervalle, horizon, config.MODELE_DEFAUT)
            modele.predire(symbole, intervalle, horizon, config.SEUIL_DEFAUT)
            return True

        self.executer(f"Chaîne complète {symbole} ({intervalle})", tache,
                      apres=lambda _: self.log(
                          "🎉 Terminé — les résultats sont visibles dans "
                          "Évaluation, Visualisation et Backtest."))

    # ----------------------------------------------------------------------
    # Outils
    # ----------------------------------------------------------------------
    def _parametres_telechargement(self, avec_symbole=True):
        """Lit et valide les champs communs aux téléchargements. None si invalide."""
        debut = self._valider_date(self.don_debut.get())
        fin = self._valider_date(self.don_fin.get())
        if debut is None or fin is None:
            self.log("❌ Date invalide — format attendu : AAAA-MM-JJ (ex : 2024-01-01).")
            return None

        symbole = self.don_symbole.get().strip().upper()
        if avec_symbole and not symbole:
            self.log("❌ Renseigne un symbole (BTC, ETH…).")
            return None

        return {"symbole": symbole, "debut": debut, "fin": fin,
                "intervalle": self.don_intervalle.get(),
                "source": self.don_source.get()}

    def _choisir_depuis_classement(self, _event):
        """Double-clic dans le classement : reporte le symbole dans le formulaire."""
        selection = self.tab_classement.selection()
        if not selection:
            return
        colonnes = list(self.tab_classement["columns"])
        if "symbol" not in colonnes:
            return
        symbole = self.tab_classement.item(selection[0], "values")[colonnes.index("symbol")]
        self.don_symbole.delete(0, "end")
        self.don_symbole.insert(0, symbole)
        self.log(f"➡️  Symbole sélectionné : {symbole}")
