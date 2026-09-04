"""Page 4 — Évaluation détaillée d'un modèle entraîné."""

from __future__ import annotations

import json
import os

import customtkinter as ctk

from ... import stockage
from ..textes import STATS, commenter
from ..theme import COULEURS


class PageEvaluation:
    """Affiche les métadonnées d'un modèle : configuration, scores, seuils."""

    def _page_evaluation(self):
        page = self._nouvelle_page("Évaluation")
        self._titre_page(page, "📋  Évaluation",
                         "Tout ce qu'a donné l'entraînement, expliqué et commenté.")

        corps = self._section(page, "Modèle à analyser")
        ligne = self._ligne(corps, espace=(0, 0))
        colonne, self.eval_fichier, self.eval_menu = self._menu(
            ligne, "Modèle entraîné", ["(aucun)"], 220)
        colonne.pack(side="left", padx=(0, 12))
        ctk.CTkButton(ligne, text="🔄", width=40,
                      command=lambda: self.afficher_page("Évaluation")
                      ).pack(side="left", pady=(18, 0))
        ctk.CTkButton(ligne, text="📋 Analyser", height=40,
                      command=self._action_evaluer
                      ).pack(side="left", padx=10, pady=(14, 0))

        self.eval_contenu = ctk.CTkFrame(page, fg_color="transparent")
        self.eval_contenu.pack(fill="both", expand=True, pady=(4, 0))
        self._message_evaluation("Sélectionne un modèle puis clique sur « Analyser ».")

        self.rafraichisseurs["Évaluation"] = lambda: self._maj_menu(
            self.eval_menu, self.eval_fichier, stockage.lister_modeles())

    # ----------------------------------------------------------------------
    def _message_evaluation(self, texte):
        for enfant in self.eval_contenu.winfo_children():
            enfant.destroy()
        ctk.CTkLabel(self.eval_contenu, text=texte,
                     text_color=COULEURS["texte_doux"]).pack(pady=30)

    def _action_evaluer(self):
        cle = self.eval_fichier.get()
        if cle in ("(aucun)", ""):
            self.log("❌ Aucun modèle sélectionné.")
            return
        symbole, intervalle, horizon = stockage.separer_cle_modele(cle)

        def charger():
            chemin = stockage.chemin_meta(symbole, intervalle, horizon)
            if not os.path.exists(chemin):
                raise FileNotFoundError(
                    f"Métadonnées absentes ({os.path.basename(chemin)}). "
                    f"Réentraîne ce modèle.")
            with open(chemin, encoding="utf-8") as fichier:
                return json.load(fichier)

        self.executer(f"Évaluation {cle}", charger, apres=self._afficher_evaluation)

    # ----------------------------------------------------------------------
    def _afficher_evaluation(self, meta):
        if not meta:
            self._message_evaluation("Aucune donnée à afficher.")
            return

        for enfant in self.eval_contenu.winfo_children():
            enfant.destroy()

        metriques = meta.get("metriques", {})
        rapport = metriques.get("rapport", {})
        hausse = rapport.get("Hausse", {})
        baisse = rapport.get("Baisse", {})

        self._bloc_identite(meta)
        self._bloc_performance(meta, metriques, hausse, baisse)
        self._bloc_seuils(metriques)
        self._bloc_configuration(meta)

        self.log(f"📋 Évaluation affichée : {meta.get('symbole')} "
                 f"({meta.get('intervalle')}, horizon {meta.get('horizon')}).")

    # -- blocs --------------------------------------------------------------
    def _bloc_identite(self, meta):
        section = self._section(self.eval_contenu, "🪪  Identité")
        self._ligne_stat(section, "Crypto", meta.get("symbole", "?"))
        self._ligne_stat(section, "Intervalle", meta.get("intervalle", "?"))
        self._ligne_stat(section, "Modèle", meta.get("modele", "?"))
        self._ligne_stat(section, "Horizon prédit",
                         f"{meta.get('horizon', '?')} période(s)")
        self._ligne_stat(section, "Date d'entraînement", meta.get("date_entrainement", "?"))
        self._ligne_stat(
            section, "Période des données",
            f"{str(meta.get('periode_debut', '?'))[:10]} → "
            f"{str(meta.get('periode_fin', '?'))[:10]}")
        self._ligne_stat(
            section, "Lignes (total / train / validation / test)",
            f"{meta.get('n_total', 0):,} / {meta.get('n_train', 0):,} / "
            f"{meta.get('n_validation', 0):,} / {meta.get('n_test', 0):,}")

    def _bloc_performance(self, meta, metriques, hausse, baisse):
        section = self._section(self.eval_contenu,
                                "🎯  Performance sur le test (données jamais vues)")
        self._stat(section, "accuracy", metriques.get("accuracy"), pourcentage=True)
        self._stat(section, "auc", metriques.get("auc"))
        self._stat(section, "auc_validation", meta.get("auc_validation"))

        # Le verdict qui compte : bat-on la réponse constante ?
        accuracy = metriques.get("accuracy")
        baseline = metriques.get("baseline_majoritaire")
        if accuracy is not None and baseline is not None:
            gagne = accuracy > baseline
            self._ligne_stat(
                section, STATS["baseline_majoritaire"]["label"],
                f"{baseline:.3f}  ({baseline * 100:.1f} %)",
                STATS["baseline_majoritaire"]["aide"],
                commentaire=("Le modèle fait mieux que la réponse constante."
                             if gagne else
                             "Le modèle ne bat pas la réponse constante : son seul "
                             "intérêt réside alors dans le filtrage par confiance."),
                niveau="bon" if gagne else "mauvais")

        self._stat(section, "part_hausse", meta.get("part_hausse"), pourcentage=True)
        self._stat(section, "precision_hausse", hausse.get("precision"), pourcentage=True)
        self._stat(section, "rappel_hausse", hausse.get("recall"), pourcentage=True)
        self._stat(section, "precision_baisse", baisse.get("precision"), pourcentage=True)
        self._stat(section, "rappel_baisse", baisse.get("recall"), pourcentage=True)

    def _bloc_seuils(self, metriques):
        """
        Tableau central : ce que gagne le filtrage par seuil de confiance.

        C'est lui qui répond à « à partir de quelle confiance mes signaux
        deviennent-ils fiables, et combien m'en reste-t-il ? ».
        """
        table = metriques.get("table_confiance", [])
        if not table:
            return

        section = self._section(self.eval_contenu, "🔎  Précision selon le seuil de confiance")
        ctk.CTkLabel(
            section,
            text=("Ne garder que les prédictions les plus sûres — dans les deux sens. "
                  "Si la précision monte avec le seuil, le score de confiance est fiable."),
            text_color=COULEURS["texte_doux"], font=ctk.CTkFont(size=12),
            wraplength=880, justify="left").pack(anchor="w", pady=(0, 10))

        entete = ctk.CTkFrame(section, fg_color=COULEURS["accent"], corner_radius=6)
        entete.pack(fill="x", pady=(0, 4))
        for texte, largeur in (("Seuil", 90), ("Retenues", 120), ("Couverture", 120),
                               ("Précision", 120), ("Dont ▲ hausse", 140)):
            ctk.CTkLabel(entete, text=texte, width=largeur, anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(side="left", padx=6, pady=6)

        reference = next((l["precision"] for l in table if l["seuil"] == 0.50), None)
        for ligne in table:
            if ligne["precision"] is None:
                continue
            self._ligne_seuil(section, ligne, reference)

    def _ligne_seuil(self, parent, ligne, reference):
        """Une ligne du tableau des seuils, colorée selon le gain apporté."""
        cadre = ctk.CTkFrame(parent, fg_color=COULEURS["carte"], corner_radius=6)
        cadre.pack(fill="x", pady=2)

        if reference is None or ligne["n"] < 30:
            couleur = COULEURS["texte_doux"]
        elif ligne["precision"] > reference + 0.01:
            couleur = COULEURS["vert"]
        elif ligne["precision"] < reference - 0.01:
            couleur = COULEURS["rouge"]
        else:
            couleur = COULEURS["texte"]

        valeurs = (
            (f"{ligne['seuil']:.0%}", 90, COULEURS["texte"]),
            (f"{ligne['n']:,}", 120, COULEURS["texte"]),
            (f"{ligne['couverture']:.1%}", 120, COULEURS["texte_doux"]),
            (f"{ligne['precision']:.2%}", 120, couleur),
            (f"{ligne['part_hausse']:.0%}" if ligne["part_hausse"] is not None else "—",
             140, COULEURS["texte_doux"]),
        )
        for texte, largeur, teinte in valeurs:
            ctk.CTkLabel(cadre, text=texte, width=largeur, anchor="w",
                         text_color=teinte, font=ctk.CTkFont(size=12)
                         ).pack(side="left", padx=6, pady=6)

        if ligne["n"] < 30:
            ctk.CTkLabel(cadre, text="échantillon trop petit pour conclure",
                         text_color=COULEURS["texte_doux"],
                         font=ctk.CTkFont(size=11)).pack(side="left", padx=10)

    def _bloc_configuration(self, meta):
        section = self._section(self.eval_contenu, "⚙️  Réglages appliqués automatiquement")
        self._stat(section, "n_arbres", meta.get("n_arbres"))
        self._stat(section, "embargo", meta.get("embargo"))
        self._ligne_stat(section, "Calibration des probabilités",
                         meta.get("calibration", "—"))
        self._ligne_stat(section, "Rééquilibrage des classes",
                         "Oui" if meta.get("reequilibrage") else
                         "Non (classes déjà équilibrées)")
        self._ligne_stat(section, "Indicateurs utilisés",
                         f"{len(meta.get('features', []))} — "
                         f"{', '.join(meta.get('features', []))}")

        hyperparametres = meta.get("hyperparametres", {})
        if hyperparametres:
            self._ligne_stat(
                section, "Hyperparamètres retenus",
                ", ".join(f"{cle}={valeur}" for cle, valeur in hyperparametres.items()))

        machine = meta.get("ressources", {})
        if machine:
            self._ligne_stat(
                section, "Ressources exploitées",
                f"{machine.get('coeurs', '?')} cœurs · budget "
                f"{machine.get('budget_go', '?')} Go · max_bin {machine.get('max_bin', '?')}")

    # -- utilitaire ---------------------------------------------------------
    def _stat(self, parent, cle, valeur, pourcentage=False):
        """Ligne de statistique documentée et commentée selon sa valeur."""
        infos = STATS.get(cle, {"label": cle, "aide": ""})
        if valeur is None:
            self._ligne_stat(parent, infos["label"], "—", infos["aide"])
            return

        if pourcentage:
            affichage = f"{float(valeur):.3f}  ({float(valeur) * 100:.1f} %)"
        elif isinstance(valeur, float):
            affichage = f"{valeur:.4f}"
        else:
            affichage = str(valeur)

        commentaire, niveau = commenter(cle, valeur)
        self._ligne_stat(parent, infos["label"], affichage, infos["aide"],
                         commentaire, niveau)
