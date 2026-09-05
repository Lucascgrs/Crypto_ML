"""
Page 5 — Évaluation détaillée d'un modèle entraîné.

Deux familles de modèles y cohabitent, avec des métriques qui n'ont rien à voir :

  * **classification** (direction, direction nette, triple barrière, amplitude)
    — justesse, AUC, table des seuils de confiance ;
  * **régression** (volatilité, quantiles) — R² face à son plafond théorique,
    couverture de l'intervalle, perte pinball.

Le préfixe du menu indique de laquelle il s'agit.
"""

from __future__ import annotations

import json
import os

import customtkinter as ctk

from ... import cibles, stockage
from ..textes import AIDES, STATS, commenter
from ..theme import COULEURS

# Préfixes du menu déroulant : ils disent d'un coup d'œil quel type de modèle
# on regarde, et évitent toute collision de nom entre les deux familles.
PREFIXE_CLASSIFICATION = "🧠 "
PREFIXE_REGRESSION = "📐 "


class PageEvaluation:
    """Affiche les métadonnées d'un modèle : configuration, scores, seuils."""

    def _page_evaluation(self):
        page = self._nouvelle_page("Évaluation")
        self._titre_page(page, "📋  Évaluation",
                         "Tout ce qu'a donné l'entraînement, expliqué et commenté.")

        corps = self._section(page, "Modèle à analyser")
        ligne = self._ligne(corps, espace=(0, 0))
        colonne, self.eval_fichier, self.eval_menu = self._menu(
            ligne, "Modèle entraîné", ["(aucun)"], 300)
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

        self.rafraichisseurs["Évaluation"] = self._rafraichir_evaluation

    # ----------------------------------------------------------------------
    def _rafraichir_evaluation(self):
        """Liste les deux familles de modèles, préfixées pour être distinguables."""
        entrees = ([PREFIXE_CLASSIFICATION + cle for cle in stockage.lister_modeles()]
                   + [PREFIXE_REGRESSION + cle for cle in stockage.lister_regressions()])
        self._maj_menu(self.eval_menu, self.eval_fichier, entrees)

    def _message_evaluation(self, texte):
        for enfant in self.eval_contenu.winfo_children():
            enfant.destroy()
        ctk.CTkLabel(self.eval_contenu, text=texte,
                     text_color=COULEURS["texte_doux"]).pack(pady=30)

    def _action_evaluer(self):
        entree = self.eval_fichier.get()
        if entree in ("(aucun)", ""):
            self.log("❌ Aucun modèle sélectionné.")
            return

        regression = entree.startswith(PREFIXE_REGRESSION)
        cle = entree[len(PREFIXE_REGRESSION if regression else PREFIXE_CLASSIFICATION):]

        def charger():
            chemin = (self._chemin_meta_regression(cle) if regression
                      else self._chemin_meta_classification(cle))
            if not os.path.exists(chemin):
                raise FileNotFoundError(
                    f"Métadonnées absentes ({os.path.basename(chemin)}). "
                    f"Réentraîne ce modèle.")
            with open(chemin, encoding="utf-8") as fichier:
                return json.load(fichier)

        self.executer(f"Évaluation {cle}", charger, apres=self._afficher_evaluation)

    @staticmethod
    def _chemin_meta_classification(cle):
        infos = stockage.analyser_cle(cle)
        return stockage.chemin_meta(infos.symbole, infos.intervalle,
                                    infos.horizon, infos.tache)

    @staticmethod
    def _chemin_meta_regression(cle):
        """'BTC_1h_h3_volatilite' -> chemin du META correspondant."""
        base, cible = cle.rsplit("_", 1)
        infos = stockage.analyser_cle(base)
        return stockage.chemin_meta_regression(infos.symbole, infos.intervalle,
                                               infos.horizon, cible)

    # ----------------------------------------------------------------------
    def _afficher_evaluation(self, meta):
        if not meta:
            self._message_evaluation("Aucune donnée à afficher.")
            return

        for enfant in self.eval_contenu.winfo_children():
            enfant.destroy()

        self._bloc_identite(meta)
        if meta.get("cible") in ("amplitude", "volatilite"):
            self._bloc_regression(meta)
        else:
            self._bloc_classification(meta)
        self._bloc_configuration(meta)
        self._bloc_utilite(meta)

        self.log(f"📋 Évaluation affichée : {meta.get('symbole')} "
                 f"({meta.get('intervalle')}, horizon {meta.get('horizon')}).")

    def _bloc_classification(self, meta):
        metriques = meta.get("metriques", {})
        objectif = cibles.obtenir(meta.get("tache"))
        rapport = metriques.get("rapport", {})
        self._bloc_performance(meta, metriques, objectif, rapport)
        self._bloc_regimes(metriques)
        self._bloc_seuils(metriques, objectif)

    # -- blocs --------------------------------------------------------------
    def _bloc_identite(self, meta):
        section = self._section(self.eval_contenu, "🪪  Identité")
        membres = meta.get("panier") or []
        if membres:
            self._ligne_stat(
                section, "Panier", f"{len(membres)} cryptos",
                AIDES["panier"],
                commentaire=", ".join(membres), niveau="bon")
        else:
            self._ligne_stat(section, "Crypto", meta.get("symbole", "?"))
        self._ligne_stat(section, "Intervalle", meta.get("intervalle", "?"))
        self._ligne_stat(section, "Modèle", meta.get("modele", "?"))
        if meta.get("cible"):
            self._ligne_stat(section, "Type", f"Régression — {meta['cible']}")
        else:
            self._ligne_stat(section, "Objectif appris",
                             meta.get("tache_libelle",
                                      cibles.obtenir(meta.get("tache")).libelle))
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

    def _bloc_performance(self, meta, metriques, objectif, rapport):
        section = self._section(self.eval_contenu,
                                "🎯  Performance sur le test (données jamais vues)")
        self._stat(section, "accuracy", metriques.get("accuracy"), pourcentage=True)

        # La marge d'erreur est LE chiffre qui décide si le reste veut dire
        # quelque chose : une justesse de 55 % sur 240 observations
        # indépendantes est encore compatible avec le hasard.
        marge = metriques.get("marge_accuracy")
        effectif = metriques.get("n_effectif")
        if marge is not None and effectif:
            significatif = metriques.get("significatif")
            self._ligne_stat(
                section, "Marge d'erreur à 95 %",
                f"+/- {marge * 100:.2f} points", AIDES["marge"],
                commentaire=(
                    f"Calculée sur {effectif:,.0f} observations INDÉPENDANTES "
                    f"pour {meta.get('n_test', 0):,} lignes de test — les cibles "
                    f"se chevauchent sur l'horizon."
                    + ("  L'écart à la réponse constante dépasse cette marge : "
                       "le modèle a appris quelque chose."
                       if significatif else
                       "  L'écart à la réponse constante tient DANS cette marge : "
                       "rien ne prouve encore un avantage réel.")),
                niveau="bon" if significatif else "moyen")

        self._stat(section, "auc", metriques.get("auc"))
        self._stat(section, "auc_validation", meta.get("auc_validation"))

        blocs = meta.get("auc_blocs") or []
        if blocs:
            ecart = (max(blocs) - min(blocs))
            self._ligne_stat(
                section, "AUC par bloc de validation croisée",
                " · ".join(f"{a:.4f}" for a in blocs),
                "Le modèle retenu est celui qui tient sur PLUSIEURS époques de\n"
                "marché, pas celui qui a eu de la chance sur une seule.\n\n"
                "L'écart entre blocs est la vraie mesure de l'incertitude : il\n"
                "est presque toujours bien plus grand que l'écart entre les\n"
                "configurations candidates, ce qui montre qu'un choix fait sur\n"
                "un seul bloc revient à tirer au sort.",
                commentaire=(f"Écart entre le meilleur et le pire bloc : "
                             f"{ecart:.4f}."),
                niveau="bon" if ecart < 0.03 else "moyen")

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

        if objectif.binaire:
            hausse = rapport.get("HAUSSE", rapport.get("Hausse", {}))
            baisse = rapport.get("BAISSE", rapport.get("Baisse", {}))
            self._stat(section, "precision_hausse", hausse.get("precision"),
                       pourcentage=True)
            self._stat(section, "rappel_hausse", hausse.get("recall"), pourcentage=True)
            self._stat(section, "precision_baisse", baisse.get("precision"),
                       pourcentage=True)
            self._stat(section, "rappel_baisse", baisse.get("recall"), pourcentage=True)
            return

        self._detail_classes(section, metriques, objectif, rapport)

    def _bloc_regimes(self, metriques):
        """
        Justesse ventilée par régime de volatilité.

        La moyenne globale peut cacher un modèle parfaitement utilisable dans
        un seul régime. C'est le tableau qui dit s'il faut trader tout le temps
        ou seulement une partie du temps.
        """
        regimes = metriques.get("regimes") or []
        if not regimes:
            return

        section = self._section(self.eval_contenu,
                                "🌡️  Justesse par régime de volatilité",
                                aide=AIDES["regimes"])
        reference = metriques.get("accuracy") or 0.0
        for ligne in regimes:
            ecart = ligne["justesse"] - reference
            self._ligne_stat(
                section, f"{ligne['regime']}  ({ligne['n']:,} bougies)",
                f"{ligne['justesse'] * 100:.2f} %  +/- {ligne['marge'] * 100:.2f}",
                commentaire=(f"{ecart * 100:+.2f} points par rapport à la "
                             f"moyenne du test."),
                niveau=("bon" if ecart > ligne["marge"] else
                        "mauvais" if ecart < -ligne["marge"] else "moyen"))

        meilleur = max(regimes, key=lambda ligne: ligne["justesse"])
        pire = min(regimes, key=lambda ligne: ligne["justesse"])
        if meilleur["justesse"] - pire["justesse"] > meilleur["marge"] + pire["marge"]:
            self._ligne_stat(
                section, "Lecture", "Modèle conditionnel",
                commentaire=(
                    f"L'écart entre « {meilleur['regime']} » "
                    f"({meilleur['justesse'] * 100:.1f} %) et « {pire['regime']} » "
                    f"({pire['justesse'] * 100:.1f} %) dépasse les marges d'erreur : "
                    f"ne suivre ce modèle qu'en régime « {meilleur['regime']} » "
                    f"vaut mieux que de le suivre partout."),
                niveau="bon")

    def _detail_classes(self, section, metriques, objectif, rapport):
        """
        Précision et rappel classe par classe, pour les objectifs multi-classes.

        C'est ici que se lit l'intérêt réel de l'objectif « amplitude » : la
        classe neutre écrase tout en effectif, et ce sont les classes extrêmes
        (« forte hausse », « forte baisse ») qui portent l'information
        exploitable — même si le modèle les trouve rarement.
        """
        repartition = metriques.get("repartition_reelle", [])
        for indice, classe in enumerate(objectif.classes):
            scores = rapport.get(classe, {})
            precision = scores.get("precision")
            rappel = scores.get("recall")
            if precision is None:
                continue
            part = repartition[indice] if indice < len(repartition) else None
            commentaire = (f"{part:.1%} des bougies réelles" if part is not None else "")
            niveau = ("bon" if precision > (part or 0) + 0.05 else
                      "moyen" if precision > (part or 0) else "faible")
            self._ligne_stat(
                section, f"Classe « {classe} »",
                f"précision {precision:.1%} · rappel {rappel:.1%}",
                "Précision : parmi les bougies annoncées dans cette classe,\n"
                "part réellement dans cette classe.\n"
                "Rappel : part des bougies de cette classe effectivement trouvées.\n"
                "Une précision supérieure à la fréquence réelle de la classe\n"
                "signifie que le modèle apporte une information.",
                commentaire=commentaire, niveau=niveau)

    # ----------------------------------------------------------------------
    def _bloc_regression(self, meta):
        """Métriques propres aux modèles de régression (volatilité, quantiles)."""
        metriques = meta.get("metriques", {})
        if metriques.get("type") == "quantiles":
            self._bloc_quantiles(metriques)
        else:
            self._bloc_volatilite(metriques)

    def _bloc_volatilite(self, metriques):
        section = self._section(self.eval_contenu,
                                "📐  Volatilité prévue (test, jamais vu)")
        self._stat(section, "r2", metriques.get("r2"))
        self._stat(section, "r2_naif", metriques.get("r2_naif"))

        plafond = metriques.get("r2_plafond")
        if plafond:
            self._ligne_stat(
                section, STATS["r2_plafond"]["label"], f"{plafond:.4f}",
                STATS["r2_plafond"]["aide"],
                commentaire=(f"Le modèle en atteint "
                             f"{metriques.get('part_du_plafond', 0):.0%}. "
                             f"C'est à ce plafond qu'il faut comparer le R², pas à 1."),
                niveau="bon" if metriques.get("part_du_plafond", 0) > 0.4 else "moyen")

        self._stat(section, "correlation_rang", metriques.get("correlation_rang"))
        self._ligne_stat(
            section, "Erreur absolue moyenne",
            f"{metriques.get('mae', 0):.4f} %",
            "Écart moyen entre l'amplitude prévue et l'amplitude observée.",
            commentaire=(f"ATR dilaté : {metriques.get('mae_naif', 0):.4f} % · "
                         f"moyenne constante : {metriques.get('mae_moyenne', 0):.4f} %"))
        self._ligne_stat(
            section, "Amplitude moyenne (prévue / réelle)",
            f"{metriques.get('moyenne_prevue', 0):.3f} % / "
            f"{metriques.get('moyenne_reelle', 0):.3f} %",
            "Un modèle bien calibré prévoit en moyenne la bonne amplitude.\n"
            "Un écart systématique vers le bas trahit une perte mal choisie\n"
            "(l'erreur absolue viserait la médiane, pas la moyenne).")

    def _bloc_quantiles(self, metriques):
        section = self._section(self.eval_contenu,
                                "📊  Intervalle de prédiction (test, jamais vu)")

        couverture = metriques.get("couverture")
        attendue = metriques.get("couverture_attendue", 0.80)
        commentaire, niveau = commenter("couverture", couverture)
        self._ligne_stat(
            section, STATS["couverture"]["label"],
            f"{couverture:.1%}  (attendu {attendue:.0%})",
            STATS["couverture"]["aide"], commentaire, niveau)

        self._ligne_stat(
            section, STATS["largeur_moyenne"]["label"],
            f"{metriques.get('largeur_moyenne', 0):.2f} %",
            STATS["largeur_moyenne"]["aide"],
            commentaire=(f"Médiane {metriques.get('largeur_mediane', 0):.2f} % · "
                         f"écart-type {metriques.get('largeur_ecart_type', 0):.2f} % — "
                         f"un écart-type élevé est une bonne nouvelle : l'intervalle "
                         f"s'élargit vraiment en marché agité."),
            niveau="bon" if metriques.get("largeur_ecart_type", 0) > 0 else "moyen")

        for cle, detail in (metriques.get("detail") or {}).items():
            gain = detail.get("gain", 0.0)
            self._ligne_stat(
                section, f"Quantile {detail['quantile']:.0%} ({cle})",
                f"perte pinball {detail['perte_pinball']:.4f}",
                "Perte pinball : la métrique propre d'une régression quantile.\n"
                "Elle pénalise différemment les écarts au-dessus et en dessous,\n"
                "ce qui force le modèle à produire un vrai quantile.",
                commentaire=(f"{gain:+.1%} face au quantile constant "
                             f"({detail['perte_reference']:.4f})"),
                niveau="bon" if gain > 0.02 else "moyen" if gain > 0 else "faible")

    def _bloc_seuils(self, metriques, objectif=None):
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

        # Le plafond de confiance explique à lui seul la plupart des « aucun
        # signal » : au-dessus, le curseur de seuil ne peut rien retenir.
        # Le « bon » plafond dépend du nombre de classes : 60 % est tranché en
        # binaire, ce serait déjà remarquable sur cinq classes (hasard = 20 %).
        hasard = metriques.get("confiance_neutre",
                               objectif.confiance_neutre if objectif else 0.5)
        repere = hasard + 0.10

        plafond = metriques.get("confiance_max")
        if plafond is not None:
            self._ligne_stat(
                section, "Confiance maximale atteinte", f"{plafond:.1%}",
                f"Confiance la plus élevée que ce modèle ait produite sur le test.\n"
                f"Régler le seuil au-dessus ne retient aucun signal — ce n'est pas\n"
                f"un défaut de réglage mais la limite de ce que le modèle sait.\n"
                f"Le hasard vaut ici {hasard:.0%} (une chance sur "
                f"{round(1 / hasard) if hasard else '?'}).",
                commentaire=("Plafond bas : le modèle ne prétend jamais être très sûr."
                             if plafond < repere else
                             "Le modèle produit des signaux nettement tranchés."),
                niveau="moyen" if plafond < repere else "bon")

        conseil = metriques.get("seuil_conseille")
        if conseil:
            self._ligne_stat(
                section, "Seuil conseillé", f"{conseil['seuil']:.0%}",
                "Seuil qui maximise la justesse tout en laissant passer assez de\n"
                "signaux pour que le chiffre soit fiable (au moins 200).",
                commentaire=(f"{conseil['n']:,} signaux ({conseil['couverture']:.1%} du "
                             f"temps) pour {conseil['precision']:.2%} de justesse."),
                niveau="bon")
        else:
            self._ligne_stat(
                section, "Seuil conseillé", "aucun",
                "Aucun seuil ne fait nettement mieux que l'absence de filtrage.",
                commentaire="Essaie un horizon plus court (3 à 6) ou un autre modèle.",
                niveau="faible")

        entete = ctk.CTkFrame(section, fg_color=COULEURS["accent"], corner_radius=6)
        entete.pack(fill="x", pady=(0, 4))
        for texte, largeur in (("Seuil", 90), ("Retenues", 120), ("Couverture", 120),
                               ("Précision", 120), ("Dont ▲ hausse", 140)):
            ctk.CTkLabel(entete, text=texte, width=largeur, anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold")
                         ).pack(side="left", padx=6, pady=6)

        # Référence = le seuil le plus bas de la grille, c'est-à-dire « aucun
        # filtrage ». Sa valeur dépend de l'objectif (0.50 en binaire, 0.20 sur
        # cinq classes), d'où la lecture du premier élément plutôt qu'un 0.50 codé en dur.
        reference = next((l["precision"] for l in table
                          if l["precision"] is not None), None)
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

        # Les régressions entraînent un modèle par quantile : `n_arbres` y est
        # un dictionnaire, contre un simple entier en classification.
        arbres = meta.get("n_arbres")
        if isinstance(arbres, dict):
            self._ligne_stat(
                section, "Arbres retenus (early stopping)",
                " · ".join(f"{cle} : {valeur}" for cle, valeur in arbres.items()),
                STATS["n_arbres"]["aide"])
        else:
            self._stat(section, "n_arbres", arbres)

        self._stat(section, "embargo", meta.get("embargo"))
        if meta.get("validation_croisee") is not None:
            self._ligne_stat(
                section, "Sélection de la configuration",
                ("Validation croisée purgée" if meta["validation_croisee"]
                 else "Bloc de validation unique"),
                "Découper la période d'apprentissage en blocs successifs donne\n"
                "plusieurs mesures au lieu d'une seule, donc un écart-type réel.\n"
                "« Purgée » : les lignes dont la cible empiète sur le bloc évalué\n"
                "sont retirées de l'apprentissage.",
                commentaire=(f"Marge de sélection : "
                             f"{meta.get('marge_selection'):.4f}."
                             if meta.get("marge_selection") else ""),
                niveau="bon" if meta.get("validation_croisee") else "moyen")
        if meta.get("normalisation"):
            self._ligne_stat(
                section, "Normalisation entre cryptos", meta["normalisation"],
                "Les features dont le niveau dépend de l'actif sont converties\n"
                "en rang de percentile glissant, calculé crypto par crypto et\n"
                "uniquement sur le passé.",
                niveau="bon")
        if meta.get("calibration"):
            self._ligne_stat(section, "Calibration des probabilités",
                             meta["calibration"])
            self._ligne_stat(section, "Rééquilibrage des classes",
                             "Oui" if meta.get("reequilibrage") else
                             "Non (classes déjà équilibrées)")

        features = meta.get("features", [])
        contexte = meta.get("features_contexte", [])
        base = [f for f in features if f not in contexte]
        self._ligne_stat(
            section, "Features utilisées", f"{len(features)}",
            "Liste blanche stricte : aucun prix, aucune colonne variation_*\n"
            "ne peut entrer ici, même par accident.",
            commentaire=f"{len(base)} indicateurs de base : {', '.join(base)}")
        if contexte:
            self._ligne_stat(
                section, "Dont contexte (multi-timeframe / exogène)",
                f"{len(contexte)}",
                "Colonnes ajoutées automatiquement quand elles sont présentes\n"
                "dans le fichier analysé et couvrent au moins 60 % des lignes.",
                commentaire=", ".join(contexte), niveau="moyen")

        if meta.get("n_configurations"):
            duree = meta.get("duree_secondes")
            self._ligne_stat(
                section, "Configurations essayées",
                f"{meta['n_configurations']}",
                "Nombre de jeux d'hyperparamètres comparés avant de choisir.\n"
                "Chacun est évalué sur tous les blocs de validation croisée,\n"
                "et c'est la plus SIMPLE des configurations statistiquement à\n"
                "égalité qui est retenue.\n\n"
                "En essayer six fois plus déplace le résultat de quelques\n"
                "millièmes d'AUC, soit bien moins que sa marge d'erreur : ce\n"
                "réglage sert à vérifier, pas à espérer un gain.",
                commentaire=(f"{meta.get('mode_recherche_libelle', '')}"
                             + (f" · {duree:,.0f} s au total" if duree else "")))

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

    # ----------------------------------------------------------------------
    def _bloc_utilite(self, meta):
        """
        Ce que chaque feature a réellement apporté, et ce qu'un filtre ferait.

        C'est la contrepartie du curseur « utilité minimale » de la page
        Modèle : le curseur décide, ce tableau montre sur quoi.

        Mesure faite sur la VALIDATION, jamais sur le test. Le graphique de la
        page Visualisation, lui, est mesuré sur le test — les deux ne sont pas
        redondants : l'un sert à choisir, l'autre à diagnostiquer, et l'écart
        entre les deux est en soi une information sur la stabilité du modèle.
        """
        utilite = meta.get("utilite") or {}
        valeurs = utilite.get("valeurs") or {}
        if not valeurs:
            return

        section = self._section(self.eval_contenu, "🏅  Utilité mesurée des features")

        classement = sorted(valeurs.items(), key=lambda paire: paire[1],
                            reverse=True)
        utiles = [nom for nom, valeur in classement if valeur > 0]
        nuisibles = [nom for nom, valeur in classement if valeur < 0]
        actives = set(meta.get("features", []))

        self._ligne_stat(
            section, "Mesure",
            f"{len(classement)} features · {utilite.get('n', 0):,} bougies "
            f"de validation",
            "On mélange chaque colonne au hasard et on regarde de combien\n"
            "l'AUC tombe. Grande perte = feature essentielle. Perte nulle =\n"
            "feature inutile. Perte NÉGATIVE = feature nuisible : le modèle\n"
            "s'appuyait dessus et cet appui se retourne contre lui.\n\n"
            "Bloc de validation, jamais le test : sélectionner ses features\n"
            "d'après le test puis annoncer un résultat sur ce même test\n"
            "reviendrait à se noter soi-même.",
            commentaire=f"{utilite.get('repetitions', 5)} permutations par feature")

        tete = classement[:8]
        for nom, valeur in tete:
            marque = "" if nom in actives else "  (écartée par le filtre)"
            niveau = ("bon" if valeur >= 0.002 else
                      "moyen" if valeur >= 0.0005 else
                      "faible" if valeur > 0 else "mauvais")
            self._ligne_stat(section, nom + marque, f"{valeur:+.4f}",
                             niveau=niveau)

        if len(classement) > len(tete):
            reste = classement[len(tete):]
            self._ligne_stat(
                section, f"… et {len(reste)} autres",
                f"de {reste[0][1]:+.4f} à {reste[-1][1]:+.4f}",
                "Le bas du classement. Une feature autour de zéro ne coûte\n"
                "presque rien à garder, mais elle n'apporte rien non plus.",
                commentaire=", ".join(nom for nom, _ in reste))

        self._ligne_stat(
            section, "Bilan",
            f"{len(utiles)} utiles · {len(classement) - len(utiles) - len(nuisibles)} "
            f"neutres · {len(nuisibles)} nuisibles",
            "Un modèle dont la moitié des features est neutre n'est pas\n"
            "cassé : ces colonnes diluent le signal sans le détruire. Les\n"
            "écarter avec le curseur « utilité minimale » de la page Modèle\n"
            "rend le modèle plus lisible, rarement plus précis.",
            commentaire=(f"Nuisibles : {', '.join(nuisibles)}"
                         if nuisibles else ""),
            niveau="moyen" if nuisibles else "bon")

        if meta.get("features_ecartees"):
            self._ligne_stat(
                section, "Filtre appliqué à cet entraînement",
                f"utilité ≥ {meta.get('seuil_utilite', 0):.4f} d'AUC",
                "Les features écartées gardent leur dernière valeur mesurée :\n"
                "abaisser le curseur les fait revenir au prochain entraînement.",
                commentaire=(f"{len(meta['features_ecartees'])} écartées sur "
                             f"{len(meta.get('features_candidates', []))} : "
                             + ", ".join(meta["features_ecartees"])),
                niveau="moyen")

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
