"""
Page 3 — Modèle et prédiction.

Volontairement dépouillée : cinq réglages seulement (crypto, modèle, objectif,
horizon, seuil de confiance). Tout le reste — découpage, embargo, équilibrage
des classes, hyperparamètres, calibration, exploitation de la RAM — est décidé
automatiquement et détaillé dans la fenêtre « Ce qui est fait automatiquement ».

L'objectif est le seul réglage ajouté depuis la refonte. Il ne change pas la
machinerie, seulement la QUESTION posée : direction simple (par défaut),
direction au-delà des frais, triple barrière, ou amplitude en 5 classes.
"""

from __future__ import annotations

import customtkinter as ctk

from ... import cibles, config, modele, stockage
from ..composants import InfoBulle
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
        # Les cinq seuls réglages
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

        colonne_objectif, self.mod_objectif, self.mod_objectif_menu = self._menu(
            ligne, "Objectif", cibles.libelles(), 240, aide=AIDES["objectif"])
        self.mod_objectif_menu.configure(command=lambda _v: self._changer_objectif())
        colonne_objectif.pack(side="left")

        ligne2 = self._ligne(corps, espace=(16, 0))
        colonne_horizon, self.mod_horizon = self._curseur(
            ligne2, "Horizon (périodes)", 1, config.HORIZON_MAX,
            config.HORIZON_DEFAUT, aide=AIDES["horizon"], largeur=280)
        colonne_horizon.pack(side="left", padx=(0, 30))

        colonne_seuil, self.mod_seuil = self._curseur(
            ligne2, "Seuil de confiance", 0.50, 0.95, config.SEUIL_DEFAUT,
            aide=AIDES["seuil"], largeur=240, format_valeur="{:.0%}", nb_pas=45)
        colonne_seuil.pack(side="left")

        # Description du modèle et de l'objectif, mises à jour à chaque changement.
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

        ctk.CTkButton(boutons, text="🔥 Entraîner", height=42, width=140,
                      command=self._action_entrainer).pack(side="left")
        ctk.CTkButton(boutons, text="🔮 Prédire", height=42, width=140,
                      fg_color=COULEURS["vert"], hover_color="#27ae60",
                      command=self._action_predire).pack(side="left", padx=10)
        bouton_wf = ctk.CTkButton(boutons, text="📏 Walk-forward", height=42, width=160,
                                  fg_color=COULEURS["bleu"], hover_color="#0097a7",
                                  command=self._action_walkforward)
        bouton_wf.pack(side="left", padx=(0, 10))
        InfoBulle(bouton_wf, AIDES["walk_forward"])
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
        objectif = self._objectif()
        self.mod_description.configure(
            text=(f"🤖 {nom} — {modele.DESCRIPTIONS_MODELES.get(nom, '')}\n"
                  f"🎯 {objectif.libelle} — {objectif.description}"))

    def _objectif(self):
        """Objectif d'apprentissage actuellement sélectionné."""
        return cibles.par_libelle(self.mod_objectif.get())

    def _changer_objectif(self):
        """
        Adapte le curseur de seuil au nombre de classes de l'objectif.

        Avec deux classes, la confiance part de 50 % (le hasard) ; avec cinq,
        elle part de 20 %. Laisser le curseur démarrer à 50 % ne retiendrait
        alors AUCUN signal, et rien ne l'expliquerait à l'écran.
        """
        objectif = self._objectif()
        plancher = objectif.confiance_neutre
        defaut = min(0.95, plancher + (config.SEUIL_DEFAUT - 0.5))

        self.mod_seuil.configure(from_=plancher, to=0.95,
                                 number_of_steps=max(1, int((0.95 - plancher) * 100)))
        self.mod_seuil.set(defaut)
        self._maj_description_modele()
        self.log(f"🎯 Objectif « {objectif.libelle} » — {objectif.n_classes} classes, "
                 f"confiance entre {plancher:.0%} et 100 %. "
                 f"Seuil replacé sur {defaut:.0%}.")

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
        objectif = self._objectif()

        def apres(meta):
            if not meta:
                return
            self._afficher_bilan_entrainement(meta, nom_modele, symbole,
                                              intervalle, horizon)
            self._rafraichir_modele()

        self.executer(
            f"Entraînement {symbole} ({intervalle}) h={horizon} — {objectif.libelle}",
            lambda: modele.entrainer(symbole, intervalle, horizon, nom_modele,
                                     objectif.cle),
            apres=apres)

    def _afficher_bilan_entrainement(self, meta, nom_modele, symbole,
                                     intervalle, horizon):
        """
        Résume l'entraînement et RÈGLE LE SEUIL à la valeur utile.

        Le plafond de confiance est propre à chaque modèle : sur un marché
        bruité il dépasse rarement 0.60, si bien qu'un seuil laissé trop haut ne
        laisse passer aucun signal — sans que rien ne l'explique. On affiche donc
        le plafond atteint et on positionne le curseur sur le seuil qui donne
        réellement la meilleure justesse.
        """
        metriques = meta["metriques"]
        plafond = metriques["confiance_max"]
        conseil = metriques.get("seuil_conseille")
        objectif = cibles.obtenir(meta.get("tache"))

        entete = (f"✅ {nom_modele} entraîné sur {symbole} ({intervalle}), "
                  f"horizon {horizon} — objectif « {objectif.libelle} ».\n"
                  f"Test : justesse {metriques['accuracy']:.1%} · "
                  f"AUC {metriques['auc']:.3f} · "
                  f"baseline {metriques['baseline_majoritaire']:.1%} · "
                  f"confiance plafonnée à {plafond:.1%}.\n")

        if conseil is None:
            self.mod_resume.configure(
                text=entete + ("⚠️ Aucun seuil n'améliore nettement la justesse. "
                               "Essaie un horizon plus court (3 à 6) ou un autre modèle."),
                text_color=COULEURS["orange"])
            return

        self.mod_seuil.set(conseil["seuil"])
        self.mod_resume.configure(
            text=entete + (
                f"👉 Seuil réglé sur {conseil['seuil']:.0%} (le meilleur pour ce modèle) : "
                f"{conseil['n']:,} signaux, soit {conseil['couverture']:.1%} du temps, "
                f"pour {conseil['precision']:.2%} de justesse contre "
                f"{metriques['accuracy']:.2%} sans filtrage."),
            text_color=COULEURS["vert"])

    def _action_walkforward(self):
        """
        Réentraîne le modèle de fenêtre en fenêtre et ne prédit que l'inconnu.

        Produit un fichier « …_wf » entièrement hors échantillon, backtestable
        sur toute sa durée — contrairement à une prédiction classique, dont les
        premiers 85 % ont servi à l'entraînement.
        """
        reglages = self._reglages()
        if reglages is None:
            return
        symbole, intervalle, horizon, seuil = reglages
        nom_modele = self.mod_type.get()
        objectif = self._objectif()
        marque = "" if objectif.cle == cibles.TACHE_DEFAUT else f"_{objectif.cle}"

        def apres(resultats):
            if resultats is None or resultats.empty:
                return
            evaluables = resultats.dropna(subset=["Correct"])
            retenues = evaluables[evaluables["Retenu"] == 1]
            self.mod_carte_retenues[1].configure(text=f"{len(retenues):,}")
            self._afficher_dernier_signal(resultats, seuil)

            detail = (f"{len(retenues):,} signaux retenus au seuil de {seuil:.0%}, "
                      f"justesse {retenues['Correct'].mean():.2%}"
                      if len(retenues) else
                      f"aucun signal au seuil de {seuil:.0%}")
            self.mod_resume.configure(
                text=(f"📏 Walk-forward terminé sur {len(evaluables):,} bougies, "
                      f"toutes hors échantillon.\n"
                      f"Justesse globale {evaluables['Correct'].mean():.2%} — {detail}.\n"
                      f"Va dans Backtest et choisis "
                      f"« {symbole}_{intervalle}_h{horizon}{marque}_wf » : "
                      f"tu peux y simuler TOUTE la période sans fausser le résultat."),
                text_color=COULEURS["vert"])

        self.executer(
            f"Walk-forward {symbole} ({intervalle}) h={horizon}",
            lambda: modele.walk_forward(symbole, intervalle, horizon, nom_modele,
                                        seuil_confiance=seuil, tache=objectif.cle),
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

        objectif = self._objectif()
        self.executer(
            f"Prédiction {symbole} ({intervalle}) h={horizon}",
            lambda: modele.predire(symbole, intervalle, horizon, seuil,
                                   symbole_modele, objectif.cle),
            apres=apres)

    # ----------------------------------------------------------------------
    def _afficher_dernier_signal(self, resultats, seuil):
        """Met à jour les cartes avec la dernière bougie et le bilan sur le test."""
        derniere = resultats.iloc[-1]
        sens = derniere["Sens_Predit"]
        # En multi-classe, l'étiquette détaillée porte l'amplitude (« Forte
        # hausse ») : c'est elle qui intéresse, pas le simple sens.
        etiquette = str(derniere.get("Classe_Predite", sens))
        confiance = float(derniere["Confiance"])

        if confiance < seuil:
            texte, couleur = f"{etiquette} (sous le seuil)", COULEURS["texte_doux"]
        elif sens == cibles.HAUSSE:
            texte, couleur = f"{etiquette} ▲", COULEURS["vert"]
        elif sens == cibles.BAISSE:
            texte, couleur = f"{etiquette} ▼", COULEURS["rouge"]
        else:
            texte, couleur = f"{etiquette} —", COULEURS["texte_doux"]

        self.mod_carte_sens[1].configure(text=texte, text_color=couleur,
                                         font=ctk.CTkFont(size=18, weight="bold"))
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

        hausses = int((retenues["Sens_Predit"] == cibles.HAUSSE).sum())
        baisses = int((retenues["Sens_Predit"] == cibles.BAISSE).sum())
        neutres = len(retenues) - hausses - baisses
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

        repartition = f"{hausses:,} ▲ et {baisses:,} ▼"
        if neutres:
            repartition += f", plus {neutres:,} « ne rien faire »"

        self.mod_resume.configure(
            text=(f"Sur le bloc test ({len(test):,} bougies jamais vues) : "
                  f"{len(retenues):,} signaux retenus au seuil de {seuil:.0%} "
                  f"({len(retenues) / len(test):.1%} du temps), "
                  f"dont {repartition}.\n"
                  f"Justesse des signaux retenus : {justesse:.2%} "
                  f"— contre {reference:.2%} sans filtrage. "
                  f"Le seuil de confiance {verdict} ici."),
            text_color=couleur_resume)
