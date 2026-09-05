"""
Fenêtre principale de Crypto Lab.

Assemble les sept pages (une par étape du pipeline), la console, la barre de
statut et le moteur d'exécution en arrière-plan.

Règle de fonctionnement : tout traitement long part dans un thread de travail
via `executer()`, et seul le thread graphique touche aux widgets. Les `print`
du code métier sont redirigés vers la console de la fenêtre.

Conséquence pratique : **aucune action ne fige l'application**. Pendant qu'un
entraînement tourne, on peut changer de page, consulter un graphique, lire la
console — et l'arrêter d'un bouton. Le thread de travail n'est jamais tué de
force : il consulte un drapeau (`suivi.MONITEUR`) entre deux étapes et à chaque
arbre construit, puis s'arrête proprement.

Trois files se croisent dans `_vider_file_console`, la boucle de service du
thread graphique :

    file_console     les `print` du code métier         -> console
    file_resultats   les tâches terminées               -> callbacks « après »
    suivi.MONITEUR   progression, courbes, diagnostics  -> fenêtre de suivi
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
import traceback

import customtkinter as ctk

from .. import config, suivi
from . import theme
from .composants import FluxConsole, MixinComposants
from .moniteur import FenetreSuivi
from .pages import (PageAmplitude, PageAnalyse, PageBacktest, PageDonnees,
                    PageEvaluation, PageModele, PageVisualisation)
from .theme import COULEURS

# Navigation : (clé interne, libellé affiché)
NAVIGATION = [
    ("Données",       "📥  1 · Extraction"),
    ("Analyse",       "🔬  2 · Analyse"),
    ("Modèle",        "🧠  3 · Prédiction"),
    ("Amplitude",     "📐  4 · Amplitude"),
    ("Évaluation",    "📋  5 · Évaluation"),
    ("Visualisation", "📊  6 · Visualisation"),
    ("Backtest",      "💰  7 · Backtest"),
]


class CryptoLab(ctk.CTk, MixinComposants, PageDonnees, PageAnalyse, PageModele,
                PageAmplitude, PageEvaluation, PageVisualisation, PageBacktest):
    """Application complète : extraction → analyse → modèle → backtest."""

    def __init__(self):
        super().__init__()
        config.preparer_dossiers()

        self.title("Crypto Lab")
        self.geometry("1360x900")
        self.minsize(1150, 760)
        self.configure(fg_color=COULEURS["fond"])

        # État partagé par les pages
        self.tache_en_cours = False
        self.fenetre_suivi = None
        self.libelle_tache = ""
        self.file_console = queue.Queue()
        self.file_resultats = queue.Queue()
        self.pages = {}
        self.boutons_navigation = {}
        self.rafraichisseurs = {}
        self.zones_graphe = {}

        theme.configurer_tableaux()

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._construire_sidebar()
        self._construire_contenu()
        self._construire_console()
        self._construire_barre_statut()

        # Construction des sept pages, dans l'ordre du pipeline.
        self._page_donnees()
        self._page_analyse()
        self._page_modele()
        self._page_amplitude()
        self._page_evaluation()
        self._page_visualisation()
        self._page_backtest()

        self.afficher_page("Données")

        sys.stdout = FluxConsole(self.file_console)
        sys.stderr = FluxConsole(self.file_console)
        self._vider_file_console()

        self.protocol("WM_DELETE_WINDOW", self._fermer)
        self.log("✅ Crypto Lab prêt.")

    # ======================================================================
    # STRUCTURE DE LA FENÊTRE
    # ======================================================================
    def _construire_sidebar(self):
        barre = ctk.CTkFrame(self, width=210, corner_radius=0,
                             fg_color=COULEURS["panneau"])
        barre.grid(row=0, column=0, sticky="nsew")
        barre.grid_propagate(False)

        ctk.CTkLabel(barre, text="⚡ CRYPTO LAB",
                     font=ctk.CTkFont(size=22, weight="bold")
                     ).pack(pady=(26, 4), padx=20)
        ctk.CTkLabel(barre, text="Hausse ou baisse, avec un niveau de confiance",
                     font=ctk.CTkFont(size=11), text_color=COULEURS["texte_doux"],
                     wraplength=170, justify="center").pack(pady=(0, 24), padx=14)

        for nom, libelle in NAVIGATION:
            bouton = ctk.CTkButton(
                barre, text=libelle, anchor="w", height=42, corner_radius=8,
                fg_color="transparent", font=ctk.CTkFont(size=14),
                hover_color=COULEURS["accent"],
                command=lambda cible=nom: self.afficher_page(cible))
            bouton.pack(fill="x", padx=12, pady=4)
            self.boutons_navigation[nom] = bouton

        bas = ctk.CTkFrame(barre, fg_color="transparent")
        bas.pack(side="bottom", fill="x", padx=12, pady=18)
        ctk.CTkLabel(bas, text="Apparence", text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=11)).pack(anchor="w")
        ctk.CTkOptionMenu(bas, values=["Dark", "Light", "System"],
                          command=lambda choix: ctk.set_appearance_mode(choix.lower())
                          ).pack(fill="x", pady=(4, 0))

    def _construire_contenu(self):
        self.conteneur = ctk.CTkFrame(self, fg_color="transparent")
        self.conteneur.grid(row=0, column=1, sticky="nsew", padx=16, pady=(16, 8))
        self.conteneur.grid_rowconfigure(0, weight=1)
        self.conteneur.grid_columnconfigure(0, weight=1)

    def _nouvelle_page(self, nom, defilante=True):
        """Crée une page (masquée) dans la zone de contenu."""
        classe = ctk.CTkScrollableFrame if defilante else ctk.CTkFrame
        page = classe(self.conteneur, fg_color="transparent")
        page.grid(row=0, column=0, sticky="nsew")
        page.grid_remove()
        self.pages[nom] = page
        return page

    def afficher_page(self, nom):
        """Affiche une page et déclenche son rafraîchissement éventuel."""
        for cle, page in self.pages.items():
            if cle == nom:
                page.grid()
            else:
                page.grid_remove()
        for cle, bouton in self.boutons_navigation.items():
            bouton.configure(fg_color=COULEURS["accent"] if cle == nom else "transparent")

        rafraichir = self.rafraichisseurs.get(nom)
        if rafraichir is not None:
            try:
                rafraichir()
            except Exception as err:                   # noqa: BLE001
                self.log(f"⚠️ Rafraîchissement de « {nom} » : {err}")

    # ======================================================================
    # CONSOLE
    # ======================================================================
    def _construire_console(self):
        cadre = ctk.CTkFrame(self, height=170, corner_radius=10,
                             fg_color=COULEURS["panneau"])
        cadre.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=4)
        cadre.grid_propagate(False)
        cadre.grid_columnconfigure(0, weight=1)
        cadre.grid_rowconfigure(1, weight=1)

        entete = ctk.CTkFrame(cadre, fg_color="transparent")
        entete.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 0))
        ctk.CTkLabel(entete, text="🖥️  Console",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(entete, text="Effacer", width=70, height=26,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self._effacer_console).pack(side="right")

        self.console = ctk.CTkTextbox(
            cadre, font=ctk.CTkFont(family="Consolas", size=12),
            fg_color=COULEURS["axe"], text_color="#cfd8dc", wrap="word")
        self.console.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        self.console.configure(state="disabled")

    def _effacer_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def log(self, message):
        """Écrit un message dans la console de l'application."""
        self.file_console.put(str(message) + "\n")

    def _vider_file_console(self):
        """
        Boucle de service du thread graphique, réveillée toutes les 80 ms.

        Elle draine les deux files alimentées par les threads de travail : les
        messages à afficher, puis les tâches terminées. C'est le SEUL endroit
        où l'on touche aux widgets depuis un résultat de tâche — Tkinter n'est
        pas conçu pour être manipulé depuis un autre thread.
        """
        morceaux = []
        try:
            while True:
                morceaux.append(self.file_console.get_nowait())
        except queue.Empty:
            pass

        if morceaux:
            self.console.configure(state="normal")
            self.console.insert("end", "".join(morceaux).replace("\r", "\n"))
            self.console.see("end")
            self.console.configure(state="disabled")

        # Événements de suivi : progression, courbe d'apprentissage, utilité
        # des features. Consommés même quand la fenêtre de suivi est fermée,
        # sinon la file se remplirait pour rien.
        evenements = suivi.MONITEUR.evenements()
        if evenements:
            self._appliquer_progression(evenements)
            if self.fenetre_suivi is not None:
                try:
                    self.fenetre_suivi.traiter(evenements)
                except tk.TclError:                    # fenêtre déjà détruite
                    self.fenetre_suivi = None

        try:
            while True:
                self._fin_tache(*self.file_resultats.get_nowait())
        except queue.Empty:
            pass

        self.after(80, self._vider_file_console)

    def _appliquer_progression(self, evenements):
        """Bascule la barre de statut en progression réelle dès qu'on en a une."""
        etapes = [e for e in evenements if e.get("type") == suivi.ETAPE]
        if not etapes or not self.tache_en_cours:
            return
        derniere = etapes[-1]
        part = float(derniere.get("part", 0.0))
        if not self.progression_determinee:
            self.progression.stop()
            self.progression.configure(mode="determinate")
            self.progression_determinee = True
        self.progression.set(part)
        self.statut.configure(
            text=f"● {self.libelle_tache} — {derniere.get('libelle', '')} "
                 f"({part:.0%})",
            text_color=COULEURS["orange"])

    # ======================================================================
    # BARRE DE STATUT
    # ======================================================================
    def _construire_barre_statut(self):
        barre = ctk.CTkFrame(self, height=34, corner_radius=10,
                             fg_color=COULEURS["panneau"])
        barre.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 12))
        barre.grid_propagate(False)
        barre.grid_columnconfigure(0, weight=1)

        self.statut = ctk.CTkLabel(barre, text="● Prêt", text_color=COULEURS["vert"],
                                   font=ctk.CTkFont(size=12))
        self.statut.grid(row=0, column=0, sticky="w", padx=14)

        self.progression = ctk.CTkProgressBar(barre, width=220, mode="indeterminate")
        self.progression.grid(row=0, column=1, sticky="e", padx=14)
        self.progression.set(0)
        self.progression_determinee = False

        # Visible seulement pendant un traitement : un bouton d'arrêt grisé en
        # permanence n'apprend rien à personne.
        self.bouton_arret = ctk.CTkButton(
            barre, text="⏹ Arrêter", width=96, height=24,
            fg_color=COULEURS["rouge"], hover_color="#c0392b",
            font=ctk.CTkFont(size=12), command=self.arreter_tache)
        self.bouton_suivi = ctk.CTkButton(
            barre, text="📺 Suivi", width=88, height=24,
            fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
            font=ctk.CTkFont(size=12), command=self.ouvrir_suivi)

    def _maj_statut(self, texte, en_cours=False, erreur=False):
        couleur = (COULEURS["rouge"] if erreur else
                   COULEURS["orange"] if en_cours else COULEURS["vert"])
        self.statut.configure(text=f"● {texte}", text_color=couleur)
        if en_cours:
            self.progression.stop()
            self.progression.configure(mode="indeterminate")
            self.progression_determinee = False
            self.progression.start()
            # Reactive a chaque tache : sinon un arret demande une fois griserait
            # le bouton pour toute la duree de la session.
            self.bouton_arret.configure(state="normal", text="⏹ Arrêter")
            self.bouton_arret.grid(row=0, column=2, padx=(0, 6))
            self.bouton_suivi.grid(row=0, column=3, padx=(0, 12))
        else:
            self.progression.stop()
            self.progression.configure(mode="determinate")
            self.progression.set(0)
            self.progression_determinee = False
            self.bouton_arret.grid_forget()
            self.bouton_suivi.grid_forget()

    # ======================================================================
    # SUIVI EN DIRECT
    # ======================================================================
    def ouvrir_suivi(self, reinitialiser=False, libelle=""):
        """Ouvre (ou remonte) la fenêtre de suivi de l'entraînement."""
        if self.fenetre_suivi is None or not self.fenetre_suivi.winfo_exists():
            self.fenetre_suivi = FenetreSuivi(self, au_arreter=self._arret_demande)
        if reinitialiser:
            self.fenetre_suivi.reinitialiser(libelle)
        self.fenetre_suivi.montrer()
        return self.fenetre_suivi

    def arreter_tache(self):
        """Demande l'arrêt du traitement en cours. Le thread n'est jamais tué."""
        if not self.tache_en_cours:
            return
        suivi.MONITEUR.demander_arret()
        self._arret_demande()

    def _arret_demande(self):
        self.bouton_arret.configure(state="disabled", text="⏹ Arrêt…")
        self.log("⏹ Arrêt demandé — le traitement s'interrompra à la fin de "
                 "l'étape en cours (quelques secondes).")

    # ======================================================================
    # EXÉCUTION EN ARRIÈRE-PLAN
    # ======================================================================
    def executer(self, libelle, fonction, apres=None, suivre=False):
        """
        Lance `fonction` dans un thread de travail, sans figer l'interface.

        Le résultat repart par une file que le thread graphique vide (voir
        `_vider_file_console`) : aucun widget n'est touché depuis le thread de
        travail. `apres` est alors rappelé avec le résultat, en cas de succès
        uniquement.

        `suivre=True` ouvre la fenêtre de suivi en direct : progression réelle,
        courbe d'apprentissage, configurations évaluées. Réservé aux traitements
        qui publient quelque chose (entraînement, walk-forward) — l'ouvrir pour
        une prédiction d'une seconde n'aurait aucun intérêt.

        Une seule tâche à la fois : les traitements se partagent les mêmes
        fichiers, les enchaîner en parallèle n'aurait pas de sens et rendrait
        les erreurs illisibles. Elle reste interruptible à tout moment
        (`arreter_tache`).
        """
        if self.tache_en_cours:
            self.log("⚠️ Une tâche est déjà en cours, patiente… "
                     "(ou clique sur « ⏹ Arrêter »)")
            return

        self.tache_en_cours = True
        self.libelle_tache = libelle
        suivi.MONITEUR.demarrer(libelle)
        self._maj_statut(f"{libelle}…", en_cours=True)
        self.log(f"\n▶️  {libelle}")
        if suivre:
            self.ouvrir_suivi(reinitialiser=True, libelle=libelle)

        def travail():
            resultat, erreur = None, None
            try:
                resultat = fonction()
            except suivi.Annulation as arret:
                erreur = arret
            except Exception as err:                   # noqa: BLE001
                erreur = err
                traceback.print_exc()
            suivi.MONITEUR.terminer(interrompu=isinstance(erreur, suivi.Annulation))
            self.file_resultats.put((libelle, resultat, erreur, apres))

        threading.Thread(target=travail, daemon=True).start()

    def _fin_tache(self, libelle, resultat, erreur, apres):
        self.tache_en_cours = False
        # Dernier passage sur les événements restants : sans lui, la fenêtre de
        # suivi resterait figée sur l'avant-dernière étape.
        restants = suivi.MONITEUR.evenements()
        if restants and self.fenetre_suivi is not None:
            try:
                self.fenetre_suivi.traiter(restants)
            except tk.TclError:                        # pragma: no cover
                self.fenetre_suivi = None

        if isinstance(erreur, suivi.Annulation):
            self._maj_statut("Interrompu")
            self.log(f"⏹ {libelle} — interrompu à ta demande. "
                     f"Aucun modèle n'a été enregistré.")
            return

        if erreur is not None:
            self._maj_statut("Erreur", erreur=True)
            self.log(f"❌ {libelle} : {erreur}")
            return

        self._maj_statut("Prêt")
        self.log(f"✅ {libelle} — terminé.")
        if apres is None:
            return
        try:
            apres(resultat)
        except Exception as err:                       # noqa: BLE001
            self.log(f"⚠️ Affichage du résultat : {err}")
            traceback.print_exc()

    # ======================================================================
    # Les réglages de la page Modèle sont lus par plusieurs pages (graphiques,
    # backtest) : ces deux accesseurs évitent d'aller chercher les curseurs à la
    # main et fonctionnent même avant que la page ne soit construite.
    def lire_horizon(self) -> int:
        """Horizon de prédiction actuellement sélectionné, en périodes."""
        curseur = getattr(self, "mod_horizon", None)
        if curseur is None:
            return config.HORIZON_DEFAUT
        return int(round(curseur.get()))

    def lire_seuil(self) -> float:
        """Seuil de confiance actuellement sélectionné."""
        curseur = getattr(self, "mod_seuil", None)
        return curseur.get() if curseur is not None else config.SEUIL_DEFAUT

    def _fermer(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


def lancer():
    """Point d'entrée de l'interface graphique."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    CryptoLab().mainloop()
