"""
Briques réutilisables de l'interface.

On y trouve trois choses :
  * `FluxConsole` — redirige les `print` du code métier vers la console de l'app ;
  * `InfoBulle` et `FenetreExplication` — l'aide contextuelle ;
  * `MixinComposants` — les fabriques de widgets (sections, champs, menus,
    cartes, tableaux, zones de graphique) partagées par toutes les pages.
"""

from __future__ import annotations

import sys
import tkinter as tk
from datetime import datetime
from tkinter import ttk

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import theme
from .theme import COULEURS


# ===========================================================================
# CONSOLE
# ===========================================================================
class FluxConsole:
    """
    Remplace `sys.stdout` pour capter les `print` du code métier.

    Les messages sont déposés dans une file thread-safe : les traitements
    tournent dans un thread de travail, mais seul le thread graphique a le
    droit de toucher aux widgets Tk.
    """

    def __init__(self, file_attente):
        self.file = file_attente
        self.terminal = sys.__stdout__

    def write(self, message):
        self.file.put(message)
        try:
            self.terminal.write(message)
        except (ValueError, OSError, UnicodeEncodeError):
            pass  # terminal fermé ou incapable d'encoder : la fenêtre suffit

    def flush(self):
        try:
            self.terminal.flush()
        except (ValueError, OSError):
            pass


# ===========================================================================
# AIDE CONTEXTUELLE
# ===========================================================================
class InfoBulle:
    """Petite bulle d'aide affichée au survol d'un widget."""

    def __init__(self, widget, texte, delai=300):
        self.widget = widget
        self.texte = texte
        self.delai = delai
        self.fenetre = None
        self.apres = None
        widget.bind("<Enter>", self._programmer, add="+")
        widget.bind("<Leave>", self._cacher, add="+")
        widget.bind("<ButtonPress>", self._cacher, add="+")

    def _programmer(self, _event=None):
        self._annuler()
        self.apres = self.widget.after(self.delai, self._afficher)

    def _annuler(self):
        if self.apres is not None:
            try:
                self.widget.after_cancel(self.apres)
            except (ValueError, tk.TclError):
                pass
            self.apres = None

    def _afficher(self):
        if self.fenetre is not None or not self.texte:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        self.fenetre = tk.Toplevel(self.widget)
        self.fenetre.wm_overrideredirect(True)
        self.fenetre.wm_geometry(f"+{x}+{y}")
        self.fenetre.configure(bg=COULEURS["accent"])
        tk.Label(
            self.fenetre, text=self.texte, justify="left",
            bg=COULEURS["carte"], fg=COULEURS["texte"],
            font=("Segoe UI", 10), padx=10, pady=8, wraplength=460,
        ).pack(padx=1, pady=1)

    def _cacher(self, _event=None):
        self._annuler()
        if self.fenetre is not None:
            self.fenetre.destroy()
            self.fenetre = None


class FenetreExplication(ctk.CTkToplevel):
    """Fenêtre de lecture pour les textes longs (« Comment ça marche ? »)."""

    def __init__(self, parent, titre, contenu):
        super().__init__(parent)
        self.title(titre)
        self.geometry("880x680")
        self.configure(fg_color=COULEURS["fond"])
        self.transient(parent)

        ctk.CTkLabel(self, text=titre, font=ctk.CTkFont(size=20, weight="bold")
                     ).pack(anchor="w", padx=24, pady=(20, 10))

        zone = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=13),
                              fg_color=COULEURS["panneau"], text_color=COULEURS["texte"],
                              wrap="word")
        zone.pack(fill="both", expand=True, padx=24, pady=(0, 14))
        zone.insert("1.0", contenu)
        zone.configure(state="disabled")

        ctk.CTkButton(self, text="Fermer", width=120, command=self.destroy
                      ).pack(pady=(0, 20))
        self.after(120, self.lift)      # passe devant la fenêtre principale


class FenetrePanier(ctk.CTkToplevel):
    """
    Choix des cryptos à empiler dans un même modèle.

    Une liste de cases à cocher, deux boutons, rien d'autre : le panier ne doit
    pas devenir un formulaire. Tout ce qui compte — l'alignement des dates, la
    normalisation entre actifs, le choix des features communes — est décidé
    automatiquement et expliqué dans l'infobulle de la page.
    """

    def __init__(self, parent, cryptos, selection, resume, au_valider):
        super().__init__(parent)
        self.title("Panier de cryptos")
        self.geometry("460x560")
        self.configure(fg_color=COULEURS["fond"])
        self.transient(parent)
        self.au_valider = au_valider
        self.cases = {}

        ctk.CTkLabel(self, text="🧺  Entraîner sur plusieurs cryptos",
                     font=ctk.CTkFont(size=18, weight="bold")
                     ).pack(anchor="w", padx=24, pady=(20, 4))
        ctk.CTkLabel(
            self, justify="left", wraplength=400,
            text=("Un seul modèle, entraîné sur toutes les cryptos cochées.\n"
                  "Les blocs d'apprentissage, de validation et de test sont "
                  "coupés aux MÊMES DATES pour chacune : chaque crypto apprend "
                  "sur tout l'historique dont elle dispose avant la frontière, "
                  "et toutes sont jugées sur la même période de marché."),
            font=ctk.CTkFont(size=12), text_color=COULEURS["texte_doux"]
            ).pack(anchor="w", padx=24, pady=(0, 12))

        liste = ctk.CTkScrollableFrame(self, fg_color=COULEURS["panneau"])
        liste.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        for symbole in cryptos:
            variable = ctk.BooleanVar(value=symbole in selection)
            case = ctk.CTkCheckBox(liste, text=symbole, variable=variable,
                                   command=self._maj_resume)
            case.pack(anchor="w", padx=12, pady=4)
            self.cases[symbole] = variable

        self.resume = resume
        self.etiquette = ctk.CTkLabel(self, text="", justify="left", wraplength=400,
                                      font=ctk.CTkFont(size=12),
                                      text_color=COULEURS["texte_doux"])
        self.etiquette.pack(anchor="w", padx=24, pady=(0, 8))

        boutons = ctk.CTkFrame(self, fg_color="transparent")
        boutons.pack(fill="x", padx=24, pady=(0, 20))
        ctk.CTkButton(boutons, text="Tout cocher", width=110,
                      fg_color=COULEURS["carte"],
                      command=lambda: self._tout(True)).pack(side="left")
        ctk.CTkButton(boutons, text="Tout décocher", width=110,
                      fg_color=COULEURS["carte"],
                      command=lambda: self._tout(False)).pack(side="left", padx=8)
        ctk.CTkButton(boutons, text="Valider", width=110,
                      command=self._valider).pack(side="right")

        self._maj_resume()
        self.after(120, self.lift)

    def _tout(self, valeur):
        for variable in self.cases.values():
            variable.set(valeur)
        self._maj_resume()

    def selection(self):
        return [symbole for symbole, variable in self.cases.items() if variable.get()]

    def _maj_resume(self):
        choisies = self.selection()
        if len(choisies) < 2:
            self.etiquette.configure(
                text="Coche au moins deux cryptos (une seule = modèle ordinaire).",
                text_color=COULEURS["orange"])
            return
        self.etiquette.configure(text=self.resume(choisies),
                                 text_color=COULEURS["texte_doux"])

    def _valider(self):
        self.au_valider(self.selection())
        self.destroy()


# ===========================================================================
# FABRIQUES DE WIDGETS
# ===========================================================================
class MixinComposants:
    """
    Constructeurs de widgets partagés par toutes les pages.

    Destiné à être hérité par la fenêtre principale : les méthodes supposent
    l'existence de `self.zones_graphe` (dictionnaire des cadres de graphique).
    """

    # -- structure ---------------------------------------------------------
    def _titre_page(self, parent, titre, sous_titre):
        cadre = ctk.CTkFrame(parent, fg_color="transparent")
        cadre.pack(fill="x", pady=(0, 14))
        ctk.CTkLabel(cadre, text=titre, font=ctk.CTkFont(size=26, weight="bold")
                     ).pack(anchor="w")
        ctk.CTkLabel(cadre, text=sous_titre, text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=13)).pack(anchor="w")
        return cadre

    def _section(self, parent, titre, aide=None):
        """Encadré titré ; retourne le conteneur dans lequel remplir la section."""
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["panneau"], corner_radius=12)
        carte.pack(fill="x", pady=8)

        entete = ctk.CTkFrame(carte, fg_color="transparent")
        entete.pack(anchor="w", fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(entete, text=titre, font=ctk.CTkFont(size=15, weight="bold")
                     ).pack(side="left")
        if aide:
            self._badge_info(entete, aide).pack(side="left", padx=(6, 0))

        corps = ctk.CTkFrame(carte, fg_color="transparent")
        corps.pack(fill="x", padx=18, pady=(0, 16))
        return corps

    def _ligne(self, parent, espace=(12, 0)):
        """Rangée horizontale de contrôles."""
        cadre = ctk.CTkFrame(parent, fg_color="transparent")
        cadre.pack(fill="x", pady=espace)
        return cadre

    # -- contrôles ---------------------------------------------------------
    def _badge_info(self, parent, texte):
        """Pastille « ⓘ » affichant une info-bulle au survol."""
        badge = ctk.CTkLabel(parent, text="ⓘ", width=16,
                             font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=COULEURS["accent_clair"], cursor="question_arrow")
        InfoBulle(badge, texte)
        return badge

    def _etiquette(self, parent, libelle, aide=None):
        """Ligne « libellé + pastille d'aide » au-dessus d'un contrôle."""
        entete = ctk.CTkFrame(parent, fg_color="transparent")
        entete.pack(anchor="w")
        ctk.CTkLabel(entete, text=libelle, font=ctk.CTkFont(size=12),
                     text_color=COULEURS["texte_doux"]).pack(side="left")
        if aide:
            self._badge_info(entete, aide).pack(side="left", padx=(4, 0))
        return entete

    def _champ(self, parent, libelle, valeur="", largeur=140, aide=None):
        """Champ de saisie titré. Retourne (colonne, widget d'entrée)."""
        colonne = ctk.CTkFrame(parent, fg_color="transparent")
        self._etiquette(colonne, libelle, aide)
        entree = ctk.CTkEntry(colonne, width=largeur)
        entree.insert(0, str(valeur))
        entree.pack(anchor="w", pady=(2, 0))
        return colonne, entree

    def _menu(self, parent, libelle, valeurs, largeur=180, aide=None):
        """Menu déroulant titré. Retourne (colonne, variable, widget)."""
        colonne = ctk.CTkFrame(parent, fg_color="transparent")
        self._etiquette(colonne, libelle, aide)
        variable = ctk.StringVar(value=valeurs[0] if valeurs else "")
        menu = ctk.CTkOptionMenu(colonne, values=valeurs or [""],
                                 variable=variable, width=largeur)
        menu.pack(anchor="w", pady=(2, 0))
        return colonne, variable, menu

    def _curseur(self, parent, libelle, mini, maxi, defaut, aide=None,
                 largeur=260, format_valeur="{:.0f}", nb_pas=None,
                 au_changement=None):
        """
        Curseur avec sa valeur affichée en direct.

        Retourne (colonne, curseur) ; la valeur se lit avec `curseur.get()`.
        Plus lisible qu'un champ texte pour une valeur bornée comme l'horizon.

        `au_changement` est rappelé à chaque déplacement, après la mise à jour
        de la valeur affichée. Il sert aux curseurs dont l'effet doit être
        visible AVANT de lancer quoi que ce soit — celui de l'utilité minimale
        annonce ainsi le nombre de features conservées à chaque cran.
        """
        colonne = ctk.CTkFrame(parent, fg_color="transparent")
        entete = self._etiquette(colonne, libelle, aide)
        valeur_affichee = ctk.CTkLabel(entete, text=format_valeur.format(defaut),
                                       font=ctk.CTkFont(size=13, weight="bold"),
                                       text_color=COULEURS["accent_clair"])
        valeur_affichee.pack(side="left", padx=(10, 0))

        # Un pas par unité par défaut ; `nb_pas` sert aux plages décimales
        # (un seuil de 0.50 à 0.95 n'a pas d'unité entière exploitable).
        pas = nb_pas if nb_pas else max(1, int(round(maxi - mini)))
        curseur = ctk.CTkSlider(colonne, from_=mini, to=maxi, width=largeur,
                                number_of_steps=pas)
        curseur.set(defaut)

        def _deplace(valeur):
            valeur_affichee.configure(text=format_valeur.format(valeur))
            if au_changement is not None:
                au_changement(valeur)

        curseur.configure(command=_deplace)
        curseur.pack(anchor="w", pady=(4, 0))
        return colonne, curseur

    def _carte_metrique(self, parent, titre):
        """Grande valeur chiffrée sur fond de carte. Retourne (cadre, label)."""
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["carte"], corner_radius=10)
        ctk.CTkLabel(carte, text=titre, text_color=COULEURS["texte_doux"],
                     font=ctk.CTkFont(size=12)).pack(anchor="w", padx=14, pady=(10, 0))
        valeur = ctk.CTkLabel(carte, text="—", font=ctk.CTkFont(size=22, weight="bold"))
        valeur.pack(anchor="w", padx=14, pady=(0, 12))
        return carte, valeur

    def _bouton_explication(self, parent, texte_bouton, titre, contenu):
        """Bouton ouvrant une fenêtre d'explication détaillée."""
        return ctk.CTkButton(
            parent, text=texte_bouton, height=32,
            fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
            command=lambda: FenetreExplication(self, titre, contenu))

    # -- statistiques commentées ------------------------------------------
    def _ligne_stat(self, parent, libelle, valeur, aide="", commentaire="",
                    niveau="moyen"):
        """Ligne « nom · valeur · commentaire » utilisée par la page Évaluation."""
        carte = ctk.CTkFrame(parent, fg_color=COULEURS["carte"], corner_radius=10)
        carte.pack(fill="x", pady=4)

        haut = ctk.CTkFrame(carte, fg_color="transparent")
        haut.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(haut, text=libelle, font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(side="left")
        if aide:
            self._badge_info(haut, aide).pack(side="left", padx=(6, 0))
        couleur = theme.couleur_niveau(niveau) if commentaire else COULEURS["texte"]
        ctk.CTkLabel(haut, text=str(valeur), font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=couleur).pack(side="right")

        if commentaire:
            bas = ctk.CTkFrame(carte, fg_color="transparent")
            bas.pack(fill="x", padx=14, pady=(0, 10))
            ctk.CTkLabel(bas, text="• " + commentaire, font=ctk.CTkFont(size=12),
                         text_color=theme.couleur_niveau(niveau),
                         wraplength=820, justify="left").pack(anchor="w")
        return carte

    # -- tableaux ----------------------------------------------------------
    def _creer_tableau(self, parent, hauteur=8):
        """Tableau défilant. Retourne (cadre à placer, widget Treeview)."""
        cadre = tk.Frame(parent, bg=COULEURS["carte"], highlightthickness=0)
        tableau = ttk.Treeview(cadre, show="headings", height=hauteur)
        ascenseur = ttk.Scrollbar(cadre, orient="vertical", command=tableau.yview)
        tableau.configure(yscrollcommand=ascenseur.set)
        tableau.pack(side="left", fill="both", expand=True)
        ascenseur.pack(side="right", fill="y")
        return cadre, tableau

    def _remplir_tableau(self, tableau, df, max_lignes=300):
        """Recharge un Treeview à partir d'un DataFrame."""
        tableau.delete(*tableau.get_children())
        if df is None or df.empty:
            tableau["columns"] = ["Info"]
            tableau.heading("Info", text="Info")
            tableau.column("Info", width=300, anchor="center")
            tableau.insert("", "end", values=["Aucune donnée"])
            return

        colonnes = [str(c) for c in df.columns]
        tableau["columns"] = colonnes
        for colonne in colonnes:
            tableau.heading(colonne, text=colonne)
            tableau.column(colonne, width=max(90, min(170, len(colonne) * 11)),
                           anchor="center")
        for _, ligne in df.head(max_lignes).iterrows():
            tableau.insert("", "end", values=[self._formater(v) for v in ligne.values])

    @staticmethod
    def _formater(valeur):
        """Affichage compact d'une valeur de tableau."""
        if isinstance(valeur, (float, np.floating)):
            if np.isnan(valeur):
                return "—"
            return f"{valeur:,.4f}" if abs(valeur) < 1000 else f"{valeur:,.2f}"
        return str(valeur)

    # -- graphiques --------------------------------------------------------
    def _zone_graphe(self, parent, cle, hauteur=380):
        """Réserve un cadre destiné à recevoir une figure matplotlib."""
        cadre = ctk.CTkFrame(parent, fg_color=COULEURS["panneau"],
                             corner_radius=12, height=hauteur)
        self.zones_graphe[cle] = cadre
        return cadre

    @staticmethod
    def _nouvelle_figure(taille=(10, 5)):
        return Figure(figsize=taille, dpi=100, facecolor=COULEURS["panneau"])

    def _afficher_figure(self, cle, figure, barre_outils=True):
        """Remplace le contenu d'une zone de graphique par une nouvelle figure."""
        cadre = self.zones_graphe[cle]
        for enfant in cadre.winfo_children():
            enfant.destroy()

        canevas = FigureCanvasTkAgg(figure, master=cadre)
        canevas.draw()
        if barre_outils:
            # Zoom, déplacement, retour à la vue initiale, export PNG.
            barre = NavigationToolbar2Tk(canevas, cadre, pack_toolbar=False)
            barre.update()
            self._styliser_barre_outils(barre)
            barre.pack(side="top", fill="x", padx=8, pady=(4, 0))
        canevas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    @staticmethod
    def _styliser_barre_outils(barre):
        """Accorde la barre d'outils matplotlib au thème sombre (au mieux)."""
        try:
            barre.configure(background=COULEURS["panneau"])
            for enfant in barre.winfo_children():
                try:
                    enfant.configure(background=COULEURS["panneau"])
                except tk.TclError:
                    pass
        except tk.TclError:
            pass

    # -- utilitaires -------------------------------------------------------
    @staticmethod
    def _valider_date(texte):
        """Retourne la date si elle est au format AAAA-MM-JJ, sinon None."""
        texte = (texte or "").strip()
        try:
            datetime.strptime(texte, "%Y-%m-%d")
            return texte
        except ValueError:
            return None

    @staticmethod
    def _maj_menu(menu, variable, valeurs):
        """Recharge les options d'un menu en préservant la sélection si possible."""
        valeurs = valeurs or ["(aucun)"]
        menu.configure(values=valeurs)
        if variable.get() not in valeurs:
            variable.set(valeurs[0])

    @staticmethod
    def _lire_float(champ, defaut):
        """Lecture tolérante d'un champ numérique (virgule décimale acceptée)."""
        try:
            return float(str(champ.get()).replace(",", "."))
        except (ValueError, AttributeError):
            return defaut

    @staticmethod
    def _lire_int(champ, defaut):
        """Lecture tolérante d'un champ entier."""
        try:
            return int(float(str(champ.get()).replace(",", ".")))
        except (ValueError, AttributeError):
            return defaut
