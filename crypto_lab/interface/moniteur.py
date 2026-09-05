"""
Fenêtre de suivi en direct de l'entraînement.

Elle affiche trois choses pendant qu'un modèle s'entraîne :

  * **la courbe d'apprentissage** — la métrique d'erreur arbre après arbre, sur
    l'apprentissage et sur la validation. C'est le seul graphique qui montre le
    surapprentissage se produire : les deux courbes descendent ensemble, puis
    celle de validation remonte pendant que celle d'apprentissage continue de
    descendre. L'arrêt anticipé coupe exactement là ;
  * **les configurations évaluées** — une barre par jeu d'hyperparamètres, avec
    la dispersion entre blocs de validation croisée. Quand les barres d'erreur
    se recouvrent toutes, c'est que le réglage ne change rien, et c'est une
    information en soi ;
  * **l'utilité des features** — le classement final, dès qu'il est mesuré.

Deux principes de fonctionnement :

  1. **Rien ne bloque.** La fenêtre ne calcule rien : elle consomme des
     événements déposés par le thread de travail dans `suivi.MONITEUR`.
  2. **Le tracé est découplé du flux.** Les événements arrivent par centaines,
     le redessin a lieu au plus `IMAGES_PAR_SECONDE` fois par seconde et
     seulement si quelque chose a changé.
  3. **Le tracé est INCRÉMENTIEL.** Les courbes ne sont pas redessinées, leurs
     données sont remplacées (`set_data`) ; la légende et les libellés d'axes
     ne sont refaits qu'au changement d'ajustement ; le graphique des
     configurations n'est retracé qu'à l'arrivée d'une configuration.

     Ce point n'est pas un détail d'optimisation. Mesuré sur la version
     naïve — `axe.clear()` puis retracé complet à chaque image — une image
     coûtait 407 ms, soit trois fois le budget de 125 ms d'un affichage à 8
     images par seconde : le graphique prenait du retard sur ses propres
     données et volait du temps au thread graphique pendant l'entraînement.
"""

from __future__ import annotations

import time
import tkinter as tk

import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .. import suivi
from . import theme
from .theme import COULEURS

# Cadence NOMINALE de redessin. Au-delà, l'œil ne voit pas la différence et le
# thread graphique commence à se battre avec le reste de l'interface.
IMAGES_PAR_SECONDE = 6

# Part maximale du thread graphique que l'affichage s'autorise à consommer.
# La boucle mesure ce que son dernier dessin a coûté et s'espace d'autant : le
# graphique ralentit au lieu de bégayer, et ne vole jamais plus de ce quota au
# reste de l'interface. Un cinquième, parce que le thread graphique a aussi une
# console à vider toutes les 80 ms et une application à garder réactive
# pendant que le calcul occupe tous les cœurs.
PART_MAX_THREAD = 0.2

# Marge de sécurité sur les limites de l'axe. Les redimensionner à chaque point
# imposerait un redessin complet à chaque image et annulerait tout le bénéfice
# du blitting ; on prend donc de l'avance, et l'axe ne bouge que par paliers.
MARGE_LIMITES = 0.08

# Taille de la figure. 80 dpi plutôt que 100 : 8 % de rendu en moins, mesuré,
# pour une différence invisible sur un graphique de suivi. Le gros du coût de
# rendu ne vient pas des pixels mais du nombre d objets à dessiner (grille,
# graduations, légende), et il est incompressible — d où la cadence adaptative
# ci-dessus, qui est la vraie réponse.
TAILLE_FIGURE = (10, 4.4)
DPI_FIGURE = 80

# Couleur de chaque courbe.
COULEURS_COURBES = {
    "apprentissage": COULEURS["accent_clair"],
    "validation": COULEURS["orange"],
}

# Au-delà, les vieux points sont sous-échantillonnés : tracer 3 000 points quand
# la fenêtre en fait 700 de large ne montre rien de plus et coûte cher.
POINTS_MAX = 900


class FenetreSuivi(ctk.CTkToplevel):
    """Suivi en direct d'un entraînement, avec bouton d'arrêt."""

    def __init__(self, parent, au_arreter=None):
        super().__init__(parent)
        self.title("Suivi de l'entraînement")
        self.geometry("1080x700")
        self.minsize(820, 560)
        self.configure(fg_color=COULEURS["fond"])
        self.au_arreter = au_arreter

        # --- état accumulé --------------------------------------------------
        self.serie = ""                 # ajustement en cours
        self.courbes: dict[str, tuple[list, list]] = {}
        self.configurations: list[dict] = []
        self.utilite: dict = {}

        # Artistes matplotlib réutilisés d'une image à l'autre. Les recréer
        # coûterait plus cher que tout le reste du dessin réuni.
        self._lignes: dict = {}         # nom de courbe -> Line2D
        self._marque_arret = None       # trait vertical du meilleur nombre d'arbres
        self._serie_tracee = None       # série dont les axes sont déjà préparés
        self._etat_barres = None        # empreinte du panneau de droite
        self._fond = None               # fond figé de la figure, pour le blitting
        self._limites = None            # (xmin, xmax, ymin, ymax) du fond capturé
        self._refonte = True            # un redessin complet est-il nécessaire ?

        self._sale_courbe = True
        self._sale_barres = True
        self._vivant = True
        self._cout_image = 0.0          # durée du dernier dessin, en secondes

        self._construire()
        self.protocol("WM_DELETE_WINDOW", self.masquer)
        self.after(120, self.lift)
        self._boucle_dessin()

    # ==================================================================
    # Construction
    # ==================================================================
    def _construire(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        entete = ctk.CTkFrame(self, fg_color="transparent")
        entete.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 4))
        ctk.CTkLabel(entete, text="📺  Entraînement en direct",
                     font=ctk.CTkFont(size=19, weight="bold")).pack(side="left")
        self.chrono = ctk.CTkLabel(entete, text="", font=ctk.CTkFont(size=13),
                                   text_color=COULEURS["texte_doux"])
        self.chrono.pack(side="right")

        barre = ctk.CTkFrame(self, fg_color="transparent")
        barre.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        barre.grid_columnconfigure(0, weight=1)

        self.etape = ctk.CTkLabel(barre, text="En attente…", anchor="w",
                                  font=ctk.CTkFont(size=13),
                                  text_color=COULEURS["accent_clair"])
        self.etape.grid(row=0, column=0, sticky="ew")
        self.progression = ctk.CTkProgressBar(barre, height=12)
        self.progression.set(0)
        self.progression.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.pourcentage = ctk.CTkLabel(barre, text="0 %", width=54,
                                        font=ctk.CTkFont(size=12, weight="bold"))
        self.pourcentage.grid(row=1, column=1, padx=(10, 0))

        cadre = ctk.CTkFrame(self, fg_color=COULEURS["panneau"], corner_radius=12)
        cadre.grid(row=2, column=0, sticky="nsew", padx=20, pady=(4, 8))

        self.figure = Figure(figsize=TAILLE_FIGURE, dpi=DPI_FIGURE,
                             facecolor=COULEURS["panneau"])
        self.axe_courbe = self.figure.add_subplot(121)
        self.axe_barres = self.figure.add_subplot(122)
        self.figure.subplots_adjust(left=0.09, right=0.97, top=0.90,
                                    bottom=0.14, wspace=0.28)
        for axe in (self.axe_courbe, self.axe_barres):
            theme.styliser_axes(axe)

        self.canevas = FigureCanvasTkAgg(self.figure, master=cadre)
        self.canevas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self.resume = ctk.CTkLabel(
            self, text="", justify="left", anchor="w", wraplength=1020,
            font=ctk.CTkFont(size=12), text_color=COULEURS["texte_doux"])
        self.resume.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 6))

        boutons = ctk.CTkFrame(self, fg_color="transparent")
        boutons.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 16))
        self.bouton_arret = ctk.CTkButton(
            boutons, text="⏹ Arrêter l'entraînement", width=210, height=36,
            fg_color=COULEURS["rouge"], hover_color="#c0392b",
            command=self._arreter)
        self.bouton_arret.pack(side="left")
        ctk.CTkLabel(boutons, text="Fermer cette fenêtre n'arrête rien : "
                                   "l'entraînement continue en arrière-plan.",
                     font=ctk.CTkFont(size=11),
                     text_color=COULEURS["texte_doux"]).pack(side="left", padx=12)
        ctk.CTkButton(boutons, text="Fermer", width=100, height=36,
                      fg_color=COULEURS["carte"], hover_color=COULEURS["accent"],
                      command=self.masquer).pack(side="right")

    # ==================================================================
    # Cycle de vie
    # ==================================================================
    def masquer(self):
        """Cache la fenêtre sans détruire son contenu ni arrêter le calcul."""
        self.withdraw()

    def montrer(self):
        self.deiconify()
        self.lift()
        self._sale_courbe = True
        self._sale_barres = True
        self._refonte = True            # le fond capturé n'est plus valable

    def reinitialiser(self, libelle: str = ""):
        """Vide les courbes au démarrage d'un nouvel entraînement."""
        self.serie = ""
        self.courbes = {}
        self.configurations = []
        self.utilite = {}
        self.progression.set(0)
        self.pourcentage.configure(text="0 %")
        self.etape.configure(text=libelle or "Démarrage…")
        self.resume.configure(text="")
        self.bouton_arret.configure(state="normal",
                                    text="⏹ Arrêter l'entraînement")
        self._lignes = {}
        self._marque_arret = None
        self._serie_tracee = None
        self._etat_barres = None
        self._fond = None
        self._limites = None
        self._refonte = True
        self._sale_courbe = True
        self._sale_barres = True

    def _arreter(self):
        suivi.MONITEUR.demander_arret()
        self.bouton_arret.configure(state="disabled", text="⏹ Arrêt demandé…")
        self.etape.configure(text="Arrêt demandé — fin de l'étape en cours…",
                             text_color=COULEURS["orange"])
        if self.au_arreter is not None:
            self.au_arreter()

    def destroy(self):                                  # noqa: D102
        self._vivant = False
        super().destroy()

    # ==================================================================
    # Consommation des événements
    # ==================================================================
    def traiter(self, evenements: list[dict]) -> None:
        """
        Range un lot d'événements. Appelée par le thread graphique seulement.

        Aucun tracé ici : on accumule, on marque « à redessiner », et la boucle
        de dessin s'en occupe à son rythme. C'est toute la différence entre une
        interface fluide et une interface qui redessine 500 fois par seconde.
        """
        for evenement in evenements:
            type_ = evenement.get("type")
            if type_ == suivi.ETAPE:
                self._sur_etape(evenement)
            elif type_ == suivi.COURBE:
                self._sur_courbe(evenement)
            elif type_ == suivi.CONFIGURATION:
                self.configurations.append(evenement)
                self._sale_barres = True
                self._refonte = True
            elif type_ == suivi.IMPORTANCE:
                self.utilite = evenement.get("valeurs") or {}
                self._sale_barres = True
                self._refonte = True
            elif type_ == suivi.FIN:
                self._sur_fin(evenement)

    def _sur_etape(self, evenement):
        part = float(evenement.get("part", 0.0))
        self.progression.set(part)
        self.pourcentage.configure(text=f"{part:.0%}")
        self.etape.configure(text=evenement.get("libelle", ""),
                             text_color=COULEURS["accent_clair"])

    def _sur_courbe(self, evenement):
        serie = evenement.get("serie", "")
        if serie != self.serie:
            # Nouvel ajustement : on repart d'une courbe vierge, sinon les
            # itérations de deux modèles différents se mélangeraient.
            self.serie = serie
            self.courbes = {}
            self._lignes = {}
            self._serie_tracee = None
            self._refonte = True
        for nom, valeur in (evenement.get("valeurs") or {}).items():
            xs, ys = self.courbes.setdefault(nom, ([], []))
            xs.append(int(evenement.get("iteration", len(xs))))
            ys.append(float(valeur))
        self._sale_courbe = True

    def _sur_fin(self, evenement):
        self.bouton_arret.configure(state="disabled", text="⏹ Terminé")
        duree = float(evenement.get("duree", 0.0))
        if evenement.get("interrompu"):
            self.etape.configure(text=f"Interrompu après {duree:,.0f} s.",
                                 text_color=COULEURS["orange"])
        else:
            self.etape.configure(text=f"Terminé en {duree:,.0f} s.",
                                 text_color=COULEURS["vert"])
            self.progression.set(1.0)
            self.pourcentage.configure(text="100 %")
        self._sale_courbe = True
        self._sale_barres = True

    # ==================================================================
    # Dessin
    # ==================================================================
    def _boucle_dessin(self):
        """
        Redessine au plus `IMAGES_PAR_SECONDE` fois par seconde, si besoin.

        Ne redessine pas quand la fenêtre est masquée : les données continuent
        d'être accumulées, mais tracer ce que personne ne regarde volerait du
        temps au thread graphique pendant que le calcul tourne.
        """
        if not self._vivant:
            return
        depart = time.perf_counter()
        try:
            if (self._sale_courbe or self._sale_barres) and self.winfo_viewable():
                self._redessiner()
            if suivi.MONITEUR.actif:
                self.chrono.configure(text=f"⏱ {suivi.MONITEUR.duree:,.0f} s")
        except tk.TclError:
            # La fenêtre (ou son parent) a été détruite : on arrête la boucle
            # au lieu de reprogrammer un rappel sur un widget mort.
            self._vivant = False
            return
        except Exception:                               # noqa: BLE001
            # Un souci d'affichage ne doit jamais interrompre la boucle ni,
            # surtout, le calcul qui tourne derrière.
            self._sale_courbe = self._sale_barres = False

        self._cout_image = time.perf_counter() - depart
        try:
            self.after(self.prochain_delai(), self._boucle_dessin)
        except tk.TclError:                             # pragma: no cover
            self._vivant = False

    def prochain_delai(self) -> int:
        """
        Millisecondes avant le prochain redessin.

        Jamais moins que la cadence nominale, et jamais assez souvent pour que
        l'affichage dépasse `PART_MAX_THREAD` du thread graphique. C'est cette
        seconde borne qui rend la fluidité indépendante de la machine.
        """
        nominal = 1000.0 / IMAGES_PAR_SECONDE
        rembourse = self._cout_image * 1000.0 / PART_MAX_THREAD
        return int(max(nominal, rembourse))

    def _redessiner(self):
        """
        Met à jour l'affichage au moindre coût possible.

        Deux régimes :

          RAFRAÎCHISSEMENT (le cas courant) — on repeint le fond mémorisé puis
          les seules courbes par-dessus. Quelques millisecondes.

          REFONTE (rare) — nouvel ajustement, nouvelle courbe, nouvelle
          configuration, ou données sorties des limites : la figure entière est
          redessinée et son fond recapturé. 160 à 250 ms, d'où l'insistance à
          ne le faire que quand c'est vraiment nécessaire.
        """
        if self._sale_courbe:
            self._sale_courbe = False
            self._tracer_courbe()
        if self._sale_barres:
            self._sale_barres = False
            self._tracer_barres()
        self._maj_resume()

        if self._refonte or self._fond is None:
            self._refonte = False
            self.canevas.draw()
            self._capturer_fond()
        else:
            self._rafraichir_courbes()

    def _capturer_fond(self):
        """Mémorise la figure telle quelle, pour repeindre par-dessus."""
        try:
            self._fond = self.canevas.copy_from_bbox(self.figure.bbox)
            self._limites = (self.axe_courbe.get_xlim()
                             + self.axe_courbe.get_ylim())
        except Exception:                               # noqa: BLE001
            # Backend sans support du blitting : on retombe sur le redessin
            # complet, plus lent mais toujours correct.
            self._fond = None

    def _rafraichir_courbes(self):
        """Repeint le fond mémorisé, puis les courbes seules par-dessus."""
        try:
            self.canevas.restore_region(self._fond)
            for ligne in self._lignes.values():
                self.axe_courbe.draw_artist(ligne)
            if self._marque_arret is not None:
                self.axe_courbe.draw_artist(self._marque_arret)
            self.canevas.blit(self.axe_courbe.bbox)
        except Exception:                               # noqa: BLE001
            self._refonte = True

    def _limites_depassees(self) -> bool:
        """
        Les données sortent-elles du cadre capturé ?

        Tant qu'elles tiennent dedans, le fond reste valable et les courbes
        peuvent être repeintes seules. Dès qu'elles en sortent, il faut
        redimensionner les axes — donc tout redessiner.
        """
        if self._limites is None:
            return True
        x_min, x_max, y_min, y_max = self._limites
        for xs, ys in self.courbes.values():
            if not xs:
                continue
            if xs[-1] > x_max or xs[0] < x_min:
                return True
            if max(ys) > y_max or min(ys) < y_min:
                return True
        return False

    @staticmethod
    def _alleger(xs, ys):
        """Sous-échantillonne une courbe trop dense pour la largeur d'écran."""
        if len(xs) <= POINTS_MAX:
            return xs, ys
        pas = len(xs) // POINTS_MAX + 1
        return xs[::pas] + xs[-1:], ys[::pas] + ys[-1:]

    def _preparer_axe_courbe(self):
        """Titres, libellés et légende : refaits seulement au changement d'ajustement."""
        axe = self.axe_courbe
        axe.clear()
        theme.styliser_axes(axe)
        axe.set_title(f"Courbe d'apprentissage — {self.serie or '…'}",
                      fontsize=11, color="white")
        axe.set_xlabel("arbres construits", fontsize=9)
        axe.set_ylabel("erreur (logloss)", fontsize=9)
        self._lignes = {}
        self._marque_arret = None
        self._serie_tracee = self.serie

    def _tracer_courbe(self):
        if not self.courbes:
            if self._serie_tracee is not None or not self.axe_courbe.texts:
                self._preparer_axe_courbe()
                self.axe_courbe.text(
                    0.5, 0.5, "En attente du premier ajustement…",
                    ha="center", va="center", color=COULEURS["texte_doux"],
                    transform=self.axe_courbe.transAxes, fontsize=10)
                self._serie_tracee = None
            return

        axe = self.axe_courbe
        if self._serie_tracee != self.serie:
            self._preparer_axe_courbe()

        nouvelle_ligne = False
        for nom, (xs, ys) in sorted(self.courbes.items()):
            if not xs:
                continue
            x_traces, y_traces = self._alleger(xs, ys)
            ligne = self._lignes.get(nom)
            if ligne is None:
                # Créée une seule fois, puis alimentée par `set_data`.
                ligne, = axe.plot(x_traces, y_traces, lw=1.6, label=nom,
                                  color=COULEURS_COURBES.get(nom,
                                                             COULEURS["texte_doux"]))
                self._lignes[nom] = ligne
                nouvelle_ligne = True
            else:
                ligne.set_data(x_traces, y_traces)

        # Le minimum de la validation, c'est l'endroit où l'early stopping
        # coupera : le montrer explicitement évite d'avoir à le deviner.
        validation = self.courbes.get("validation")
        if validation and validation[1]:
            ys = validation[1]
            meilleur = validation[0][min(range(len(ys)), key=lambda i: ys[i])]
            if self._marque_arret is None:
                self._marque_arret = axe.axvline(
                    meilleur, color=COULEURS["vert"], ls="--", lw=1.0, alpha=0.7,
                    label="meilleur")
                nouvelle_ligne = True
            else:
                self._marque_arret.set_xdata([meilleur, meilleur])

        if nouvelle_ligne:
            self._refonte = True
            # Position FIXE : loc="best" relance une minimisation de
            # recouvrement à chaque rendu, ce qui coûtait à lui seul plus cher
            # que le tracé des courbes.
            theme.legende(axe, fontsize=8, loc="upper right")

        # Les axes ne sont redimensionnés que lorsque les données sortent du
        # cadre, et alors avec de la marge : chaque redimensionnement impose un
        # redessin complet, donc on prend de l'avance pour en avoir peu.
        if self._limites_depassees():
            # `set_xlim` / `set_ylim` plus bas desactivent l'auto-echelle : sans
            # la reactiver, `autoscale_view` ne ferait plus rien des la deuxieme
            # fois et le cadre resterait fige sur des donnees sorties depuis
            # longtemps.
            axe.set_autoscale_on(True)
            axe.relim()
            axe.autoscale_view()
            x_min, x_max = axe.get_xlim()
            y_min, y_max = axe.get_ylim()
            largeur = max(x_max - x_min, 1e-9) * MARGE_LIMITES
            hauteur = max(y_max - y_min, 1e-12) * MARGE_LIMITES
            axe.set_xlim(x_min - largeur, x_max + largeur * 4)
            axe.set_ylim(y_min - hauteur, y_max + hauteur)
            self._refonte = True

    def _tracer_barres(self):
        """
        Panneau de droite : configurations évaluées, puis utilité des features.

        Contrairement à la courbe, ce panneau change rarement — une fois par
        configuration terminée. Il est donc entièrement retracé, mais seulement
        quand son contenu a réellement bougé.
        """
        empreinte = (len(self.configurations), len(self.utilite))
        if empreinte == self._etat_barres:
            return
        self._etat_barres = empreinte

        axe = self.axe_barres
        axe.clear()
        theme.styliser_axes(axe)

        # Tant qu'aucune utilité n'est mesurée, la place sert aux configurations.
        if self.utilite:
            self._tracer_utilite(axe)
            return

        axe.set_title("Configurations évaluées", fontsize=11, color="white")
        axe.set_xlabel("AUC (validation croisée)", fontsize=9)

        if not self.configurations:
            axe.text(0.5, 0.5, "Aucune configuration terminée.",
                     ha="center", va="center", color=COULEURS["texte_doux"],
                     transform=axe.transAxes, fontsize=10)
            return

        # Les vingt dernières suffisent : au-delà, les barres sont illisibles.
        lot = self.configurations[-20:]
        positions = list(range(len(lot)))
        valeurs = [c["auc"] for c in lot]
        ecarts = [c.get("ecart", 0.0) for c in lot]
        meilleure = max(valeurs)
        couleurs = [COULEURS["vert"] if v >= meilleure - 1e-12
                    else COULEURS["accent"] for v in valeurs]

        axe.barh(positions, valeurs, xerr=ecarts, color=couleurs,
                 height=0.7, error_kw={"ecolor": "#888888", "lw": 1})
        axe.set_yticks(positions)
        axe.set_yticklabels([f"n°{c['numero']}" for c in lot], fontsize=8)
        axe.axvline(0.5, color=COULEURS["rouge"], ls=":", lw=1.2, alpha=0.8)
        bas = min(0.495, min(v - e for v, e in zip(valeurs, ecarts)) - 0.002)
        axe.set_xlim(bas, max(v + e for v, e in zip(valeurs, ecarts)) + 0.003)
        axe.invert_yaxis()

    def _tracer_utilite(self, axe):
        """Classement final des features, quand il est disponible."""
        axe.set_title("Utilité des features (validation)", fontsize=11,
                      color="white")
        axe.set_xlabel("perte d'AUC si on la mélange", fontsize=9)

        classement = sorted(self.utilite.items(), key=lambda paire: paire[1])[-14:]
        noms = [nom for nom, _ in classement]
        valeurs = [valeur for _, valeur in classement]
        couleurs = [COULEURS["vert"] if v > 0 else COULEURS["rouge"]
                    for v in valeurs]
        axe.barh(range(len(noms)), valeurs, color=couleurs, height=0.72)
        axe.set_yticks(range(len(noms)))
        axe.set_yticklabels(noms, fontsize=8)
        axe.axvline(0, color="#888888", lw=1)

    def _maj_resume(self):
        morceaux = []
        if self.configurations:
            meilleure = max(self.configurations, key=lambda c: c["auc"])
            derniere = self.configurations[-1]
            morceaux.append(
                f"{len(self.configurations)}/{derniere.get('total', '?')} "
                f"configurations · meilleure : n°{meilleure['numero']} à "
                f"AUC {meilleure['auc']:.4f}"
                + (f" ± {meilleure['ecart']:.4f}" if meilleure.get("ecart") else ""))
        if self.utilite:
            classement = sorted(self.utilite.items(), key=lambda p: p[1],
                                reverse=True)
            utiles = sum(1 for _, v in classement if v > 0)
            morceaux.append(
                f"{utiles} feature(s) utiles sur {len(classement)} · "
                f"en tête : "
                + ", ".join(f"{nom} {valeur:+.4f}"
                            for nom, valeur in classement[:3]))
        self.resume.configure(text="   —   ".join(morceaux))
