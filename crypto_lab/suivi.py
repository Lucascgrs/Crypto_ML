"""
Suivi en direct des traitements longs, et arrêt propre.

Ce module est la boîte aux lettres entre le code métier (qui calcule, dans un
thread de travail) et l'interface (qui affiche, dans le thread graphique). Il
ne connaît ni Tkinter ni matplotlib : `modele.py` publie des événements,
quelqu'un les consomme ou personne ne les consomme, ça ne change rien au
calcul.

Trois besoins, un seul objet :

  1. **Savoir où on en est.** `etape()` publie une progression globale, ce qui
     permet une vraie barre de progression au lieu d'un défilement décoratif.
  2. **Voir le modèle apprendre.** `courbe()` publie un point de la courbe
     d'apprentissage à chaque poignée d'arbres. C'est ce qui rend visible la
     seule chose qui compte pendant un boosting : le moment où la validation
     cesse de progresser alors que l'apprentissage continue de descendre.
  3. **Pouvoir arrêter.** `demander_arret()` lève un drapeau ; le code métier
     le consulte entre deux étapes (`verifier()`) et les callbacks de XGBoost
     ou LightGBM le consultent à chaque arbre. Aucun thread n'est tué de
     force : le calcul s'arrête de lui-même, à un endroit sûr.

La file a une taille maximale et les événements les plus anciens sont jetés
quand personne ne consomme : un entraînement lancé en ligne de commande, sans
interface, ne doit pas se mettre à consommer de la mémoire pour rien.
"""

from __future__ import annotations

import queue
import threading
import time

# Types d'événements publiés. Ce sont les seuls que l'interface a besoin de
# connaître ; en ajouter un ne casse rien, un consommateur ignore ce qu'il ne
# sait pas afficher.
ETAPE = "etape"                # progression globale
COURBE = "courbe"              # un point de la courbe d'apprentissage
CONFIGURATION = "configuration"  # une configuration d'hyperparamètres évaluée
IMPORTANCE = "importance"      # utilité mesurée des features
FIN = "fin"                    # le traitement est terminé (ou interrompu)

# Au-delà, les plus vieux événements sont jetés : mieux vaut une courbe avec un
# trou qu'une file qui gonfle indéfiniment.
TAILLE_FILE = 4000

# Intervalle minimum entre deux points de courbe, en nombre d'arbres. Publier à
# chaque arbre saturerait la file sans rien apporter à l'œil.
PAS_COURBE = 5


class Annulation(Exception):
    """Levée quand l'utilisateur a demandé l'arrêt du traitement en cours."""


class Moniteur:
    """
    Canal de communication à sens unique, sûr entre threads.

    Un seul producteur (le thread de travail) et un seul consommateur (le
    thread graphique) en pratique, mais rien n'en dépend : `queue.Queue` et
    `threading.Event` font tout le travail.
    """

    def __init__(self, taille_max: int = TAILLE_FILE):
        self.file: queue.Queue = queue.Queue(maxsize=taille_max)
        self._arret = threading.Event()
        self._actif = threading.Event()
        self._libelle = ""
        self._part = 0.0
        self._debut = 0.0

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def demarrer(self, libelle: str = "") -> "Moniteur":
        """Vide la file, efface le drapeau d'arrêt, arme le chronomètre."""
        self.vider()
        self._arret.clear()
        self._actif.set()
        self._libelle = libelle
        self._part = 0.0
        self._debut = time.monotonic()
        return self

    def terminer(self, interrompu: bool = False) -> None:
        self._actif.clear()
        self.publier(FIN, interrompu=bool(interrompu),
                     duree=time.monotonic() - self._debut if self._debut else 0.0)

    @property
    def actif(self) -> bool:
        return self._actif.is_set()

    @property
    def duree(self) -> float:
        """Secondes écoulées depuis `demarrer()`."""
        return time.monotonic() - self._debut if self._debut else 0.0

    def vider(self) -> None:
        try:
            while True:
                self.file.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # Arrêt coopératif
    # ------------------------------------------------------------------
    def demander_arret(self) -> None:
        """Demande l'arrêt. Le calcul s'arrêtera au prochain point de contrôle."""
        self._arret.set()

    def arret_demande(self) -> bool:
        return self._arret.is_set()

    def verifier(self) -> None:
        """Point de contrôle : lève `Annulation` si l'arrêt a été demandé."""
        if self._arret.is_set():
            raise Annulation("Entraînement interrompu à la demande.")

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------
    def publier(self, type_evenement: str, **donnees) -> None:
        """Dépose un événement. Jette le plus ancien si la file est pleine."""
        evenement = {"type": type_evenement, "instant": time.monotonic(), **donnees}
        try:
            self.file.put_nowait(evenement)
        except queue.Full:
            try:
                self.file.get_nowait()
                self.file.put_nowait(evenement)
            except (queue.Empty, queue.Full):        # pragma: no cover
                pass

    def etape(self, libelle: str, part: float | None = None,
              detail: str = "") -> None:
        """
        Progression globale. `part` va de 0 à 1 ; None conserve la précédente.

        C'est ce qui permet une barre déterminée : sans elle, l'interface ne
        peut afficher qu'un défilement qui ne dit rien du temps restant.
        """
        if part is not None:
            self._part = float(min(max(part, 0.0), 1.0))
        self._libelle = libelle
        self.publier(ETAPE, libelle=libelle, part=self._part, detail=detail)

    def courbe(self, iteration: int, valeurs: dict, serie: str = "") -> None:
        """
        Un point de la courbe d'apprentissage.

        `valeurs` associe un nom de jeu ('apprentissage', 'validation') à la
        valeur de la métrique d'arrêt. `serie` identifie l'ajustement en cours
        (« config 2 · bloc 3 ») : changer de série remet la courbe à zéro côté
        interface.
        """
        self.publier(COURBE, iteration=int(iteration),
                     valeurs={str(k): float(v) for k, v in valeurs.items()},
                     serie=str(serie))

    def configuration(self, numero: int, total: int, params: dict,
                      auc: float, ecart: float = 0.0,
                      aucs: list | None = None) -> None:
        """Une configuration d'hyperparamètres vient d'être évaluée."""
        self.publier(CONFIGURATION, numero=int(numero), total=int(total),
                     params={str(k): v for k, v in params.items()},
                     auc=float(auc), ecart=float(ecart),
                     aucs=[float(a) for a in (aucs or [])])

    def importance(self, valeurs: dict, bloc: str = "validation") -> None:
        """Utilité mesurée de chaque feature (perte d'AUC par permutation)."""
        self.publier(IMPORTANCE, bloc=str(bloc),
                     valeurs={str(k): float(v) for k, v in valeurs.items()})

    # ------------------------------------------------------------------
    # Consommation
    # ------------------------------------------------------------------
    def evenements(self, maximum: int = 500) -> list[dict]:
        """Retire et retourne jusqu'à `maximum` événements en attente."""
        lot = []
        try:
            for _ in range(maximum):
                lot.append(self.file.get_nowait())
        except queue.Empty:
            pass
        return lot


# Instance partagée. L'application n'exécute qu'une tâche longue à la fois
# (voir `interface/app.py::executer`), un seul moniteur suffit donc — et cela
# évite de faire transiter un objet à travers toute la pile d'appels.
MONITEUR = Moniteur()


# ---------------------------------------------------------------------------
# Raccourcis, pour que le code métier reste lisible
# ---------------------------------------------------------------------------
def etape(libelle: str, part: float | None = None, detail: str = "") -> None:
    MONITEUR.etape(libelle, part, detail)


def verifier() -> None:
    MONITEUR.verifier()


def arret_demande() -> bool:
    return MONITEUR.arret_demande()
