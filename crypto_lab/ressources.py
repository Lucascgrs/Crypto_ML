"""
Dimensionnement automatique des ressources machine (RAM / CPU).

Objectif : ne pas laisser la moitié de la RAM inutilisée pendant l'entraînement.
La mémoire libre est convertie en VITESSE et en PRÉCISION, pas gaspillée :

  1. `max_bin` élevé   -> les modèles à arbres construisent des histogrammes plus
                          fins (donc des coupures plus précises). Le coût est
                          uniquement mémoire, d'où l'intérêt d'en profiter.
  2. Parallélisme      -> chaque modèle utilise tous les cœurs. Le calcul
                          réellement séquentiel (importance par permutation)
                          est réparti sur autant de processus que la RAM le
                          permet, chacun recopiant le jeu de données.
  3. Cache en mémoire  -> les fichiers analysés restent chargés d'un écran à
                          l'autre au lieu d'être relus depuis le disque : c'est
                          de loin le gain le plus visible (20 s -> 0.1 s).

Tout est calculé à partir de la RAM RÉELLEMENT disponible au moment de l'appel,
et plafonné pour laisser respirer le système.

À noter : avec 8 indicateurs seulement, le jeu d'entraînement pèse quelques
mégaoctets. Il reste donc beaucoup de RAM libre pendant l'entraînement, et
c'est normal — la remplir artificiellement ne rendrait rien plus rapide. Les
réglages ci-dessus servent à ne jamais être LIMITÉ par la mémoire, pas à la
consommer pour la consommer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_OK = True
except ImportError:                                    # pragma: no cover
    PSUTIL_OK = False


# Part de la RAM disponible qu'on s'autorise à consommer.
FRACTION_UTILISABLE = 0.75

# Marge de sécurité (Go) toujours laissée au système d'exploitation.
MARGE_SYSTEME_GO = 2.0


@dataclass
class Ressources:
    """Photographie des ressources machine + réglages qui en découlent."""

    coeurs: int             # cœurs logiques disponibles
    ram_totale_go: float    # RAM physique de la machine
    ram_dispo_go: float     # RAM réellement libre maintenant
    budget_go: float        # ce qu'on s'autorise à utiliser
    n_jobs: int             # threads pour l'entraînement d'un modèle
    n_jobs_recherche: int   # processus parallèles pour les calculs séquentiels
    max_bin: int            # finesse des histogrammes (coût = mémoire)
    pool_histogramme_mo: int  # budget histogramme LightGBM, en Mo

    def resume(self) -> str:
        """Ligne de rapport affichée dans la console au démarrage d'un entraînement."""
        return (
            f"🧠 Ressources — {self.coeurs} cœurs | RAM {self.ram_dispo_go:.1f} Go libres "
            f"sur {self.ram_totale_go:.1f} Go | budget {self.budget_go:.1f} Go\n"
            f"   Réglages : n_jobs={self.n_jobs}, max_bin={self.max_bin} "
            f"(histogrammes plus fins = RAM convertie en précision)"
        )


def _ram_go() -> tuple[float, float]:
    """(RAM totale, RAM disponible) en Go. Valeurs prudentes si psutil manque."""
    if PSUTIL_OK:
        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    return 8.0, 4.0


def detecter(octets_dataset: int = 0) -> Ressources:
    """
    Calcule les réglages adaptés à la machine.

    `octets_dataset` : empreinte mémoire approximative du jeu d'entraînement.
    Elle sert à limiter le nombre de processus parallèles : chacun recopie le
    jeu de données, donc en lancer 16 sur un gros dataset ferait swapper la
    machine.
    """
    coeurs = os.cpu_count() or 4
    ram_totale, ram_dispo = _ram_go()

    budget = max(0.5, (ram_dispo - MARGE_SYSTEME_GO) * FRACTION_UTILISABLE)

    # --- Finesse des histogrammes : le principal levier « RAM -> qualité » ---
    if budget >= 8:
        max_bin = 1024
    elif budget >= 4:
        max_bin = 512
    else:
        max_bin = 256

    # --- Parallélisme des calculs multi-processus ---
    # Coût mémoire ~3× le dataset par processus (copie + structures internes).
    go_dataset = max(octets_dataset, 1) / 1e9
    cout_par_processus = go_dataset * 3 + 0.25
    n_recherche = int(budget // cout_par_processus) if cout_par_processus > 0 else coeurs
    n_jobs_recherche = max(1, min(coeurs, n_recherche))

    # LightGBM : budget histogramme explicite (Mo), plafonné à 2 Go.
    pool_mo = int(min(2048, budget * 1024 * 0.25))

    return Ressources(
        coeurs=coeurs,
        ram_totale_go=ram_totale,
        ram_dispo_go=ram_dispo,
        budget_go=budget,
        n_jobs=coeurs,
        n_jobs_recherche=n_jobs_recherche,
        max_bin=max_bin,
        pool_histogramme_mo=max(256, pool_mo),
    )


def memoire_processus_go() -> float:
    """RAM actuellement consommée par le processus Python (0 si psutil absent)."""
    if not PSUTIL_OK:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1e9
