"""
Amplitude et espérance de gain, en ligne de commande.

    python Amplitude.py BTC_1h --horizon 3               # les deux régressions
    python Amplitude.py BTC_1h --horizon 3 --esperance   # + le score de décision
    python Amplitude.py BTC_1h --cible volatilite        # une seule des deux
    python Amplitude.py BTC_1h --esperance-seulement     # réutilise l'existant
    python Amplitude.py BTC_1h --esperance --objectif direction_nette

Le modèle de direction répond à « ça monte ou ça descend ? ». Il ne dit rien de
l'AMPLEUR du mouvement — or un signal juste à 55 % sur un mouvement de 3 % ne
vaut pas un signal juste à 60 % sur 0.2 %. Ce script entraîne les deux modèles
qui manquaient, puis les combine avec la direction :

    espérance = (2 × P(hausse) − 1) × amplitude attendue − frais

Le code est dans `crypto_lab/amplitude.py`. Depuis l'interface : onglet
« 4 · Amplitude ».
"""

import argparse
import sys

from crypto_lab import amplitude, cibles, config, stockage


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Entraîne les régressions d'amplitude et calcule l'espérance de gain.")
    parseur.add_argument("fichier", help="Crypto analysée (ex : BTC_1h).")
    parseur.add_argument("--horizon", type=int, default=config.HORIZON_DEFAUT,
                         help=f"Périodes à prédire, 1 à {config.HORIZON_MAX} "
                              f"(défaut : {config.HORIZON_DEFAUT}).")
    parseur.add_argument("--modele", default="LightGBM",
                         choices=amplitude.modeles_disponibles(),
                         help="Algorithme de régression (défaut : LightGBM). Seuls "
                              "les modèles sachant faire de la régression quantile "
                              "figurent ici.")
    parseur.add_argument("--cible", default="tout",
                         choices=["tout"] + list(config.CIBLES_REGRESSION),
                         help="volatilite = l'ampleur attendue ; amplitude = les "
                              "quantiles 10/50/90 ; tout = les deux (défaut).")
    parseur.add_argument("--seuil", type=float, default=config.SEUIL_DEFAUT,
                         help=f"Seuil de confiance du modèle de direction "
                              f"(défaut : {config.SEUIL_DEFAUT}).")
    parseur.add_argument("--objectif", default=None,
                         choices=list(cibles.TACHES),
                         help="Objectif du modèle de direction à combiner à "
                              "l'amplitude. Par défaut : le seul entraîné à cet "
                              "horizon, s'il n'y en a qu'un.")
    parseur.add_argument("--esperance", action="store_true",
                         help="Calculer aussi l'espérance de gain après entraînement.")
    parseur.add_argument("--esperance-seulement", action="store_true",
                         help="Ne rien réentraîner : combiner les modèles existants.")
    return parseur.parse_args()


def main():
    arguments = analyser_arguments()
    config.preparer_dossiers()
    symbole, intervalle = stockage.separer_cle(arguments.fichier)

    if not arguments.esperance_seulement:
        cibles_a_entrainer = (list(config.CIBLES_REGRESSION)
                              if arguments.cible == "tout" else [arguments.cible])
        for cible in cibles_a_entrainer:
            amplitude.entrainer(symbole, intervalle, arguments.horizon,
                                arguments.modele, cible)

    if arguments.esperance or arguments.esperance_seulement:
        amplitude.esperance(symbole, intervalle, arguments.horizon, arguments.seuil,
                            tache=arguments.objectif)
    return 0


if __name__ == "__main__":
    sys.exit(main())
