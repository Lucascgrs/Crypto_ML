"""
Entraînement et prédiction, en ligne de commande.

    python Predict.py BTC_1h                          # horizon et modèle par défaut
    python Predict.py BTC_1h --horizon 6 --modele LightGBM
    python Predict.py BTC_1h --horizon 24 --seuil 0.65
    python Predict.py BTC_1h --predire-seulement       # réutilise le modèle existant

Le modèle apprend à répondre : « dans X périodes, le prix sera-t-il plus haut ou
plus bas ? », et fournit un niveau de confiance exploitable dans les deux sens.

Le code est dans `crypto_lab/modele.py`. Depuis l'interface : onglet
« 3 · Prédiction ».
"""

import argparse
import sys

from crypto_lab import config, modele, stockage


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Entraîne un modèle de direction et génère les signaux.")
    parseur.add_argument("fichier", help="Crypto analysée (ex : BTC_1h).")
    parseur.add_argument("--horizon", type=int, default=config.HORIZON_DEFAUT,
                         help=f"Périodes à prédire, 1 à {config.HORIZON_MAX} "
                              f"(défaut : {config.HORIZON_DEFAUT}).")
    parseur.add_argument("--modele", default=config.MODELE_DEFAUT,
                         choices=modele.modeles_disponibles(),
                         help=f"Algorithme (défaut : {config.MODELE_DEFAUT}).")
    parseur.add_argument("--seuil", type=float, default=config.SEUIL_DEFAUT,
                         help=f"Seuil de confiance (défaut : {config.SEUIL_DEFAUT}).")
    parseur.add_argument("--predire-seulement", action="store_true",
                         help="Ne pas réentraîner : réutiliser le modèle existant.")
    return parseur.parse_args()


def main():
    arguments = analyser_arguments()
    config.preparer_dossiers()
    symbole, intervalle = stockage.separer_cle(arguments.fichier)

    if not arguments.predire_seulement:
        modele.entrainer(symbole, intervalle, arguments.horizon, arguments.modele)

    modele.predire(symbole, intervalle, arguments.horizon, arguments.seuil)
    return 0


if __name__ == "__main__":
    sys.exit(main())
