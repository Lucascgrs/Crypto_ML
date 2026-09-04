"""
Calcul des indicateurs et des variations, en ligne de commande.

    python CryptoAnalysis.py                 # analyse tout le dossier data_crypto
    python CryptoAnalysis.py BTC_1h ETH_1h   # analyse des fichiers précis
    python CryptoAnalysis.py BTC_1h --sans-contexte   # les 8 indicateurs seuls

Produit, pour chaque crypto, un fichier `analysis_crypto/{SYMBOLE}_{INTERVALLE}_analyzed.xlsx`
contenant les prix, les 8 indicateurs, le contexte disponible (les 8 mêmes
indicateurs sur l'intervalle supérieur, funding rate, open interest) et les
24 colonnes variation_x.

Le code est dans `crypto_lab/indicateurs.py`. Depuis l'interface : onglet
« 2 · Analyse ».
"""

import argparse
import sys

from crypto_lab import config, indicateurs, stockage


def main():
    parseur = argparse.ArgumentParser(
        description="Calcule les 8 indicateurs et les 24 variations futures.")
    parseur.add_argument("fichiers", nargs="*", default=[],
                         help="Clés à analyser (BTC_1h…). Vide = tout le dossier.")
    parseur.add_argument("--sans-contexte", action="store_true",
                         help="N'ajouter ni le contexte multi-timeframe ni les "
                              "colonnes exogènes. Utile pour comparer.")
    arguments = parseur.parse_args()

    contexte = not arguments.sans_contexte
    config.preparer_dossiers()
    print(f"🔬 Indicateurs : {', '.join(config.INDICATEURS)}")
    if contexte:
        print(f"🔭 Contexte : intervalle supérieur (_MTF) + "
              f"{', '.join(config.COLONNES_EXOGENES)} si disponibles")
    print(f"🎯 Cibles : variation_1 … variation_{config.HORIZON_MAX} (en %)\n")

    if not arguments.fichiers:
        indicateurs.analyser_tout(contexte)
        return 0

    for cle in arguments.fichiers:
        symbole, intervalle = stockage.separer_cle(cle)
        indicateurs.analyser_fichier(symbole, intervalle, contexte)
    return 0


if __name__ == "__main__":
    sys.exit(main())
