"""
Téléchargement des données de marché, en ligne de commande.

    python GatherData.py BTC ETH --intervalle 1h --debut 2022-01-01
    python GatherData.py --top 5 --intervalle 4h
    python GatherData.py BTC --source Yahoo --intervalle 1d

Le code est dans `crypto_lab/extraction.py` ; ce fichier n'est que l'entrée
en ligne de commande. Depuis l'interface : onglet « 1 · Extraction ».
"""

import argparse
import sys

from crypto_lab import config, extraction


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Télécharge l'historique OHLCV de cryptomonnaies.")
    parseur.add_argument("symboles", nargs="*", default=[],
                         help="Symboles à télécharger (BTC ETH SOL…).")
    parseur.add_argument("--top", type=int, default=0,
                         help="Télécharger plutôt le Top N par capitalisation.")
    parseur.add_argument("--intervalle", default="1h", choices=config.INTERVALLES,
                         help="Durée d'une bougie (défaut : 1h).")
    parseur.add_argument("--debut", default="2022-01-01", help="Date de début (AAAA-MM-JJ).")
    parseur.add_argument("--fin", default="2026-01-01", help="Date de fin (AAAA-MM-JJ).")
    parseur.add_argument("--source", default="Binance", choices=["Binance", "Yahoo"],
                         help="Source des données (défaut : Binance).")
    return parseur.parse_args()


def main():
    arguments = analyser_arguments()
    config.preparer_dossiers()

    if arguments.top:
        extraction.telecharger_top_n(arguments.top, arguments.debut, arguments.fin,
                                     arguments.intervalle, arguments.source)
        return 0

    if not arguments.symboles:
        print("❌ Indique au moins un symbole, ou utilise --top N.")
        return 1

    for symbole in arguments.symboles:
        df = extraction.telecharger(symbole.upper(), arguments.debut, arguments.fin,
                                    arguments.intervalle, arguments.source)
        extraction.sauvegarder(df, symbole.upper(), arguments.intervalle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
