"""
Crypto Lab — pipeline complet d'analyse et de prédiction crypto.

Chaîne de traitement :

    extraction  →  indicateurs  →  modele  →  backtest
    (OHLCV)        (8 features    (hausse /   (simulation
                    + variations)  baisse)     de trades)

Chaque module est utilisable seul, en script comme depuis l'interface
graphique (`python Dashboard.py`).
"""

from . import console

console.securiser()   # emojis/accents lisibles même dans un terminal Windows

from . import (amplitude, backtest, cibles, config, exogene,  # noqa: E402
               extraction, indicateurs, modele, ressources, stockage)

__all__ = ["amplitude", "backtest", "cibles", "config", "console", "exogene",
           "extraction", "indicateurs", "modele", "ressources", "stockage"]

__version__ = "2.1"
