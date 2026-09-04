"""
Interface graphique de Crypto Lab (CustomTkinter).

    app.py         fenêtre principale, navigation, console, exécution en tâche de fond
    composants.py  briques de widgets réutilisées par toutes les pages
    theme.py       palette et style des tableaux et graphiques
    textes.py      textes d'aide et explications
    pages/         une page par étape du pipeline

Lancement : `python Dashboard.py` à la racine du projet.
"""

from .app import CryptoLab, lancer

__all__ = ["CryptoLab", "lancer"]
