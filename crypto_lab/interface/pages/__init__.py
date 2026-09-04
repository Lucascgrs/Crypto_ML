"""
Les six pages de l'application, une par étape du pipeline.

Chaque page est une classe « mixin » : elle apporte sa méthode `_page_X` de
construction et ses actions, mais s'appuie sur la fenêtre principale pour la
navigation, la console et l'exécution en arrière-plan. Cela garde un fichier
court et autonome par écran, au lieu d'un unique module de plusieurs milliers
de lignes.
"""

from .amplitude import PageAmplitude
from .analyse import PageAnalyse
from .backtest import PageBacktest
from .donnees import PageDonnees
from .evaluation import PageEvaluation
from .modele import PageModele
from .visualisation import PageVisualisation

__all__ = ["PageAmplitude", "PageAnalyse", "PageBacktest", "PageDonnees",
           "PageEvaluation", "PageModele", "PageVisualisation"]
