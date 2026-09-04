"""
Compatibilité de la console Windows.

Les rapports du projet utilisent des emojis et des accents. Or un terminal
Windows démarre souvent en cp1252, qui ne sait pas les encoder : le moindre
`print` lève alors une `UnicodeEncodeError` et interrompt un traitement par
ailleurs correct. On bascule donc la sortie en UTF-8 dès l'import du package.
"""

from __future__ import annotations

import sys


def securiser() -> None:
    """Force stdout / stderr en UTF-8 tolérant, si ce n'est pas déjà le cas."""
    for flux in (sys.stdout, sys.stderr):
        encodage = (getattr(flux, "encoding", "") or "").lower()
        if "utf" in encodage:
            continue
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass  # flux non reconfigurable (déjà redirigé) : on n'insiste pas
