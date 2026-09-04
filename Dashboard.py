"""
Crypto Lab — lancement de l'interface graphique.

    python Dashboard.py

Tout le code vit dans le paquet `crypto_lab/` ; ce fichier ne fait qu'ouvrir la
fenêtre. Les traitements sont aussi utilisables en ligne de commande via
GatherData.py, CryptoAnalysis.py et Predict.py.
"""

import os
import sys

# Le projet manipule des dossiers relatifs (data_crypto, models…) : on se place
# systématiquement à la racine du projet, quel que soit l'endroit d'où on lance.
RACINE = os.path.dirname(os.path.abspath(__file__))
os.chdir(RACINE)
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)


def main():
    try:
        from crypto_lab.interface import lancer
    except ImportError as err:
        print(f"❌ Dépendance manquante : {err}")
        print("   Installe le nécessaire : pip install -r requirements.txt")
        return 1
    lancer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
