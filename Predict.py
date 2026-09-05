"""
Entraînement et prédiction, en ligne de commande.

    python Predict.py BTC_1h                          # horizon et modèle par défaut
    python Predict.py BTC_1h --horizon 6 --modele LightGBM
    python Predict.py BTC_1h --horizon 24 --seuil 0.65
    python Predict.py BTC_1h --objectif direction_nette   # n'apprend que le rentable
    python Predict.py BTC_1h --objectif amplitude         # 5 classes d'amplitude
    python Predict.py BTC_1h --walk-forward               # évaluation hors échantillon
    python Predict.py BTC_1h --predire-seulement       # réutilise le modèle existant
    python Predict.py BTC_1h --panier BTC,ETH,SOL,BNB  # un modèle sur 4 cryptos
    python Predict.py BTC_1h --recherche approfondie   # 18 configurations au lieu de 3
    python Predict.py BTC_1h --utilite 0.0005          # ne garder que les features utiles

L'option --utilite ne garde que les features dont l'utilité mesurée au
PRÉCÉDENT entraînement atteint le seuil donné (une perte d'AUC : 0.0005 est un
bon point de départ). Elle est sans effet au premier entraînement, puisqu'il
faut bien une première mesure. L'utilité servant de filtre est relevée sur le
bloc de validation, jamais sur le test.

L'option --panier entraîne UN modèle sur plusieurs cryptos empilées, puis
l'applique à la crypto passée en premier argument. Les blocs train/validation/
test sont coupés aux mêmes DATES pour toutes, et les features dont le niveau
dépend de l'actif sont converties en rang glissant (voir `crypto_lab/panier.py`).

Le modèle apprend à répondre : « dans X périodes, que va-t-il se passer ? ». La
forme exacte de la question dépend de l'objectif ; dans tous les cas il fournit
un niveau de confiance exploitable dans les deux sens.

Le code est dans `crypto_lab/modele.py`. Depuis l'interface : onglet
« 3 · Prédiction ».
"""

import argparse
import sys

from crypto_lab import cibles, config, modele, stockage


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
                         help=f"Seuil de confiance (défaut : {config.SEUIL_DEFAUT}). "
                              f"Attention : il part de 1/nombre de classes, soit "
                              f"0.20 pour l'objectif « amplitude ».")
    parseur.add_argument("--objectif", default=cibles.TACHE_DEFAUT,
                         choices=list(cibles.TACHES),
                         help=f"Ce que le modèle apprend (défaut : "
                              f"{cibles.TACHE_DEFAUT}).")
    parseur.add_argument("--panier", default=None,
                         help="Cryptos à empiler dans un seul modèle, séparées "
                              "par des virgules (ex : BTC,ETH,SOL). Le modèle "
                              "est enregistré sous PANIER-… et appliqué ensuite "
                              "à la crypto passée en argument.")
    parseur.add_argument("--recherche", default=config.RECHERCHE_DEFAUT,
                         choices=list(config.MODES_RECHERCHE),
                         help=f"Nombre de configurations d'hyperparamètres "
                              f"essayées : rapide (3), approfondie (18) ou "
                              f"exhaustive (40). Défaut : "
                              f"{config.RECHERCHE_DEFAUT}. Attention, le gain "
                              f"attendu reste inférieur à la marge d'erreur de "
                              f"l'AUC.")
    parseur.add_argument("--utilite", type=float, default=0.0,
                         help="Utilité minimale d'une feature, en perte d'AUC "
                              "(ex : 0.0005). Les features en dessous sont "
                              "écartées de l'entraînement. Sans effet tant "
                              "qu'aucune mesure n'existe.")
    parseur.add_argument("--walk-forward", action="store_true",
                         help="Réentraîner de fenêtre en fenêtre et ne prédire que "
                              "l'inconnu. Produit un fichier « _wf » backtestable "
                              "sur toute sa durée.")
    parseur.add_argument("--predire-seulement", action="store_true",
                         help="Ne pas réentraîner : réutiliser le modèle existant.")
    return parseur.parse_args()


def main():
    arguments = analyser_arguments()
    config.preparer_dossiers()
    symbole, intervalle = stockage.separer_cle(arguments.fichier)
    objectif = cibles.obtenir(arguments.objectif)

    # Le panier tient lieu de « symbole » à l'entraînement ; la prédiction,
    # elle, porte toujours sur une crypto précise.
    membres = [s.strip().upper() for s in (arguments.panier or "").split(",") if s.strip()]
    source = config.nom_panier(membres) if len(membres) >= 2 else symbole

    if arguments.walk_forward:
        modele.walk_forward(symbole, intervalle, arguments.horizon, arguments.modele,
                            seuil_confiance=arguments.seuil, tache=objectif.cle)
        return 0

    if not arguments.predire_seulement:
        modele.entrainer(source, intervalle, arguments.horizon, arguments.modele,
                         objectif.cle,
                         mode_recherche=arguments.recherche,
                         seuil_utilite=arguments.utilite)

    modele.predire(symbole, intervalle, arguments.horizon, arguments.seuil,
                   symbole_modele=None if source == symbole else source,
                   tache=objectif.cle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
