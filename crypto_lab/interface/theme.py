"""
Apparence de l'interface : palette, styles des tableaux, style des graphiques.

Tout ce qui touche aux couleurs est centralisé ici pour que l'application reste
cohérente d'un écran à l'autre.
"""

from __future__ import annotations

from tkinter import ttk

COULEURS = {
    "fond":         "#1a1a1a",
    "panneau":      "#242424",
    "carte":        "#2b2b2b",
    "accent":       "#1f6aa5",
    "accent_clair": "#2e86c1",
    "vert":         "#2ecc71",
    "rouge":        "#e74c3c",
    "orange":       "#f39c12",
    "jaune":        "#f1c40f",
    "bleu":         "#00bcd4",
    "texte":        "#e6e6e6",
    "texte_doux":   "#9aa0a6",
    "axe":          "#1e1e1e",
}

# Couleur associée au jugement porté sur une statistique.
NIVEAUX = {
    "bon":     COULEURS["vert"],
    "moyen":   COULEURS["jaune"],
    "faible":  COULEURS["orange"],
    "mauvais": COULEURS["rouge"],
}


def couleur_niveau(niveau: str) -> str:
    """Couleur d'un niveau de qualité ('bon', 'moyen', 'faible', 'mauvais')."""
    return NIVEAUX.get(niveau, COULEURS["texte"])


def configurer_tableaux() -> None:
    """Accorde les tableaux ttk (Treeview) au thème sombre de CustomTkinter."""
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Treeview",
        background=COULEURS["carte"], foreground=COULEURS["texte"],
        fieldbackground=COULEURS["carte"], rowheight=26, borderwidth=0,
        font=("Segoe UI", 10))
    style.configure(
        "Treeview.Heading",
        background=COULEURS["accent"], foreground="white",
        relief="flat", font=("Segoe UI Semibold", 10))
    style.map("Treeview", background=[("selected", COULEURS["accent_clair"])])
    style.map("Treeview.Heading", background=[("active", COULEURS["accent_clair"])])


def styliser_axes(ax) -> None:
    """Applique le thème sombre à un axe matplotlib."""
    ax.set_facecolor(COULEURS["axe"])
    ax.tick_params(colors="#bbbbbb", labelsize=9)
    for bordure in ax.spines.values():
        bordure.set_color("#555555")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("#bbbbbb")
    ax.yaxis.label.set_color("#bbbbbb")
    ax.grid(alpha=0.15, color="#888888")


def legende(ax, **kwargs):
    """Légende matplotlib lisible sur fond sombre."""
    return ax.legend(facecolor=COULEURS["carte"], labelcolor="white",
                     edgecolor="#555555", **kwargs)
