"""Shared presentation style for the DFTorch diatomic figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1f3864"
SLATE = "#6b7280"
CRIMSON = "#b3282d"
GREEN = "#2e8b57"
GOLD = "#b8860b"

RC = {
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "axes.labelsize": 12.5,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 10.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
}


def apply():
    plt.rcParams.update(RC)


def finish(ax):
    """Trim chartjunk: drop the top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
