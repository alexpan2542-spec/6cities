"""Generate publication-quality Figure_SceneFrequency.png for Remote Sensing papers."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "Figure_SceneFrequency.png"

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)

# Full name (code) — sorted high → low frequency
DATA = [
    ("Bare Land (BL)", 55),
    ("Road Edge (RE)", 50),
    ("Cropland Transition (CT)", 44),
    ("Water Edge (WE)", 41),
    ("Cropland Edge (CE)", 36),
    ("Rural Settlement (RS)", 34),
    ("Building Edge (BE)", 31),
    ("Turbid Water (TW)", 26),
    ("Forest Edge (FE)", 26),
]


def main() -> None:
    # Already sorted descending; reverse for horizontal bars (top = highest)
    labels = [d[0] for d in DATA][::-1]
    values = [d[1] for d in DATA][::-1]

    cmap = LinearSegmentedColormap.from_list(
        "blue_grad",
        ["#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )
    # Higher frequency → darker blue
    norm = plt.Normalize(min(values), max(values))
    colors = [cmap(norm(v)) for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    y = np.arange(len(labels))
    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.55,
        height=0.72,
        zorder=3,
    )

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.8,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Scene Type")
    ax.set_xlim(0, max(values) + 10)
    ax.set_title("Frequency of Error-Prone Scene Types", pad=12)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="0.75", alpha=0.85, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {OUT}")

    # Suggested caption:
    # Figure X. Frequency of error-prone scene types identified from the
    # uncertainty-selected Top100 samples across four cities. Bare land (BL),
    # road edge (RE), cropland transition (CT), water edge (WE), and cropland
    # edge (CE) were the most common sources of label uncertainty, indicating
    # that WorldCover errors are concentrated in boundary and transitional
    # environments.


if __name__ == "__main__":
    main()
