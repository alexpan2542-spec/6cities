"""Generate publication-quality Figure_Top100_Accuracy.png for Remote Sensing papers."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "Figure_Top100_Accuracy.png"

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def main() -> None:
    labels = [
        "Overall\nWorldCover",
        "Nanchang",
        "Wuhan",
        "Changsha",
        "Hefei",
    ]
    values = [87, 55, 43, 38, 35]
    colors = ["#1f77b4"] + ["#d62728"] * 4

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x = range(len(labels))
    bars = ax.bar(
        x,
        values,
        color=colors,
        width=0.65,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.axhline(
        87,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        label="Overall WC Accuracy",
        zorder=2,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Accuracy Comparison Between WorldCover Labels and\n"
        "Top100 Uncertainty-Selected Samples",
        pad=12,
    )

    ax.yaxis.grid(
        True, linestyle="--", linewidth=0.6, color="0.75", alpha=0.8, zorder=0
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {OUT}")

    # Suggested caption for the manuscript:
    # Figure X. Comparison between overall WorldCover label accuracy and the
    # accuracy of the 100 most uncertain samples identified by margin-based
    # uncertainty sampling across four cities. The uncertainty-selected samples
    # exhibit substantially lower accuracy (35–55%) than the overall WorldCover
    # accuracy (~87%), indicating that margin sampling effectively concentrates
    # label errors in a small subset of samples.


if __name__ == "__main__":
    main()
