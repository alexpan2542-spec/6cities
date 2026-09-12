#!/usr/bin/env python3
"""
Step 3 of the C2 example-chip figure (Supplementary).

Tiles the Sentinel-2 chips fetched by c2_chip_fetch.py (or exported by
gee_c2_chip_export.js and dropped into .../c2_chip_figure/chips/) into one
labelled gallery:

    group A  C2  WC=built, expert=non-built            -- directional error
    group B  C1  DW=Esri=built, expert=non-built,      -- coupled error
                 WC dissents (correct here)
    group C  counter-examples  WC=built, expert=built

Each cell: true-colour chip with a false-colour (NIR) inset, and a caption
naming every product's call and the expert's.  Missing chips render as a
placeholder so the layout is visible before all images are in.

Output: submission/figures/FigureS5_c2_chips.{pdf,png}

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/c2_chip_assemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2_chip_select import EXCLUDE_IDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "data/analysis_outputs/c2_chip_figure"
CHIPS = FIGDIR / "chips"
OUT = ROOT / "submission/figures/FigureS5_c2_chips"

NCOL = 6
CORRECT = "#1a7f37"   # product call matches the expert
WRONG = "#c1443c"     # product call differs from the expert
NEUTRAL = "#555"
GROUP_HEAD = {
    "A": ("A  —  C1 + C2: product consensus errs together",
          "WorldCover = Dynamic World = Esri = built-up, expert = non-built "
          "(vegetation). Agreement is not corroboration; and once WC calls "
          "“built”, the expert reads vegetation ~7 times in 10."),
    "B": ("B  —  C1: the coupled pair errs, WorldCover dissents and is right",
          "Dynamic World = Esri = built-up (the pair that agrees most), expert = "
          "non-built, and WorldCover — least coupled to either — dissents and is "
          "correct here."),
    "C": ("C  —  counter-examples",
          "WorldCover = built-up, expert = built-up. WC is right on the boundary too; "
          "the gallery is not selected on WC being wrong."),
}
CAPTION = (
    "Sentinel-2 chips ({win} m window), 2021 S2_SR_HARMONIZED median, "
    "CLOUDY_PIXEL_PERCENTAGE < 20 — the sampling pipeline's imagery recipe. "
    "Inset: false colour B8 B4 B3 (vegetation red, water black). Selection "
    "(scripts/c2_chip_select.py): ranked by selection margin descending, "
    "top {a}/{b}/{c} per city; Conf ∈ {{high, med}} for A and B, high only for C "
    "(med where a city has no high-conf built–built point); ids {excl} dropped "
    "on review. Each product call is green where it matches the expert, red where "
    "it differs. Illustration only — the quantitative claims rest on §4.5–4.6."
)


def load_img(pid: str, oid: int, kind: str):
    # filename carries panel id AND point id, so a re-selection never
    # collides with a stale cached chip
    pats = [f"c2_{pid}_{oid}_*_{kind}.*", f"c2_{pid}_*_{kind}.*", f"c2_{pid}_{kind}.*"]
    for pat in pats:
        for h in sorted(CHIPS.glob(pat)):
            if h.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
                try:
                    return mpimg.imread(h)
                except Exception:  # noqa: BLE001
                    pass
    return None


def draw_cell(ax, rec):
    tc = load_img(rec.panel_id, rec.Original_ID, "tc")
    fc = load_img(rec.panel_id, rec.Original_ID, "fc")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    if tc is None:
        ax.set_facecolor("#ececec")
        ax.text(0.5, 0.5, "chip pending", ha="center", va="center",
                fontsize=8, color="#888", transform=ax.transAxes)
    else:
        ax.imshow(tc)
        ax.plot(0.5, 0.5, marker="+", ms=11, mew=1.4, color="#ffdd00",
                transform=ax.transAxes)
        if fc is not None:
            ins = ax.inset_axes([0.63, 0.02, 0.35, 0.35])
            ins.imshow(fc); ins.set_xticks([]); ins.set_yticks([])
            for s in ins.spines.values():
                s.set_color("white"); s.set_linewidth(0.8)

    ax.set_title(f"{rec.City}  ·  id {rec.Original_ID}", fontsize=8, pad=3)
    ax.add_artist(AnnotationBbox(
        _caption_box(rec), (0.5, -0.045), xycoords="axes fraction",
        box_alignment=(0.5, 1.0), frameon=False, pad=0))


def _tok(text, color, bold=False):
    return TextArea(text, textprops=dict(
        color=color, fontsize=7.3, fontweight="bold" if bold else "normal"))


def _caption_box(rec):
    """Two stacked lines; each product token green if it matches the expert
    call, red if it differs."""
    exp = rec.expert_c3

    def col(v):
        return CORRECT if v == exp else WRONG

    row1 = HPacker(pad=0, sep=0, align="baseline", children=[
        _tok(f"WC {rec.wc_c3}", col(rec.wc_c3)),
        _tok("  ·  ", NEUTRAL),
        _tok(f"DW {rec.dw_c3}", col(rec.dw_c3)),
        _tok("  ·  ", NEUTRAL),
        _tok(f"Esri {rec.esri_c3}", col(rec.esri_c3)),
    ])
    row2 = HPacker(pad=0, sep=0, align="baseline", children=[
        _tok("expert: ", NEUTRAL),
        _tok(rec.expert_c3, NEUTRAL, bold=True),
        _tok(f"   [{rec.Conf}]", NEUTRAL),
    ])
    return VPacker(pad=0, sep=3, align="center", children=[row1, row2])


def main() -> None:
    man = pd.read_csv(FIGDIR / "chip_manifest.csv")
    counts = man.groupby("group").size().to_dict()
    rows_per = {g: -(-counts.get(g, 0) // NCOL) for g in ("A", "B", "C")}
    n_img_rows = sum(rows_per.values())

    # gridspec: a thin header row before each group's chip rows
    height_ratios, kind = [], []
    for g in ("A", "B", "C"):
        height_ratios.append(0.30); kind.append(("hdr", g))
        for _ in range(rows_per[g]):
            height_ratios.append(1.0); kind.append(("img", g))

    fig = plt.figure(figsize=(2.45 * NCOL, sum(height_ratios) * 2.55 + 0.6))
    gs = fig.add_gridspec(len(height_ratios), NCOL, height_ratios=height_ratios,
                          hspace=0.62, wspace=0.06)

    img_row_cursor = {g: [] for g in ("A", "B", "C")}
    for i, (k, g) in enumerate(kind):
        if k == "img":
            img_row_cursor[g].append(i)

    for g in ("A", "B", "C"):
        # header
        hdr_i = kind.index(("hdr", g))
        hax = fig.add_subplot(gs[hdr_i, :]); hax.axis("off")
        t, sub = GROUP_HEAD[g]
        hax.text(0.0, 0.55, t, fontsize=11, fontweight="bold", va="center")
        hax.text(0.0, -0.35, sub, fontsize=8.2, color="#555", va="center")
        # chips
        sub_man = man[man.group == g].reset_index(drop=True)
        for j, rec in enumerate(sub_man.itertuples(index=False)):
            r = img_row_cursor[g][j // NCOL]
            c = j % NCOL
            ax = fig.add_subplot(gs[r, c])
            draw_cell(ax, rec)

    win = 2 * 375
    fig.text(0.5, 0.012,
             CAPTION.format(win=win, a=2, b=1, c=1,
                            excl="/".join(str(i) for i in sorted(EXCLUDE_IDS))),
             ha="center", va="bottom", fontsize=7.4, color="#444", wrap=True)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.965, bottom=0.085)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    present = sum(load_img(r.panel_id, r.Original_ID, "tc") is not None
                  for r in man.itertuples(index=False))
    print(f"chips present: {present}/{len(man)}")
    print(f"wrote {OUT.with_suffix('.pdf')}")
    print(f"wrote {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()
