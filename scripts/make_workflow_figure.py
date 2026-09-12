#!/usr/bin/env python3
"""
Figure 2 - Three-stage analysis workflow.

Redrawn for the diagnostic-first paper (2026-09-09 pivot). Structure follows
docs/manuscript_sec3.tex:
  Stage 1  Boundary-aware candidate sampling            (Sec 3.1)
  Stage 2  Multi-reference agreement diagnostic  [core] (Sec 3.2)
  Stage 3  LOCO correction operator  [supporting]       (Sec 3.3)
  spanning: Evaluation protocol & leakage controls      (Sec 3.4)
  footer:   conditioning caveat

Output : submission/figures/Figure2_workflow.png  (+ .pdf)
Run    : /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 scripts/make_workflow_figure.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 400})

C_S1 = "#eef4fb"; C_S1E = "#4b78ad"
C_S2 = "#fdf3e3"; C_S2E = "#c98a1e"
C_S3 = "#f0f0ee"; C_S3E = "#8a8a84"
C_OUT = "#e5f0e6"; C_OUTE = "#4f8f5f"
C_Q = "#f6e0dd"; C_QE = "#b5402f"
C_EVAL = "#eef0f4"; C_EVALE = "#7b8494"
C_CAV = "#f4efe6"; C_CAVE = "#9a7b3f"
INK = "#1b1b1b"; SUB = "#3c3c3c"

fig = plt.figure(figsize=(7.5, 10.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 130)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.1, r=2.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z,
                                boxstyle=f"round,pad=0,rounding_size={r}"))


def txt(x, y, s, size=8, weight="normal", color=INK, ha="center", va="center",
        z=5, bbox=None):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=z, bbox=bbox)


def arrow(x, y0, y1, lw=1.9, color=INK, dash=None):
    p = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                        lw=lw, color=color, zorder=3, shrinkA=0, shrinkB=0)
    if dash:
        p.set_linestyle(dash)
    ax.add_patch(p)


TAG = dict(boxstyle="round,pad=0.28", ec="none")

# ---------------------------------------------------------------- title
txt(50, 127, "Three-stage analysis workflow", size=13, weight="bold")

# ================================================================ STAGE 1
box(4, 91, 92, 33, C_S1, C_S1E, lw=1.4)
txt(7, 121.5, "Stage 1", size=9.5, weight="bold", color=C_S1E, ha="left")
txt(50, 121.5, "Boundary-aware candidate sampling", size=9.5, weight="bold")
txt(93, 121.5, "§3.1", size=8, color=SUB, ha="right")

box(7, 108.5, 41, 9, "white", C_S1E, lw=0.8)
txt(27.5, 113, "Per-city AOI  =  ADM2 prefecture polygon\n"
    "6 cities  ·  taken whole  ·  no urban mask", size=7.2, color=SUB)

box(52, 108.5, 41, 9, "white", C_S1E, lw=0.8)
txt(72.5, 113, "ESA WorldCover 2021  →  3 classes\n"
    "stratified sample 5,000 / class  →  15,000 / city", size=7.2, color=SUB)

box(7, 98, 41, 9, "white", C_S1E, lw=0.8)
txt(27.5, 102.5, "Sentinel-2 2021 composite\n"
    "9 spectral features (6 bands + NDVI, NDBI, MNDWI)\n"
    "point value  +  3×3 mean & SD", size=6.9, color=SUB)

box(52, 98, 41, 9, "white", C_S1E, lw=0.8)
txt(72.5, 104.4, "BAMS score", size=7.2, weight="bold", color=SUB)
txt(72.5, 102.4, "RF margin  m = p₁ − p₂   +   BoundaryScore", size=6.9, color=SUB)
txt(72.5, 100.4, "BoundaryScore = Σ σ₃ₓ₃(B2, B3, B4, B8)   →   pick 150 / city",
    size=6.4, color=SUB)

txt(50, 94, "the random forest is a proposal mechanism only — it assigns no class labels",
    size=6.8, color="#7a5a2a")

# ---- output pill
arrow(50, 91, 87.3)
box(9, 80.5, 82, 6.4, C_OUT, C_OUTE, lw=1.3, r=3)
txt(50, 83.7, "900 boundary points  ·  independent expert photo-interpretation  ·  "
    "QC: 100-pt κ = 0.89 (92.7 %)", size=7.2, weight="bold", color="#2f5d3a")

# ================================================================ STAGE 2
arrow(50, 80.5, 76.3)
box(4, 43, 92, 32.5, C_S2, C_S2E, lw=1.9)
txt(7, 72.7, "Stage 2", size=9.5, weight="bold", color=C_S2E, ha="left")
txt(50, 72.7, "Multi-reference agreement diagnostic", size=9.5, weight="bold")
txt(82, 72.7, "core", size=7.3, weight="bold", color="white",
    bbox=dict(fc=C_S2E, **TAG))
txt(93, 72.7, "§3.2", size=8, color=SUB, ha="right")

box(7, 66.5, 86, 4.4, C_Q, C_QE, lw=1.0, r=1.8)
txt(50, 68.7, "Does consulting a 2nd / 3rd product (Dynamic World, Esri) "
    "arbitrate truth at the urban boundary?", size=7.5, weight="bold",
    color="#8a2f22")
txt(50, 63.8, "compare  expert  vs  WorldCover · Dynamic World · Esri      (n = 900)",
    size=7.0, color=SUB)

r1 = 53.0        # bottom of row-1 sub-boxes
r2 = 44.5        # bottom of row-2 sub-boxes
hh = 8.0
w3 = 27.3
xs = [7, 36.35, 65.7]
row1 = [("(a) Four-way agreement", "“how much”", "T1"),
        ("(b) Confidence stratification", "annotator noise?", "T2"),
        ("(c) Direction of WC error", "“which way”", "T3")]
for x, (t, m, tag) in zip(xs, row1):
    box(x, r1, w3, hh, "white", C_S2E, lw=0.8)
    txt(x + w3 / 2, r1 + 6.0, t, size=6.9, weight="bold")
    txt(x + w3 / 2, r1 + 3.8, m, size=6.4, color=SUB)
    txt(x + w3 / 2, r1 + 1.6, tag, size=6.0, color=C_S2E, weight="bold")

box(7, r2, w3, hh, "white", C_S2E, lw=0.8)
txt(7 + w3 / 2, r2 + 6.0, "(d) Physical scene\nstratification", size=6.9, weight="bold")
txt(7 + w3 / 2, r2 + 2.9, "“where”", size=6.4, color=SUB)
txt(7 + w3 / 2, r2 + 1.1, "T4", size=6.0, color=C_S2E, weight="bold")

we = 56.65
box(36.35, r2, we, hh, "white", C_S2E, lw=0.8)
txt(36.35 + we / 2, r2 + 6.1, "(e) Independence & arbitration test", size=6.9, weight="bold")
txt(36.35 + we / 2, r2 + 3.8,
    "excess agreement  ·  error dependence (Yule Q, κ)  ·  arbitration value  ·  confounds",
    size=6.0, color=SUB)
txt(36.35 + we / 2, r2 + 1.6, "Tables S2–S3 · Fig 3", size=6.0, color=C_S2E, weight="bold")

# ================================================================ STAGE 3
arrow(50, 43, 37.3, lw=1.5, color=C_S3E, dash=(0, (5, 3)))
box(4, 15.5, 92, 21, C_S3, C_S3E, lw=1.2)
txt(7, 33.7, "Stage 3", size=9.5, weight="bold", color=C_S3E, ha="left")
txt(50, 33.7, "LOCO correction operator", size=9.5, weight="bold")
txt(80, 33.7, "supporting", size=7.3, weight="bold", color="white",
    bbox=dict(fc=C_S3E, **TAG))
txt(93, 33.7, "§3.3", size=8, color=SUB, ha="right")

txt(50, 30.0, "leave-one-city-out: fit expert corrections on 5 cities  →  apply to the "
    "held-out 6th (no labels of its own)", size=7.0, color=SUB)
txt(50, 27.4, "the operator overwrites the WorldCover label only where it is confident "
    "(gate τ)", size=7.0, color=SUB)

phw = 27.3
for x, lab in zip([7, 36.35, 65.7],
                  ["Phase 1\noperator transfer",
                   "Phase 2\ndownstream map correction",
                   "Phase 3\nmulti-reference referee"]):
    box(x, 20.0, phw, 5.4, "white", C_S3E, lw=0.7)
    txt(x + phw / 2, 22.7, lab, size=6.6)
txt(50, 17.6, "before / after correction maps:   Fig 4 (main)   ·   Fig S1–S4",
    size=6.4, color=SUB)

# ================================================================ EVAL + CAVEAT
box(4, 8, 92, 5.6, C_EVAL, C_EVALE, lw=1.0)
txt(50, 11.9, "Evaluation protocol & leakage controls    §3.4", size=7.5,
    weight="bold", color="#4a5262")
txt(50, 9.5, "fixed downstream classifier  ·  spatial-block partitioning  ·  "
    "target-city isolation  ·  block-bootstrap CIs", size=6.7, color=SUB)

box(4, 1.4, 92, 5.2, C_CAV, C_CAVE, lw=1.3)
txt(50, 4.9, "All agreement / disagreement rates are conditional on the "
    "boundary-candidate population;", size=6.9, weight="bold", color="#6d5320")
txt(50, 2.9, "they are not map-wide accuracies of any product.", size=6.9,
    weight="bold", color="#6d5320")

fig.savefig("submission/figures/Figure2_workflow.png", bbox_inches="tight")
fig.savefig("submission/figures/Figure2_workflow.pdf", bbox_inches="tight")
print("wrote submission/figures/Figure2_workflow.png / .pdf")
