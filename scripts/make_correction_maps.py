#!/usr/bin/env python3
"""
Before/after boundary-correction maps for 1-2 example cities (Nanjing,
Changsha), for the Results 4.6 slot in docs/Paper_Writing_Plan.md.

Reuses the exact LOCO operator from scripts/loco_correction_operator.py
(same feature set, same tightened gate: tau=0.85, lowest-10%-margin) so the
map is consistent with the T5 / Phase3 numbers already frozen in the plan.

For each city:
  Panel A - WorldCover raw class map (15000 sample points)
  Panel B - LOCO-corrected class map, points the operator changed circled
  Panel C - the 150 hand-labelled BAMS points: green = human agrees with WC,
            red = human disagrees, over a faint WC backdrop

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/make_correction_maps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from loco_correction_operator import (  # noqa: E402
    CITIES, FEATURE_SETS, PHASE2_FEATURE_SET, CORR_CONF_TAU,
    BOUNDARY_GATE_Q, fit_operator, load_city,
)

FIG = ROOT / "data" / "analysis_outputs" / "correction_maps"
CLASS_COLORS = {0: "#b35806", 1: "#1a9850", 2: "#4575b4"}  # built, non-built, water
CLASS_NAMES = {0: "built", 1: "non-built", 2: "water"}
# all 6 cities: 2 go in the main text (Nanjing = lowest WC-human agreement
# of the six + largest LOCO transfer gain, Changsha = a mid-range city with
# high operator fire-precision), the other 4 are supplementary -- generating
# all six costs nothing and pre-empts a "why only 2 cities" reviewer question.
# NOTE: do not call Nanjing a "failure case reversal" -- that is leftover
# framing from the dead Prototype-Expansion narrative (see Paper_Writing_Plan
# sec 1.2/4.6). Under the diagnostic-first framing Nanjing is simply the city
# with the lowest raw WC-vs-human agreement; there is no "failure" to reverse.
MAP_CITIES = CITIES


def make_map(city: str, city_df: dict):
    df = city_df[city]
    feats = FEATURE_SETS[PHASE2_FEATURE_SET]
    op = fit_operator(city_df, city, feats, CITIES)

    X = df[feats].to_numpy(float)
    proba = op.predict_proba(X)
    pred = op.classes_[proba.argmax(1)]
    pmax = proba.max(1)
    margin = df["margin"].to_numpy(float)
    wc = df["wc"].to_numpy(int)

    gate_thr = np.quantile(margin, BOUNDARY_GATE_Q)
    gated = margin <= gate_thr
    fire = gated & (pmax >= CORR_CONF_TAU) & (pred != wc)
    corrected = wc.copy()
    corrected[fire] = pred[fire]

    lat = df["lat"].to_numpy(float)
    lon = df["lon"].to_numpy(float)

    bams = df[df["is_bams"]]
    agree = (bams["wc"] == bams["human"]).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    # Panel A -- WC raw
    ax = axes[0]
    for c in (0, 1, 2):
        m = wc == c
        ax.scatter(lon[m], lat[m], s=3, c=CLASS_COLORS[c], alpha=0.55, linewidths=0)
    ax.set_title(f"{city} — WorldCover (raw)")

    # Panel B -- corrected, fired points circled
    ax = axes[1]
    for c in (0, 1, 2):
        m = corrected == c
        ax.scatter(lon[m], lat[m], s=3, c=CLASS_COLORS[c], alpha=0.55, linewidths=0)
    ax.scatter(lon[fire], lat[fire], s=22, facecolors="none", edgecolors="black",
               linewidths=0.8, label=f"corrected (n={int(fire.sum())})")
    ax.set_title(f"{city} — LOCO-corrected (τ={CORR_CONF_TAU}, lowest {int(BOUNDARY_GATE_Q*100)}% margin)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    # Panel C -- BAMS150 human vs WC agreement over a faint backdrop
    ax = axes[2]
    ax.scatter(lon, lat, s=2, c="#d9d9d9", alpha=0.5, linewidths=0)
    ax.scatter(bams.loc[agree, "lon"], bams.loc[agree, "lat"], s=26, c="#1a9850",
               marker="o", label=f"human = WC (n={int(agree.sum())})", edgecolors="black", linewidths=0.3)
    ax.scatter(bams.loc[~agree, "lon"], bams.loc[~agree, "lat"], s=26, c="#c1443c",
               marker="^", label=f"human ≠ WC (n={int((~agree).sum())})", edgecolors="black", linewidths=0.3)
    ax.set_title(f"{city} — 150 hand-labelled boundary points")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    for ax in axes:
        ax.set_xlabel("lon"); ax.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("lat")

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=CLASS_COLORS[c],
                      markersize=8, label=CLASS_NAMES[c]) for c in (0, 1, 2)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / f"Fig_diag_map_{city}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{city}: n_corrected={int(fire.sum())}/{len(df)}  "
          f"bams_disagree={int((~agree).sum())}/{len(bams)}  -> {out}")


def main():
    city_df = {c: load_city(c) for c in CITIES}
    for city in MAP_CITIES:
        make_map(city, city_df)


if __name__ == "__main__":
    main()
