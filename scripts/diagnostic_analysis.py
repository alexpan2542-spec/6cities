#!/usr/bin/env python3
"""
Diagnostic analysis -- centrepiece for the MDPI Remote Sensing paper
(diagnostic-first framing; the LOCO operator is a supporting section).

Question: how badly, where, and in which direction do global land-cover
products disagree with expert judgement on urban built <-> vegetation
boundaries -- and do additional products (Dynamic World, ESRI) resolve it?

Anchor data: the hand-labelled BAMS150 boundary points.
  - 6 cities have Human labels + WC (Wuhan, Hefei, Nanchang, Nanjing,
    Changsha, Hangzhou).
  - all 6 additionally have Dynamic World + ESRI
    (data/shared_reference/cross_city_second_ref_dw_esri/
     all_cities_BAMS150_second_ref.csv). Hangzhou's DW/ESRI were added
     2026-09-07; the multi-reference subset is now n = 900, not 750.

Outputs -> data/analysis_outputs/diagnostic_analysis/  (tables and PNGs, co-located).

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/diagnostic_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DATA2 = ROOT / "data"
CITIES_DIR = DATA2 / "cities"
REF_DIR = DATA2 / "shared_reference"
OUT = DATA2 / "analysis_outputs" / "diagnostic_analysis"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT  # co-located with this script's own tables
SECOND_REF = REF_DIR / "cross_city_second_ref_dw_esri" / "all_cities_BAMS150_second_ref.csv"
SCENE_CSV = REF_DIR / "cross_city_scene_localization" / "bams150_scene_grouped.csv"
LOCO = DATA2 / "analysis_outputs" / "loco_correction_operator"

CITIES6 = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]
# All 6 now carry DW+ESRI (n=900). T4 scene stratification stays 5-city
# regardless -- Hangzhou has no scene-code table (SCENE_CSV is 750 rows).
CLASSES = {0: "built", 1: "non-built", 2: "water"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_manual_wc() -> pd.DataFrame:
    """One row per hand-labelled point: city, Original_ID, human, wc, conf."""
    rows = []
    for c in CITIES6:
        m = pd.read_csv(CITIES_DIR / c / "04_manual" / f"{c}_BAMS150_Manual.csv")
        m["Original_ID"] = m["Original_ID"].astype(int)
        m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
        m = m[m["Human_Class"].notna()].copy()
        m["human"] = m["Human_Class"].astype(int) - 1
        m["conf"] = (m["Confidence"].astype(str).str.strip().str.lower()
                     .where(lambda s: s.isin(["h", "m", "l"]), "m"))
        orig = pd.read_csv(CITIES_DIR / c / "01_original" / f"{c}_WC_Samples_15000.csv")
        orig["Original_ID"] = orig["Original_ID"].astype(int)
        wc = orig.set_index("Original_ID")["Class"].astype(int) - 1
        m["wc"] = m["Original_ID"].map(wc)
        m["city"] = c
        rows.append(m[["city", "Original_ID", "human", "wc", "conf"]])
    return pd.concat(rows, ignore_index=True)


def load_dwesri() -> pd.DataFrame:
    df = pd.read_csv(SECOND_REF)
    df["Original_ID"] = df["Original_ID"].astype(int)
    df["dw"] = df["DW_C3"].round().astype(int) - 1
    df["esri"] = df["ESRI_C3"].round().astype(int) - 1
    return df.rename(columns={"City": "city"})[["city", "Original_ID", "dw", "esri"]]


# ---------------------------------------------------------------------------
# T1  four-way agreement
# ---------------------------------------------------------------------------
def table_fourway(mw: pd.DataFrame, de: pd.DataFrame) -> pd.DataFrame:
    d = mw.merge(de, on=["city", "Original_ID"], how="left")
    out = []
    for city in CITIES6 + ["ALL(6)"]:
        if city == "ALL(6)":
            sub = d
        else:
            sub = d[d["city"] == city]
        n = len(sub)
        he_wc = (sub["human"] == sub["wc"]).mean()
        lo, hi = wilson(int((sub["human"] == sub["wc"]).sum()), n)
        row = {"city": city, "n": n,
               "human=WC": round(he_wc, 3),
               "human=WC_CI": f"[{lo:.2f},{hi:.2f}]"}
        has = sub["dw"].notna()
        if has.any():
            s = sub[has]
            ns = len(s)
            tri = (s["human"] == s["dw"]) & (s["dw"] == s["esri"])
            quad = tri & (s["human"] == s["wc"])
            row.update({
                "human=DW": round((s["human"] == s["dw"]).mean(), 3),
                "human=ESRI": round((s["human"] == s["esri"]).mean(), 3),
                "WC=DW": round((s["wc"] == s["dw"]).mean(), 3),
                "WC=ESRI": round((s["wc"] == s["esri"]).mean(), 3),
                "DW=ESRI": round((s["dw"] == s["esri"]).mean(), 3),
                "tri(H,DW,ESRI)_agree": round(tri.mean(), 3),
                "all4_agree": round(quad.mean(), 3),
                "n_dwesri": ns,
            })
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# T2  confidence-stratified WC<->Human
# ---------------------------------------------------------------------------
def table_confidence(mw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for city in CITIES6 + ["ALL(6)"]:
        sub = mw if city == "ALL(6)" else mw[mw["city"] == city]
        row = {"city": city}
        for tier, mask in [
            ("all", np.ones(len(sub), bool)),
            ("high", (sub["conf"] == "h").to_numpy()),
            ("high+med", sub["conf"].isin(["h", "m"]).to_numpy()),
        ]:
            s = sub[mask]
            n = len(s)
            dis = (s["human"] != s["wc"]).mean() if n else np.nan
            k = int((s["human"] != s["wc"]).sum())
            lo, hi = wilson(k, n)
            row[f"{tier}_n"] = n
            row[f"{tier}_WC_disagree"] = round(dis, 3) if n else np.nan
            row[f"{tier}_CI"] = f"[{lo:.2f},{hi:.2f}]" if n else ""
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# T3  direction of the WC error  (WC -> Human confusion on boundary points)
# ---------------------------------------------------------------------------
def table_wc_error_direction(mw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conf_rows, dir_rows = [], []
    for city in CITIES6 + ["ALL(6)"]:
        sub = mw if city == "ALL(6)" else mw[mw["city"] == city]
        cm = pd.crosstab(sub["wc"].map(CLASSES), sub["human"].map(CLASSES),
                         dropna=False).reindex(index=list(CLASSES.values()),
                                               columns=list(CLASSES.values()),
                                               fill_value=0)
        cm.index = [f"WC={i}" for i in cm.index]
        cm.insert(0, "city", city)
        conf_rows.append(cm.reset_index(names="row"))

        wc_built = sub[sub["wc"] == 0]
        wc_veg = sub[sub["wc"] == 1]
        dir_rows.append({
            "city": city, "n": len(sub),
            "n_WC=built": len(wc_built),
            "P(expert=non-built | WC=built)": round((wc_built["human"] == 1).mean(), 3) if len(wc_built) else np.nan,
            "P(expert=water | WC=built)": round((wc_built["human"] == 2).mean(), 3) if len(wc_built) else np.nan,
            "n_WC=non-built": len(wc_veg),
            "P(expert=built | WC=non-built)": round((wc_veg["human"] == 0).mean(), 3) if len(wc_veg) else np.nan,
            "P(expert=water | WC=non-built)": round((wc_veg["human"] == 2).mean(), 3) if len(wc_veg) else np.nan,
            "net_built_over_call": round(
                ((wc_built["human"] == 1).sum() - (wc_veg["human"] == 0).sum()) / max(len(sub), 1), 3),
        })
    return pd.concat(conf_rows, ignore_index=True), pd.DataFrame(dir_rows)


# ---------------------------------------------------------------------------
# T4  scene x disagreement
# ---------------------------------------------------------------------------
def table_scene() -> pd.DataFrame:
    s = pd.read_csv(SCENE_CSV)
    # Nanchang: 26 points have no Scene text recorded (annotator only wrote a
    # scene note when flagging a WC disagreement -> these 26 are systematically
    # all Human=WC agreements, disagree_rate=0.0, not a random missingness).
    # Excluded here rather than reported as a meaningless "Unlabeled" bucket.
    s = s[s["Scene_group"] != "Unlabeled"]
    g = s.groupby("Scene_group").agg(n=("Disagree", "size"),
                                     n_disagree=("Disagree", "sum")).reset_index()
    g["disagree_pct"] = (100 * g["n_disagree"] / g["n"]).round(1)
    ci = g.apply(lambda r: wilson(int(r["n_disagree"]), int(r["n"])), axis=1)
    g["CI95"] = ci.map(lambda t: f"[{100*t[0]:.0f},{100*t[1]:.0f}]")
    return g.sort_values("disagree_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# T5  operator transfer digest
# ---------------------------------------------------------------------------
def table_operator() -> pd.DataFrame | None:
    p1 = LOCO / "phase1_corrector_transfer.csv"
    p3 = LOCO / "phase3_referee_eval.csv"
    if not p1.exists():
        return None
    a = pd.read_csv(p1)
    a = a[a["feature_set"] == "all27"][["held_out", "wc_vs_human_acc",
                                        "operator_acc", "n_operator_fires",
                                        "fire_precision_vs_human"]]
    if p3.exists():
        b = pd.read_csv(p3)
        b = b[b["method"] == "LOCO_corrected"][["city", "OA_vs_Human",
                                               "changed_toward_Human_frac",
                                               "n_changed"]]
        a = a.merge(b, left_on="held_out", right_on="city", how="left").drop(columns=["city"])
    return a


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_fourway(t1: pd.DataFrame):
    d = t1[t1["city"].isin(CITIES6)].set_index("city").reindex(CITIES6)
    x = np.arange(len(CITIES6)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w, d["human=WC"], w, label="Human = WorldCover", color="#c1443c")
    ax.bar(x, d["human=DW"], w, label="Human = Dynamic World", color="#e0983c")
    ax.bar(x + w, d["human=ESRI"], w, label="Human = ESRI", color="#4c78a8")
    ax.axhline(1 / 3, ls=":", c="grey", lw=1)
    ax.text(len(CITIES6) - 0.5, 0.34, "chance (3-class)", fontsize=8, c="grey", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(CITIES6)
    ax.set_ylabel("agreement with expert label")
    ax.set_ylim(0, 0.75)
    ax.set_title("Global products vs expert on urban built/vegetation boundaries")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "Fig_diag_fourway.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_confidence(t2: pd.DataFrame):
    d = t2[t2["city"].isin(CITIES6)].set_index("city").reindex(CITIES6)
    x = np.arange(len(CITIES6)); w = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w, d["all_WC_disagree"], w, label="all human labels", color="#9ecae1")
    ax.bar(x, d["high+med_WC_disagree"], w, label="high+medium conf.", color="#4c78a8")
    ax.bar(x + w, d["high_WC_disagree"], w, label="high confidence only", color="#08306b")
    ax.set_xticks(x); ax.set_xticklabels(CITIES6)
    ax.set_ylabel("WorldCover disagrees with expert")
    ax.set_ylim(0, 1)
    ax.set_title("Where the expert is most certain, WorldCover is not more right")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_diag_confidence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_scene(t4: pd.DataFrame):
    d = t4[t4["n"] >= 15].sort_values("disagree_pct")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d["disagree_pct"], color="#c1443c")
    ax.set_yticks(y); ax.set_yticklabels(d["Scene_group"], fontsize=8)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["disagree_pct"] + 1, i, f"n={int(r['n'])}", va="center", fontsize=7)
    ax.set_xlabel("Human ≠ WorldCover  (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Which physical boundary types drive the disagreement (5 cities pooled)")
    fig.tight_layout()
    fig.savefig(FIG / "Fig_diag_scene.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_error_direction(dir_df: pd.DataFrame):
    d = dir_df[dir_df["city"].isin(CITIES6)].set_index("city").reindex(CITIES6)
    x = np.arange(len(CITIES6)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w / 2, d["P(expert=non-built | WC=built)"], w,
           label="WC says built, expert says vegetation", color="#2e7d32")
    ax.bar(x + w / 2, d["P(expert=built | WC=non-built)"], w,
           label="WC says vegetation, expert says built", color="#8d6e63")
    ax.set_xticks(x); ax.set_xticklabels(CITIES6)
    ax.set_ylabel("conditional error rate")
    ax.set_title("Direction of the WorldCover boundary error")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "Fig_diag_error_direction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    mw = load_manual_wc()
    de = load_dwesri()
    print(f"loaded {len(mw)} hand-labelled points ({mw.city.nunique()} cities); "
          f"{len(de)} with DW+ESRI")

    t1 = table_fourway(mw, de)
    t2 = table_confidence(mw)
    conf_cm, dir_df = table_wc_error_direction(mw)
    t4 = table_scene()
    t5 = table_operator()

    t1.to_csv(OUT / "T1_fourway_agreement.csv", index=False)
    t2.to_csv(OUT / "T2_confidence_stratified.csv", index=False)
    conf_cm.to_csv(OUT / "T3a_WC_to_human_confusion.csv", index=False)
    dir_df.to_csv(OUT / "T3b_WC_error_direction.csv", index=False)
    t4.to_csv(OUT / "T4_scene_disagreement.csv", index=False)
    if t5 is not None:
        t5.to_csv(OUT / "T5_operator_digest.csv", index=False)

    fig_fourway(t1)
    fig_confidence(t2)
    fig_scene(t4)
    fig_error_direction(dir_df)

    with (OUT / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        for name, df in [("T1 four-way agreement", t1),
                         ("T2 confidence-stratified WC<->Human", t2),
                         ("T3a WC->Human confusion (boundary points)", conf_cm),
                         ("T3b direction of WC error", dir_df),
                         ("T4 scene x disagreement (pooled)", t4),
                         ("T5 LOCO operator digest", t5 if t5 is not None else pd.DataFrame())]:
            f.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
            f.write(df.to_string(index=False) + "\n")
    print((OUT / "SUMMARY.txt").read_text())
    print("figures ->", FIG / "Fig_diag_*.png")


if __name__ == "__main__":
    main()
