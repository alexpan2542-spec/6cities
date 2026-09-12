#!/usr/bin/env python3
"""
Independence diagnostic -- turns the "DW and ESRI agree with each other but not
with the expert" observation into a formal statement, and rules out the cheap
confounds, so the paper can lead with the *statistical* result (violation of the
conditional-independence assumption that multi-reference arbitration relies on)
and leave "shared training priors" as a discussion-level hypothesis.

Companion to scripts/diagnostic_analysis.py -- same anchor data, same loaders.
All 6 cities have DW + ESRI (Hangzhou added 2026-09-07); n = 900 boundary
candidate points.
Class codes after -1 shift: 0 built, 1 non-built, 2 water.

Outputs -> data/analysis_outputs/independence_diagnostic/  (tables + SUMMARY.txt)
           data/analysis_outputs/independence_diagnostic/Fig_diag_independence.png

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/independence_diagnostic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic_analysis import (  # reuse the frozen loaders
    load_manual_wc, load_dwesri, wilson, CITIES6, CLASSES,
)

ROOT = Path(__file__).resolve().parents[1]
DATA2 = ROOT / "data"
OUT = DATA2 / "analysis_outputs" / "independence_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT  # co-located with this script's own tables
SCENE_CSV = DATA2 / "shared_reference" / "cross_city_scene_localization" / "bams150_scene_grouped.csv"

RNG = np.random.default_rng(20260906)
NBOOT = 4000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def yules_q(a: np.ndarray, b: np.ndarray) -> float:
    """Yule's Q on the 2x2 of two boolean error indicators."""
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    num = n11 * n00 - n10 * n01
    den = n11 * n00 + n10 * n01
    return num / den if den else np.nan


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa between two boolean vectors (agreement corrected for chance)."""
    n = len(a)
    po = np.mean(a == b)
    pa1 = np.mean(a); pb1 = np.mean(b)
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) else np.nan


def expected_agree_cond_indep(pa: np.ndarray, pb: np.ndarray,
                              y: np.ndarray, k_classes: int = 3) -> float:
    """P(A == B) expected if A, B are independent *given the true class y*.

    Conditional class distributions P(A = c | Y = j) are plugged in from the
    sample itself (slight optimism toward the independence fit -- noted in the
    write-up); y is the expert label used as the truth proxy.
    """
    exp = 0.0
    n = len(y)
    for j in range(k_classes):
        mask = y == j
        nj = int(mask.sum())
        if nj == 0:
            continue
        pj = nj / n
        for c in range(k_classes):
            pac = np.mean(pa[mask] == c)
            pbc = np.mean(pb[mask] == c)
            exp += pj * pac * pbc
    return exp


def boot_ci(fn, *arrays, nboot: int = NBOOT, alpha: float = 0.05):
    """Point-level bootstrap CI for a scalar statistic fn(*resampled_arrays)."""
    n = len(arrays[0])
    stats = np.empty(nboot)
    for i in range(nboot):
        idx = RNG.integers(0, n, n)
        stats[i] = fn(*[arr[idx] for arr in arrays])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def city_jackknife(fn, df: pd.DataFrame, *cols):
    """Leave-one-city-out values for a scalar statistic."""
    vals = {}
    for c in CITIES6:
        s = df[df["city"] != c]
        vals[c] = fn(*[s[col].to_numpy() for col in cols])
    v = np.array(list(vals.values()))
    return vals, float(v.min()), float(v.max())


# ---------------------------------------------------------------------------
def load_all() -> pd.DataFrame:
    mw = load_manual_wc()
    de = load_dwesri()
    d = mw.merge(de, on=["city", "Original_ID"], how="inner")
    d = d[d["city"].isin(CITIES6)].reset_index(drop=True)
    for col in ["human", "wc", "dw", "esri"]:
        d[col] = d[col].astype(int)
    return d


# ---------------------------------------------------------------------------
# S2-a  observed vs expected-under-conditional-independence agreement
# ---------------------------------------------------------------------------
def table_excess_agreement(d: pd.DataFrame) -> pd.DataFrame:
    y = d["human"].to_numpy()
    pairs = [("DW", "ESRI", "dw", "esri"),
             ("WC", "DW", "wc", "dw"),
             ("WC", "ESRI", "wc", "esri")]
    rows = []
    for na, nb, ca, cb in pairs:
        pa = d[ca].to_numpy(); pb = d[cb].to_numpy()
        obs = float(np.mean(pa == pb))
        exp = expected_agree_cond_indep(pa, pb, y)
        excess = obs - exp

        def _excess(pa_, pb_, y_):
            return float(np.mean(pa_ == pb_)) - expected_agree_cond_indep(pa_, pb_, y_)

        lo, hi = boot_ci(_excess, pa, pb, y)
        _, jklo, jkhi = city_jackknife(
            lambda a, b, yy: float(np.mean(a == b)) - expected_agree_cond_indep(a, b, yy),
            d, ca, cb, "human")
        rows.append({
            "pair": f"{na}={nb}",
            "observed_agree": round(obs, 3),
            "expected_if_cond_indep": round(exp, 3),
            "excess": round(excess, 3),
            "excess_boot_CI": f"[{lo:.2f},{hi:.2f}]",
            "excess_LOCO_range": f"[{jklo:.2f},{jkhi:.2f}]",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S2-b  error-dependence / ensemble-diversity statistics
# ---------------------------------------------------------------------------
def table_error_dependence(d: pd.DataFrame) -> pd.DataFrame:
    y = d["human"].to_numpy()
    err = {k: (d[k].to_numpy() != y) for k in ["wc", "dw", "esri"]}
    pairs = [("DW", "ESRI", "dw", "esri"),
             ("WC", "DW", "wc", "dw"),
             ("WC", "ESRI", "wc", "esri")]
    rows = []
    for na, nb, ca, cb in pairs:
        ea, eb = err[ca], err[cb]
        both_wrong = ea & eb
        n_bw = int(both_wrong.sum())
        # among points where BOTH are wrong: how often the SAME wrong label
        same_when_both_wrong = (float(np.mean(d.loc[both_wrong, ca].to_numpy()
                                              == d.loc[both_wrong, cb].to_numpy()))
                                if n_bw else np.nan)
        co_err = float(np.mean(both_wrong))
        indep_co_err = float(np.mean(ea) * np.mean(eb))
        q = yules_q(ea, eb)
        qlo, qhi = boot_ci(yules_q, ea, eb)
        rows.append({
            "pair": f"{na},{nb} errors",
            "err_rate_A": round(float(np.mean(ea)), 3),
            "err_rate_B": round(float(np.mean(eb)), 3),
            "co_error_obs": round(co_err, 3),
            "co_error_if_indep": round(indep_co_err, 3),
            "co_error_ratio": round(co_err / indep_co_err, 2) if indep_co_err else np.nan,
            "yules_Q": round(q, 3),
            "yules_Q_CI": f"[{qlo:.2f},{qhi:.2f}]",
            "kappa_err": round(cohen_kappa(ea, eb), 3),
            "n_both_wrong": n_bw,
            "P(same wrong label | both wrong)": round(same_when_both_wrong, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S2-c  does product consensus carry information about truth?
#       (the arbitration assumption, tested directly)
# ---------------------------------------------------------------------------
def table_consensus_ppv(d: pd.DataFrame) -> pd.DataFrame:
    y = d["human"].to_numpy()
    base = {c: float(np.mean(y == c)) for c in CLASSES}
    rows = []

    # unconditional agreement with expert
    for name, col in [("DW", "dw"), ("ESRI", "esri"), ("WC", "wc")]:
        rows.append({
            "condition": f"(baseline) P(expert == {name})",
            "n": len(d),
            "P(expert agrees)": round(float(np.mean(d[col].to_numpy() == y)), 3),
            "lift_vs_prior": "",
        })

    # DW == ESRI  -> does expert follow?
    con = d[d["dw"] == d["esri"]]
    p = float(np.mean(con["human"].to_numpy() == con["dw"].to_numpy()))
    lo, hi = wilson(int((con["human"].to_numpy() == con["dw"].to_numpy()).sum()), len(con))
    # prior probability of the consensus class, averaged over consensus points
    prior_of_call = float(np.mean([base[c] for c in con["dw"].to_numpy()]))
    rows.append({
        "condition": "DW == ESRI  -> P(expert == that label)",
        "n": len(con),
        "P(expert agrees)": f"{p:.3f} [{lo:.2f},{hi:.2f}]",
        "lift_vs_prior": f"{p - prior_of_call:+.3f}  (prior {prior_of_call:.2f})",
    })

    # DW == ESRI == WC  (all three products agree)
    con3 = d[(d["dw"] == d["esri"]) & (d["dw"] == d["wc"])]
    if len(con3):
        p3 = float(np.mean(con3["human"].to_numpy() == con3["dw"].to_numpy()))
        lo3, hi3 = wilson(int((con3["human"].to_numpy() == con3["dw"].to_numpy()).sum()), len(con3))
        prior3 = float(np.mean([base[c] for c in con3["dw"].to_numpy()]))
        rows.append({
            "condition": "DW == ESRI == WC -> P(expert == that label)",
            "n": len(con3),
            "P(expert agrees)": f"{p3:.3f} [{lo3:.2f},{hi3:.2f}]",
            "lift_vs_prior": f"{p3 - prior3:+.3f}  (prior {prior3:.2f})",
        })

    # contrast: DW == WC (a pair that does NOT share the deep-seg/label lineage)
    con_wcdw = d[d["wc"] == d["dw"]]
    p_wd = float(np.mean(con_wcdw["human"].to_numpy() == con_wcdw["dw"].to_numpy()))
    lo_wd, hi_wd = wilson(int((con_wcdw["human"].to_numpy() == con_wcdw["dw"].to_numpy()).sum()), len(con_wcdw))
    prior_wd = float(np.mean([base[c] for c in con_wcdw["dw"].to_numpy()]))
    rows.append({
        "condition": "WC == DW  -> P(expert == that label)",
        "n": len(con_wcdw),
        "P(expert agrees)": f"{p_wd:.3f} [{lo_wd:.2f},{hi_wd:.2f}]",
        "lift_vs_prior": f"{p_wd - prior_wd:+.3f}  (prior {prior_wd:.2f})",
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S3-a  ontology confound: collapse to coarser schemes, re-test
# ---------------------------------------------------------------------------
def table_ontology_confound(d: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def block(name, dd, mapper):
        m = {k: dd[k].map(mapper).to_numpy() for k in ["human", "wc", "dw", "esri"]}
        return {
            "scheme": name,
            "n": len(dd),
            "Human=DW": round(float(np.mean(m["human"] == m["dw"])), 3),
            "Human=ESRI": round(float(np.mean(m["human"] == m["esri"])), 3),
            "DW=ESRI": round(float(np.mean(m["dw"] == m["esri"])), 3),
            "WC=DW": round(float(np.mean(m["wc"] == m["dw"])), 3),
            "WC=ESRI": round(float(np.mean(m["wc"] == m["esri"])), 3),
            "gap(DW=ESRI  -  Human=DW)": round(
                float(np.mean(m["dw"] == m["esri"]) - np.mean(m["human"] == m["dw"])), 3),
        }

    rows.append(block("3-class (built/non-built/water)", d, {0: 0, 1: 1, 2: 2}))
    rows.append(block("2-class (built vs rest)", d, {0: 0, 1: 1, 2: 1}))
    rows.append(block("built vs vegetation (water pts dropped)",
                      d[(d["human"] != 2) & (d["wc"] != 2) & (d["dw"] != 2) & (d["esri"] != 2)],
                      {0: 0, 1: 1, 2: 1}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# S3-b  WorldCover as negative control  (same S2 input, different model family)
# ---------------------------------------------------------------------------
def table_negative_control(excess: pd.DataFrame) -> pd.DataFrame:
    note = {
        "DW=ESRI": "deep semantic seg + shared point-label lineage (S2)",
        "WC=DW": "WC = boosted-tree + rules (S2); DW = deep seg (S2)",
        "WC=ESRI": "WC = boosted-tree + rules (S2); ESRI = deep seg (S2)",
    }
    out = excess.copy()
    out["shares_with"] = out["pair"].map(note)
    return out[["pair", "shares_with", "observed_agree",
               "expected_if_cond_indep", "excess", "excess_boot_CI"]]


# ---------------------------------------------------------------------------
# S3-c  is the DW=ESRI coupling uniform across scene types?
#       (argues against "same image, same day")
# ---------------------------------------------------------------------------
def table_coupling_by_scene(d: pd.DataFrame) -> pd.DataFrame:
    # Scene stratification is 5-city (the scene-code table has no Hangzhou) and,
    # within Nanchang, the ~26 points with no scene note are a systematic
    # missingness (annotator only wrote a note when flagging a disagreement,
    # see Paper_Writing_Plan.md 4.4 / 7.2). Both are dropped here rather than
    # pooled into a meaningless "Unlabeled" bucket -- same handling as T4.
    s = pd.read_csv(SCENE_CSV)[["City", "Original_ID", "Scene_group"]]
    s = s.rename(columns={"City": "city"})
    m = d.merge(s, on=["city", "Original_ID"], how="inner")
    m = m[m["Scene_group"] != "Unlabeled"]
    rows = []
    for g, sub in m.groupby("Scene_group"):
        if len(sub) < 15:
            continue
        dweq = float(np.mean(sub["dw"].to_numpy() == sub["esri"].to_numpy()))
        hdw = float(np.mean(sub["human"].to_numpy() == sub["dw"].to_numpy()))
        rows.append({"scene": g, "n": len(sub),
                     "DW=ESRI": round(dweq, 3),
                     "Human=DW": round(hdw, 3),
                     "gap": round(dweq - hdw, 3)})
    return pd.DataFrame(rows).sort_values("DW=ESRI", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def fig_independence(excess: pd.DataFrame, dep: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    ax = axes[0]
    pairs = excess["pair"].tolist()
    x = np.arange(len(pairs)); w = 0.38
    ax.bar(x - w / 2, excess["expected_if_cond_indep"], w,
           label="expected if conditionally independent", color="#b8b8b0")
    ax.bar(x + w / 2, excess["observed_agree"], w,
           label="observed", color="#c1443c")
    ax.set_xticks(x); ax.set_xticklabels(pairs)
    ax.set_ylabel("pairwise agreement")
    ax.set_ylim(0, 1)
    ax.set_title("Observed vs conditional-independence expectation")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    ax = axes[1]
    q = dep["yules_Q"].to_numpy()
    ax.bar(x, q, 0.5, color=["#c1443c", "#8d6e63", "#8d6e63"])
    ax.set_xticks(x); ax.set_xticklabels([p.replace(" errors", "") for p in dep["pair"]])
    ax.set_ylabel("Yule's Q  (error co-occurrence)")
    ymin, ymax = 0.0, 1.0              # same scale/ticks as the left panel
    ax.set_ylim(ymin, ymax)
    ax.set_title("Dependence between product errors vs expert")
    for i, v in enumerate(q):
        if v >= 0:
            if v + 0.08 > ymax:        # no room above the bar -> label inside it
                ax.text(i, v - 0.04, f"{v:.2f}", ha="center", va="top",
                        fontsize=9, color="white")
            else:
                ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(i, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "Fig_diag_independence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    d = load_all()
    print(f"loaded {len(d)} boundary points across {d.city.nunique()} cities "
          f"({', '.join(sorted(d.city.unique()))})")

    excess = table_excess_agreement(d)
    dep = table_error_dependence(d)
    ppv = table_consensus_ppv(d)
    onto = table_ontology_confound(d)
    negctl = table_negative_control(excess)
    byscene = table_coupling_by_scene(d)

    excess.to_csv(OUT / "S2a_excess_agreement.csv", index=False)
    dep.to_csv(OUT / "S2b_error_dependence.csv", index=False)
    ppv.to_csv(OUT / "S2c_consensus_ppv.csv", index=False)
    onto.to_csv(OUT / "S3a_ontology_confound.csv", index=False)
    negctl.to_csv(OUT / "S3b_negative_control.csv", index=False)
    byscene.to_csv(OUT / "S3c_coupling_by_scene.csv", index=False)

    fig_independence(excess, dep)

    blocks = [
        ("S2a  observed vs expected-if-conditionally-independent agreement\n"
         "     (truth proxied by expert label; 6 cities, n=900)", excess),
        ("S2b  dependence between product errors (vs expert)\n"
         "     co_error_ratio = observed / (marginal-independent); Q,kappa on error indicators", dep),
        ("S2c  does product consensus predict the expert label?\n"
         "     lift_vs_prior = P(expert agrees) - prior prob. of the agreed class", ppv),
        ("S3a  ontology confound: collapse to coarser schemes, re-test", onto),
        ("S3b  WorldCover as negative control (shared S2 input, different model family)", negctl),
        ("S3c  is the DW=ESRI coupling uniform across scene types?", byscene),
    ]
    with (OUT / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("INDEPENDENCE DIAGNOSTIC  --  data/independence_diagnostic/\n")
        f.write("anchor: BAMS150 boundary points, 6 cities with DW+ESRI, n=900\n")
        f.write("classes: 0 built / 1 non-built / 2 water\n")
        for name, df in blocks:
            f.write(f"\n{'='*74}\n{name}\n{'='*74}\n")
            f.write(df.to_string(index=False) + "\n")
    print((OUT / "SUMMARY.txt").read_text())
    print("figure ->", FIG / "Fig_diag_independence.png")


if __name__ == "__main__":
    main()
