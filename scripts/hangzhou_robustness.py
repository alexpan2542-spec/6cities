#!/usr/bin/env python3
"""
Hangzhou-exclusion robustness check (Option 1 for reviewer concern:
"why is Hangzhou sampled with a different BAMS rule?").

Hangzhou's 150 boundary points were drawn with the early weighted-blend
variant S_HZ = 0.7*(1-m)_norm + 0.3*BoundaryScore_norm; the other five
cities use the frozen two-stage cut (1000 lowest-margin -> 150 highest
BoundaryScore). This script recomputes every *pooled headline* diagnostic
on the five two-stage cities only (n = 750) and prints it beside the
published six-city pooled value (n = 900), with the delta.

Reuses the frozen loaders from diagnostic_analysis.py / independence_diagnostic.py
so the numbers are directly comparable to docs/results_snapshot/.

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/hangzhou_robustness.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diagnostic_analysis import load_manual_wc, load_dwesri, wilson, CLASSES
from independence_diagnostic import (
    yules_q, cohen_kappa, expected_agree_cond_indep, boot_ci,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "analysis_outputs" / "hangzhou_robustness"
OUT.mkdir(parents=True, exist_ok=True)

TWO_STAGE = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"]  # frozen rule
ALL6 = TWO_STAGE + ["Hangzhou"]


def build() -> pd.DataFrame:
    mw = load_manual_wc()
    de = load_dwesri()
    d = mw.merge(de, on=["city", "Original_ID"], how="inner")
    for c in ["human", "wc", "dw", "esri"]:
        d[c] = d[c].astype(int)
    return d


def pooled_headline(d: pd.DataFrame) -> dict:
    """Every pooled number the manuscript leads with, for one city subset."""
    y = d["human"].to_numpy()
    wc, dw, es = d["wc"].to_numpy(), d["dw"].to_numpy(), d["esri"].to_numpy()
    n = len(d)
    r = {"n": n}

    # --- T1 four-way agreement -------------------------------------------------
    r["T1 human=WC"] = np.mean(y == wc)
    r["T1 human=DW"] = np.mean(y == dw)
    r["T1 human=ESRI"] = np.mean(y == es)
    r["T1 DW=ESRI"] = np.mean(dw == es)
    tri = (y == dw) & (dw == es)
    r["T1 tri(H,DW,ESRI)"] = np.mean(tri)
    r["T1 all4_agree"] = np.mean(tri & (y == wc))

    # --- T2 confidence-stratified WC disagreement ----------------------------
    conf = d["conf"].to_numpy()
    dis = (y != wc)
    r["T2 WC-disagree all"] = np.mean(dis)
    hm = np.isin(conf, ["h", "m"])
    r["T2 WC-disagree high+med"] = np.mean(dis[hm])
    hi = conf == "h"
    r["T2 WC-disagree high-only"] = np.mean(dis[hi])
    r["T2 n_high"] = int(hi.sum())

    # --- T3b direction of the WC error -------------------------------------
    wcb = wc == 0
    wcv = wc == 1
    r["T3b P(expert=non-built | WC=built)"] = np.mean(y[wcb] == 1)
    r["T3b P(expert=built | WC=non-built)"] = np.mean(y[wcv] == 0)
    r["T3b net_built_over_call"] = ((y[wcb] == 1).sum() - (y[wcv] == 0).sum()) / n

    # --- S2a excess agreement over conditional independence ---------------
    for na, a, b in [("DW=ESRI", dw, es), ("WC=DW", wc, dw), ("WC=ESRI", wc, es)]:
        obs = float(np.mean(a == b))
        exp = expected_agree_cond_indep(a, b, y)
        r[f"S2a {na} observed"] = obs
        r[f"S2a {na} expected_ci"] = exp
        r[f"S2a {na} excess"] = obs - exp

    # --- S2b error dependence --------------------------------------------
    e = {"wc": wc != y, "dw": dw != y, "esri": es != y}
    for na, ka, kb in [("DW,ESRI", "dw", "esri"), ("WC,DW", "wc", "dw"), ("WC,ESRI", "wc", "esri")]:
        ea, eb = e[ka], e[kb]
        co = float(np.mean(ea & eb))
        indep = float(np.mean(ea) * np.mean(eb))
        r[f"S2b {na} Yule Q"] = yules_q(ea, eb)
        r[f"S2b {na} co_error_ratio"] = co / indep if indep else np.nan
        r[f"S2b {na} kappa_err"] = cohen_kappa(ea, eb)

    # --- S2c does product consensus predict the expert label? -------------
    base = {c: float(np.mean(y == c)) for c in CLASSES}
    r["S2c baseline P(expert=DW)"] = np.mean(dw == y)
    r["S2c baseline P(expert=ESRI)"] = np.mean(es == y)
    con = dw == es
    p = float(np.mean(y[con] == dw[con]))
    prior = float(np.mean([base[c] for c in dw[con]]))
    r["S2c DW==ESRI -> P(expert agrees)"] = p
    r["S2c DW==ESRI lift_vs_prior"] = p - prior
    conwd = wc == dw
    pwd = float(np.mean(y[conwd] == dw[conwd]))
    priorwd = float(np.mean([base[c] for c in dw[conwd]]))
    r["S2c WC==DW -> P(expert agrees)"] = pwd
    r["S2c WC==DW lift_vs_prior"] = pwd - priorwd

    # --- S3a ontology confound: 2-class (built vs rest) -------------------
    m = {"y": np.where(y == 2, 1, y), "wc": np.where(wc == 2, 1, wc),
         "dw": np.where(dw == 2, 1, dw), "esri": np.where(es == 2, 1, es)}
    r["S3a 2class DW=ESRI"] = np.mean(m["dw"] == m["esri"])
    r["S3a 2class Human=DW"] = np.mean(m["y"] == m["dw"])
    r["S3a 2class gap(DW=ESRI - Human=DW)"] = np.mean(m["dw"] == m["esri"]) - np.mean(m["y"] == m["dw"])

    return r


def main():
    d = build()
    d6 = d[d["city"].isin(ALL6)].reset_index(drop=True)
    d5 = d[d["city"].isin(TWO_STAGE)].reset_index(drop=True)
    assert len(d6) == 900 and len(d5) == 750, (len(d6), len(d5))

    h6 = pooled_headline(d6)
    h5 = pooled_headline(d5)

    rows = []
    for k in h6:
        v6, v5 = h6[k], h5[k]
        if k.startswith("n") or k.startswith("T2 n"):
            rows.append({"metric": k, "6-city (n=900)": v6, "5-city (n=750)": v5, "delta (5-6)": v5 - v6})
        else:
            rows.append({"metric": k,
                         "6-city (n=900)": round(float(v6), 3),
                         "5-city (n=750)": round(float(v5), 3),
                         "delta (5-6)": round(float(v5 - v6), 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "headline_5city_vs_6city.csv", index=False)

    # headline Yule's Q, 5-city, with bootstrap CI (matches independence script)
    y5 = d5["human"].to_numpy()
    edw = d5["dw"].to_numpy() != y5
    ees = d5["esri"].to_numpy() != y5
    qlo, qhi = boot_ci(yules_q, edw, ees)

    with (OUT / "SUMMARY.txt").open("w", encoding="utf-8") as f:
        f.write("HANGZHOU-EXCLUSION ROBUSTNESS  (Option 1)\n")
        f.write("=" * 74 + "\n")
        f.write("Five two-stage-rule cities (Wuhan, Hefei, Nanchang, Nanjing, Changsha),\n")
        f.write("n = 750, vs the published six-city pooled value, n = 900.\n")
        f.write("Per-city numbers are unchanged (frozen snapshot); only the POOLED\n")
        f.write("headline figures can move, and this is the check for that.\n\n")
        f.write(tbl.to_string(index=False) + "\n\n")
        f.write(f"Headline DW,ESRI error coupling, 5-city: Yule's Q = "
                f"{yules_q(edw, ees):.3f}  boot95% [{qlo:.2f},{qhi:.2f}]\n")
        f.write(f"  (published 6-city: Q = 0.971 [0.96,0.98])\n\n")
        mx = tbl.loc[~tbl['metric'].str.startswith(('n', 'T2 n')), 'delta (5-6)'].abs().max()
        f.write(f"Largest absolute shift in any pooled headline metric: {mx:.3f}\n")

    print((OUT / "SUMMARY.txt").read_text())
    print("written ->", OUT)


if __name__ == "__main__":
    main()
