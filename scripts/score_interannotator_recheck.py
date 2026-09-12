#!/usr/bin/env python3
"""Score the blind inter-annotator recheck once colleagues send filled sheets back.

Expects filled CSVs (same columns as the blank templates, My_Class/My_Confidence/
My_Scene now filled in) placed under:
    data/interannotator_recheck/returned/annotator_{A,B,C}/{City}_Recheck20_BLANK.csv

Compares against data/interannotator_recheck/answer_key/recheck_answer_key.csv
(the reference labels, kept private from annotators).

Reports, per annotator and overall:
  - raw agreement vs reference
  - Cohen's kappa vs reference
And across the three annotators (independent of the reference):
  - Fleiss' kappa

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/score_interannotator_recheck.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "interannotator_recheck"
KEY = BASE / "answer_key" / "recheck_answer_key.csv"
RETURNED = BASE / "returned"
CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]
ANNOTATORS = ["A", "B", "C"]


def fleiss_kappa(table: np.ndarray) -> float:
    """table: (n_items, n_categories) counts of raters choosing each category."""
    n_items, n_cats = table.shape
    n_raters = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / (n_items * n_raters)
    P_i = (np.sum(table * table, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j ** 2)
    return (P_bar - P_e) / (1 - P_e)


def load_returned() -> pd.DataFrame:
    rows = []
    for person in ANNOTATORS:
        for city in CITIES:
            f = RETURNED / f"annotator_{person}" / f"{city}_Recheck20_BLANK.csv"
            if not f.exists():
                print(f"[missing] {f}")
                continue
            d = pd.read_csv(f)
            d["City"] = city
            d["Annotator"] = person
            rows.append(d[["City", "Annotator", "CheckID", "My_Class", "My_Confidence", "My_Scene"]])
    if not rows:
        raise SystemExit(
            f"No returned files found under {RETURNED}/annotator_{{A,B,C}}/. "
            "Place the filled sheets there first (same filenames as the blanks)."
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    key = pd.read_csv(KEY)
    returned = load_returned()

    merged = returned.merge(key, on=["City", "CheckID"], how="left")
    merged["My_Class"] = pd.to_numeric(merged["My_Class"], errors="coerce")
    missing = merged["My_Class"].isna()
    if missing.any():
        print(f"[warn] {missing.sum()} rows have no numeric My_Class (unfilled?) — dropped from scoring")
        merged = merged[~missing]

    print("=" * 60)
    print("Cohen's kappa vs reference (per annotator)")
    print("=" * 60)
    for person in ANNOTATORS:
        sub = merged[merged["Annotator"] == person]
        if sub.empty:
            continue
        agree = (sub["My_Class"] == sub["Ref_Class"]).mean()
        kappa = cohen_kappa_score(sub["Ref_Class"], sub["My_Class"])
        print(f"  {person}: n={len(sub):3d}  raw agreement={agree:.3f}  kappa={kappa:.3f}")

    print()
    print("Overall (all annotators pooled) vs reference")
    agree_all = (merged["My_Class"] == merged["Ref_Class"]).mean()
    kappa_all = cohen_kappa_score(merged["Ref_Class"], merged["My_Class"])
    print(f"  n={len(merged)}  raw agreement={agree_all:.3f}  kappa={kappa_all:.3f}")

    print()
    print("=" * 60)
    print("Fleiss' kappa across the 3 annotators (reference NOT included)")
    print("=" * 60)
    wide = merged.pivot_table(index=["City", "CheckID"], columns="Annotator", values="My_Class", aggfunc="first")
    wide = wide.dropna()
    cats = sorted(set(wide.values.flatten().astype(int)))
    table = np.zeros((len(wide), len(cats)))
    for i, (_, row) in enumerate(wide.iterrows()):
        for v in row.values:
            table[i, cats.index(int(v))] += 1
    fk = fleiss_kappa(table)
    print(f"  n_items={len(wide)} (points all 3 annotators completed)  categories={cats}")
    print(f"  Fleiss' kappa = {fk:.3f}")

    print()
    print("Disagreement detail (any annotator != reference):")
    disagree = merged[merged["My_Class"] != merged["Ref_Class"]]
    cols = ["City", "CheckID", "Annotator", "Ref_Class", "My_Class", "Ref_Scene", "My_Scene"]
    print(disagree[cols].to_string(index=False))

    out = BASE / "results"
    out.mkdir(exist_ok=True)
    merged.to_csv(out / "merged_scored.csv", index=False)
    disagree[cols].to_csv(out / "disagreements.csv", index=False)
    print(f"\nSaved -> {out}/merged_scored.csv, {out}/disagreements.csv")


if __name__ == "__main__":
    main()
