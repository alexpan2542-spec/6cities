#!/usr/bin/env python3
"""
Apply the Hangzhou standardisation once the 104 new points are labelled.

Inputs
  data/analysis_outputs/hangzhou_standardise/keep_labelled.csv   (46, auto)
  data/analysis_outputs/hangzhou_standardise/labelled_104.csv    (104, YOU create
      this: paste the GEE "Dump log" block -- header
      PointID,Original_ID,Human_Class,Scene,Confidence,Notes -- into this file)

Output
  data/cities/Hangzhou/04_manual/Hangzhou_BAMS150_Manual.csv     (rewritten, 150)
  old file backed up to  ..._Manual.blend.bak.csv

Then:
  1. edit scripts/reproduce_bams_selection.py -> SELECTION_RULE["Hangzhou"] = "two_stage"
  2. python scripts/reproduce_bams_selection.py --cities Hangzhou --in-place
  3. re-run diagnostic_analysis.py, independence_diagnostic.py,
     loco_correction_operator.py, and the figure scripts
  4. refresh docs/results_snapshot/  (copy data/analysis_outputs/* back)
  5. delete the Hangzhou special-case text (Eq. hzblend, the "five of six"
     split, Supplementary S4) from manuscript_sec3.tex / manuscript.md /
     manuscript_supplementary.md

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/hangzhou_standardise_apply.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "data" / "analysis_outputs" / "hangzhou_standardise"
REPRO = ROOT / "data" / "analysis_outputs" / "bams_reproduction" / "Hangzhou"
MANUAL = ROOT / "data" / "cities" / "Hangzhou" / "04_manual" / "Hangzhou_BAMS150_Manual.csv"

COLS = ["PointID", "Original_ID", "Human_Class", "Scene", "Confidence", "Notes"]


def die(msg: str) -> None:
    print("ERROR:", msg, file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    keep = pd.read_csv(WORK / "keep_labelled.csv")
    new_path = WORK / "labelled_104.csv"
    if not new_path.exists():
        die(f"{new_path} not found -- paste the GEE Dump-log block into it first "
            f"(header: {','.join(COLS)})")
    new = pd.read_csv(new_path)

    for name, df in (("keep_labelled", keep), ("labelled_104", new)):
        miss = [c for c in COLS if c not in df.columns]
        if miss:
            die(f"{name}.csv missing columns {miss}")
        df["Original_ID"] = df["Original_ID"].astype(int)
        df["Human_Class"] = pd.to_numeric(df["Human_Class"], errors="coerce")
        if df["Human_Class"].isna().any():
            bad = df.loc[df["Human_Class"].isna(), "Original_ID"].tolist()
            die(f"{name}.csv has blank/non-numeric Human_Class for Original_ID {bad}")
        df["Human_Class"] = df["Human_Class"].astype(int)
        if not df["Human_Class"].isin([1, 2, 3]).all():
            die(f"{name}.csv Human_Class not in {{1,2,3}}")
        if "Confidence" in df:
            df["Confidence"] = df["Confidence"].astype(str).str.strip().str.lower()
            if not df["Confidence"].isin(["h", "m", "l"]).all():
                die(f"{name}.csv Confidence not all in {{h,m,l}}")

    if len(new) != 104:
        die(f"labelled_104.csv has {len(new)} rows, expected 104")
    if len(keep) != 46:
        die(f"keep_labelled.csv has {len(keep)} rows, expected 46")

    two_stage = pd.read_csv(REPRO / "Hangzhou_BAMS150_two_stage.csv")
    two_stage["Original_ID"] = two_stage["Original_ID"].astype(int)
    ts_ids = set(two_stage["Original_ID"])

    merged_ids = set(keep["Original_ID"]) | set(new["Original_ID"])
    if merged_ids != ts_ids:
        die(f"keep+new Original_IDs do not match the two-stage set "
            f"(only_here={sorted(merged_ids - ts_ids)[:10]}, "
            f"only_two_stage={sorted(ts_ids - merged_ids)[:10]})")
    if set(keep["Original_ID"]) & set(new["Original_ID"]):
        die("keep_labelled and labelled_104 overlap on Original_ID")

    out = pd.concat([keep[COLS], new[COLS]], ignore_index=True)
    # renumber PointID 1..150 in the two-stage BAMS150 order
    order = {oid: i + 1 for i, oid in enumerate(two_stage["Original_ID"])}
    out["PointID"] = out["Original_ID"].map(order)
    out = out.sort_values("PointID").reset_index(drop=True)
    out["Scene"] = out["Scene"].fillna("").astype(str)
    out["Notes"] = out["Notes"].fillna("").astype(str)

    if len(out) != 150 or out["Original_ID"].duplicated().any():
        die(f"assembled file is {len(out)} rows / dup Original_ID")

    bak = MANUAL.with_suffix(".blend.bak.csv")
    if MANUAL.exists() and not bak.exists():
        bak.write_text(MANUAL.read_text())
        print(f"backed up old manual -> {bak.name}")
    out.to_csv(MANUAL, index=False)

    conf = out["Confidence"].str.lower().value_counts().to_dict()
    cls = out["Human_Class"].value_counts().to_dict()
    print(f"wrote {MANUAL.relative_to(ROOT)}  ({len(out)} rows)")
    print(f"  Human_Class: built={cls.get(1,0)} non-built={cls.get(2,0)} water={cls.get(3,0)}")
    print(f"  Confidence : {conf}")
    print("\nnext: flip SELECTION_RULE['Hangzhou']='two_stage' in "
          "reproduce_bams_selection.py, then --in-place, then re-run the analyses.")


if __name__ == "__main__":
    main()
