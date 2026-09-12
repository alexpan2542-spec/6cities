#!/usr/bin/env python3
"""
Build the re-labelling worklist for standardising Hangzhou onto the frozen
two-stage BAMS150 rule (Option 2).

Splits Hangzhou's new two-stage 150-point set into:
  - keep_labelled.csv : points also in the current blend set -> expert label
                        already exists, carried over verbatim
  - to_label.csv      : points new under the two-stage rule -> need expert
                        photo-interpretation (GEE, 1:2000, same protocol)
  - dropped.csv       : blend-set points not in the two-stage set -> their
                        existing labels are retired (kept for the record)

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/hangzhou_standardise_worklist.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / "data" / "cities" / "Hangzhou"
REPRO = ROOT / "data" / "analysis_outputs" / "bams_reproduction" / "Hangzhou"
OUT = ROOT / "data" / "analysis_outputs" / "hangzhou_standardise"
OUT.mkdir(parents=True, exist_ok=True)

two_stage = pd.read_csv(REPRO / "Hangzhou_BAMS150_two_stage.csv")
two_stage["Original_ID"] = two_stage["Original_ID"].astype(int)

manual = pd.read_csv(CITY / "04_manual" / "Hangzhou_BAMS150_Manual.csv")
manual["Original_ID"] = manual["Original_ID"].astype(int)
labelled_ids = set(manual["Original_ID"])

blend = pd.read_csv(CITY / "03_top150" / "Hangzhou_BAMS150.csv")
blend["Original_ID"] = blend["Original_ID"].astype(int)
blend_ids = set(blend["Original_ID"])

ts_ids = set(two_stage["Original_ID"])

keep = two_stage[two_stage["Original_ID"].isin(labelled_ids)].copy()
to_label = two_stage[~two_stage["Original_ID"].isin(labelled_ids)].copy()
dropped = manual[~manual["Original_ID"].isin(ts_ids)].copy()

# attach the existing expert label to the carry-over set
keep = keep.merge(manual[["Original_ID", "Human_Class", "Scene", "Confidence", "Notes"]],
                  on="Original_ID", how="left")

# GEE-ready columns for the annotator: id, coords, WC class, the two BAMS
# signals, RF prediction. No expert-label column -> to be filled in GEE.
label_cols = ["PointID", "Original_ID", "lon", "lat", "Class", "RF_Pred",
              "margin", "BoundaryScore", "prob_1", "prob_2", "prob_3"]
to_label = to_label[label_cols].rename(columns={"Class": "WC_Class"})
to_label.insert(len(to_label.columns), "Human_Class", "")
to_label.insert(len(to_label.columns), "Scene", "")
to_label.insert(len(to_label.columns), "Confidence", "")
to_label.insert(len(to_label.columns), "Notes", "")
to_label = to_label.sort_values("Original_ID").reset_index(drop=True)
to_label["PointID"] = range(1, len(to_label) + 1)

keep.to_csv(OUT / "keep_labelled.csv", index=False)
to_label.to_csv(OUT / "to_label.csv", index=False)
dropped.to_csv(OUT / "dropped.csv", index=False)

wc_map = {1: "built", 2: "non-built", 3: "water"}
tl_wc = to_label["WC_Class"].map(wc_map).value_counts().to_dict()

print(f"""
Hangzhou -> two-stage BAMS150 standardisation worklist
=====================================================
current blend set        : 150 points, all expert-labelled
new two-stage set        : 150 points
  carried over (labelled) : {len(keep):>3}   ({OUT.name}/keep_labelled.csv)
  NEW, need expert label  : {len(to_label):>3}   ({OUT.name}/to_label.csv)
  retired blend-only pts   : {len(dropped):>3}   ({OUT.name}/dropped.csv)

to_label WC-class mix     : {tl_wc}
margin range (to_label)   : {to_label['margin'].min():.3f} - {to_label['margin'].max():.3f}

next steps
  1. label the {len(to_label)} points in to_label.csv in GEE (1:2000, 2021 S2
     composite, same protocol as the other cities; fill Human_Class / Scene /
     Confidence / Notes)
  2. concat keep_labelled.csv + the filled to_label.csv -> new
     Hangzhou_BAMS150_Manual.csv (150 rows)
  3. flip SELECTION_RULE['Hangzhou'] to 'two_stage' in
     reproduce_bams_selection.py, run with --in-place
  4. re-run diagnostic_analysis.py, independence_diagnostic.py,
     loco_correction_operator.py (Phase 3), and the figure scripts
  5. refresh docs/results_snapshot/ ; drop the Hangzhou special-case text
     (Eq. hzblend, the "five of six" split) from manuscript_sec3.tex and
     manuscript.md
""")
