#!/usr/bin/env python3
"""
One-off fix: all_cities_BAMS150_second_ref.csv carries Nanchang's Human_Class
from a pre-correction snapshot for 14 points -- these are part of the
documented 26-point Nanchang re-annotation (Section 3.2(d): points initially
placeholder-filled from WorldCover, later re-annotated on sub-metre imagery).
The stale copy still shows Human_Class == WC_C3 (the placeholder) for these
14 points instead of the corrected label in 04_manual/Nanchang_BAMS150_Manual.csv.

WC_C3, DW_C3, ESRI_C3 and all other columns are untouched -- only Human_Class
is recomputed against the current canonical label. This file feeds
scripts/c2_chip_select.py (Figure S5 chip gallery) directly off its own
Human_Class column, so this fix can change which points that script selects.

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/fix_second_ref_nanchang.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITIES_DIR = ROOT / "data" / "cities"
SECOND_REF = (ROOT / "data" / "shared_reference" / "cross_city_second_ref_dw_esri"
              / "all_cities_BAMS150_second_ref.csv")
CITIES6 = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]


def canonical_human() -> pd.DataFrame:
    rows = []
    for c in CITIES6:
        m = pd.read_csv(CITIES_DIR / c / "04_manual" / f"{c}_BAMS150_Manual.csv")
        m["Original_ID"] = m["Original_ID"].astype(int)
        m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
        m = m[m["Human_Class"].notna()].copy()
        m["human"] = m["Human_Class"].astype(int)
        m["city"] = c
        rows.append(m[["city", "Original_ID", "human"]])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    canon = canonical_human()
    sr = pd.read_csv(SECOND_REF)
    sr["Human_Class"] = sr["Human_Class"].astype(int)

    merged = sr.merge(canon, left_on=["City", "Original_ID"],
                       right_on=["city", "Original_ID"], how="left")
    assert merged["human"].notna().all(), "some SECOND_REF rows have no canonical match"

    n_changed = int((merged["Human_Class"] != merged["human"]).sum())
    print(f"Human_Class changed on {n_changed} rows")
    print(merged.loc[merged["Human_Class"] != merged["human"], "City"].value_counts())

    bak = SECOND_REF.with_suffix(".pre_nanchang_fix.bak.csv")
    if not bak.exists():
        bak.write_text(SECOND_REF.read_text())
        print(f"backed up -> {bak.name}")
    else:
        print(f"backup already exists, not overwriting: {bak.name}")

    sr["Human_Class"] = merged["human"].astype(int)
    sr.to_csv(SECOND_REF, index=False)
    print(f"wrote {SECOND_REF}")


if __name__ == "__main__":
    main()
