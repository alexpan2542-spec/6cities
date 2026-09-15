#!/usr/bin/env python3
"""
One-off fix: bams150_scene_grouped.csv carried Changsha's and Wuhan's
Human_Class/Disagree from a pre-correction snapshot (matching
data/cities/{city}/04_manual/{city}_BAMS150_Manual_bk.csv, Aug 8) instead of
the current, corrected 04_manual labels (Aug 9 self-QC pass: 36 Changsha
points and 7 Wuhan points changed class). Scene_group/Scene_raw/Confidence/
PointID are untouched -- only Human_Class and the derived Disagree flag are
recomputed against the current canonical labels. All other four cities are
already consistent with 04_manual and are left as-is (values just get
overwritten with the same numbers).

Backs up the pre-fix file, then rewrites bams150_scene_grouped.csv in place.

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/fix_scene_csv_changsha_wuhan.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITIES_DIR = ROOT / "data" / "cities"
SCENE_CSV = ROOT / "data" / "shared_reference" / "cross_city_scene_localization" / "bams150_scene_grouped.csv"
CITIES6 = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]


def canonical_human_wc() -> pd.DataFrame:
    rows = []
    for c in CITIES6:
        m = pd.read_csv(CITIES_DIR / c / "04_manual" / f"{c}_BAMS150_Manual.csv")
        m["Original_ID"] = m["Original_ID"].astype(int)
        m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
        m = m[m["Human_Class"].notna()].copy()
        m["human"] = m["Human_Class"].astype(int)
        orig = pd.read_csv(CITIES_DIR / c / "01_original" / f"{c}_WC_Samples_15000.csv")
        orig["Original_ID"] = orig["Original_ID"].astype(int)
        wc = orig.set_index("Original_ID")["Class"].astype(int)
        m["wc"] = m["Original_ID"].map(wc)
        m["city"] = c
        rows.append(m[["city", "Original_ID", "human", "wc"]])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    canon = canonical_human_wc()
    scene = pd.read_csv(SCENE_CSV)

    merged = scene.merge(canon, left_on=["City", "Original_ID"],
                          right_on=["city", "Original_ID"], how="left")
    assert merged["human"].notna().all(), "some scene rows have no canonical match"

    n_human_changed = int((merged["Human_Class"] != merged["human"]).sum())
    n_wc_changed = int((merged["WC_Class"] != merged["wc"]).sum())
    print(f"Human_Class changed on {n_human_changed} rows")
    print(f"WC_Class changed on {n_wc_changed} rows (expected 0)")
    print(merged.loc[merged["Human_Class"] != merged["human"], "City"].value_counts())

    bak = SCENE_CSV.with_suffix(".pre_changsha_wuhan_fix.bak.csv")
    if not bak.exists():
        bak.write_text(SCENE_CSV.read_text())
        print(f"backed up -> {bak.name}")
    else:
        print(f"backup already exists, not overwriting: {bak.name}")

    scene["Human_Class"] = merged["human"].astype(int)
    scene["Disagree"] = (scene["Human_Class"] != scene["WC_Class"]).astype(int)
    scene.to_csv(SCENE_CSV, index=False)
    print(f"wrote {SCENE_CSV}")


if __name__ == "__main__":
    main()
