#!/usr/bin/env python3
"""
One-shot data prep: fold Hangzhou's DW / ESRI / WorldCover-v200 samples into
the cross-city second-reference file, so the four-way / independence analyses
can run on all 6 cities (n = 900) instead of 5 (n = 750).

Input  (put it here after the GEE export finishes):
    data/shared_reference/cross_city_second_ref_dw_esri/Hangzhou_BAMS150_dwesri_raw.csv
      columns: PointID, Original_ID, lon, lat, DW_raw, ESRI_raw, WC_ee_raw
      (produced by data/gee_shp/Hangzhou_second_ref_DW_ESRI.js)

Outputs (overwrites):
    .../cross_city_second_ref_dw_esri/Hangzhou_BAMS150_second_ref.csv
    .../cross_city_second_ref_dw_esri/all_cities_BAMS150_second_ref.csv   (5 -> 6 cities)

Safety: the raw->3-class remap below is re-derived from, and asserted against,
the existing 5-city rows before any Hangzhou row is written. If the assert
trips, the mapping assumption is wrong -- stop and inspect, don't ship.

Run:
    /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 scripts/add_hangzhou_second_ref.py

After it succeeds, widen the DW/ESRI city gate in the two analysis scripts
(CITIES5 -> include "Hangzhou"; docstrings / "n = 750" strings) and re-run:
    scripts/diagnostic_analysis.py   scripts/independence_diagnostic.py
then refreeze docs/results_snapshot/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / "data" / "cities" / "Hangzhou"
REF = ROOT / "data" / "shared_reference" / "cross_city_second_ref_dw_esri"

RAW_IN = REF / "Hangzhou_BAMS150_dwesri_raw.csv"
HZ_OUT = REF / "Hangzhou_BAMS150_second_ref.csv"
ALL_OUT = REF / "all_cities_BAMS150_second_ref.csv"

COLS = [
    "Original_ID", "lon", "lat", "Class", "margin", "Human_Class", "Conf",
    "WC_local", "WC_bams", "City",
    "DW_raw", "ESRI_raw", "WC_ee_raw", "WC_C3", "DW_C3", "ESRI_C3", "WC_ee_C3",
]

# --- raw product code -> 3-class {1 built, 2 non-built, 3 water} --------------
# Matches the mapping already baked into the 5-city file (verified below).
# flooded / wetland vegetation -> water, consistent across all three products
# (DW 3, ESRI 4, WC 90/95).
def wc_to_c3(v: int) -> int:
    if v == 50:
        return 1
    if v in (80, 90, 95):
        return 3
    return 2  # 10,20,30,40,60,70,100


def dw_to_c3(v: int) -> int:
    if v == 6:
        return 1
    if v in (0, 3):
        return 3
    return 2  # 1,2,4,5,7,8,9


def esri_to_c3(v: int) -> int:
    if v == 7:
        return 1
    if v in (1, 4):
        return 3
    return 2  # 2,5,8,9,10,11


def _verify_against_5city() -> None:
    """Re-derive DW_C3 / ESRI_C3 / WC_ee_C3 from the raw columns of the existing
    5-city file and require an exact match. Guards the mapping assumption."""
    if not ALL_OUT.exists():
        sys.exit(f"missing {ALL_OUT}")
    df = pd.read_csv(ALL_OUT)
    if set(df["City"].unique()) - {"Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"}:
        sys.exit("all_cities file already contains a 6th city -- refusing to "
                 "re-append. Restore the 5-city version first.")
    checks = [
        ("WC_ee_raw", "WC_ee_C3", wc_to_c3),
        ("DW_raw", "DW_C3", dw_to_c3),
        ("ESRI_raw", "ESRI_C3", esri_to_c3),
    ]
    for raw_col, c3_col, fn in checks:
        got = df[raw_col].round().astype(int).map(fn)
        exp = df[c3_col].round().astype(int)
        bad = df.loc[got.values != exp.values, [raw_col, c3_col]]
        if len(bad):
            sys.exit(f"remap mismatch on {raw_col}->{c3_col}:\n{bad.drop_duplicates()}")
    print(f"remap verified against {len(df)} existing 5-city rows -- exact match")


def main() -> None:
    _verify_against_5city()

    if not RAW_IN.exists():
        sys.exit(f"missing GEE export: {RAW_IN}\n"
                 f"run data/gee_shp/Hangzhou_second_ref_DW_ESRI.js and drop "
                 f"the CSV there.")

    gee = pd.read_csv(RAW_IN)
    gee["Original_ID"] = gee["Original_ID"].astype(int)
    for c in ("DW_raw", "ESRI_raw", "WC_ee_raw"):
        if gee[c].isna().any():
            miss = gee.loc[gee[c].isna(), "Original_ID"].tolist()
            sys.exit(f"{c} is null for Original_IDs {miss} -- GEE returned no "
                     f"value at those points; inspect before proceeding.")
        gee[c] = gee[c].round().astype(int)
    if len(gee) != 150:
        sys.exit(f"expected 150 Hangzhou rows in the export, got {len(gee)}")

    # flag ESRI flooded-veg (code 4): not present in the 5-city data, so the
    # 4 -> water choice is an inference. Show them; they are still written.
    esri4 = gee.loc[gee["ESRI_raw"] == 4, "Original_ID"].tolist()
    if esri4:
        print(f"note: ESRI_raw == 4 (flooded veg) at Original_IDs {esri4} "
              f"-> mapped to water (3); review if that ontology is wrong here")

    manual = pd.read_csv(CITY / "04_manual" / "Hangzhou_BAMS150_Manual.csv")
    manual["Original_ID"] = manual["Original_ID"].astype(int)
    manual = manual[pd.to_numeric(manual["Human_Class"], errors="coerce").notna()]
    conf_map = {"h": "high", "m": "med", "l": "low"}
    manual["Conf"] = (manual["Confidence"].astype(str).str.strip().str.lower()
                      .map(conf_map))
    if manual["Conf"].isna().any():
        sys.exit("unmapped Confidence value in Hangzhou manual file")
    manual["Human_Class"] = pd.to_numeric(manual["Human_Class"]).map(
        lambda x: f"{float(x):.1f}")

    orig = pd.read_csv(CITY / "01_original" / "Hangzhou_WC_Samples_15000.csv")
    orig["Original_ID"] = orig["Original_ID"].astype(int)
    wc_class = orig.set_index("Original_ID")["Class"].round().astype(int)

    top150 = pd.read_csv(CITY / "03_top150" / "Hangzhou_BAMS150.csv")
    top150["Original_ID"] = top150["Original_ID"].astype(int)
    margin = top150.set_index("Original_ID")["margin"]

    df = gee.merge(manual[["Original_ID", "Human_Class", "Conf"]],
                   on="Original_ID", how="left")
    if df["Human_Class"].isna().any():
        sys.exit("some Hangzhou export points have no manual label -- id mismatch")

    df["Class"] = df["Original_ID"].map(wc_class)
    df["WC_local"] = df["Class"]
    df["WC_bams"] = df["Class"]
    df["WC_C3"] = df["Class"].astype(float)
    df["margin"] = df["Original_ID"].map(margin)
    if df["margin"].isna().any() or df["Class"].isna().any():
        sys.exit("Hangzhou Original_ID not found in 15000-pt / top150 file")
    df["City"] = "Hangzhou"

    df["DW_C3"] = df["DW_raw"].map(dw_to_c3)
    df["ESRI_C3"] = df["ESRI_raw"].map(esri_to_c3)
    df["WC_ee_C3"] = df["WC_ee_raw"].map(wc_to_c3)

    hz = df[COLS].sort_values("Original_ID").reset_index(drop=True)
    hz.to_csv(HZ_OUT, index=False)
    print(f"wrote {HZ_OUT}  ({len(hz)} rows)")

    all5 = pd.read_csv(ALL_OUT)
    all6 = pd.concat([all5, hz[all5.columns]], ignore_index=True)
    all6.to_csv(ALL_OUT, index=False)
    print(f"wrote {ALL_OUT}  ({len(all5)} -> {len(all6)} rows, "
          f"{all6['City'].nunique()} cities)")

    # quick headline for the new city
    m = (hz["Human_Class"].astype(float).astype(int))
    dw = hz["DW_C3"].astype(int)
    es = hz["ESRI_C3"].astype(int)
    wc = hz["WC_C3"].astype(int)
    print("\nHangzhou (n=150), boundary-candidate points:")
    print(f"  Human = WC     {(m == wc).mean():.3f}")
    print(f"  Human = DW     {(m == dw).mean():.3f}")
    print(f"  Human = ESRI   {(m == es).mean():.3f}")
    print(f"  DW = ESRI      {(dw == es).mean():.3f}")


if __name__ == "__main__":
    main()
