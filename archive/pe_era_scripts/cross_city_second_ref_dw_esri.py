#!/usr/bin/env python3
"""
Cross-city second-reference check: Human vs WC vs Dynamic World vs ESRI.

For each city BAMS150 point, sample DW (2021 mode) and ESRI 10m LULC (2021),
remap to the project's 3-class scheme, and measure:
  - human==ref agreement
  - WC==DW / WC==ESRI
  - human==DW≠WC / human==ESRI≠WC  (key: is Nanjing highest?)

Requires Earth Engine: ee.Initialize(project='healthy-area-463312-i4')

Outputs -> data2/cross_city_second_ref_dw_esri/
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA2 = ROOT / "data2"
OUT = DATA2 / "cross_city_second_ref_dw_esri"
CITIES = ["Wuhan", "Changsha", "Nanchang", "Hefei", "Nanjing"]
EE_PROJECT = "healthy-area-463312-i4"
YEAR = 2021
SCALE = 10

# Project 3-class: 1=built, 2=non-built, 3=water
WC_RAW_TO_C3 = {
    10: 2,
    20: 2,
    30: 2,
    40: 2,
    50: 1,
    60: 2,
    70: 2,
    80: 3,
    90: 3,
    95: 3,
    100: 2,
}
# Dynamic World label: 0 water,1 trees,2 grass,3 flooded_veg,4 crops,5 shrub,6 built,7 bare,8 snow
DW_TO_C3 = {0: 3, 1: 2, 2: 2, 3: 3, 4: 2, 5: 2, 6: 1, 7: 2, 8: 2}
# ESRI 10m: 1 Water,2 Trees,4 Flooded,5 Crops,7 Built,8 Bare,9 Snow,10 Clouds,11 Rangeland
ESRI_TO_C3 = {1: 3, 2: 2, 4: 3, 5: 2, 7: 1, 8: 2, 9: 2, 11: 2}  # 10 clouds -> NaN


def norm_conf(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().lower()
    if s in ("h", "high"):
        return "high"
    if s in ("m", "med", "medium"):
        return "med"
    if s in ("l", "low"):
        return "low"
    return None


def load_bams_city(city: str) -> pd.DataFrame:
    city_root = DATA2 / city
    bams = pd.read_csv(city_root / "03_top150" / f"{city}_BAMS150.csv")
    man = pd.read_csv(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")
    feat = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    id2wc = {int(r.Original_ID): int(r.Class) for _, r in feat.iterrows()}

    man = man.copy()
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce").astype("Int64")
    if "Confidence" in man.columns:
        man["Conf"] = man["Confidence"].map(norm_conf)
    else:
        man["Conf"] = None

    keep = ["Original_ID", "lon", "lat", "Class", "margin"]
    keep = [c for c in keep if c in bams.columns]
    df = bams[keep].copy()
    df["Original_ID"] = df["Original_ID"].astype(int)
    df = df.merge(
        man[["Original_ID", "Human_Class", "Conf"]],
        on="Original_ID",
        how="left",
    )
    df["WC_local"] = df["Original_ID"].map(id2wc)
    if "Class" in df.columns:
        # BAMS Class is usually WC
        df["WC_bams"] = df["Class"].astype(int)
    df["City"] = city
    return df


def remap_series(s: pd.Series, mapping: dict) -> pd.Series:
    return s.map(mapping)


def sample_city_ee(city_df: pd.DataFrame, city: str) -> pd.DataFrame:
    import ee

    ee.Initialize(project=EE_PROJECT)

    feats = []
    for _, r in city_df.iterrows():
        feats.append(
            ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]),
                {
                    "Original_ID": int(r.Original_ID),
                    "City": city,
                },
            )
        )
    fc = ee.FeatureCollection(feats)
    bounds = fc.geometry().bounds()

    # DW 2021 mode label
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(bounds)
        .select("label")
        .mode()
        .rename("DW_raw")
    )

    # ESRI 2021 mosaic
    esri = (
        ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
        .filterDate(f"{YEAR}-01-01", f"{YEAR + 1}-01-01")
        .filterBounds(bounds)
        .mosaic()
        .select("b1")
        .rename("ESRI_raw")
    )

    # WC for QA
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("WC_ee_raw")

    stacked = dw.addBands(esri).addBands(wc)
    sampled = stacked.sampleRegions(
        collection=fc,
        properties=["Original_ID", "City"],
        scale=SCALE,
        geometries=False,
    )
    # getInfo in one shot for 150 pts
    raw = sampled.getInfo()
    rows = []
    for f in raw["features"]:
        p = f["properties"]
        rows.append(
            {
                "Original_ID": int(p["Original_ID"]),
                "City": p["City"],
                "DW_raw": p.get("DW_raw"),
                "ESRI_raw": p.get("ESRI_raw"),
                "WC_ee_raw": p.get("WC_ee_raw"),
            }
        )
    return pd.DataFrame(rows)


def agreement_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in df.groupby("City", sort=False):
        n = len(g)
        h = g["Human_Class"]
        wc = g["WC_C3"]
        dw = g["DW_C3"]
        es = g["ESRI_C3"]
        valid_dw = dw.notna() & h.notna()
        valid_es = es.notna() & h.notna()
        valid_wc = wc.notna() & h.notna()

        def rate(mask):
            return float(mask.sum() / max(int(mask.sum() + (~mask & True).sum() if False else len(mask)), 1))

        # cleaner rates
        def frac(num, den):
            den = int(den.sum()) if hasattr(den, "sum") else int(den)
            num = int(num.sum()) if hasattr(num, "sum") else int(num)
            return num / den if den else np.nan

        m_h_wc = valid_wc
        m_h_dw = valid_dw
        m_h_es = valid_es
        m_wc_dw = wc.notna() & dw.notna()
        m_wc_es = wc.notna() & es.notna()

        h_eq_wc = (h == wc) & m_h_wc
        h_eq_dw = (h == dw) & m_h_dw
        h_eq_es = (h == es) & m_h_es
        wc_eq_dw = (wc == dw) & m_wc_dw
        wc_eq_es = (wc == es) & m_wc_es

        # key patterns
        h_dw_not_wc = (h == dw) & (h != wc) & m_h_dw & m_h_wc
        h_es_not_wc = (h == es) & (h != wc) & m_h_es & m_h_wc
        h_eq_both_not_wc = (h == dw) & (h == es) & (h != wc) & m_h_dw & m_h_es & m_h_wc

        # among human≠WC, what fraction agree with DW/ESRI
        disagree = (h != wc) & m_h_wc
        among_dis_dw = disagree & m_h_dw
        among_dis_es = disagree & m_h_es

        rows.append(
            {
                "City": city,
                "n": n,
                "human_eq_WC": frac(h_eq_wc, m_h_wc),
                "human_eq_DW": frac(h_eq_dw, m_h_dw),
                "human_eq_ESRI": frac(h_eq_es, m_h_es),
                "WC_eq_DW": frac(wc_eq_dw, m_wc_dw),
                "WC_eq_ESRI": frac(wc_eq_es, m_wc_es),
                "human_eq_DW_neq_WC": frac(h_dw_not_wc, m_h_dw & m_h_wc),
                "human_eq_ESRI_neq_WC": frac(h_es_not_wc, m_h_es & m_h_wc),
                "human_eq_DW_and_ESRI_neq_WC": frac(h_eq_both_not_wc, m_h_dw & m_h_es & m_h_wc),
                "among_HneqWC_eq_DW": frac(among_dis_dw & (h == dw), among_dis_dw),
                "among_HneqWC_eq_ESRI": frac(among_dis_es & (h == es), among_dis_es),
                "n_HneqWC": int(disagree.sum()),
                "n_missing_DW": int(dw.isna().sum()),
                "n_missing_ESRI": int(es.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def high_only_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Conf" not in df.columns:
        return pd.DataFrame()
    sub = df[df["Conf"] == "high"].copy()
    if sub.empty:
        return pd.DataFrame()
    return agreement_table(sub)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_parts = []

    for city in CITIES:
        print(f"\n=== {city}: load local BAMS ===")
        local = load_bams_city(city)
        print(f"  n={len(local)} human labeled={local['Human_Class'].notna().sum()}")

        t0 = time.time()
        print(f"  sampling EE DW/ESRI/WC ({YEAR}) ...")
        ee_df = sample_city_ee(local, city)
        print(f"  EE done in {time.time() - t0:.1f}s; got {len(ee_df)}")

        merged = local.merge(ee_df, on=["Original_ID", "City"], how="left")
        merged["WC_C3"] = merged["WC_local"].astype("float")
        merged["Human_Class"] = merged["Human_Class"].astype("float")
        merged["DW_C3"] = remap_series(pd.to_numeric(merged["DW_raw"], errors="coerce"), DW_TO_C3)
        merged["ESRI_C3"] = remap_series(pd.to_numeric(merged["ESRI_raw"], errors="coerce"), ESRI_TO_C3)
        merged["WC_ee_C3"] = remap_series(pd.to_numeric(merged["WC_ee_raw"], errors="coerce"), WC_RAW_TO_C3)

        # QA: local WC vs EE WC
        qa = (merged["WC_C3"] == merged["WC_ee_C3"]).mean()
        print(f"  QA local WC == EE WC: {qa:.3f}")

        merged.to_csv(OUT / f"{city}_BAMS150_second_ref.csv", index=False)
        all_parts.append(merged)

    df = pd.concat(all_parts, ignore_index=True)
    df.to_csv(OUT / "all_cities_BAMS150_second_ref.csv", index=False)

    summary = agreement_table(df)
    summary.to_csv(OUT / "summary_by_city.csv", index=False)
    summary_h = high_only_table(df)
    if len(summary_h):
        summary_h.to_csv(OUT / "summary_by_city_high_only.csv", index=False)

    # ranks for key metric
    key = "human_eq_DW_neq_WC"
    ranked = summary.sort_values(key, ascending=False).reset_index(drop=True)
    ranked["rank_" + key] = np.arange(1, len(ranked) + 1)

    lines = [
        "# Cross-city second reference: Human × WC × Dynamic World × ESRI\n",
        f"\n- Points: BAMS150 × 5 cities\n",
        f"- Year: {YEAR}\n",
        f"- DW: GOOGLE/DYNAMICWORLD/V1 label mode\n",
        f"- ESRI: sat-io ESRI_Global-LULC_10m_TS 2021 mosaic\n",
        f"- Remap: built=1, non-built=2, water=3 (see config.json)\n",
        "\n## All Confidence\n\n",
        summary.to_string(index=False),
        "\n\n## Key question: is Nanjing highest on human≈DW≠WC?\n\n",
        ranked[["City", key, "human_eq_ESRI_neq_WC", "human_eq_WC", "human_eq_DW", "WC_eq_DW"]].to_string(
            index=False
        ),
        "\n",
    ]
    if len(summary_h):
        lines += [
            "\n## Confidence=high only\n\n",
            summary_h.to_string(index=False),
            "\n",
        ]

    nj = summary.loc[summary["City"] == "Nanjing"].iloc[0]
    top = ranked.iloc[0]
    verdict = (
        f"Nanjing human≈DW≠WC = {nj[key]:.3f}; "
        f"highest city = {top['City']} ({top[key]:.3f}). "
    )
    if top["City"] == "Nanjing":
        verdict += "YES — Nanjing is highest → supports WC isolation under human/DW agreement."
    else:
        verdict += (
            f"NO — {top['City']} higher. "
            "Cannot claim Nanjing WC uniquely isolated by this metric alone."
        )

    lines += ["\n## Verdict\n\n", verdict, "\n"]
    md = "".join(lines)
    (OUT / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT / "VERDICT.md").write_text(
        "# Second-reference VERDICT\n\n" + verdict + "\n\nSee SUMMARY.md for full tables.\n",
        encoding="utf-8",
    )
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "ee_project": EE_PROJECT,
                "year": YEAR,
                "scale": SCALE,
                "cities": CITIES,
                "dw_to_c3": DW_TO_C3,
                "esri_to_c3": ESRI_TO_C3,
                "wc_raw_to_c3": WC_RAW_TO_C3,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
