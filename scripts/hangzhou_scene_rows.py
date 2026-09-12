#!/usr/bin/env python3
"""
Build Hangzhou's rows for bams150_scene_grouped.csv so the physical-scene
stratification (T4, independence S3c) can run six-city, n = 900.

Scene -> Scene_group map: the 43 vegetation-edge points use the manual
sub-classification (data/analysis_outputs/hangzhou_standardise/veg_subclass.csv:
ug -> Urban green space, fe -> Vegetation / forest edge). "ot" = the
vegetation-edge code did not survive review and the annotator recorded no
replacement type; these 9 points get no Scene_group and are dropped from the
grouping (like any point without a resolvable physical-scene note), so Hangzhou
contributes 141 codeable points and the pooled scene set is n = 891, not 900.
The other 107 points use the same keyword map the five existing cities use.

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 scripts/hangzhou_scene_rows.py
Appends Hangzhou (150 rows) to the scene CSV; backs up the 5-city version.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / "data" / "cities" / "Hangzhou"
REF = ROOT / "data" / "shared_reference" / "cross_city_scene_localization"
SCENE_CSV = REF / "bams150_scene_grouped.csv"
VEGSUB = ROOT / "data" / "analysis_outputs" / "hangzhou_standardise" / "veg_subclass.csv"

# raw scene note -> Scene_group  (five-city convention + Hangzhou's own codes)
SCENE_MAP = {
    # built-up
    "be": "Built-up edge / urban fabric", "br": "Built-up edge / urban fabric",
    "建筑": "Built-up edge / urban fabric", "car park": "Built-up edge / urban fabric",
    # bare
    "bl": "Bare land / construction", "土地": "Bare land / construction",
    "废土": "Bare land / construction", "荒地": "Bare land / construction",
    # water
    "we": "Water / water edge", "tw": "Water / water edge",
    "水体": "Water / water edge", "水体边缘": "Water / water edge",
    # road
    "re": "Road / road edge", "road": "Road / road edge", "土路": "Road / road edge",
    "路面": "Road / road edge", "高铁站": "Road / road edge",
    # rural
    "农村路面": "Rural settlement / surfaces",
    # cropland
    "ct": "Cropland / paddy transition", "田地": "Paddy / cropland / field",
    # mixed
    "mix": "Mixed / complex",
}
VEGSUB_MAP = {
    "ug": "Urban green space",
    "fe": "Vegetation / forest edge",
    "ot": "Unlabeled",   # no resolvable physical-scene type -> dropped downstream
}


def main() -> None:
    m = pd.read_csv(CITY / "04_manual" / "Hangzhou_BAMS150_Manual.csv")
    m["Original_ID"] = m["Original_ID"].astype(int)
    m["Human_Class"] = pd.to_numeric(m["Human_Class"]).astype(int)
    m["Confidence"] = m["Confidence"].astype(str).str.strip().str.lower()

    # WC class from the 15k pool
    orig = pd.read_csv(CITY / "01_original" / "Hangzhou_WC_Samples_15000.csv")
    orig["Original_ID"] = orig["Original_ID"].astype(int)
    wc = orig.set_index("Original_ID")["Class"].astype(int)
    m["WC_Class"] = m["Original_ID"].map(wc)

    # veg-edge sub-classification
    vs = pd.read_csv(VEGSUB)
    vs["Original_ID"] = vs["Original_ID"].astype(int)
    sub = vs.set_index("Original_ID")["subclass"].to_dict()

    rows = []
    unmapped = []
    for _, r in m.iterrows():
        oid = int(r["Original_ID"])
        raw = str(r["Scene"]).strip()
        if oid in sub:
            grp = VEGSUB_MAP[sub[oid]]
        elif raw in SCENE_MAP:
            grp = SCENE_MAP[raw]
        else:
            unmapped.append((oid, raw))
            grp = None
        rows.append({
            "City": "Hangzhou",
            "PointID": int(r["PointID"]),
            "Original_ID": oid,
            "Human_Class": int(r["Human_Class"]),
            "WC_Class": int(r["WC_Class"]),
            "Scene_raw": raw,
            "Scene_group": grp,
            "Disagree": int(r["Human_Class"] != r["WC_Class"]),
            "Confidence": r["Confidence"],
        })
    if unmapped:
        raise SystemExit(f"unmapped Hangzhou scene notes: {unmapped}")

    hz = pd.DataFrame(rows)
    assert len(hz) == 150 and hz["Scene_group"].notna().all()

    old = pd.read_csv(SCENE_CSV)
    if "Hangzhou" in set(old["City"]):
        raise SystemExit("scene CSV already has Hangzhou -- restore the 5-city "
                         "version (bams150_scene_grouped.5city.bak.csv) first")
    bak = SCENE_CSV.with_suffix(".5city.bak.csv")
    if not bak.exists():
        bak.write_text(SCENE_CSV.read_text())
        print(f"backed up 5-city scene CSV -> {bak.name}")

    out = pd.concat([old, hz[old.columns]], ignore_index=True)
    out.to_csv(SCENE_CSV, index=False)
    print(f"wrote {SCENE_CSV.name}: {len(old)} -> {len(out)} rows, "
          f"{out['City'].nunique()} cities")

    print("\nHangzhou Scene_group counts (disagree / n):")
    for grp, g in hz.groupby("Scene_group"):
        print(f"  {grp:32s} {int(g['Disagree'].sum()):3d} / {len(g):3d}"
              f"   {100*g['Disagree'].mean():.1f}%")

    print("\nSix-city T4 (groups n>=15), pooled:")
    six = out[out["Scene_group"] != "Unlabeled"]
    t = (six.groupby("Scene_group")
         .agg(n=("Disagree", "size"), d=("Disagree", "sum")).reset_index())
    t["pct"] = (100 * t["d"] / t["n"]).round(1)
    t = t[t["n"] >= 15].sort_values("pct", ascending=False)
    for _, r in t.iterrows():
        print(f"  {r['Scene_group']:32s} n={int(r['n']):3d}  {r['pct']:.1f}%")


if __name__ == "__main__":
    main()
