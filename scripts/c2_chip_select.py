#!/usr/bin/env python3
"""
Step 1 of the C2 example-chip figure (Supplementary).

Deterministically selects the boundary points whose Sentinel-2 chips go into
the gallery that illustrates:

  Group A  C2 -- WorldCover directional error
           WC = built-up, expert = non-built (vegetation).  The paper's C2
           claim: once WC says "built", the expert reads vegetation ~7/10.

  Group B  C1 -- coupled product error
           Dynamic World = Esri = built-up, expert = non-built, and WC
           dissents (WC = non-built, i.e. correct here).  Shows the two
           products that agree most err *together*; their agreement is not
           corroboration.

  Group C  counter-examples
           WC = built-up, expert = built-up.  Included so the gallery cannot
           be read as cherry-picked -- WC is right on the boundary too.

Selection rule (identical for every group, stated so it can go in the caption):
  * restrict to Conf in {high, med}; drop low-confidence expert calls
  * rank by BAMS selection margin, DESCENDING
      -> the points furthest from the product-consensus decision boundary,
         i.e. the cases where the map was *most* committed to its label
  * ties broken by Original_ID ascending (fully deterministic)
  * take the top N_PER_CITY[group] per city

Nothing here touches Earth Engine.  Outputs:
  data/analysis_outputs/c2_chip_figure/chip_manifest.csv   -- one row per chip
  data/analysis_outputs/c2_chip_figure/points_for_gee.js   -- paste into the
                                                               GEE export script
Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/c2_chip_select.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/shared_reference/cross_city_second_ref_dw_esri/all_cities_BAMS150_second_ref.csv"
OUT = ROOT / "data/analysis_outputs/c2_chip_figure"

# chips per city, per group.  Dial down here if the figure runs too large.
N_PER_CITY = {"A": 2, "B": 1, "C": 1}

# points dropped after visual review -- the next-ranked candidate for that city
# takes the slot.  A: 558 ambiguous scene; 4251, 1831 -- chip reads as a
# built structure (greenhouse film / metal-roof shed), not clearly vegetation;
# 4271, 1537, 2217, 4053 -- ranked above 853 but either shows an unambiguous
# built complex (1537, 2217) or is a weaker vegetation read than 853 (4053).
# C: 378, 927, 2183, 3557 -- built, but not obviously an urban core; 3173,
# 4886 -- chip is a weaker/patchier example of a built counter-case; 3996,
# 2961, 4440 -- ranked above 442 but weaker/sparser as a dense built example.
#
# 2026-09-12 re-pick pass: user reviewed 6-8 ranked candidates per city (not
# just the auto top-N) via rendered comparison grids and hand-picked the
# final slot per city/group. Excludes below are the ranks *not* chosen in
# that review, so pick() lands on the chosen rank. 442 (old C-Wuhan
# exclusion, "weaker/sparser") is the new Wuhan C pick, so it is removed from
# this set.
EXCLUDE_IDS = {558, 378, 927, 2183, 3557, 4251, 1831, 3173, 4886, 4271, 1537,
               2217, 4053, 3996, 2961, 4440,
               # Group A 2026-09-12 re-pick (dropped ranks per city)
               1804, 1061, 2863, 4930, 3382, 1919, 3030, 951,
               4904, 3900, 1719, 4317, 853, 3273, 3806, 2972, 1874,
               # Group B 2026-09-12 re-pick
               9555}

# 2026-09-14: SRC's Human_Class for 14 Nanchang points was stale (still the
# WC-placeholder value from before the documented 26-point Nanchang
# re-annotation, Section 3.2(d)) and has since been corrected in place
# (scripts/fix_second_ref_nanchang.py). Two of those 14 points (1041, 3553)
# now satisfy Group A's mask and outrank the currently-published picks
# (1412, 4546) by margin, so re-running this script today would swap
# Nanchang's Group A chips. The published Figure S5 predates that fix and
# still shows 1412/4546 -- both remain valid Group A instances (their labels
# were never among the 14 corrected points), so the figure was deliberately
# left as-is rather than re-fetching new Sentinel-2 chips for 1041/3553.
# The 159/191 summary statistic quoted alongside the figure (Supplementary
# S6) *was* recomputed against the corrected data and now reads 164/197.
# If this script is ever re-run to regenerate the figure from scratch, expect
# Nanchang's Group A slot to change to 1041/3553 and update the S6 text and
# figure together at that point.

CITY_ORDER = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]

# per-group confidence tiers, tried in order per city (a city short on the first
# tier tops up from the next).  Group C = high-confidence only, so the
# counter-examples sit clearly inside built fabric; falls back to med only where
# a city has no high-conf built=built point (Nanchang).
CONF_TIERS = {
    "A": [("high", "med")],
    "B": [("high", "med")],
    "C": [("high",), ("med",)],
}

C3 = {1: "built", 2: "non-built", 3: "water"}
DW_RAW = {0: "water", 1: "trees", 2: "grass", 3: "flooded veg", 4: "crops",
          5: "shrub/scrub", 6: "built", 7: "bare", 8: "snow/ice"}
ESRI_RAW = {1: "water", 2: "trees", 4: "flooded veg", 5: "crops", 7: "built area",
            8: "bare ground", 9: "snow/ice", 10: "clouds", 11: "rangeland"}
WC_RAW = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
          50: "built-up", 60: "bare/sparse", 70: "snow/ice", 80: "water",
          90: "herb. wetland", 95: "mangroves", 100: "moss/lichen"}


def _decode(v, table):
    try:
        return table.get(int(v), f"?{v}")
    except (TypeError, ValueError):
        return "?"


def pick(df: pd.DataFrame, mask: pd.Series, n_per_city: int,
         tiers: list[tuple[str, ...]]) -> pd.DataFrame:
    # key on the frame's own (unique) index -- Original_ID is only unique
    # within a city, not across the merged six-city table
    base = df[mask & ~df["Original_ID"].isin(EXCLUDE_IDS)]
    keep_idx: list[int] = []
    for _, g in base.groupby("City"):
        chosen: list[int] = []
        for tier in tiers:
            if len(chosen) >= n_per_city:
                break
            cand = (g.loc[g["Conf"].isin(tier)]
                    .drop(index=chosen, errors="ignore")
                    .sort_values(["margin", "Original_ID"], ascending=[False, True]))
            chosen += cand.index[: n_per_city - len(chosen)].tolist()
        keep_idx += chosen
    return base.loc[keep_idx]


def main() -> None:
    df = pd.read_csv(SRC)
    df["Human_Class"] = df["Human_Class"].astype(int)
    df["WC_C3"] = df["WC_C3"].astype(int)

    groups = {
        # A: all three products call built-up, expert reads vegetation
        "A": (df.WC_C3 == 1) & (df.DW_C3 == 1) & (df.ESRI_C3 == 1) & (df.Human_Class == 2),
        # B: the coupled pair (DW=Esri) calls built-up, WC dissents and is right
        "B": (df.DW_C3 == 1) & (df.ESRI_C3 == 1) & (df.WC_C3 == 2) & (df.Human_Class == 2),
        "C": (df.WC_C3 == 1) & (df.Human_Class == 1),
    }
    group_title = {
        "A": "C1+C2  WC=DW=Esri=built, expert=non-built (consensus error)",
        "B": "C1  DW=Esri=built, expert=non-built, WC dissents correctly (coupled error)",
        "C": "counter-example  WC=built, expert=built",
    }

    rows = []
    for g, mask in groups.items():
        chosen = pick(df, mask, N_PER_CITY[g], CONF_TIERS[g])
        chosen["_ord"] = chosen["City"].map(
            {c: i for i, c in enumerate(CITY_ORDER)}).fillna(99)
        chosen = chosen.sort_values(["_ord", "margin", "Original_ID"],
                                    ascending=[True, False, True])
        for k, r in enumerate(chosen.itertuples(index=False), 1):
            expert = C3[r.Human_Class]
            wc = C3[int(r.WC_C3)]
            dw = C3[int(r.DW_C3)]
            esri = C3[int(r.ESRI_C3)]
            cap = (f"{r.City} · WC={wc} DW={dw} Esri={esri} · expert={expert} "
                   f"({r.Conf}) · id {int(r.Original_ID)}")
            rows.append(dict(
                group=g, group_title=group_title[g],
                panel_id=f"{g}{k:02d}",
                Original_ID=int(r.Original_ID), City=r.City,
                lon=float(r.lon), lat=float(r.lat),
                Conf=r.Conf, margin=round(float(r.margin), 4),
                expert_c3=expert, wc_c3=wc, dw_c3=dw, esri_c3=esri,
                wc_raw=_decode(r.WC_ee_raw, WC_RAW),
                dw_raw=_decode(r.DW_raw, DW_RAW),
                esri_raw=_decode(r.ESRI_raw, ESRI_RAW),
                caption=cap,
            ))

    man = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    man.to_csv(OUT / "chip_manifest.csv", index=False)

    # JS array literal for the GEE export script
    js_pts = [
        dict(pid=x["panel_id"], id=x["Original_ID"], city=x["City"],
             lon=round(x["lon"], 7), lat=round(x["lat"], 7),
             grp=x["group"], expert=x["expert_c3"],
             wc=x["wc_c3"], dw=x["dw_c3"], esri=x["esri_c3"])
        for x in rows
    ]
    js = ("// generated by scripts/c2_chip_select.py -- do not hand-edit\n"
          "var PTS = " + json.dumps(js_pts, indent=2) + ";\n")
    (OUT / "points_for_gee.js").write_text(js)

    # console summary
    print(f"source      : {SRC.relative_to(ROOT)}")
    print(f"N_PER_CITY  : {N_PER_CITY}")
    print(f"chips total : {len(man)}\n")
    for g in groups:
        sub = man[man.group == g]
        print(f"  group {g} ({group_title[g]}):  n={len(sub)}  "
              f"cities={sub['City'].tolist()}")
    print(f"\nwrote {OUT/'chip_manifest.csv'}")
    print(f"wrote {OUT/'points_for_gee.js'}  <- paste into scripts/gee_c2_chip_export.js")


if __name__ == "__main__":
    main()
