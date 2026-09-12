#!/usr/bin/env python3
"""
Step 2 of the C2 example-chip figure (Supplementary) -- Python / ee path.

Alternative to scripts/gee_c2_chip_export.js: pulls the Sentinel-2 chips straight
from Earth Engine to data/analysis_outputs/c2_chip_figure/chips/ so
c2_chip_assemble.py can run without any Drive round-trip.

Needs the `gee` conda env (has `ee`) and an Earth Engine *cloud project* id
(the local credentials file does not carry one):
    scripts/c2_chip_fetch.py --project ee-YOURPROJECT
or  EE_PROJECT=ee-YOURPROJECT scripts/c2_chip_fetch.py

Imagery recipe = the BAMS selection pipeline:
    COPERNICUS/S2_SR_HARMONIZED, 2021, CLOUDY_PIXEL_PERCENTAGE < 20, median.
Two renders per point:  _tc true colour (B4 B3 B2), _fc false colour (B8 B4 B3).

Run:
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3 \
      scripts/c2_chip_fetch.py --project ee-YOURPROJECT
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "data/analysis_outputs/c2_chip_figure"
CHIPS = FIGDIR / "chips"

YEAR = 2021
HALF_M = 375
DIMENSIONS = 600
V_TRUE = {"bands": ["B4", "B3", "B2"], "min": 200, "max": 2500}
V_FALSE = {"bands": ["B8", "B4", "B3"], "min": 200, "max": 4000}


def get_project(cli: str | None) -> str:
    proj = cli or os.environ.get("EE_PROJECT")
    if not proj:
        sys.exit("no Earth Engine cloud project -- pass --project ee-XXXX "
                 "or set EE_PROJECT (see your code.earthengine.google.com URL)")
    return proj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import ee
    ee.Initialize(project=get_project(args.project))

    man = pd.read_csv(FIGDIR / "chip_manifest.csv")
    CHIPS.mkdir(parents=True, exist_ok=True)

    def composite(region):
        return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(region)
                .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .median())

    n_done = n_skip = 0
    for r in man.itertuples(index=False):
        region = ee.Geometry.Point([r.lon, r.lat]).buffer(HALF_M).bounds()
        s2 = composite(region)
        for kind, vis in (("tc", V_TRUE), ("fc", V_FALSE)):
            out = CHIPS / f"c2_{r.panel_id}_{r.Original_ID}_{r.City}_{kind}.png"
            if out.exists() and not args.overwrite:
                n_skip += 1
                continue
            url = s2.visualize(**vis).getThumbURL(
                {"region": region, "dimensions": DIMENSIONS, "format": "png"})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=120) as resp:
                        out.write_bytes(resp.read())
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            n_done += 1
            print(f"  {out.name}")

    print(f"\nfetched {n_done}, skipped {n_skip} (already present)")
    print(f"chips in {CHIPS.relative_to(ROOT)}  ->  run c2_chip_assemble.py")


if __name__ == "__main__":
    main()
