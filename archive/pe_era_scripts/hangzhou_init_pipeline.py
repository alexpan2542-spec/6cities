#!/usr/bin/env python3
"""Initialize Hangzhou data2 folder: 15k WC samples, 3x3, margin, BAMS150, Manual template.

Matches existing city protocol:
  Class 1=built, 2=non-built, 3=water (ESA WorldCover v200 remap)
  S2 SR Harmonized 2021 median, cloud < 20%
  BAMS = 0.7 * norm(1-margin) + 0.3 * norm(sum B2/B3/B4/B8 stdDev)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
CITY = "Hangzhou"
OUT = ROOT / "data2" / CITY
EE_PROJECT = "healthy-area-463312-i4"

FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "MNDWI"]
N_PER_CLASS = 5000
RF_NESTIMATORS = 300
SEED = 42
SCALE = 10
YEAR = 2021
CHUNK = 1000

WC_FROM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_TO = [2, 2, 2, 2, 1, 2, 2, 3, 3, 3, 2]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "original": OUT / "01_original",
        "margin": OUT / "02_margin",
        "top150": OUT / "03_top150",
        "manual": OUT / "04_manual",
        "corrected": OUT / "05_corrected",
        "features_3x3": OUT / "06_3x3",
        "results": OUT / "07_results",
        "logs": OUT / "08_logs",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def hangzhou_roi() -> ee.Geometry:
    fc = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "China"))
        .filter(ee.Filter.eq("ADM1_NAME", "Zhejiang Sheng"))
        .filter(ee.Filter.eq("ADM2_NAME", "Hangzhou"))
    )
    return fc.geometry()


def build_c3(roi: ee.Geometry) -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(roi)
    return wc.remap(WC_FROM, WC_TO).rename("Class")


def build_optical(roi: ee.Geometry) -> ee.Image:
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(roi)
    )
    optical = s2.select(["B2", "B3", "B4", "B8", "B11", "B12"])
    ndvi = optical.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = optical.normalizedDifference(["B11", "B8"]).rename("NDBI")
    mndwi = optical.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    return optical.addBands([ndvi, ndbi, mndwi])


def wait_task(task: ee.batch.Task, label: str) -> None:
    print(f"[EE Export] started {label} id={task.id}", flush=True)
    while True:
        st = task.status()
        state = st.get("state")
        print(f"  {label}: {state}", flush=True)
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            if state != "COMPLETED":
                raise RuntimeError(f"Export {label} failed: {st}")
            return
        time.sleep(30)


def export_fc_to_asset(fc: ee.FeatureCollection, asset_id: str, description: str) -> str:
    # Delete existing asset if present
    try:
        ee.data.deleteAsset(asset_id)
        print(f"  deleted existing {asset_id}", flush=True)
    except Exception:
        pass
    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description=description[:100],
        assetId=asset_id,
    )
    task.start()
    wait_task(task, description)
    return asset_id


def asset_to_df(asset_id: str) -> pd.DataFrame:
    fc = ee.FeatureCollection(asset_id)
    n = int(fc.size().getInfo())
    print(f"[Download] {asset_id} n={n}", flush=True)
    rows: list[dict] = []
    for start in range(0, n, CHUNK):
        subset = ee.FeatureCollection(fc.toList(CHUNK, start))
        info = subset.getInfo()["features"]
        for f in info:
            d = dict(f["properties"])
            geom = f.get("geometry")
            if geom and geom.get("coordinates"):
                coords = geom["coordinates"]
                d["lon"] = float(coords[0])
                d["lat"] = float(coords[1])
            rows.append(d)
        print(f"  {min(start + CHUNK, n)}/{n}", flush=True)
        time.sleep(0.15)
    return pd.DataFrame(rows)


def sample_class_points(c3: ee.Image, roi: ee.Geometry) -> ee.FeatureCollection:
    """Stratified points on Class only (fast), then export via asset."""
    return c3.stratifiedSample(
        numPoints=N_PER_CLASS,
        classBand="Class",
        region=roi,
        scale=SCALE,
        seed=SEED,
        geometries=True,
        classValues=[1, 2, 3],
        classPoints=[N_PER_CLASS, N_PER_CLASS, N_PER_CLASS],
        tileScale=8,
    )


def points_batch_fc(df: pd.DataFrame) -> ee.FeatureCollection:
    feats = []
    for _, r in df.iterrows():
        feats.append(
            ee.Feature(
                ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                {
                    "Original_ID": int(r["Original_ID"]),
                    "Class": int(r["Class"]),
                },
            )
        )
    return ee.FeatureCollection(feats)


def sample_image_at_points(
    image: ee.Image,
    points_df: pd.DataFrame,
    batch_size: int = 400,
    geometries: bool = False,
) -> pd.DataFrame:
    """sampleRegions in small batches (avoids timeout / huge payloads)."""
    rows: list[dict] = []
    n = len(points_df)
    for start in range(0, n, batch_size):
        batch = points_df.iloc[start : start + batch_size]
        fc = points_batch_fc(batch)
        sampled = image.sampleRegions(
            collection=fc,
            properties=["Original_ID", "Class"],
            scale=SCALE,
            geometries=geometries,
            tileScale=4,
        )
        # Retry once on transient errors
        for attempt in range(3):
            try:
                info = sampled.getInfo()["features"]
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retry {attempt+1} after error: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        for f in info:
            d = dict(f["properties"])
            if geometries and f.get("geometry"):
                coords = f["geometry"]["coordinates"]
                d["lon"] = float(coords[0])
                d["lat"] = float(coords[1])
            rows.append(d)
        print(f"  sampleRegions {min(start + batch_size, n)}/{n}", flush=True)
        time.sleep(0.25)
    return pd.DataFrame(rows)


def compute_margin(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES]
    y = df["Class"].astype(int)
    rf = RandomForestClassifier(
        n_estimators=RF_NESTIMATORS, random_state=SEED, n_jobs=-1
    )
    rf.fit(X, y)
    prob = rf.predict_proba(X)
    pred = rf.predict(X)
    classes = list(rf.classes_)
    out = df.copy()
    out["RF_Pred"] = pred
    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = prob[:, i]
    sorted_p = np.sort(prob, axis=1)
    out["margin"] = sorted_p[:, -1] - sorted_p[:, -2]
    return out


def compute_bams150(margin_df: pd.DataFrame, f3: pd.DataFrame) -> pd.DataFrame:
    std_cols = ["B2_stdDev", "B3_stdDev", "B4_stdDev", "B8_stdDev"]
    df = margin_df.merge(f3[["Original_ID"] + std_cols], on="Original_ID", how="inner")
    df["BoundaryScore"] = df[std_cols].sum(axis=1)
    df["MarginScore"] = 1.0 - df["margin"]
    for col in ("MarginScore", "BoundaryScore"):
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn) if mx > mn else 0.0
    df["BAMS_Score"] = 0.7 * df["MarginScore"] + 0.3 * df["BoundaryScore"]
    bams = df.sort_values("BAMS_Score", ascending=False).head(150).copy()
    bams.insert(0, "PointID", np.arange(1, len(bams) + 1))
    bams["Method"] = "BAMS150"
    cols = [
        "PointID",
        "Original_ID",
        "lon",
        "lat",
        "Class",
        "margin",
        "BoundaryScore",
        "RF_Pred",
        "prob_1",
        "prob_2",
        "prob_3",
        "Method",
    ]
    return bams[cols]


def write_manual(bams: pd.DataFrame, path: Path) -> None:
    man = pd.DataFrame(
        {
            "PointID": bams["PointID"].astype(int),
            "Original_ID": bams["Original_ID"].astype(int),
            "Human_Class": "",
            "Scene": "",
            "Confidence": "",
            "Notes": bams["Class"].map(lambda c: f"WC_Class={int(c)}"),
        }
    )
    man.to_csv(path, index=False)


def write_inspect_js(path: Path) -> None:
    path.write_text(
        """// =====================================
// Hangzhou BAMS150 Manual Inspection
// =====================================
// Upload data2/Hangzhou/03_top150/Hangzhou_BAMS150.csv as Asset:
//   projects/healthy-area-463312-i4/assets/Hangzhou_BAMS150
// Then set p = PointID (1..150) while labeling
// Manual file: data2/Hangzhou/04_manual/Hangzhou_BAMS150_Manual.csv
// Columns: PointID, Original_ID, Human_Class, Scene, Confidence, Notes
// Human_Class: 1=built, 2=non-built, 3=water
// Confidence: H / M / L

var table = ee.FeatureCollection(
  'projects/healthy-area-463312-i4/assets/Hangzhou_BAMS150'
);

var points = table.map(function(f) {
  return ee.Feature(
    ee.Geometry.Point([
      ee.Number(f.get('lon')),
      ee.Number(f.get('lat'))
    ]),
    f.toDictionary()
  );
});

var sortedPoints = points.sort('PointID');

Map.setOptions('HYBRID');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(sortedPoints)
  .filterDate('2021-01-01', '2021-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

Map.addLayer(
  s2,
  {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000},
  'Sentinel-2 RGB'
);

Map.addLayer(
  s2,
  {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000},
  'Sentinel-2 False Color',
  false
);

Map.addLayer(sortedPoints, {color: 'red'}, 'Hangzhou_BAMS150');

// ---- only change p ----
var p = 1;  // 1..150

var point = points.filter(ee.Filter.eq('PointID', p)).first();

print('---- Current point ----');
print('PointID:', p);
print('Original_ID:', point.get('Original_ID'));
print('WC Class:', point.get('Class'));
print('margin:', point.get('margin'));
print('lon:', point.get('lon'));
print('lat:', point.get('lat'));

Map.centerObject(point, 16);
Map.addLayer(ee.FeatureCollection([point]), {color: 'yellow'}, 'Current');
""",
        encoding="utf-8",
    )


def main() -> None:
    dirs = ensure_dirs()
    ee.Initialize(project=EE_PROJECT)
    roi = hangzhou_roi()
    area = float(roi.area().divide(1e6).getInfo())
    print(f"Hangzhou AOI area_km2={area:.1f}", flush=True)

    c3 = build_c3(roi)
    optical = build_optical(roi)

    # --- Step A: class points via Export (avoids interactive timeout) ---
    pts_asset = f"projects/{EE_PROJECT}/assets/{CITY}_class_pts_15k"
    print("[EE] stratified Class points → asset", flush=True)
    export_fc_to_asset(sample_class_points(c3, roi), pts_asset, f"{CITY}_class_pts_15k")
    pts = asset_to_df(pts_asset)
    if "lon" not in pts.columns or pts["lon"].isna().any():
        raise RuntimeError("Points missing lon/lat after download")
    pts = pts.dropna(subset=["Class", "lon", "lat"]).copy()
    pts["Class"] = pts["Class"].astype(int)
    # Enforce 5k/class
    parts = []
    for c in (1, 2, 3):
        part = pts[pts["Class"] == c].sort_values(["lat", "lon"]).head(N_PER_CLASS)
        if len(part) < N_PER_CLASS:
            raise RuntimeError(f"Class {c}: only {len(part)} points")
        parts.append(part)
    pts = pd.concat(parts, ignore_index=True)
    pts.insert(0, "Original_ID", np.arange(len(pts), dtype=int))
    print("class counts", pts["Class"].value_counts().sort_index().to_dict())

    # --- Step B: pixel features (batched getInfo) ---
    print("[EE] sampleRegions optical features (batched)", flush=True)
    feat = sample_image_at_points(optical, pts, batch_size=400, geometries=False)
    feat = feat.merge(pts[["Original_ID", "lon", "lat"]], on="Original_ID", how="left")
    feat = feat.sort_values("Original_ID").reset_index(drop=True)
    missing = [c for c in FEATURES if c not in feat.columns]
    if missing:
        raise RuntimeError(f"Missing feature cols: {missing}")

    original = feat[["Original_ID"] + FEATURES + ["Class", "lon", "lat"]].copy()
    original["system:index"] = original["Original_ID"]
    original = original.dropna(subset=FEATURES + ["Class"]).reset_index(drop=True)
    if len(original) != 15000:
        print(f"WARNING: original n={len(original)} (expected 15000)")
        if len(original) < 14000:
            raise RuntimeError(f"Too few feature rows: {len(original)}")

    out_orig = dirs["original"] / f"{CITY}_WC_Samples_15000.csv"
    original.to_csv(out_orig, index=False)
    print(f"[Step1] {out_orig} n={len(original)}")

    # --- Step C: 3x3 neighborhood ---
    kernel = ee.Kernel.square(radius=1, units="pixels")
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    neigh = optical.reduceNeighborhood(reducer=reducer, kernel=kernel)
    print("[EE] 3x3 sampleRegions (batched)", flush=True)
    f3 = sample_image_at_points(neigh, original, batch_size=300, geometries=False)
    f3 = f3.sort_values("Original_ID").reset_index(drop=True)
    out_f3 = dirs["features_3x3"] / f"{CITY}_3x3_Features.csv"
    f3.to_csv(out_f3, index=False)
    print(f"[Step1b] {out_f3} n={len(f3)}")

    # --- Margin + BAMS ---
    margin = compute_margin(original)
    out_m = dirs["margin"] / f"{CITY}_MarginScores.csv"
    margin.to_csv(out_m, index=False)
    print(f"[Step2] {out_m} margin mean={margin['margin'].mean():.4f}")

    top = margin.nsmallest(150, "margin").copy().reset_index(drop=True)
    top.insert(0, "PointID", np.arange(1, 151))
    top_cols = [
        "PointID",
        "Original_ID",
        "lon",
        "lat",
        "Class",
        "margin",
        "RF_Pred",
        "prob_1",
        "prob_2",
        "prob_3",
    ]
    top[top_cols].to_csv(dirs["top150"] / f"{CITY}_Top150.csv", index=False)

    bams = compute_bams150(margin, f3)
    out_bams = dirs["top150"] / f"{CITY}_BAMS150.csv"
    bams.to_csv(out_bams, index=False)
    print(f"[Step3] {out_bams}")
    print(
        bams[["PointID", "Original_ID", "Class", "margin", "BoundaryScore"]]
        .head(10)
        .to_string(index=False)
    )

    man_path = dirs["manual"] / f"{CITY}_BAMS150_Manual.csv"
    write_manual(bams, man_path)
    print(f"[Manual] {man_path}")

    js_path = ROOT / "figures" / "gee_shp" / f"{CITY}_BAMS150_inspect.js"
    write_inspect_js(js_path)
    print(f"[GEE] {js_path}")

    meta = {
        "city": CITY,
        "aoi": "FAO/GAUL/2015/level2 Hangzhou, Zhejiang Sheng",
        "area_km2": area,
        "year": YEAR,
        "n_samples": int(len(original)),
        "class_counts": {int(k): int(v) for k, v in original["Class"].value_counts().sort_index().items()},
        "rf_n_estimators": RF_NESTIMATORS,
        "bams_formula": "0.7*norm(1-margin)+0.3*norm(B2+B3+B4+B8 stdDev)",
        "manual": str(man_path),
        "bams": str(out_bams),
        "assets": {"pts": pts_asset},
    }
    (dirs["logs"] / "init_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
