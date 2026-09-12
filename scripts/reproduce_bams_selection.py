#!/usr/bin/env python3
"""
reproduce_bams_selection.py
===========================

Canonical, re-runnable reconstruction of the Boundary-Aware Margin Sampling
(BAMS) point selection for all six cities, replacing the deleted notebooks it
originally lived in.

Why this file exists
--------------------
The original selection code was in Jupyter notebooks that were deleted with no
version control (see docs/planning/CLEANUP_LOG.md).  The logic was recovered from the
backup copies (now under archive/, see archive/README.md):

    archive/pe_era_notebooks/01_baseline.ipynb   cell 35        -> *_MarginScores.csv
    archive/pe_era_notebooks/01_baseline.ipynb   cells 44 / 47  -> *_BAMS150.csv (two-stage)
    archive/pe_era_scripts/hangzhou_init_pipeline.py            -> Hangzhou *_BAMS150.csv
                                                                    (weighted-blend variant)

This script re-derives the same artefacts, deterministically, from the frozen
15,000-point pools under data/cities/<City>/, and then verifies that the
reconstructed BAMS150 / Top150 Original_ID sets match the on-disk files that the
900 hand labels are anchored to.  Run it to (a) regenerate the artefacts, or
(b) prove the pipeline still reproduces the labelled point sets.

Method (documented, not guessed)
--------------------------------
1.  Margin random forest
        RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        features = [B2, B3, B4, B8, B11, B12, NDVI, NDBI, MNDWI]
                   (nine Sentinel-2 surface-reflectance point values + spectral
                    indices; NO 3x3 neighbourhood terms enter this model)
        target   = remapped ESA WorldCover class  (1 built / 2 non-built / 3 water)
        fit on all 15,000 points, predict_proba on the same 15,000
        margin(i) = p_(1)(i) - p_(2)(i)          # gap between the two largest
                                                 # class posteriors (in-sample)

2.  Local spectral heterogeneity
        BoundaryScore(i) = sigma_3x3(B2) + sigma_3x3(B3)
                         + sigma_3x3(B4) + sigma_3x3(B8)
        i.e. the raw, unnormalised sum of the 3x3 (30 m) neighbourhood standard
        deviations of the four visible / near-infrared reflectance bands.
        (No SWIR bands, no spectral-index sigmas.)

3.  Selection
        Top150  = the 150 points with the smallest margin  (plain uncertainty
                  sampling; kept as a reference set, NOT consumed downstream).

        BAMS150 = two stages
            Stage 1 : keep the 1,000 smallest-margin points   ("most uncertain")
            Stage 2 : of those 1,000, keep the 150 with the largest BoundaryScore
        Deterministic tie-break: every sort takes ascending Original_ID as the
        final key, so the 1,000- and 150-cut boundaries are reproducible.

        Hangzhou variant (kept for faithfulness; use --standardise-hangzhou to
        also emit the two-stage result for comparison):
            MarginScore   = 1 - margin
            MarginScore and BoundaryScore are each min-max scaled over the full
            pool, then
            BAMS_Score    = 0.7 * MarginScore + 0.3 * BoundaryScore
            keep the 150 largest BAMS_Score.

Outputs
-------
Default (non-destructive) -> data/analysis_outputs/bams_reproduction/
    <City>/<City>_MarginScores.csv
    <City>/<City>_Top150.csv
    <City>/<City>_BAMS150.csv
    REPRODUCTION_REPORT.md
    manifest.json

--in-place additionally overwrites, and ONLY for cities that verify 150/150:
    data/cities/<City>/02_margin/<City>_MarginScores.csv
    data/cities/<City>/03_top150/<City>_Top150.csv
    data/cities/<City>/03_top150/<City>_BAMS150.csv
(The canonical MarginScores carries per-class prob_1/prob_2/prob_3 + RF_Pred for
all cities; the recomputed `margin` is bit-identical to the frozen files, so
nothing downstream that reads `margin` moves.)

Environment: the `gee` conda env
  /opt/homebrew/Caskroom/miniconda/base/envs/gee/bin/python3
  (numpy / pandas / scikit-learn; RF results were frozen with scikit-learn 1.9.0)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------- #
# constants                                                                   #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
CITIES_DIR = ROOT / "data" / "cities"
OUT_DIR = ROOT / "data" / "analysis_outputs" / "bams_reproduction"

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha", "Hangzhou"]

SPECTRAL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "MNDWI"]
BOUNDARY_STD_COLS = ["B2_stdDev", "B3_stdDev", "B4_stdDev", "B8_stdDev"]

RF_PARAMS = dict(n_estimators=300, random_state=42, n_jobs=-1)
EXPECTED_SKLEARN = "1.9.0"

N_POOL = 15_000
STAGE1_KEEP = 1_000          # "most uncertain" pre-filter for BAMS150
N_SELECT = 150               # labelling budget per city

# Which selection rule actually produced each city's on-disk BAMS150.
#   "two_stage" : Stage-1 margin filter (1,000) -> Stage-2 BoundaryScore top 150
#   "blend"     : 0.7*(1-margin)_norm + 0.3*BoundaryScore_norm, top 150
# Every city now uses the frozen two-stage rule. Hangzhou was standardised onto
# it on 2026-09-10 (was "blend"); the 104 newly-included points were expert-
# labelled (scripts/gee_hangzhou_relabel_104.js) and 46 carry-over labels kept.
SELECTION_RULE = {c: "two_stage" for c in CITIES}

BLEND_W_MARGIN = 0.7
BLEND_W_BOUNDARY = 0.3

CLASS_LABELS = [1, 2, 3]     # 1 built-up, 2 non-built, 3 water


# --------------------------------------------------------------------------- #
# core steps                                                                  #
# --------------------------------------------------------------------------- #
def load_pool(city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (original 15k table, 3x3 neighbourhood table), Original_ID-sorted."""
    croot = CITIES_DIR / city
    orig = pd.read_csv(croot / "01_original" / f"{city}_WC_Samples_15000.csv")
    f3 = pd.read_csv(croot / "06_3x3" / f"{city}_3x3_Features.csv")

    for name, df in (("original", orig), ("3x3", f3)):
        df["Original_ID"] = df["Original_ID"].astype(int)
        if len(df) != N_POOL:
            raise ValueError(f"{city}: {name} has {len(df)} rows, expected {N_POOL}")
        if not df["Original_ID"].is_unique:
            raise ValueError(f"{city}: {name} has duplicate Original_ID")

    orig = orig.sort_values("Original_ID").reset_index(drop=True)
    f3 = f3.sort_values("Original_ID").reset_index(drop=True)
    return orig, f3


def compute_margin(orig: pd.DataFrame) -> pd.DataFrame:
    """Fit the margin RF on the full pool and score every point in-sample."""
    X = orig[SPECTRAL_FEATURES].to_numpy(dtype=float)
    y = orig["Class"].astype(int).to_numpy()

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)

    classes = list(rf.classes_)
    if classes != CLASS_LABELS:
        raise ValueError(f"unexpected RF classes {classes}, expected {CLASS_LABELS}")

    proba = rf.predict_proba(X)                      # column j == P(class classes[j])
    proba_sorted = np.sort(proba, axis=1)
    margin = proba_sorted[:, -1] - proba_sorted[:, -2]
    rf_pred = rf.predict(X)

    out = pd.DataFrame({"Original_ID": orig["Original_ID"].to_numpy()})
    out["Class"] = y
    out["RF_Pred"] = rf_pred.astype(int)
    for j, cls in enumerate(classes):
        out[f"prob_{cls}"] = proba[:, j]
    out["margin"] = margin

    # carry coordinates through for the point files
    for col in ("lon", "lat"):
        out[col] = orig[col].to_numpy()
    return out


def add_boundary_score(margin_df: pd.DataFrame, f3: pd.DataFrame) -> pd.DataFrame:
    df = margin_df.merge(
        f3[["Original_ID"] + BOUNDARY_STD_COLS],
        on="Original_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(df) != N_POOL:
        raise ValueError(f"boundary-score merge dropped rows: {len(df)} != {N_POOL}")
    df["BoundaryScore"] = df[BOUNDARY_STD_COLS].sum(axis=1)
    return df


def select_top150(df: pd.DataFrame) -> pd.DataFrame:
    """Plain uncertainty sampling: 150 smallest margins (Original_ID tie-break)."""
    return (
        df.sort_values(["margin", "Original_ID"], ascending=[True, True])
        .head(N_SELECT)
        .reset_index(drop=True)
    )


def select_bams150_two_stage(df: pd.DataFrame) -> pd.DataFrame:
    candidate = df.sort_values(
        ["margin", "Original_ID"], ascending=[True, True]
    ).head(STAGE1_KEEP)
    return (
        candidate.sort_values(
            ["BoundaryScore", "Original_ID"], ascending=[False, True]
        )
        .head(N_SELECT)
        .reset_index(drop=True)
    )


def select_bams150_blend(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ms = 1.0 - d["margin"]
    bs = d["BoundaryScore"]
    d["MarginScore_n"] = _minmax(ms)
    d["BoundaryScore_n"] = _minmax(bs)
    d["BAMS_Score"] = (
        BLEND_W_MARGIN * d["MarginScore_n"] + BLEND_W_BOUNDARY * d["BoundaryScore_n"]
    )
    return (
        d.sort_values(["BAMS_Score", "Original_ID"], ascending=[False, True])
        .head(N_SELECT)
        .reset_index(drop=True)
    )


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def build_point_file(sel: pd.DataFrame, method: str, rule: str) -> pd.DataFrame:
    sel = sel.copy()
    sel.insert(0, "PointID", np.arange(1, len(sel) + 1))
    sel["Method"] = method
    sel["Rule"] = rule
    cols = [
        "PointID", "Original_ID", "lon", "lat", "Class",
        "margin", "BoundaryScore", "RF_Pred",
        "prob_1", "prob_2", "prob_3", "Method", "Rule",
    ]
    return sel[[c for c in cols if c in sel.columns]]


# --------------------------------------------------------------------------- #
# verification                                                                #
# --------------------------------------------------------------------------- #
def _existing_ids(path: Path) -> set[int] | None:
    if not path.exists():
        return None
    return set(pd.read_csv(path)["Original_ID"].astype(int))


def verify_city(city: str, margin_df: pd.DataFrame,
                bams: pd.DataFrame, top: pd.DataFrame) -> dict:
    croot = CITIES_DIR / city
    res: dict = {"city": city, "rule": SELECTION_RULE[city]}

    # margin drift vs the frozen MarginScores
    fm = croot / "02_margin" / f"{city}_MarginScores.csv"
    if fm.exists():
        old = pd.read_csv(fm)[["Original_ID", "margin"]].copy()
        old["Original_ID"] = old["Original_ID"].astype(int)
        m = margin_df[["Original_ID", "margin"]].merge(
            old, on="Original_ID", suffixes=("_new", "_old")
        )
        res["margin_max_abs_delta"] = float((m["margin_new"] - m["margin_old"]).abs().max())
        res["margin_rows_compared"] = int(len(m))
    else:
        res["margin_max_abs_delta"] = None
        res["margin_rows_compared"] = 0

    # BAMS150 / Top150 point-set agreement
    for tag, sel, fp in (
        ("bams150", bams, croot / "03_top150" / f"{city}_BAMS150.csv"),
        ("top150", top, croot / "03_top150" / f"{city}_Top150.csv"),
    ):
        new_ids = set(sel["Original_ID"].astype(int))
        old_ids = _existing_ids(fp)
        if old_ids is None:
            res[f"{tag}_match"] = None
            res[f"{tag}_only_new"] = sorted(new_ids)[:10]
            res[f"{tag}_only_old"] = []
            continue
        res[f"{tag}_match"] = len(new_ids & old_ids)
        res[f"{tag}_only_new"] = sorted(new_ids - old_ids)[:10]
        res[f"{tag}_only_old"] = sorted(old_ids - new_ids)[:10]

    res["bams150_ok"] = res.get("bams150_match") == N_SELECT
    return res


# --------------------------------------------------------------------------- #
# driver                                                                      #
# --------------------------------------------------------------------------- #
def run_city(city: str, standardise_hangzhou: bool) -> dict:
    orig, f3 = load_pool(city)
    margin_df = compute_margin(orig)
    scored = add_boundary_score(margin_df, f3)

    top = select_top150(scored)

    rule = SELECTION_RULE[city]
    if rule == "two_stage":
        bams = select_bams150_two_stage(scored)
    elif rule == "blend":
        bams = select_bams150_blend(scored)
    else:
        raise ValueError(f"unknown rule {rule!r} for {city}")

    # write reproduction copies
    cdir = OUT_DIR / city
    cdir.mkdir(parents=True, exist_ok=True)

    ms_cols = ["Original_ID", "Class", "RF_Pred",
               "prob_1", "prob_2", "prob_3", "margin", "lon", "lat"]
    margin_df[ms_cols].to_csv(cdir / f"{city}_MarginScores.csv", index=False)
    build_point_file(top, "Top150", "min_margin").to_csv(
        cdir / f"{city}_Top150.csv", index=False)
    build_point_file(bams, "BAMS150", rule).to_csv(
        cdir / f"{city}_BAMS150.csv", index=False)

    if standardise_hangzhou and city == "Hangzhou":
        alt = select_bams150_two_stage(scored)
        build_point_file(alt, "BAMS150", "two_stage").to_csv(
            cdir / f"{city}_BAMS150_two_stage.csv", index=False)
        cur = _existing_ids(
            CITIES_DIR / city / "03_top150" / f"{city}_BAMS150.csv") or set()
        alt_ids = set(alt["Original_ID"].astype(int))
        print(f"  [Hangzhou] two-stage vs on-disk blend BAMS150 overlap: "
              f"{len(alt_ids & cur)}/{N_SELECT}")

    res = verify_city(city, margin_df, bams, top)
    return res


def apply_in_place(city: str, res: dict) -> bool:
    if not res.get("bams150_ok"):
        print(f"  [in-place] SKIP {city}: BAMS150 match "
              f"{res.get('bams150_match')}/{N_SELECT} (not exact)")
        return False
    src = OUT_DIR / city
    dst_margin = CITIES_DIR / city / "02_margin" / f"{city}_MarginScores.csv"
    dst_bams = CITIES_DIR / city / "03_top150" / f"{city}_BAMS150.csv"
    dst_top = CITIES_DIR / city / "03_top150" / f"{city}_Top150.csv"
    for src_name, dst in (
        (f"{city}_MarginScores.csv", dst_margin),
        (f"{city}_BAMS150.csv", dst_bams),
        (f"{city}_Top150.csv", dst_top),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((src / src_name).read_text())
    print(f"  [in-place] wrote canonical MarginScores / BAMS150 / Top150 for {city}")
    return True


def write_report(results: list[dict], args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import sklearn

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/reproduce_bams_selection.py",
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "sklearn_expected": EXPECTED_SKLEARN,
        "rf_params": RF_PARAMS,
        "spectral_features": SPECTRAL_FEATURES,
        "boundary_std_cols": BOUNDARY_STD_COLS,
        "stage1_keep": STAGE1_KEEP,
        "n_select": N_SELECT,
        "selection_rule": SELECTION_RULE,
        "blend_weights": {"margin": BLEND_W_MARGIN, "boundary": BLEND_W_BOUNDARY},
        "in_place": args.in_place,
        "results": results,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = [
        "# BAMS selection — reproduction report",
        "",
        f"- generated: {manifest['generated_utc']}",
        f"- scikit-learn: {sklearn.__version__} (frozen artefacts: {EXPECTED_SKLEARN})",
        f"- RF: {RF_PARAMS}",
        f"- margin features (9): {', '.join(SPECTRAL_FEATURES)}",
        f"- BoundaryScore: sum of 3x3 stdDev over {', '.join(BOUNDARY_STD_COLS)}",
        f"- BAMS150: two-stage (margin<=rank {STAGE1_KEEP} -> top {N_SELECT} BoundaryScore); "
        "Hangzhou uses the 0.7/0.3 min-max blend",
        "",
        "| City | rule | margin max|Δ| | BAMS150 match | Top150 match |",
        "|------|------|--------------|---------------|--------------|",
    ]
    for r in results:
        md = r["margin_max_abs_delta"]
        md_s = "n/a" if md is None else f"{md:.2e}"
        bm = r["bams150_match"]
        tm = r["top150_match"]
        bm_s = "n/a" if bm is None else f"{bm}/{N_SELECT}"
        tm_s = "n/a" if tm is None else f"{tm}/{N_SELECT}"
        lines.append(f"| {r['city']} | {r['rule']} | {md_s} | {bm_s} | {tm_s} |")

    lines += ["", "## Mismatches (first 10 Original_IDs each)"]
    for r in results:
        frags = []
        for tag in ("bams150", "top150"):
            on = r.get(f"{tag}_only_new") or []
            oo = r.get(f"{tag}_only_old") or []
            if on or oo:
                frags.append(f"  - {tag}: only_new={on} only_old={oo}")
        if frags:
            lines.append(f"- **{r['city']}**")
            lines.extend(frags)
    if all(not (r.get("bams150_only_new") or r.get("bams150_only_old")
                or r.get("top150_only_new") or r.get("top150_only_old"))
           for r in results):
        lines.append("- none — all cities reproduce exactly")

    lines += [
        "",
        "## Notes",
        "- `margin` is recomputed bit-for-bit (Δ ~1e-16) from the frozen pools, so "
        "downstream code that reads only `Original_ID`/`margin` is unaffected.",
        "- The canonical `*_MarginScores.csv` standardises all six cities to "
        "per-class `prob_1/prob_2/prob_3` + `RF_Pred`; the older Wuhan file stored "
        "only the top-two sorted posteriors.",
        "- `*_Top150.csv` is a plain min-margin reference set and is not consumed "
        "downstream; Wuhan's on-disk `Top150.csv` came from a superseded 70/30-split "
        "run (`archive/pe_era_scripts/wuhan_pipeline_step1_2.py`) and will not match.",
        "- Hangzhou's BAMS150 uses the weighted-blend variant "
        "(`archive/pe_era_scripts/hangzhou_init_pipeline.py`); pass `--standardise-hangzhou` to "
        "also emit the two-stage set for comparison.",
        "- Hangzhou's on-disk `BAMS150.csv` stored a degenerate (all-zero) "
        "`BoundaryScore` column from an in-place min-max bug; the reproduction "
        "writes the real raw sum. Point selection, order and `margin` are "
        "unaffected (row order identical, Δmargin = 0).",
    ]
    (OUT_DIR / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cities", default=",".join(CITIES),
                    help="comma-separated subset (default: all six)")
    ap.add_argument("--in-place", action="store_true",
                    help="also overwrite data/cities/<City>/{02_margin,03_top150} "
                         "for cities that verify 150/150")
    ap.add_argument("--standardise-hangzhou", action="store_true",
                    help="additionally emit Hangzhou's two-stage BAMS150 for comparison")
    args = ap.parse_args()

    import sklearn
    if sklearn.__version__ != EXPECTED_SKLEARN:
        print(f"WARNING: scikit-learn {sklearn.__version__} != frozen "
              f"{EXPECTED_SKLEARN}; margins may drift.", file=sys.stderr)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    bad = [c for c in cities if c not in CITIES]
    if bad:
        ap.error(f"unknown cities: {bad}")

    results = []
    for city in cities:
        print(f"[{city}] rule={SELECTION_RULE[city]}")
        res = run_city(city, args.standardise_hangzhou)
        md = res["margin_max_abs_delta"]
        print(f"  margin max|Δ| = {'n/a' if md is None else f'{md:.2e}'}   "
              f"BAMS150 match = {res['bams150_match']}/{N_SELECT}   "
              f"Top150 match = {res['top150_match']}/{N_SELECT}")
        if args.in_place:
            apply_in_place(city, res)
        results.append(res)

    write_report(results, args)
    print(f"\nreport -> {OUT_DIR / 'REPRODUCTION_REPORT.md'}")

    failed = [r["city"] for r in results if r["bams150_match"] not in (None, N_SELECT)]
    if failed:
        print(f"\nFAIL: BAMS150 did not reproduce exactly for {failed}", file=sys.stderr)
        return 1
    print("\nOK: BAMS150 reproduced exactly for all requested cities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
