#!/usr/bin/env python3
"""Wuhan reproducible pipeline: Step 1 (original) + Step 2 (margin) + Step 3 (top150)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data" / "Wuhan_WC_Samples_15000.csv"
OUT_ROOT = ROOT / "data2" / "Wuhan"

FEATURES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NDBI",
    "MNDWI",
]

RF_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

SPLIT_PARAMS = {
    "test_size": 0.3,
    "random_state": 42,
    "stratify": True,
}

TOP_K = 150


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "original": OUT_ROOT / "01_original",
        "margin": OUT_ROOT / "02_margin",
        "top150": OUT_ROOT / "03_top150",
        "manual": OUT_ROOT / "04_manual",
        "corrected": OUT_ROOT / "05_corrected",
        "features_3x3": OUT_ROOT / "06_3x3",
        "results": OUT_ROOT / "07_results",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def step1_load_original(dirs: dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(DATA_IN)
    if len(df) != 15000:
        raise ValueError(f"Expected 15000 rows, got {len(df)}")

    # Original_ID = row index in source file (0-based, matches system:index)
    df = df.copy()
    df.insert(0, "Original_ID", np.arange(len(df), dtype=int))

    if not (df["Original_ID"].values == df["system:index"].values).all():
        raise ValueError("Original_ID does not match system:index")

    out = dirs["original"] / "Wuhan_WC_Samples_15000.csv"
    df.to_csv(out, index=False)

    summary = {
        "step": 1,
        "input": str(DATA_IN),
        "output": str(out),
        "n_rows": len(df),
        "columns": df.columns.tolist(),
        "class_counts": df["Class"].value_counts().sort_index().to_dict(),
    }
    (dirs["original"] / "step1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[Step 1] saved {out} ({len(df)} rows)")
    return df


def step2_margin_sampling(df: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    X = df[FEATURES]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SPLIT_PARAMS["test_size"],
        random_state=SPLIT_PARAMS["random_state"],
        stratify=y if SPLIT_PARAMS["stratify"] else None,
    )

    train_ids = df.loc[X_train.index, "Original_ID"].astype(int)
    test_ids = df.loc[X_test.index, "Original_ID"].astype(int)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    prob = rf.predict_proba(X_test)
    classes = rf.classes_
    pred = rf.predict(X_test)

    sorted_prob = np.sort(prob, axis=1)
    margin = sorted_prob[:, -1] - sorted_prob[:, -2]

    margin_df = df.loc[X_test.index].copy()
    margin_df["RF_Pred"] = pred
    margin_df["margin"] = margin
    for i, cls in enumerate(classes):
        margin_df[f"prob_{cls}"] = prob[:, i]

    margin_df = margin_df.sort_values("margin", ascending=True).reset_index(drop=True)
    margin_df.insert(0, "Margin_Rank", np.arange(1, len(margin_df) + 1))

    # Save split IDs
    pd.DataFrame({"Original_ID": train_ids.sort_values().values}).to_csv(
        dirs["margin"] / "Wuhan_train_Original_IDs.csv", index=False
    )
    pd.DataFrame({"Original_ID": test_ids.sort_values().values}).to_csv(
        dirs["margin"] / "Wuhan_test_Original_IDs.csv", index=False
    )

    margin_df.to_csv(dirs["margin"] / "Wuhan_test_margin.csv", index=False)

    meta = {
        "step": 2,
        "features": FEATURES,
        "rf_params": RF_PARAMS,
        "split_params": SPLIT_PARAMS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_oa": float((pred == y_test.values).mean()),
        "margin_min": float(margin.min()),
        "margin_max": float(margin.max()),
        "margin_mean": float(margin.mean()),
        "rf_classes": classes.tolist(),
    }
    (dirs["margin"] / "step2_summary.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"[Step 2] saved margin for {len(margin_df)} test samples "
        f"(train={len(X_train)}, test OA={meta['test_oa']:.4f})"
    )
    return margin_df


def step3_top150(margin_df: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    top = margin_df.nsmallest(TOP_K, "margin").copy()
    top = top.reset_index(drop=True)
    top.insert(0, "PointID", np.arange(1, len(top) + 1))

    cols = [
        "PointID",
        "Original_ID",
        "lon",
        "lat",
        "Class",
        "margin",
        "RF_Pred",
    ]
    extra = [c for c in top.columns if c.startswith("prob_")]
    out_cols = cols + extra
    top_out = top[out_cols]

    out = dirs["top150"] / "Wuhan_Top150.csv"
    top_out.to_csv(out, index=False)

    summary = {
        "step": 3,
        "top_k": TOP_K,
        "output": str(out),
        "margin_range": [float(top["margin"].min()), float(top["margin"].max())],
        "class_counts": top["Class"].value_counts().sort_index().to_dict(),
    }
    (dirs["top150"] / "step3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[Step 3] saved {out} ({len(top_out)} rows)")
    return top_out


def main() -> None:
    dirs = ensure_dirs()
    df = step1_load_original(dirs)
    margin_df = step2_margin_sampling(df, dirs)
    step3_top150(margin_df, dirs)
    print("\nPipeline complete. Outputs under:", OUT_ROOT)


if __name__ == "__main__":
    main()
