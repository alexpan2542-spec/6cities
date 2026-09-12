#!/usr/bin/env python3
"""Reusable margin-correction evaluation for data2 city experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Change only this variable to evaluate another city.
CITY = "Nanjing"

ROOT = Path(__file__).resolve().parents[1]
CITY_ROOT = ROOT / "data2" / CITY

ORIGINAL_PATH = CITY_ROOT / "01_original" / f"{CITY}_WC_Samples_15000.csv"
MANUAL_PATH = CITY_ROOT / "04_manual" / f"{CITY}_Top150_Manual.csv"
FEATURES_3X3_PATH = CITY_ROOT / "06_3x3" / f"{CITY}_3x3_Features.csv"
CORRECTED_PATH = CITY_ROOT / "05_corrected" / f"{CITY}_Corrected_15000.csv"
FEATURES_3X3_CORRECTED_PATH = CITY_ROOT / "06_3x3" / f"{CITY}_3x3_Features_Corrected.csv"
RESULTS_DIR = CITY_ROOT / "07_results"
REPORT_PATH = RESULTS_DIR / f"{CITY}_Margin_Report.txt"
SUMMARY_PATH = RESULTS_DIR / f"{CITY}_Margin_Summary.csv"

PIXEL_FEATURES = [
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (ORIGINAL_PATH, MANUAL_PATH, FEATURES_3X3_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    original = pd.read_csv(ORIGINAL_PATH)
    manual = pd.read_csv(MANUAL_PATH)
    features_3x3 = pd.read_csv(FEATURES_3X3_PATH)

    if len(original) != 15000:
        raise ValueError(f"Expected 15000 original rows, got {len(original)}")
    if len(features_3x3) != 15000:
        raise ValueError(f"Expected 15000 3x3 rows, got {len(features_3x3)}")
    if "Original_ID" not in original.columns:
        raise ValueError("Original file missing Original_ID")
    if "Original_ID" not in features_3x3.columns:
        raise ValueError("3x3 file missing Original_ID")

    return original, manual, features_3x3


def get_labeled_manual(manual: pd.DataFrame) -> pd.DataFrame:
    labeled = manual.copy()
    labeled["Human_Class"] = pd.to_numeric(labeled["Human_Class"], errors="coerce")
    labeled = labeled[labeled["Human_Class"].notna()].copy()
    labeled["Human_Class"] = labeled["Human_Class"].astype(int)
    labeled["Original_ID"] = labeled["Original_ID"].astype(int)
    return labeled


def apply_corrections(original: pd.DataFrame, manual: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    corrected = original.copy()
    labeled = get_labeled_manual(manual)
    if labeled.empty:
        raise ValueError("No Human_Class labels found in Top150 manual file")

    original_lookup = corrected.set_index("Original_ID")["Class"]
    changes = 0

    for _, row in labeled.iterrows():
        original_id = int(row["Original_ID"])
        human_class = int(row["Human_Class"])
        old_class = int(original_lookup.loc[original_id])

        if old_class != human_class:
            corrected.loc[corrected["Original_ID"] == original_id, "Class"] = human_class
            changes += 1

    return corrected, changes


def get_3x3_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_stdDev")]
    if not cols:
        raise ValueError("No 3x3 mean/stdDev feature columns found")
    return cols


def build_corrected_3x3(features_3x3: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    corrected_classes = corrected[["Original_ID", "Class"]].rename(columns={"Class": "Class_corrected"})
    features_corrected = features_3x3.merge(
        corrected_classes,
        on="Original_ID",
        how="left",
        validate="one_to_one",
    )
    if features_corrected["Class_corrected"].isna().any():
        raise ValueError("Missing corrected Class values after 3x3 merge")

    features_corrected["Class"] = features_corrected["Class_corrected"].astype(int)
    features_corrected = features_corrected.drop(columns=["Class_corrected"])
    return features_corrected


def evaluate_oa(X: pd.DataFrame, y: pd.Series) -> float:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SPLIT_PARAMS["test_size"],
        random_state=SPLIT_PARAMS["random_state"],
        stratify=y if SPLIT_PARAMS["stratify"] else None,
    )

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    return float(accuracy_score(y_test, pred))


def format_gain(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def build_report(
    labels_changed: int,
    original_pixel_oa: float,
    margin_pixel_oa: float,
    original_3x3_oa: float,
    margin_3x3_oa: float,
) -> str:
    pixel_gain = margin_pixel_oa - original_pixel_oa
    three_by_three_gain = margin_3x3_oa - original_3x3_oa

    lines = [
        "=" * 50,
        f"CITY: {CITY}",
        "=" * 50,
        "",
        f"Labels Changed: {labels_changed}",
        "",
        "Original + Pixel OA:",
        f"{original_pixel_oa:.6f}",
        "",
        "Margin Corrected + Pixel OA:",
        f"{margin_pixel_oa:.6f}",
        "",
        "Pixel Improvement:",
        format_gain(pixel_gain),
        "",
        "Original + 3x3 OA:",
        f"{original_3x3_oa:.6f}",
        "",
        "Margin Corrected + 3x3 OA:",
        f"{margin_3x3_oa:.6f}",
        "",
        "3x3 Improvement:",
        format_gain(three_by_three_gain),
        "",
        "=" * 50,
    ]
    return "\n".join(lines)


def main() -> None:
    original, manual, features_3x3 = load_inputs()

    corrected, labels_changed = apply_corrections(original, manual)
    features_3x3_corrected = build_corrected_3x3(features_3x3, corrected)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (CITY_ROOT / "05_corrected").mkdir(parents=True, exist_ok=True)
    (CITY_ROOT / "06_3x3").mkdir(parents=True, exist_ok=True)

    corrected.to_csv(CORRECTED_PATH, index=False)
    features_3x3_corrected.to_csv(FEATURES_3X3_CORRECTED_PATH, index=False)

    original_pixel_oa = evaluate_oa(original[PIXEL_FEATURES], original["Class"])
    margin_pixel_oa = evaluate_oa(corrected[PIXEL_FEATURES], corrected["Class"])

    feature_cols_3x3 = get_3x3_feature_columns(features_3x3)
    original_3x3_oa = evaluate_oa(features_3x3[feature_cols_3x3], features_3x3["Class"])
    margin_3x3_oa = evaluate_oa(
        features_3x3_corrected[feature_cols_3x3],
        features_3x3_corrected["Class"],
    )

    pixel_gain = margin_pixel_oa - original_pixel_oa
    three_by_three_gain = margin_3x3_oa - original_3x3_oa

    report = build_report(
        labels_changed=labels_changed,
        original_pixel_oa=original_pixel_oa,
        margin_pixel_oa=margin_pixel_oa,
        original_3x3_oa=original_3x3_oa,
        margin_3x3_oa=margin_3x3_oa,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "City": CITY,
                "LabelsChanged": labels_changed,
                "OriginalPixelOA": original_pixel_oa,
                "MarginPixelOA": margin_pixel_oa,
                "PixelGain": pixel_gain,
                "Original3x3OA": original_3x3_oa,
                "Margin3x3OA": margin_3x3_oa,
                "ThreeByThreeGain": three_by_three_gain,
            }
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print(report)
    print()
    print(f"Saved corrected pixel file: {CORRECTED_PATH}")
    print(f"Saved corrected 3x3 file: {FEATURES_3X3_CORRECTED_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
