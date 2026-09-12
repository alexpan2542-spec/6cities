#!/usr/bin/env python3
"""Reusable city evaluation for Margin150 and BAMS150 experiments."""

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
TOP150_MANUAL_PATH = CITY_ROOT / "04_manual" / f"{CITY}_Top150_Manual.csv"
BAMS150_MANUAL_PATH = CITY_ROOT / "04_manual" / f"{CITY}_BAMS150_Manual.csv"
FEATURES_3X3_PATH = CITY_ROOT / "06_3x3" / f"{CITY}_3x3_Features.csv"

CORRECTED_DIR = CITY_ROOT / "05_corrected"
FEATURES_3X3_DIR = CITY_ROOT / "06_3x3"
RESULTS_DIR = CITY_ROOT / "07_results"

CORRECTED_MARGIN_PATH = CORRECTED_DIR / f"{CITY}_Corrected_15000.csv"
CORRECTED_BAMS_PATH = CORRECTED_DIR / f"{CITY}_BAMS_Corrected_15000.csv"
FEATURES_3X3_MARGIN_PATH = FEATURES_3X3_DIR / f"{CITY}_3x3_Features_Corrected.csv"
FEATURES_3X3_BAMS_PATH = FEATURES_3X3_DIR / f"{CITY}_BAMS_Corrected_3x3.csv"

REPORT_PATH = RESULTS_DIR / f"{CITY}_Full_Report.txt"
SUMMARY_PATH = RESULTS_DIR / f"{CITY}_Summary.csv"

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

FEATURES_3X3 = [
    "B2_mean",
    "B2_stdDev",
    "B3_mean",
    "B3_stdDev",
    "B4_mean",
    "B4_stdDev",
    "B8_mean",
    "B8_stdDev",
    "B11_mean",
    "B11_stdDev",
    "B12_mean",
    "B12_stdDev",
    "NDVI_mean",
    "NDVI_stdDev",
    "NDBI_mean",
    "NDBI_stdDev",
    "MNDWI_mean",
    "MNDWI_stdDev",
]

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "n_jobs": -1,
}

SPLIT_PARAMS = {
    "test_size": 0.3,
    "random_state": 42,
    "stratify": True,
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (ORIGINAL_PATH, TOP150_MANUAL_PATH, BAMS150_MANUAL_PATH, FEATURES_3X3_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    original = pd.read_csv(ORIGINAL_PATH)
    top_manual = pd.read_csv(TOP150_MANUAL_PATH)
    bams_manual = pd.read_csv(BAMS150_MANUAL_PATH)
    features_3x3 = pd.read_csv(FEATURES_3X3_PATH)

    if len(original) != 15000:
        raise ValueError(f"Expected 15000 original rows, got {len(original)}")
    if len(features_3x3) != 15000:
        raise ValueError(f"Expected 15000 3x3 rows, got {len(features_3x3)}")
    if "Original_ID" not in original.columns or "Class" not in original.columns:
        raise ValueError("Original file missing Original_ID or Class")
    if "Original_ID" not in features_3x3.columns or "Class" not in features_3x3.columns:
        raise ValueError("3x3 file missing Original_ID or Class")

    missing_pixel = [c for c in PIXEL_FEATURES if c not in original.columns]
    if missing_pixel:
        raise ValueError(f"Missing pixel features: {missing_pixel}")

    missing_3x3 = [c for c in FEATURES_3X3 if c not in features_3x3.columns]
    if missing_3x3:
        raise ValueError(f"Missing 3x3 features: {missing_3x3}")

    return original, top_manual, bams_manual, features_3x3


def get_labeled_manual(manual: pd.DataFrame) -> pd.DataFrame:
    labeled = manual.copy()
    labeled["Human_Class"] = pd.to_numeric(labeled["Human_Class"], errors="coerce")
    labeled = labeled[labeled["Human_Class"].notna()].copy()
    if labeled.empty:
        raise ValueError("No Human_Class labels found in manual file")
    labeled["Human_Class"] = labeled["Human_Class"].astype(int)
    labeled["Original_ID"] = labeled["Original_ID"].astype(int)
    return labeled


def evaluate_manual_accuracy(manual: pd.DataFrame, original: pd.DataFrame) -> dict:
    labeled = get_labeled_manual(manual)
    class_lookup = original.set_index("Original_ID")["Class"].astype(int)

    missing = sorted(set(labeled["Original_ID"]) - set(class_lookup.index))
    if missing:
        raise ValueError(f"Manual Original_ID not found in original: {missing[:10]}")

    wc = labeled["Original_ID"].map(class_lookup).astype(int)
    human = labeled["Human_Class"].astype(int)

    correct = int((human == wc).sum())
    wrong = int((human != wc).sum())
    total = len(labeled)
    accuracy = correct / total
    error_rate = wrong / total

    return {
        "Correct": correct,
        "Wrong": wrong,
        "Accuracy": accuracy,
        "ErrorRate": error_rate,
        "Total": total,
    }


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


def apply_corrections(base: pd.DataFrame, manual: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    corrected = base.copy()
    labeled = get_labeled_manual(manual)
    class_lookup = corrected.set_index("Original_ID")["Class"].astype(int)

    labels_changed = 0
    for _, row in labeled.iterrows():
        original_id = int(row["Original_ID"])
        human_class = int(row["Human_Class"])
        old_class = int(class_lookup.loc[original_id])
        if old_class != human_class:
            corrected.loc[corrected["Original_ID"] == original_id, "Class"] = human_class
            labels_changed += 1

    return corrected, labels_changed


def build_corrected_3x3(features_3x3: pd.DataFrame, corrected_pixel: pd.DataFrame) -> pd.DataFrame:
    corrected_classes = corrected_pixel[["Original_ID", "Class"]].rename(
        columns={"Class": "Class_corrected"}
    )
    out = features_3x3.merge(
        corrected_classes,
        on="Original_ID",
        how="left",
        validate="one_to_one",
    )
    if out["Class_corrected"].isna().any():
        raise ValueError("Missing corrected Class after 3x3 merge")
    out["Class"] = out["Class_corrected"].astype(int)
    out = out.drop(columns=["Class_corrected"])
    return out


def format_gain(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def margin_accuracy(top_manual: pd.DataFrame, original: pd.DataFrame) -> dict:
    return evaluate_manual_accuracy(top_manual, original)


def bams_accuracy(bams_manual: pd.DataFrame, original: pd.DataFrame) -> dict:
    return evaluate_manual_accuracy(bams_manual, original)


def original_pixel_oa(original: pd.DataFrame) -> float:
    return evaluate_oa(original[PIXEL_FEATURES], original["Class"])


def margin_pixel_oa(original: pd.DataFrame, top_manual: pd.DataFrame, baseline_oa: float) -> dict:
    corrected, labels_changed = apply_corrections(original, top_manual)
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(CORRECTED_MARGIN_PATH, index=False)
    print(f"{CITY}_Corrected_15000.csv UPDATED")

    oa = evaluate_oa(corrected[PIXEL_FEATURES], corrected["Class"])
    return {
        "OA": oa,
        "LabelsChanged": labels_changed,
        "PixelGain": oa - baseline_oa,
        "Corrected": corrected,
    }


def bams_pixel_oa(original: pd.DataFrame, bams_manual: pd.DataFrame, baseline_oa: float) -> dict:
    corrected, labels_changed = apply_corrections(original, bams_manual)
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(CORRECTED_BAMS_PATH, index=False)
    print(f"{CITY}_BAMS_Corrected_15000.csv UPDATED")

    oa = evaluate_oa(corrected[PIXEL_FEATURES], corrected["Class"])
    return {
        "OA": oa,
        "LabelsChanged": labels_changed,
        "PixelGain": oa - baseline_oa,
        "Corrected": corrected,
    }


def original_3x3_oa(features_3x3: pd.DataFrame) -> float:
    return evaluate_oa(features_3x3[FEATURES_3X3], features_3x3["Class"])


def margin_3x3_oa(
    features_3x3: pd.DataFrame,
    corrected_pixel: pd.DataFrame,
    baseline_oa: float,
    top_manual: pd.DataFrame,
) -> dict:
    corrected_3x3 = build_corrected_3x3(features_3x3, corrected_pixel)
    FEATURES_3X3_DIR.mkdir(parents=True, exist_ok=True)
    corrected_3x3.to_csv(FEATURES_3X3_MARGIN_PATH, index=False)
    print(f"{CITY}_3x3_Features_Corrected.csv UPDATED")

    # LabelsChanged for 3x3 should match the manual-driven pixel corrections
    labels_changed = apply_corrections(features_3x3[["Original_ID", "Class"]].copy(), top_manual)[1]
    oa = evaluate_oa(corrected_3x3[FEATURES_3X3], corrected_3x3["Class"])
    return {
        "OA": oa,
        "LabelsChanged": labels_changed,
        "ThreeByThreeGain": oa - baseline_oa,
        "Corrected": corrected_3x3,
    }


def bams_3x3_oa(
    features_3x3: pd.DataFrame,
    corrected_pixel: pd.DataFrame,
    baseline_oa: float,
    bams_manual: pd.DataFrame,
) -> dict:
    corrected_3x3 = build_corrected_3x3(features_3x3, corrected_pixel)
    FEATURES_3X3_DIR.mkdir(parents=True, exist_ok=True)
    corrected_3x3.to_csv(FEATURES_3X3_BAMS_PATH, index=False)
    print(f"{CITY}_BAMS_Corrected_3x3.csv UPDATED")

    labels_changed = apply_corrections(features_3x3[["Original_ID", "Class"]].copy(), bams_manual)[1]
    oa = evaluate_oa(corrected_3x3[FEATURES_3X3], corrected_3x3["Class"])
    return {
        "OA": oa,
        "LabelsChanged": labels_changed,
        "ThreeByThreeGain": oa - baseline_oa,
        "Corrected": corrected_3x3,
    }


def pick_best(models: dict[str, float]) -> str:
    best_name = max(models, key=models.get)
    return f"{best_name} ({models[best_name]:.6f})"


def build_report(results: dict) -> str:
    lines = [
        "=" * 50,
        f"CITY: {CITY}",
        "=" * 50,
        "",
        "MARGIN150",
        "",
        f"Correct: {results['MarginCorrect']}",
        f"Wrong: {results['MarginWrong']}",
        f"Accuracy: {results['MarginAccuracy']:.6f}",
        f"ErrorRate: {results['MarginErrorRate']:.6f}",
        "",
        f"LabelsChanged: {results['MarginLabelsChanged']}",
        "",
        "Original + Pixel OA",
        f"{results['OriginalPixelOA']:.6f}",
        "",
        "Margin Corrected + Pixel OA",
        f"{results['MarginPixelOA']:.6f}",
        "",
        "Pixel Gain",
        format_gain(results["MarginPixelGain"]),
        "",
        "Original + 3x3 OA",
        f"{results['Original3x3OA']:.6f}",
        "",
        "Margin Corrected + 3x3 OA",
        f"{results['Margin3x3OA']:.6f}",
        "",
        "3x3 Gain",
        format_gain(results["Margin3x3Gain"]),
        "",
        "-" * 50,
        "",
        "BAMS150",
        "",
        f"Correct: {results['BAMSCorrect']}",
        f"Wrong: {results['BAMSWrong']}",
        f"Accuracy: {results['BAMSAccuracy']:.6f}",
        f"ErrorRate: {results['BAMSErrorRate']:.6f}",
        "",
        f"LabelsChanged: {results['BAMSLabelsChanged']}",
        "",
        "Original + Pixel OA",
        f"{results['OriginalPixelOA']:.6f}",
        "",
        "BAMS Corrected + Pixel OA",
        f"{results['BAMSPixelOA']:.6f}",
        "",
        "Pixel Gain",
        format_gain(results["BAMSPixelGain"]),
        "",
        "Original + 3x3 OA",
        f"{results['Original3x3OA']:.6f}",
        "",
        "BAMS Corrected + 3x3 OA",
        f"{results['BAMS3x3OA']:.6f}",
        "",
        "3x3 Gain",
        format_gain(results["BAMS3x3Gain"]),
        "",
        "-" * 50,
        "",
        "Best Pixel Model",
        results["BestPixelModel"],
        "",
        "Best 3x3 Model",
        results["Best3x3Model"],
        "",
        "=" * 50,
    ]
    return "\n".join(lines)


def main() -> None:
    original, top_manual, bams_manual, features_3x3 = load_inputs()

    margin_acc = margin_accuracy(top_manual, original)
    bams_acc = bams_accuracy(bams_manual, original)

    orig_pixel = original_pixel_oa(original)
    margin_pixel = margin_pixel_oa(original, top_manual, orig_pixel)
    bams_pixel = bams_pixel_oa(original, bams_manual, orig_pixel)

    orig_3x3 = original_3x3_oa(features_3x3)
    margin_3x3 = margin_3x3_oa(
        features_3x3,
        margin_pixel["Corrected"],
        orig_3x3,
        top_manual,
    )
    bams_3x3 = bams_3x3_oa(
        features_3x3,
        bams_pixel["Corrected"],
        orig_3x3,
        bams_manual,
    )

    best_pixel = pick_best(
        {
            "Original": orig_pixel,
            "Margin": margin_pixel["OA"],
            "BAMS": bams_pixel["OA"],
        }
    )
    best_3x3 = pick_best(
        {
            "Original": orig_3x3,
            "Margin": margin_3x3["OA"],
            "BAMS": bams_3x3["OA"],
        }
    )

    results = {
        "City": CITY,
        "MarginCorrect": margin_acc["Correct"],
        "MarginWrong": margin_acc["Wrong"],
        "MarginAccuracy": margin_acc["Accuracy"],
        "MarginErrorRate": margin_acc["ErrorRate"],
        "MarginLabelsChanged": margin_pixel["LabelsChanged"],
        "BAMSCorrect": bams_acc["Correct"],
        "BAMSWrong": bams_acc["Wrong"],
        "BAMSAccuracy": bams_acc["Accuracy"],
        "BAMSErrorRate": bams_acc["ErrorRate"],
        "BAMSLabelsChanged": bams_pixel["LabelsChanged"],
        "OriginalPixelOA": orig_pixel,
        "MarginPixelOA": margin_pixel["OA"],
        "BAMSPixelOA": bams_pixel["OA"],
        "Original3x3OA": orig_3x3,
        "Margin3x3OA": margin_3x3["OA"],
        "BAMS3x3OA": bams_3x3["OA"],
        "MarginPixelGain": margin_pixel["PixelGain"],
        "BAMSPixelGain": bams_pixel["PixelGain"],
        "Margin3x3Gain": margin_3x3["ThreeByThreeGain"],
        "BAMS3x3Gain": bams_3x3["ThreeByThreeGain"],
        "BestPixelModel": best_pixel,
        "Best3x3Model": best_3x3,
    }

    report = build_report(results)
    print()
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    summary_cols = [
        "City",
        "MarginCorrect",
        "MarginWrong",
        "MarginAccuracy",
        "MarginErrorRate",
        "MarginLabelsChanged",
        "BAMSCorrect",
        "BAMSWrong",
        "BAMSAccuracy",
        "BAMSErrorRate",
        "BAMSLabelsChanged",
        "OriginalPixelOA",
        "MarginPixelOA",
        "BAMSPixelOA",
        "Original3x3OA",
        "Margin3x3OA",
        "BAMS3x3OA",
        "MarginPixelGain",
        "BAMSPixelGain",
        "Margin3x3Gain",
        "BAMS3x3Gain",
    ]
    summary = pd.DataFrame([{k: results[k] for k in summary_cols}])
    summary.to_csv(SUMMARY_PATH, index=False)

    print()
    print("✅ Evaluation Complete")
    print("✅ Report Saved")
    print("✅ Summary CSV Saved")
    print("✅ Corrected Files Updated")
    print()
    print("Final summary table:")
    print(summary.to_string(index=False))
    print()
    print(f"Report: {REPORT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
