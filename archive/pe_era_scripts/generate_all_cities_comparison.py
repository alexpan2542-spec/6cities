#!/usr/bin/env python3
"""Run Baseline/Margin/BAMS/TeacherStudent comparison for all cities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_pkgs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"]
DATA2 = ROOT / "data2"

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

RF_PARAMS = dict(n_estimators=300, random_state=42, n_jobs=-1)
SPLIT = dict(test_size=0.3, random_state=42)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class WeightedDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.w = torch.from_numpy(w.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]


def get_labeled_manual(path: Path) -> pd.DataFrame:
    manual = pd.read_csv(path)
    manual["Human_Class"] = pd.to_numeric(manual["Human_Class"], errors="coerce")
    manual = manual[manual["Human_Class"].notna()].copy()
    manual["Human_Class"] = manual["Human_Class"].astype(int)
    manual["Original_ID"] = manual["Original_ID"].astype(int)
    if manual.empty:
        raise ValueError(f"No Human_Class in {path}")
    return manual


def apply_corrections(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = out.set_index("Original_ID")["Class"].astype(int)
    for _, row in manual.iterrows():
        oid = int(row["Original_ID"])
        human = int(row["Human_Class"])
        if int(lookup.loc[oid]) != human:
            out.loc[out["Original_ID"] == oid, "Class"] = human
    return out


def compute_metrics(y_true, y_pred, class3_label=2):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    aa = float(per_class_acc.mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "Class3 Recall": float(recall[class3_label]),
        "Class3 Precision": float(precision[class3_label]),
        "Class3 F1": float(f1[class3_label]),
    }


def train_mlp(X, y, epochs=100, lr=0.001, batch_size=256, sample_weight=None, device="cpu"):
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    loader = DataLoader(
        WeightedDataset(X, y, sample_weight),
        batch_size=batch_size,
        shuffle=True,
    )
    model = MLP(in_dim=X.shape[1], n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss_each = crit(model(xb), yb)
            loss = (loss_each * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(xt)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.cpu().numpy().astype(np.int64), conf.cpu().numpy().astype(np.float64)


def evaluate_city(city: str, device: str) -> pd.DataFrame:
    city_root = DATA2 / city
    out_dir = city_root / "07_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    top_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_Top150_Manual.csv")
    bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")

    missing = [c for c in FEATURES_3X3 if c not in feat3.columns]
    if missing:
        raise ValueError(f"{city}: missing 3x3 features {missing}")
    if "Original_ID" not in feat3.columns:
        raise ValueError(f"{city}: missing Original_ID in 3x3")

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1

    margin_df = apply_corrections(feat3, top_manual)
    bams_df = apply_corrections(feat3, bams_manual)
    y_margin = margin_df["Class"].astype(int).to_numpy() - 1
    y_bams_full = bams_df["Class"].astype(int).to_numpy() - 1

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=SPLIT["test_size"],
        random_state=SPLIT["random_state"],
        stratify=y_all,
    )

    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_true = y_all[idx_test]

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)
    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]

    rf_base = RandomForestClassifier(**RF_PARAMS)
    rf_base.fit(X_train, y_all[idx_train])
    y_pred_baseline = rf_base.predict(X_test).astype(np.int64)

    rf_margin = RandomForestClassifier(**RF_PARAMS)
    rf_margin.fit(X_train, y_margin[idx_train])
    y_pred_margin = rf_margin.predict(X_test).astype(np.int64)

    rf_bams = RandomForestClassifier(**RF_PARAMS)
    rf_bams.fit(X_train, y_bams_full[idx_train])
    y_pred_bams = rf_bams.predict(X_test).astype(np.int64)

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    bams_pos, bams_y = [], []
    for _, row in bams_manual.iterrows():
        oid = int(row["Original_ID"])
        bams_pos.append(id_to_pos[oid])
        bams_y.append(int(row["Human_Class"]) - 1)
    bams_pos = np.asarray(bams_pos, dtype=int)
    X_bams = X_all_s[bams_pos]
    y_bams = np.asarray(bams_y, dtype=np.int64)

    X_teacher = np.concatenate([X_train_s, X_bams], axis=0)
    y_teacher = np.concatenate([y_all[idx_train], y_bams], axis=0)
    teacher = train_mlp(X_teacher, y_teacher, epochs=100, device=device)
    torch.save(teacher.state_dict(), out_dir / "teacher.pth")

    pred_pool, conf_pool = predict_mlp(teacher, X_all_s, device=device)
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True
    pseudo_mask = (~used) & (pred_pool == 2) & (conf_pool > 0.9)
    pseudo_x = X_all_s[pseudo_mask]
    pseudo_y = pred_pool[pseudo_mask]

    X_student = np.concatenate([X_train_s, X_bams, pseudo_x], axis=0)
    y_student = np.concatenate([y_all[idx_train], y_bams, pseudo_y], axis=0)
    w_student = np.concatenate(
        [
            np.ones(len(idx_train) + len(y_bams), dtype=np.float32),
            np.full(len(pseudo_y), 0.3, dtype=np.float32),
        ]
    )
    student = train_mlp(
        X_student, y_student, epochs=100, sample_weight=w_student, device=device
    )
    torch.save(student.state_dict(), out_dir / "student.pth")
    y_pred_student, _ = predict_mlp(student, X_test_s, device=device)

    np.save(out_dir / "feature.npy", X_all)
    np.save(out_dir / "id.npy", ids)
    np.save(out_dir / "y_true.npy", y_true.astype(np.int64))
    np.save(out_dir / "y_pred_baseline.npy", y_pred_baseline)
    np.save(out_dir / "y_pred_margin.npy", y_pred_margin)
    np.save(out_dir / "y_pred_bams.npy", y_pred_bams)
    np.save(out_dir / "y_pred_student.npy", y_pred_student)

    methods = {
        "Baseline": y_pred_baseline,
        "Margin150": y_pred_margin,
        "BAMS150": y_pred_bams,
        "BAMS150 + TeacherStudent": y_pred_student,
    }
    rows = []
    for name, pred in methods.items():
        m = compute_metrics(y_true, pred)
        m["City"] = city
        m["Method"] = name
        m["PseudoClass3"] = int(pseudo_mask.sum())
        rows.append(m)

    df_city = pd.DataFrame(rows)[
        [
            "City",
            "Method",
            "OA",
            "AA",
            "Kappa",
            "Class3 Recall",
            "Class3 Precision",
            "Class3 F1",
            "PseudoClass3",
        ]
    ]
    df_city.to_csv(out_dir / f"{city}_method_comparison.csv", index=False)
    df_city.to_excel(out_dir / f"{city}_method_comparison.xlsx", index=False)
    print(f"[{city}] done | pseudo={int(pseudo_mask.sum())}")
    print(df_city[["Method", "OA", "Kappa", "Class3 F1"]].to_string(index=False))
    return df_city


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    all_rows = []
    for city in CITIES:
        print("\n" + "=" * 60)
        print("CITY:", city)
        print("=" * 60)
        df_city = evaluate_city(city, device=device)
        all_rows.append(df_city)

    df_all = pd.concat(all_rows, ignore_index=True)
    out_csv = DATA2 / "AllCities_method_comparison.csv"
    out_xlsx = DATA2 / "AllCities_method_comparison.xlsx"
    df_all.to_csv(out_csv, index=False)
    df_all.to_excel(out_xlsx, index=False)

    # Also a wide OA-only pivot for quick view
    oa_pivot = df_all.pivot(index="City", columns="Method", values="OA")
    oa_pivot = oa_pivot[
        ["Baseline", "Margin150", "BAMS150", "BAMS150 + TeacherStudent"]
    ]
    oa_pivot.to_csv(DATA2 / "AllCities_OA_pivot.csv")
    oa_pivot.to_excel(DATA2 / "AllCities_OA_pivot.xlsx")

    print("\n" + "=" * 60)
    print("ALL CITIES COMPARISON")
    print("=" * 60)
    print(df_all.to_string(index=False))
    print("\nOA pivot:")
    print(oa_pivot.to_string())
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_xlsx}")
    print(f"Saved: {DATA2 / 'AllCities_OA_pivot.xlsx'}")


if __name__ == "__main__":
    main()
