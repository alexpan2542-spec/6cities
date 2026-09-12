#!/usr/bin/env python3
"""Generate Changsha prediction npy files and method comparison Excel."""

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
from torch.utils.data import DataLoader, Dataset, TensorDataset

CITY = "Changsha"
CITY_ROOT = ROOT / "data2" / CITY
OUT_DIR = CITY_ROOT / "07_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_FEATURES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "MNDWI"]
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
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_w = 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss_each = crit(model(xb), yb)
            loss = (loss_each * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
            total_loss += float((loss_each * wb).sum().item())
            total_w += float(wb.sum().item())
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:03d} | loss {total_loss / total_w:.4f}")
    return model


@torch.no_grad()
def predict_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(xt)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.cpu().numpy().astype(np.int64), conf.cpu().numpy().astype(np.float64), probs.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    original = pd.read_csv(CITY_ROOT / "01_original" / f"{CITY}_WC_Samples_15000.csv")
    feat3 = pd.read_csv(CITY_ROOT / "06_3x3" / f"{CITY}_3x3_Features.csv")
    top_manual = get_labeled_manual(CITY_ROOT / "04_manual" / f"{CITY}_Top150_Manual.csv")
    bams_manual = get_labeled_manual(CITY_ROOT / "04_manual" / f"{CITY}_BAMS150_Manual.csv")

    # Use 3x3 features for all methods (shared representation)
    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all_1based = feat3["Class"].astype(int).to_numpy()
    y_all = y_all_1based - 1  # 0/1/2

    # Save feature/id for reproducibility
    np.save(OUT_DIR / "feature.npy", X_all)
    np.save(OUT_DIR / "id.npy", ids)

    # Corrected label arrays (0/1/2)
    margin_df = apply_corrections(feat3, top_manual)
    bams_df = apply_corrections(feat3, bams_manual)
    y_margin = margin_df["Class"].astype(int).to_numpy() - 1
    y_bams_full = bams_df["Class"].astype(int).to_numpy() - 1

    # Fixed split indices on Original_ID order
    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=SPLIT["test_size"],
        random_state=SPLIT["random_state"],
        stratify=y_all,
    )

    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_true = y_all[idx_test]

    # Scale features for MLP
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)
    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    X_pool_s = X_all_s  # all 15000 for pseudo labeling

    # ---- Baseline RF ----
    rf_base = RandomForestClassifier(**RF_PARAMS)
    rf_base.fit(X_train, y_all[idx_train])
    y_pred_baseline = rf_base.predict(X_test).astype(np.int64)

    # ---- Margin150 RF ----
    rf_margin = RandomForestClassifier(**RF_PARAMS)
    rf_margin.fit(X_train, y_margin[idx_train])
    y_pred_margin = rf_margin.predict(X_test).astype(np.int64)

    # ---- BAMS150 RF ----
    rf_bams = RandomForestClassifier(**RF_PARAMS)
    rf_bams.fit(X_train, y_bams_full[idx_train])
    y_pred_bams = rf_bams.predict(X_test).astype(np.int64)

    # ---- Teacher-Student ----
    # BAMS150 samples for teacher
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

    print("Training Teacher...")
    teacher = train_mlp(X_teacher, y_teacher, epochs=100, device=device)
    torch.save(teacher.state_dict(), OUT_DIR / "teacher.pth")

    pred_pool, conf_pool, _ = predict_mlp(teacher, X_pool_s, device=device)
    # Exclude train+bams from pool for cleaner pseudo labels
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True
    pseudo_mask = (~used) & (pred_pool == 2) & (conf_pool > 0.9)
    pseudo_x = X_pool_s[pseudo_mask]
    pseudo_y = pred_pool[pseudo_mask]
    print(f"Pseudo class3 samples: {len(pseudo_y)}")

    X_student = np.concatenate([X_train_s, X_bams, pseudo_x], axis=0)
    y_student = np.concatenate([y_all[idx_train], y_bams, pseudo_y], axis=0)
    w_student = np.concatenate(
        [
            np.ones(len(idx_train) + len(y_bams), dtype=np.float32),
            np.full(len(pseudo_y), 0.3, dtype=np.float32),
        ]
    )

    print("Training Student...")
    student = train_mlp(
        X_student, y_student, epochs=100, sample_weight=w_student, device=device
    )
    torch.save(student.state_dict(), OUT_DIR / "student.pth")
    y_pred_student, _, _ = predict_mlp(student, X_test_s, device=device)

    # ---- Save npy ----
    np.save(OUT_DIR / "y_true.npy", y_true.astype(np.int64))
    np.save(OUT_DIR / "y_pred_baseline.npy", y_pred_baseline)
    np.save(OUT_DIR / "y_pred_margin.npy", y_pred_margin)
    np.save(OUT_DIR / "y_pred_bams.npy", y_pred_bams)
    np.save(OUT_DIR / "y_pred_student.npy", y_pred_student)

    # ---- Comparison table ----
    methods = {
        "Baseline": y_pred_baseline,
        "Margin150": y_pred_margin,
        "BAMS150": y_pred_bams,
        "BAMS150 + TeacherStudent": y_pred_student,
    }
    rows = []
    for name, pred in methods.items():
        m = compute_metrics(y_true, pred)
        m["Method"] = name
        rows.append(m)

    df_compare = pd.DataFrame(rows)[
        ["Method", "OA", "AA", "Kappa", "Class3 Recall", "Class3 Precision", "Class3 F1"]
    ]
    xlsx_path = OUT_DIR / f"{CITY}_method_comparison.xlsx"
    csv_path = OUT_DIR / f"{CITY}_method_comparison.csv"
    df_compare.to_excel(xlsx_path, index=False)
    df_compare.to_csv(csv_path, index=False)

    print("\n" + df_compare.to_string(index=False))
    print(f"\nSaved npy + models + tables under: {OUT_DIR}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    main()
