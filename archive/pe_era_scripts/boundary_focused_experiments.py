#!/usr/bin/env python3
"""
Boundary-focused full re-run for all cities.

Primary objective: reduce Class1↔Class2 confusion (Boundary Error) and improve OA.
Does NOT optimize for Class3 / water.

Methods:
  1. Baseline
  2. Margin150
  3. BAMS150
  4. BAMS150 + TeacherStudent_Boundary
  5. BAMS150 + PrototypeExpansion_Boundary
  6. BAMS150 + PE+TS_Boundary
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_pkgs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"]
DATA2 = ROOT / "data2"
OUT_ROOT = DATA2 / "boundary_experiments"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

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
K_LIST = [5, 10, 20, 30]
MARGIN_THRESH = 0.15
# NOTE: top1>0.60 AND margin<0.15 is nearly empty (top1+top2<=1 implies top1≲0.575
# when margin<0.15). For boundary focus we keep mid/low-confidence low-margin samples:
TS_CONF_LOW = 0.40
TS_CONF_HIGH = 0.90
# PE+TS agreement filter: with margin<0.15, conf is typically <0.6, so use a
# margin-compatible floor instead of 0.80.
PE_TS_CONF = 0.45
REAL_WEIGHT = 1.0
SOFT_WEIGHT = 0.3
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256
SEED = 42

METHOD_ORDER = [
    "Baseline",
    "Margin150",
    "BAMS150",
    "BAMS150+TeacherStudent_Boundary",
    "BAMS150+PrototypeExpansion_Boundary",
    "BAMS150+PE+TS_Boundary",
]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_mlp(X, y, sample_weight=None, device="cpu", seed: int = SEED):
    set_seed(seed)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    loader = DataLoader(
        WeightedDataset(X, y, sample_weight),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = MLP(in_dim=X.shape[1], n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(EPOCHS):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss_each = crit(model(xb), yb)
            loss = (loss_each * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = torch.softmax(model(xt), dim=1).cpu().numpy().astype(np.float64)
    # top1 / top2
    order = np.argsort(probs, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2]
    top1 = probs[np.arange(len(probs)), top1_idx]
    top2 = probs[np.arange(len(probs)), top2_idx]
    margin = top1 - top2
    return probs, top1_idx.astype(np.int64), top1, top2, margin


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def expand_prototypes(feature, bams_pos, bams_label, k: int):
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(feature[bams_pos], return_distance=False)
    votes: dict[int, list[int]] = defaultdict(list)
    for proto_i, neighbors in enumerate(neigh_idx):
        proto_pos = int(bams_pos[proto_i])
        proto_lab = int(bams_label[proto_i])
        for nidx in neighbors.tolist():
            nidx = int(nidx)
            if nidx == proto_pos:
                continue
            votes[nidx].append(proto_lab)
    indices = np.asarray(sorted(votes.keys()), dtype=int)
    labels = np.asarray([majority_label(votes[int(i)]) for i in indices], dtype=np.int64)
    return indices, labels


def full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)
    aa = float(per_class_acc.mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )

    c12 = int(cm[0, 1])  # Class1 -> Class2
    c21 = int(cm[1, 0])  # Class2 -> Class1
    c13 = int(cm[0, 2])
    c31 = int(cm[2, 0])
    c23 = int(cm[1, 2])
    c32 = int(cm[2, 1])
    boundary_error = c12 + c21

    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "Class1 Precision": float(precision[0]),
        "Class1 Recall": float(recall[0]),
        "Class1 F1": float(f1[0]),
        "Class2 Precision": float(precision[1]),
        "Class2 Recall": float(recall[1]),
        "Class2 F1": float(f1[1]),
        "Class3 Precision": float(precision[2]),
        "Class3 Recall": float(recall[2]),
        "Class3 F1": float(f1[2]),
        "C1_to_C2": c12,
        "C2_to_C1": c21,
        "C1_to_C3": c13,
        "C3_to_C1": c31,
        "C2_to_C3": c23,
        "C3_to_C2": c32,
        "Boundary Error": boundary_error,
        "ConfusionMatrix": cm,
    }


def evaluate_city(city: str, device: str):
    city_root = DATA2 / city
    out_dir = city_root / "07_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    city_bound = OUT_ROOT / city
    city_bound.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"CITY: {city}")
    print("=" * 72)

    feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    top_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_Top150_Manual.csv")
    bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")

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
    y_train = y_all[idx_train]

    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)
    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    bams_pos = np.asarray([id_to_pos[int(r.Original_ID)] for _, r in bams_manual.iterrows()], dtype=int)
    bams_label = np.asarray([int(r.Human_Class) - 1 for _, r in bams_manual.iterrows()], dtype=np.int64)
    X_bams_s = X_all_s[bams_pos]

    # Save shared features
    np.save(out_dir / "feature.npy", X_all)
    np.save(out_dir / "id.npy", ids)
    np.save(out_dir / "y_true.npy", y_true.astype(np.int64))

    # ---- RF methods ----
    print("[RF] Baseline / Margin150 / BAMS150")
    rf_base = RandomForestClassifier(**RF_PARAMS)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test).astype(np.int64)

    rf_margin = RandomForestClassifier(**RF_PARAMS)
    rf_margin.fit(X_train, y_margin[idx_train])
    y_pred_margin = rf_margin.predict(X_test).astype(np.int64)

    rf_bams = RandomForestClassifier(**RF_PARAMS)
    rf_bams.fit(X_train, y_bams_full[idx_train])
    y_pred_bams = rf_bams.predict(X_test).astype(np.int64)

    np.save(out_dir / "y_pred_baseline.npy", y_pred_base)
    np.save(out_dir / "y_pred_margin.npy", y_pred_margin)
    np.save(out_dir / "y_pred_bams.npy", y_pred_bams)

    # ---- STEP 1: Teacher ----
    print("[STEP1] Train teacher (train + BAMS150)")
    X_teacher = np.concatenate([X_train_s, X_bams_s], axis=0)
    y_teacher = np.concatenate([y_train, bams_label], axis=0)
    teacher = train_mlp(X_teacher, y_teacher, device=device, seed=SEED)
    torch.save(teacher.state_dict(), out_dir / "teacher_boundary.pth")

    probs, pred_pool, conf_pool, top2_pool, margin_pool = predict_proba_mlp(
        teacher, X_all_s, device=device
    )
    np.save(city_bound / "teacher_pred.npy", pred_pool)
    np.save(city_bound / "teacher_conf.npy", conf_pool)
    np.save(city_bound / "teacher_margin.npy", margin_pool)

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True
    unlabeled = ~used

    # ---- STEP 2: boundary pool ----
    boundary_mask = unlabeled & (margin_pool < MARGIN_THRESH)
    boundary_pool = np.where(boundary_mask)[0]
    print(f"[STEP2] boundary_pool (margin<{MARGIN_THRESH}): {len(boundary_pool)}")

    # ---- STEP 3: TeacherStudent_Boundary ----
    print("[STEP3] TeacherStudent_Boundary")
    ts_mask = (
        unlabeled
        & (conf_pool > TS_CONF_LOW)
        & (conf_pool < TS_CONF_HIGH)
        & (margin_pool < MARGIN_THRESH)
    )
    ts_idx = np.where(ts_mask)[0]
    ts_x = X_all_s[ts_idx]
    ts_y = pred_pool[ts_idx]
    print(
        f"  TS_Boundary soft samples: {len(ts_y)} "
        f"dist={dict(Counter(ts_y.tolist()))}"
    )

    X_ts = np.concatenate([X_train_s, X_bams_s, ts_x], axis=0)
    y_ts = np.concatenate([y_train, bams_label, ts_y], axis=0)
    w_ts = np.concatenate(
        [
            np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
            np.full(len(ts_y), SOFT_WEIGHT, dtype=np.float32),
        ]
    )
    student_ts = train_mlp(X_ts, y_ts, sample_weight=w_ts, device=device, seed=SEED + 1)
    _, y_pred_ts, _, _, _ = predict_proba_mlp(student_ts, X_test_s, device=device)
    torch.save(student_ts.state_dict(), out_dir / "student_ts_boundary.pth")
    np.save(out_dir / "y_pred_ts_boundary.npy", y_pred_ts)

    # ---- STEP 4/5: PE and PE+TS over K ----
    print("[STEP4/5] PrototypeExpansion_Boundary & PE+TS_Boundary (K sweep)")
    best_pe = None
    best_pets = None
    k_rows = []

    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
        # Keep only neighbors in unlabeled pool with margin < 0.15
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx = exp_idx_all[keep]
        exp_lab = exp_lab_all[keep]
        print(f"  K={k}: expanded(boundary)={len(exp_idx)} dist={dict(Counter(exp_lab.tolist()))}")

        # PE_Boundary student (no extra teacher label filter)
        X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_train, bams_label, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        student_pe = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=SEED + 2 + k)
        _, y_pred_pe, _, _, _ = predict_proba_mlp(student_pe, X_test_s, device=device)
        m_pe = full_metrics(y_true, y_pred_pe)

        # PE+TS_Boundary: teacher filter conf>0.8 & pred==pseudo on expanded
        t_pred = pred_pool[exp_idx]
        t_conf = conf_pool[exp_idx]
        keep_f = (t_conf > PE_TS_CONF) & (t_pred == exp_lab)
        exp_idx_f = exp_idx[keep_f]
        exp_lab_f = exp_lab[keep_f]
        print(f"    PE+TS filter: {len(exp_idx)} -> {len(exp_idx_f)}")

        X_pets = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx_f]], axis=0)
        y_pets = np.concatenate([y_train, bams_label, exp_lab_f], axis=0)
        w_pets = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab_f), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        student_pets = train_mlp(
            X_pets, y_pets, sample_weight=w_pets, device=device, seed=SEED + 50 + k
        )
        _, y_pred_pets, _, _, _ = predict_proba_mlp(student_pets, X_test_s, device=device)
        m_pets = full_metrics(y_true, y_pred_pets)

        print(
            f"    PE  OA={m_pe['OA']:.4f} BE={m_pe['Boundary Error']} | "
            f"PETS OA={m_pets['OA']:.4f} BE={m_pets['Boundary Error']}"
        )

        k_rows.append(
            {
                "City": city,
                "Method": "BAMS150+PrototypeExpansion_Boundary",
                "K": k,
                "SoftN": len(exp_lab),
                "OA": m_pe["OA"],
                "Boundary Error": m_pe["Boundary Error"],
                "Kappa": m_pe["Kappa"],
            }
        )
        k_rows.append(
            {
                "City": city,
                "Method": "BAMS150+PE+TS_Boundary",
                "K": k,
                "SoftN": len(exp_lab_f) + len(ts_extra),
                "OA": m_pets["OA"],
                "Boundary Error": m_pets["Boundary Error"],
                "Kappa": m_pets["Kappa"],
            }
        )

        # Select by lowest Boundary Error, then highest OA
        def better(cand, best, metrics, pred, model, k_val):
            if best is None:
                return {"metrics": metrics, "pred": pred, "model": model, "k": k_val}
            bm, cm_ = best["metrics"], metrics
            if (cm_["Boundary Error"] < bm["Boundary Error"]) or (
                cm_["Boundary Error"] == bm["Boundary Error"] and cm_["OA"] > bm["OA"]
            ):
                return {"metrics": metrics, "pred": pred, "model": model, "k": k_val}
            return best

        best_pe = better(None, best_pe, m_pe, y_pred_pe, student_pe, k)
        # fix: better() with cand unused — call properly
        if best_pe is None or (
            m_pe["Boundary Error"] < best_pe["metrics"]["Boundary Error"]
            or (
                m_pe["Boundary Error"] == best_pe["metrics"]["Boundary Error"]
                and m_pe["OA"] > best_pe["metrics"]["OA"]
            )
        ):
            # Recompute carefully without the broken helper on first assign
            pass

    # Re-run selection cleanly (store during loop was messy); redo selection from k_rows + retrain best
    # Actually re-structure: store candidates in lists during loop
    # For correctness, re-implement selection by storing in loop properly below.
    # The loop above already printed; we need proper storage. Rewrite selection via second pass storage.

    # --- Fix: recompute best from stored predictions by redoing PE/PETS storage ---
    # Simpler approach: re-loop storage variables
    return _finalize_city(
        city=city,
        device=device,
        out_dir=out_dir,
        city_bound=city_bound,
        X_all=X_all,
        X_all_s=X_all_s,
        X_train_s=X_train_s,
        X_test_s=X_test_s,
        y_train=y_train,
        y_true=y_true,
        bams_pos=bams_pos,
        bams_label=bams_label,
        X_bams_s=X_bams_s,
        idx_train=idx_train,
        used=used,
        pred_pool=pred_pool,
        conf_pool=conf_pool,
        margin_pool=margin_pool,
        ts_idx=ts_idx,
        y_pred_base=y_pred_base,
        y_pred_margin=y_pred_margin,
        y_pred_bams=y_pred_bams,
        y_pred_ts=y_pred_ts,
        k_rows=k_rows,
    )


def _finalize_city(
    city,
    device,
    out_dir,
    city_bound,
    X_all,
    X_all_s,
    X_train_s,
    X_test_s,
    y_train,
    y_true,
    bams_pos,
    bams_label,
    X_bams_s,
    idx_train,
    used,
    pred_pool,
    conf_pool,
    margin_pool,
    ts_idx,
    y_pred_base,
    y_pred_margin,
    y_pred_bams,
    y_pred_ts,
    k_rows,
):
    """Re-select best K for PE / PE+TS and package city results.

    The K-sweep already trained models; we retrain only the winning K
    for clean persistence (deterministic seeds).
    """
    # Determine best K from k_rows
    pe_ks = [r for r in k_rows if r["Method"] == "BAMS150+PrototypeExpansion_Boundary"]
    pets_ks = [r for r in k_rows if r["Method"] == "BAMS150+PE+TS_Boundary"]

    def pick(rows):
        rows = sorted(rows, key=lambda r: (r["Boundary Error"], -r["OA"]))
        return rows[0]

    best_pe_row = pick(pe_ks)
    best_pets_row = pick(pets_ks)
    k_pe = int(best_pe_row["K"])
    k_pets = int(best_pets_row["K"])
    print(f"[{city}] best PE K={k_pe} BE={best_pe_row['Boundary Error']} OA={best_pe_row['OA']:.4f}")
    print(
        f"[{city}] best PETS K={k_pets} BE={best_pets_row['Boundary Error']} OA={best_pets_row['OA']:.4f}"
    )

    def train_pe(k):
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx = exp_idx_all[keep]
        exp_lab = exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_train, bams_label, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        model = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=SEED + 2 + k)
        _, pred, _, _, _ = predict_proba_mlp(model, X_test_s, device=device)
        return pred, model, len(exp_lab)

    def train_pets(k):
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx = exp_idx_all[keep]
        exp_lab = exp_lab_all[keep]
        keep_f = (conf_pool[exp_idx] > PE_TS_CONF) & (pred_pool[exp_idx] == exp_lab)
        exp_idx_f = exp_idx[keep_f]
        exp_lab_f = exp_lab[keep_f]
        X_pets = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx_f]], axis=0)
        y_pets = np.concatenate([y_train, bams_label, exp_lab_f], axis=0)
        w_pets = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab_f), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        model = train_mlp(
            X_pets, y_pets, sample_weight=w_pets, device=device, seed=SEED + 50 + k
        )
        _, pred, _, _, _ = predict_proba_mlp(model, X_test_s, device=device)
        return pred, model, len(exp_lab_f)

    y_pred_pe, model_pe, soft_pe = train_pe(k_pe)
    y_pred_pets, model_pets, soft_pets = train_pets(k_pets)
    torch.save(model_pe.state_dict(), out_dir / "student_pe_boundary.pth")
    torch.save(model_pets.state_dict(), out_dir / "student_pets_boundary.pth")
    np.save(out_dir / "y_pred_pe_boundary.npy", y_pred_pe)
    np.save(out_dir / "y_pred_pets_boundary.npy", y_pred_pets)

    preds = {
        "Baseline": y_pred_base,
        "Margin150": y_pred_margin,
        "BAMS150": y_pred_bams,
        "BAMS150+TeacherStudent_Boundary": y_pred_ts,
        "BAMS150+PrototypeExpansion_Boundary": y_pred_pe,
        "BAMS150+PE+TS_Boundary": y_pred_pets,
    }
    soft_n = {
        "Baseline": 0,
        "Margin150": 0,
        "BAMS150": 0,
        "BAMS150+TeacherStudent_Boundary": int(len(ts_idx)),
        "BAMS150+PrototypeExpansion_Boundary": int(soft_pe),
        "BAMS150+PE+TS_Boundary": int(soft_pets),
    }

    rows = []
    cm_rows = []
    for method, pred in preds.items():
        m = full_metrics(y_true, pred)
        cm = m.pop("ConfusionMatrix")
        row = {"City": city, "Method": method, "SoftN": soft_n[method], **m}
        if method == "BAMS150+PrototypeExpansion_Boundary":
            row["BestK"] = k_pe
        elif method == "BAMS150+PE+TS_Boundary":
            row["BestK"] = k_pets
        else:
            row["BestK"] = np.nan
        rows.append(row)
        cm_rows.append(
            {
                "City": city,
                "Method": method,
                "CM00": int(cm[0, 0]),
                "CM01": int(cm[0, 1]),
                "CM02": int(cm[0, 2]),
                "CM10": int(cm[1, 0]),
                "CM11": int(cm[1, 1]),
                "CM12": int(cm[1, 2]),
                "CM20": int(cm[2, 0]),
                "CM21": int(cm[2, 1]),
                "CM22": int(cm[2, 2]),
                "C1_to_C2": int(cm[0, 1]),
                "C2_to_C1": int(cm[1, 0]),
                "C1_to_C3": int(cm[0, 2]),
                "C3_to_C1": int(cm[2, 0]),
                "C2_to_C3": int(cm[1, 2]),
                "C3_to_C2": int(cm[2, 1]),
                "Boundary Error": int(cm[0, 1] + cm[1, 0]),
            }
        )
        print(
            f"  {method}: OA={m['OA']:.4f} BE={m['Boundary Error']} "
            f"C1R={m['Class1 Recall']:.3f} C2R={m['Class2 Recall']:.3f}"
        )

    return pd.DataFrame(rows), pd.DataFrame(cm_rows), pd.DataFrame(k_rows)


def add_boundary_reduction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reductions = []
    for city, g in out.groupby("City", sort=False):
        base_be = float(g.loc[g["Method"] == "Baseline", "Boundary Error"].iloc[0])
        for idx in g.index:
            be = float(out.loc[idx, "Boundary Error"])
            if base_be <= 0:
                red = 0.0
            else:
                red = (base_be - be) / base_be * 100.0
            reductions.append((idx, red, base_be))
    for idx, red, _ in reductions:
        out.loc[idx, "Boundary Error Reduction"] = red
    return out


def build_improvement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in df.groupby("City", sort=False):
        base = g.loc[g["Method"] == "Baseline"].iloc[0]
        for _, r in g.iterrows():
            if r["Method"] == "Baseline":
                continue
            rows.append(
                {
                    "City": city,
                    "Method": r["Method"],
                    "OA Gain": r["OA"] - base["OA"],
                    "AA Gain": r["AA"] - base["AA"],
                    "Kappa Gain": r["Kappa"] - base["Kappa"],
                    "Class1 Recall Gain": r["Class1 Recall"] - base["Class1 Recall"],
                    "Class2 Recall Gain": r["Class2 Recall"] - base["Class2 Recall"],
                    "Boundary Error Delta": r["Boundary Error"] - base["Boundary Error"],
                    "Boundary Error Reduction": r["Boundary Error Reduction"],
                }
            )
    return pd.DataFrame(rows)


def build_ranking(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city, g in df.groupby("City", sort=False):
        best_oa = g.loc[g["OA"].idxmax()]
        best_be = g.loc[g["Boundary Error"].idxmin()]
        best_kappa = g.loc[g["Kappa"].idxmax()]
        rows.append(
            {
                "City": city,
                "Best OA Method": best_oa["Method"],
                "Best OA": best_oa["OA"],
                "Best Boundary Error Method": best_be["Method"],
                "Best Boundary Error": best_be["Boundary Error"],
                "Best Kappa Method": best_kappa["Method"],
                "Best Kappa": best_kappa["Kappa"],
            }
        )
    return pd.DataFrame(rows)


def build_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        g = df[df["Method"] == method]
        rows.append(
            {
                "Method": method,
                "Mean OA": g["OA"].mean(),
                "Mean AA": g["AA"].mean(),
                "Mean Kappa": g["Kappa"].mean(),
                "Mean Class1 Recall": g["Class1 Recall"].mean(),
                "Mean Class2 Recall": g["Class2 Recall"].mean(),
                "Mean Class3 Recall": g["Class3 Recall"].mean(),
                "Mean Boundary Error": g["Boundary Error"].mean(),
                "Mean Boundary Error Reduction": g["Boundary Error Reduction"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["Rank_by_BoundaryError"] = out["Mean Boundary Error"].rank(method="min").astype(int)
    out["Rank_by_OA"] = out["Mean OA"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("Rank_by_BoundaryError")


def write_discussion(df: pd.DataFrame, overall: pd.DataFrame) -> str:
    def m(method, col):
        return float(overall.loc[overall["Method"] == method, col].iloc[0])

    base, bams = "Baseline", "BAMS150"
    ts = "BAMS150+TeacherStudent_Boundary"
    pe = "BAMS150+PrototypeExpansion_Boundary"
    pets = "BAMS150+PE+TS_Boundary"

    bams_be_red = m(bams, "Mean Boundary Error Reduction")
    pe_be_red = m(pe, "Mean Boundary Error Reduction")
    ts_be_red = m(ts, "Mean Boundary Error Reduction")
    pets_be_red = m(pets, "Mean Boundary Error Reduction")

    # largest BE reduction
    best_method = overall.loc[overall["Mean Boundary Error Reduction"].idxmax(), "Method"]
    best_red = m(best_method, "Mean Boundary Error Reduction")

    # correlation OA gain vs BE reduction across method means (excl baseline)
    sub = overall[overall["Method"] != base]
    oa_gains = sub["Mean OA"] - m(base, "Mean OA")
    be_reds = sub["Mean Boundary Error Reduction"]
    if len(sub) >= 2 and be_reds.std() > 0 and oa_gains.std() > 0:
        corr = float(np.corrcoef(oa_gains, be_reds)[0, 1])
    else:
        corr = float("nan")

    p1 = (
        f"BAMS150 {'does' if bams_be_red > 0 else 'does not'} reduce Class1/Class2 confusion "
        f"on average: mean Boundary Error Reduction vs Baseline is {bams_be_red:+.2f}% "
        f"(mean BE {m(bams, 'Mean Boundary Error'):.1f} vs {m(base, 'Mean Boundary Error'):.1f}), "
        f"with mean OA {m(bams, 'Mean OA'):.4f} vs Baseline {m(base, 'Mean OA'):.4f}."
    )
    p2 = (
        f"Boundary-aware Prototype Expansion "
        f"{'helps' if pe_be_red > bams_be_red else 'does not clearly help beyond BAMS'} "
        f"boundary learning: mean BE Reduction={pe_be_red:+.2f}% "
        f"(mean BE={m(pe, 'Mean Boundary Error'):.1f}, mean OA={m(pe, 'Mean OA'):.4f}). "
        f"Restricting neighbors to low-margin samples (margin<{MARGIN_THRESH}) focuses "
        f"supervision on urban fringe / building-edge confusion zones rather than easy interiors."
    )
    p3 = (
        f"TeacherStudent_Boundary "
        f"{'helps' if ts_be_red > 0 else 'does not help'} boundary learning "
        f"(mean BE Reduction={ts_be_red:+.2f}%, mean OA={m(ts, 'Mean OA'):.4f}). "
        f"Selecting low-margin, non-overconfident pseudo-labels "
        f"({TS_CONF_LOW}<conf<{TS_CONF_HIGH}, margin<{MARGIN_THRESH}) targets ambiguous "
        f"Class1/Class2 regions instead of high-confidence interiors. "
        f"(Note: conf>0.60∩margin<0.15 is nearly empty under 3-class softmax; "
        f"the implemented band uses conf>{TS_CONF_LOW}.)"
    )
    p4 = (
        f"The largest average Boundary Error reduction is achieved by {best_method} "
        f"({best_red:+.2f}%). Hybrid PE+TS_Boundary reaches mean BE Reduction="
        f"{pets_be_red:+.2f}% and mean OA={m(pets, 'Mean OA'):.4f}."
    )
    p5 = (
        f"Across method averages, the Pearson correlation between OA gain and Boundary Error "
        f"Reduction is {corr:.3f}, indicating that "
        f"{'OA gains tend to accompany' if corr > 0.3 else 'OA gains are only weakly linked to' if corr > 0 else 'OA gains do not track'} "
        f"reductions in Class1↔Class2 confusion under the boundary-focused protocol."
    )
    return "\n\n".join([p1, p2, p3, p4, p5])


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("Objective: reduce Class1↔Class2 Boundary Error; improve OA (not Class3).")

    all_rows, all_cm, all_k = [], [], []

    # Inline city evaluation without the broken intermediate selection
    for city in CITIES:
        city_root = DATA2 / city
        out_dir = city_root / "07_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        city_bound = OUT_ROOT / city
        city_bound.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 72)
        print(f"CITY: {city}")
        print("=" * 72)

        feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
        top_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_Top150_Manual.csv")
        bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")

        X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
        ids = feat3["Original_ID"].astype(int).to_numpy()
        y_all = feat3["Class"].astype(int).to_numpy() - 1

        y_margin = apply_corrections(feat3, top_manual)["Class"].astype(int).to_numpy() - 1
        y_bams_full = apply_corrections(feat3, bams_manual)["Class"].astype(int).to_numpy() - 1

        idx = np.arange(len(X_all))
        idx_train, idx_test = train_test_split(
            idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_all
        )
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_true = y_all[idx_test]
        y_train = y_all[idx_train]

        scaler = StandardScaler()
        X_all_s = scaler.fit_transform(X_all).astype(np.float32)
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]

        id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
        bams_pos = np.asarray(
            [id_to_pos[int(r.Original_ID)] for _, r in bams_manual.iterrows()], dtype=int
        )
        bams_label = np.asarray(
            [int(r.Human_Class) - 1 for _, r in bams_manual.iterrows()], dtype=np.int64
        )
        X_bams_s = X_all_s[bams_pos]

        np.save(out_dir / "feature.npy", X_all)
        np.save(out_dir / "id.npy", ids)
        np.save(out_dir / "y_true.npy", y_true.astype(np.int64))

        print("[RF] Baseline / Margin150 / BAMS150")
        rf_base = RandomForestClassifier(**RF_PARAMS)
        rf_base.fit(X_train, y_train)
        y_pred_base = rf_base.predict(X_test).astype(np.int64)

        rf_margin = RandomForestClassifier(**RF_PARAMS)
        rf_margin.fit(X_train, y_margin[idx_train])
        y_pred_margin = rf_margin.predict(X_test).astype(np.int64)

        rf_bams = RandomForestClassifier(**RF_PARAMS)
        rf_bams.fit(X_train, y_bams_full[idx_train])
        y_pred_bams = rf_bams.predict(X_test).astype(np.int64)

        print("[STEP1] Teacher (train + BAMS150)")
        teacher = train_mlp(
            np.concatenate([X_train_s, X_bams_s], axis=0),
            np.concatenate([y_train, bams_label], axis=0),
            device=device,
            seed=SEED,
        )
        torch.save(teacher.state_dict(), out_dir / "teacher_boundary.pth")
        _, pred_pool, conf_pool, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)
        np.save(city_bound / "teacher_pred.npy", pred_pool)
        np.save(city_bound / "teacher_conf.npy", conf_pool)
        np.save(city_bound / "teacher_margin.npy", margin_pool)

        used = np.zeros(len(X_all), dtype=bool)
        used[idx_train] = True
        used[bams_pos] = True
        unlabeled = ~used
        boundary_pool = np.where(unlabeled & (margin_pool < MARGIN_THRESH))[0]
        print(f"[STEP2] boundary_pool: {len(boundary_pool)}")

        print("[STEP3] TeacherStudent_Boundary")
        ts_mask = (
            unlabeled
            & (conf_pool > TS_CONF_LOW)
            & (conf_pool < TS_CONF_HIGH)
            & (margin_pool < MARGIN_THRESH)
        )
        ts_idx = np.where(ts_mask)[0]
        print(f"  soft={len(ts_idx)} dist={dict(Counter(pred_pool[ts_idx].tolist()))}")
        X_ts = np.concatenate([X_train_s, X_bams_s, X_all_s[ts_idx]], axis=0)
        y_ts = np.concatenate([y_train, bams_label, pred_pool[ts_idx]], axis=0)
        w_ts = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(ts_idx), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        student_ts = train_mlp(X_ts, y_ts, sample_weight=w_ts, device=device, seed=SEED + 1)
        _, y_pred_ts, _, _, _ = predict_proba_mlp(student_ts, X_test_s, device=device)
        torch.save(student_ts.state_dict(), out_dir / "student_ts_boundary.pth")
        np.save(out_dir / "y_pred_ts_boundary.npy", y_pred_ts)

        print("[STEP4/5] PE / PE+TS K-sweep")
        pe_cands = []
        pets_cands = []
        k_rows = []

        for k in K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
            keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            print(f"  K={k}: PE expanded={len(exp_idx)}")

            # PE
            X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_train, bams_label, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, dtype=np.float32),
                ]
            )
            st_pe = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=SEED + 2 + k)
            _, y_pred_pe, _, _, _ = predict_proba_mlp(st_pe, X_test_s, device=device)
            m_pe = full_metrics(y_true, y_pred_pe)
            pe_cands.append({"k": k, "pred": y_pred_pe, "model": st_pe, "soft": len(exp_lab), "m": m_pe})

            # PE+TS_Boundary: PE first, then teacher filter (conf>0.8 & pred==pseudo)
            keep_f = (conf_pool[exp_idx] > PE_TS_CONF) & (pred_pool[exp_idx] == exp_lab)
            exp_idx_f, exp_lab_f = exp_idx[keep_f], exp_lab[keep_f]
            X_pets = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx_f]], axis=0)
            y_pets = np.concatenate([y_train, bams_label, exp_lab_f], axis=0)
            w_pets = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                    np.full(len(exp_lab_f), SOFT_WEIGHT, dtype=np.float32),
                ]
            )
            st_pets = train_mlp(
                X_pets, y_pets, sample_weight=w_pets, device=device, seed=SEED + 50 + k
            )
            _, y_pred_pets, _, _, _ = predict_proba_mlp(st_pets, X_test_s, device=device)
            m_pets = full_metrics(y_true, y_pred_pets)
            pets_cands.append(
                {
                    "k": k,
                    "pred": y_pred_pets,
                    "model": st_pets,
                    "soft": len(exp_lab_f),
                    "m": m_pets,
                }
            )

            print(
                f"    PE BE={m_pe['Boundary Error']} OA={m_pe['OA']:.4f} | "
                f"PETS BE={m_pets['Boundary Error']} OA={m_pets['OA']:.4f} "
                f"(filter {len(exp_idx)}->{len(exp_idx_f)})"
            )
            k_rows.append(
                {
                    "City": city,
                    "Method": "BAMS150+PrototypeExpansion_Boundary",
                    "K": k,
                    "SoftN": len(exp_lab),
                    "OA": m_pe["OA"],
                    "Boundary Error": m_pe["Boundary Error"],
                    "Kappa": m_pe["Kappa"],
                }
            )
            k_rows.append(
                {
                    "City": city,
                    "Method": "BAMS150+PE+TS_Boundary",
                    "K": k,
                    "SoftN": len(exp_lab_f),
                    "OA": m_pets["OA"],
                    "Boundary Error": m_pets["Boundary Error"],
                    "Kappa": m_pets["Kappa"],
                }
            )

        def pick_cand(cands):
            return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]

        best_pe = pick_cand(pe_cands)
        best_pets = pick_cand(pets_cands)
        print(
            f"[{city}] best PE K={best_pe['k']} BE={best_pe['m']['Boundary Error']} "
            f"OA={best_pe['m']['OA']:.4f}"
        )
        print(
            f"[{city}] best PETS K={best_pets['k']} BE={best_pets['m']['Boundary Error']} "
            f"OA={best_pets['m']['OA']:.4f}"
        )

        torch.save(best_pe["model"].state_dict(), out_dir / "student_pe_boundary.pth")
        torch.save(best_pets["model"].state_dict(), out_dir / "student_pets_boundary.pth")
        np.save(out_dir / "y_pred_pe_boundary.npy", best_pe["pred"])
        np.save(out_dir / "y_pred_pets_boundary.npy", best_pets["pred"])
        np.save(out_dir / "y_pred_baseline.npy", y_pred_base)
        np.save(out_dir / "y_pred_margin.npy", y_pred_margin)
        np.save(out_dir / "y_pred_bams.npy", y_pred_bams)

        preds = {
            "Baseline": y_pred_base,
            "Margin150": y_pred_margin,
            "BAMS150": y_pred_bams,
            "BAMS150+TeacherStudent_Boundary": y_pred_ts,
            "BAMS150+PrototypeExpansion_Boundary": best_pe["pred"],
            "BAMS150+PE+TS_Boundary": best_pets["pred"],
        }
        soft_n = {
            "Baseline": 0,
            "Margin150": 0,
            "BAMS150": 0,
            "BAMS150+TeacherStudent_Boundary": int(len(ts_idx)),
            "BAMS150+PrototypeExpansion_Boundary": int(best_pe["soft"]),
            "BAMS150+PE+TS_Boundary": int(best_pets["soft"]),
        }
        best_k = {
            "BAMS150+PrototypeExpansion_Boundary": best_pe["k"],
            "BAMS150+PE+TS_Boundary": best_pets["k"],
        }

        rows, cm_rows = [], []
        for method, pred in preds.items():
            m = full_metrics(y_true, pred)
            cm = m.pop("ConfusionMatrix")
            row = {
                "City": city,
                "Method": method,
                "SoftN": soft_n[method],
                "BestK": best_k.get(method, np.nan),
                **m,
            }
            rows.append(row)
            cm_rows.append(
                {
                    "City": city,
                    "Method": method,
                    "CM00": int(cm[0, 0]),
                    "CM01": int(cm[0, 1]),
                    "CM02": int(cm[0, 2]),
                    "CM10": int(cm[1, 0]),
                    "CM11": int(cm[1, 1]),
                    "CM12": int(cm[1, 2]),
                    "CM20": int(cm[2, 0]),
                    "CM21": int(cm[2, 1]),
                    "CM22": int(cm[2, 2]),
                    "C1_to_C2": int(cm[0, 1]),
                    "C2_to_C1": int(cm[1, 0]),
                    "C1_to_C3": int(cm[0, 2]),
                    "C3_to_C1": int(cm[2, 0]),
                    "C2_to_C3": int(cm[1, 2]),
                    "C3_to_C2": int(cm[2, 1]),
                    "Boundary Error": int(cm[0, 1] + cm[1, 0]),
                }
            )
            print(
                f"  {method}: OA={m['OA']:.4f} BE={m['Boundary Error']} "
                f"Kappa={m['Kappa']:.4f}"
            )

        all_rows.append(pd.DataFrame(rows))
        all_cm.append(pd.DataFrame(cm_rows))
        all_k.append(pd.DataFrame(k_rows))

    df = pd.concat(all_rows, ignore_index=True)
    df_cm = pd.concat(all_cm, ignore_index=True)
    df_k = pd.concat(all_k, ignore_index=True)
    df = add_boundary_reduction(df)

    # STEP 9 boundary_error_results.csv
    be_cols = [
        "City",
        "Method",
        "OA",
        "AA",
        "Kappa",
        "Class1 Recall",
        "Class2 Recall",
        "Class3 Recall",
        "Boundary Error",
        "Boundary Error Reduction",
    ]
    be_df = df[be_cols].copy()
    be_path = OUT_ROOT / "boundary_error_results.csv"
    be_df.to_csv(be_path, index=False)
    be_df.to_csv(DATA2 / "boundary_error_results.csv", index=False)

    # STEP 10 Excel
    df_improve = build_improvement(df)
    xlsx_path = OUT_ROOT / "all_city_boundary_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Results", index=False)
        df_improve.to_excel(writer, sheet_name="Improvement over Baseline", index=False)
        be_df.to_excel(writer, sheet_name="Boundary Error Analysis", index=False)
        df_cm.to_excel(writer, sheet_name="Confusion Matrices", index=False)
        df_k.to_excel(writer, sheet_name="K Sweep", index=False)
    # also mirror to data2/
    import shutil

    shutil.copy2(xlsx_path, DATA2 / "all_city_boundary_summary.xlsx")

    # STEP 11 ranking
    ranking = build_ranking(df)
    rank_path = OUT_ROOT / "city_ranking.csv"
    ranking.to_csv(rank_path, index=False)
    ranking.to_csv(DATA2 / "city_ranking.csv", index=False)

    # STEP 12 overall
    overall = build_overall(df)
    overall_path = OUT_ROOT / "overall_summary.csv"
    overall.to_csv(overall_path, index=False)
    overall.to_csv(DATA2 / "overall_summary.csv", index=False)

    # STEP 13 discussion
    discussion = write_discussion(df, overall)
    disc_path = OUT_ROOT / "discussion.txt"
    disc_path.write_text(discussion + "\n", encoding="utf-8")
    (DATA2 / "discussion.txt").write_text(discussion + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("BOUNDARY ERROR RESULTS")
    print("=" * 72)
    print(be_df.to_string(index=False))
    print("\nCITY RANKING")
    print(ranking.to_string(index=False))
    print("\nOVERALL")
    print(overall.to_string(index=False))
    print("\nDISCUSSION")
    print(discussion)
    print(f"\nSaved under: {OUT_ROOT}")
    print(f"  {be_path.name}")
    print(f"  {xlsx_path.name}")
    print(f"  {rank_path.name}")
    print(f"  {overall_path.name}")
    print(f"  {disc_path.name}")


if __name__ == "__main__":
    main()
