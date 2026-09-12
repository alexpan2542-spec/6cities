#!/usr/bin/env python3
"""
BAMS Prototype Expansion (BPE) for Changsha.

Uses BAMS150 human-labeled samples as prototypes, expands supervision to
cosine nearest neighbors in feature space, filters with the trained teacher,
then trains a weighted student and compares against existing methods.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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

CITY = "Changsha"
CITY_ROOT = ROOT / "data2" / CITY
OUT_DIR = CITY_ROOT / "07_results"
MANUAL_PATH = CITY_ROOT / "04_manual" / f"{CITY}_BAMS150_Manual.csv"
FEAT3_PATH = CITY_ROOT / "06_3x3" / f"{CITY}_3x3_Features.csv"

K_NEIGHBORS = 20
CONF_THRESH = 0.9
EXPANDED_WEIGHT = 0.3
REAL_WEIGHT = 1.0
SPLIT = dict(test_size=0.3, random_state=42)
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256


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
        "OA": float(oa),
        "AA": aa,
        "Kappa": float(kappa),
        "Class3 Recall": float(recall[class3_label]),
        "Class3 Precision": float(precision[class3_label]),
        "Class3 F1": float(f1[class3_label]),
    }


def train_mlp(X, y, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, sample_weight=None, device="cpu"):
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
            print(f"  epoch {epoch:03d} | loss {total_loss / max(total_w, 1e-8):.4f}")
    return model


@torch.no_grad()
def predict_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    logits = model(xt)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.cpu().numpy().astype(np.int64), conf.cpu().numpy().astype(np.float64)


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    # majority; ties broken by highest count then smallest class id
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def class_dist(y: np.ndarray, title: str) -> None:
    y = np.asarray(y, dtype=int)
    counts = Counter(y.tolist())
    total = len(y)
    print(f"\n{title} (n={total})")
    for c in sorted(counts):
        print(f"  class {c} (WC={c + 1}): {counts[c]} ({100.0 * counts[c] / total:.2f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("BAMS Prototype Expansion (BPE) — Changsha")
    print("=" * 60)
    print("device:", device)

    # ------------------------------------------------------------------
    # STEP 1 — Load features, IDs, BAMS manual; extract prototypes
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Load & match BAMS prototypes")
    print("=" * 60)

    feature = np.load(OUT_DIR / "feature.npy")  # (N, D)
    ids = np.load(OUT_DIR / "id.npy").astype(int)  # (N,)
    assert feature.shape[0] == ids.shape[0], "feature/id length mismatch"

    bams_manual = pd.read_csv(MANUAL_PATH)
    bams_manual["Original_ID"] = bams_manual["Original_ID"].astype(int)
    bams_manual["Human_Class"] = pd.to_numeric(bams_manual["Human_Class"], errors="coerce")
    bams_manual = bams_manual[bams_manual["Human_Class"].notna()].copy()
    bams_manual["Human_Class"] = bams_manual["Human_Class"].astype(int)

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    bams_pos, bams_label = [], []
    missing = []
    for _, row in bams_manual.iterrows():
        oid = int(row["Original_ID"])
        if oid not in id_to_pos:
            missing.append(oid)
            continue
        bams_pos.append(id_to_pos[oid])
        bams_label.append(int(row["Human_Class"]) - 1)
    if missing:
        raise ValueError(f"BAMS Original_ID not in id.npy: {missing[:10]}")

    bams_pos = np.asarray(bams_pos, dtype=int)
    bams_label = np.asarray(bams_label, dtype=np.int64)
    bams_feature = feature[bams_pos]

    print(f"feature shape: {feature.shape}")
    print(f"id shape:      {ids.shape}")
    print(f"bams_feature:  {bams_feature.shape}")
    print(f"bams_label:    {bams_label.shape}")
    class_dist(bams_label, "BAMS prototype class distribution")

    # ------------------------------------------------------------------
    # STEP 2 — Cosine nearest neighbors (K=20)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Cosine nearest neighbors (K=20)")
    print("=" * 60)

    # n_neighbors=K+1 so after dropping self we keep K neighbors
    nn_model = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(bams_feature, return_distance=False)  # (150, 21)

    bams_set = set(bams_pos.tolist())
    votes: dict[int, list[int]] = defaultdict(list)
    for proto_i, neighbors in enumerate(neigh_idx):
        proto_label = int(bams_label[proto_i])
        proto_pos = int(bams_pos[proto_i])
        for nidx in neighbors.tolist():
            nidx = int(nidx)
            if nidx == proto_pos:
                continue  # skip self
            votes[nidx].append(proto_label)

    expanded_indices_raw = np.asarray(sorted(votes.keys()), dtype=int)
    print(f"total expanded samples (unique neighbors, excl. self): {len(expanded_indices_raw)}")
    print(f"  of which are other BAMS prototypes: {sum(i in bams_set for i in expanded_indices_raw)}")

    # ------------------------------------------------------------------
    # STEP 3 — Pseudo labels via majority vote
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Pseudo labels (majority vote)")
    print("=" * 60)

    expanded_label_raw = np.asarray(
        [majority_label(votes[int(i)]) for i in expanded_indices_raw], dtype=np.int64
    )
    expanded_feature_raw = feature[expanded_indices_raw]
    print(f"expanded_feature: {expanded_feature_raw.shape}")
    print(f"expanded_label:   {expanded_label_raw.shape}")
    class_dist(expanded_label_raw, "Expanded (before teacher filter) class distribution")

    # ------------------------------------------------------------------
    # Shared split / scaler / labels (same protocol as Teacher–Student)
    # ------------------------------------------------------------------
    feat3 = pd.read_csv(FEAT3_PATH)
    y_all = feat3["Class"].astype(int).to_numpy() - 1
    idx = np.arange(len(feature))
    idx_train, idx_test = train_test_split(
        idx,
        test_size=SPLIT["test_size"],
        random_state=SPLIT["random_state"],
        stratify=y_all,
    )

    scaler = StandardScaler()
    feature_s = scaler.fit_transform(feature).astype(np.float32)
    X_train_s = feature_s[idx_train]
    X_test_s = feature_s[idx_test]
    y_train = y_all[idx_train]
    y_true = y_all[idx_test]

    # Exclude original train + BAMS from expansion (same spirit as TS pool)
    used = np.zeros(len(feature), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True
    keep_pool = ~used[expanded_indices_raw]
    expanded_indices = expanded_indices_raw[keep_pool]
    expanded_label = expanded_label_raw[keep_pool]
    expanded_feature = feature[expanded_indices]
    print(
        f"\nAfter excluding train+BAMS from expansion pool: {len(expanded_indices)} samples"
    )
    class_dist(expanded_label, "Expanded pool (excl. train+BAMS)")

    # ------------------------------------------------------------------
    # STEP 4 — Teacher filtering
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Teacher filtering (conf > 0.9 & pred == pseudo)")
    print("=" * 60)

    teacher = MLP(in_dim=feature.shape[1], n_classes=3).to(device)
    teacher.load_state_dict(torch.load(OUT_DIR / "teacher.pth", map_location=device))
    teacher.eval()

    expanded_s = feature_s[expanded_indices]
    teacher_pred, teacher_conf = predict_mlp(teacher, expanded_s, device=device)

    before_n = len(expanded_label)
    keep = (teacher_conf > CONF_THRESH) & (teacher_pred == expanded_label)
    after_n = int(keep.sum())

    expanded_indices_f = expanded_indices[keep]
    expanded_feature_f = expanded_feature[keep]
    expanded_label_f = expanded_label[keep]
    expanded_s_f = expanded_s[keep]

    print(f"Before filtering: {before_n}")
    print(f"After filtering:  {after_n}")
    class_dist(expanded_label_f, "Expanded (after teacher filter) class distribution")

    # ------------------------------------------------------------------
    # STEP 5 — Student training
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Student training (train + BAMS150 + expanded)")
    print("=" * 60)

    X_bams_s = feature_s[bams_pos]
    X_student = np.concatenate([X_train_s, X_bams_s, expanded_s_f], axis=0)
    y_student = np.concatenate([y_train, bams_label, expanded_label_f], axis=0)
    w_student = np.concatenate(
        [
            np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
            np.full(len(expanded_label_f), EXPANDED_WEIGHT, dtype=np.float32),
        ]
    )
    print(
        f"Student set size: {len(y_student)} "
        f"(train={len(idx_train)}, BAMS={len(bams_label)}, expanded={len(expanded_label_f)})"
    )

    print("Training BPE student...")
    student = train_mlp(
        X_student, y_student, epochs=EPOCHS, sample_weight=w_student, device=device
    )
    torch.save(student.state_dict(), OUT_DIR / "student_bpe.pth")
    y_pred_bpe, _ = predict_mlp(student, X_test_s, device=device)
    np.save(OUT_DIR / "y_pred_bpe.npy", y_pred_bpe)

    # ------------------------------------------------------------------
    # STEP 6–8 — Evaluate, compare, export
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6–8: Evaluate & comparison table")
    print("=" * 60)

    # Prefer saved predictions for prior methods (same split / protocol)
    y_true_saved = np.load(OUT_DIR / "y_true.npy")
    if not np.array_equal(y_true_saved, y_true):
        print("WARNING: y_true.npy differs from recomputed split; using recomputed y_true.")
    else:
        y_true = y_true_saved

    pred_files = {
        "Baseline": OUT_DIR / "y_pred_baseline.npy",
        "Margin150": OUT_DIR / "y_pred_margin.npy",
        "BAMS150": OUT_DIR / "y_pred_bams.npy",
        "BAMS150+TeacherStudent": OUT_DIR / "y_pred_student.npy",
    }
    methods = {}
    for name, path in pred_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run generate_changsha_comparison.py first."
            )
        methods[name] = np.load(path)

    methods["BAMS150+PrototypeExpansion"] = y_pred_bpe

    rows = []
    for name, pred in methods.items():
        m = compute_metrics(y_true, pred)
        m["Method"] = name
        rows.append(m)

    cols = ["Method", "OA", "AA", "Kappa", "Class3 Recall", "Class3 Precision", "Class3 F1"]
    df = pd.DataFrame(rows)[cols]

    xlsx_path = OUT_DIR / "prototype_expansion_results.xlsx"
    csv_path = OUT_DIR / "prototype_expansion_results.csv"
    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)

    print("\n" + df.to_string(index=False))
    print(f"\nSaved: {xlsx_path}")
    print(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # STEP 9 — Gains
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: Gains")
    print("=" * 60)

    m_bams = df.loc[df["Method"] == "BAMS150"].iloc[0]
    m_ts = df.loc[df["Method"] == "BAMS150+TeacherStudent"].iloc[0]
    m_bpe = df.loc[df["Method"] == "BAMS150+PrototypeExpansion"].iloc[0]

    oa_vs_bams = m_bpe["OA"] - m_bams["OA"]
    oa_vs_ts = m_bpe["OA"] - m_ts["OA"]
    c3r_vs_bams = m_bpe["Class3 Recall"] - m_bams["Class3 Recall"]
    c3r_vs_ts = m_bpe["Class3 Recall"] - m_ts["Class3 Recall"]
    c3f_vs_bams = m_bpe["Class3 F1"] - m_bams["Class3 F1"]
    c3f_vs_ts = m_bpe["Class3 F1"] - m_ts["Class3 F1"]

    print(f"OA gain vs BAMS150:          {oa_vs_bams:+.4f}")
    print(f"OA gain vs TeacherStudent:   {oa_vs_ts:+.4f}")
    print(f"Class3 Recall gain vs BAMS:  {c3r_vs_bams:+.4f}")
    print(f"Class3 Recall gain vs TS:    {c3r_vs_ts:+.4f}")
    print(f"Class3 F1 gain vs BAMS:      {c3f_vs_bams:+.4f}")
    print(f"Class3 F1 gain vs TS:        {c3f_vs_ts:+.4f}")

    # ------------------------------------------------------------------
    # STEP 10 — Publication-ready summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10: Summary paragraph")
    print("=" * 60)

    oa_improve_bams = "improves" if oa_vs_bams > 0 else ("matches" if oa_vs_bams == 0 else "does not improve")
    oa_improve_ts = "improves upon" if oa_vs_ts > 0 else ("matches" if oa_vs_ts == 0 else "does not surpass")
    c3_improve = (
        "also strengthens Class-3 detection"
        if (c3r_vs_bams > 0 or c3f_vs_bams > 0)
        else "does not clearly strengthen Class-3 detection relative to BAMS150"
    )

    summary = (
        f"BAMS Prototype Expansion (BPE) is introduced to amplify scarce human "
        f"corrections by propagating prototype labels to nearby samples in feature "
        f"space, thereby enlarging the supervised signal without additional manual "
        f"annotation. BAMS-selected samples are treated as prototypes because they "
        f"combine high model uncertainty (margin) with strong local spatial "
        f"heterogeneity (boundary score), and their Human_Class labels provide "
        f"trusted anchors at decision-boundary regions where WorldCover errors are "
        f"concentrated. On Changsha, BPE achieves OA={m_bpe['OA']:.4f}, which "
        f"{oa_improve_bams} BAMS150 (OA={m_bams['OA']:.4f}, Δ={oa_vs_bams:+.4f}) and "
        f"{oa_improve_ts} BAMS150+TeacherStudent (OA={m_ts['OA']:.4f}, Δ={oa_vs_ts:+.4f}). "
        f"For the minority/target Class 3, BPE reaches Recall={m_bpe['Class3 Recall']:.4f} "
        f"and F1={m_bpe['Class3 F1']:.4f} (vs BAMS150 Recall={m_bams['Class3 Recall']:.4f}, "
        f"F1={m_bams['Class3 F1']:.4f}; ΔRecall={c3r_vs_bams:+.4f}, ΔF1={c3f_vs_bams:+.4f}), "
        f"indicating that prototype expansion {c3_improve}. Overall, cosine-neighborhood "
        f"label propagation from BAMS prototypes, followed by teacher confidence "
        f"filtering (conf>{CONF_THRESH} and agreement with the assigned pseudo label) "
        f"and weighted student training (real={REAL_WEIGHT}, expanded={EXPANDED_WEIGHT}), "
        f"offers a practical route to densify boundary-aware supervision from a compact "
        f"set of 150 human labels."
    )
    print("\n" + summary)

    # Save summary text alongside tables
    (OUT_DIR / "prototype_expansion_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved summary: {OUT_DIR / 'prototype_expansion_summary.txt'}")


if __name__ == "__main__":
    main()
