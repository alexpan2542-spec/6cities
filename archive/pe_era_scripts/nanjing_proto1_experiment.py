#!/usr/bin/env python3
"""
Nanjing revised human-protocol experiment (point OA only)
---------------------------------------------------------
Force Human_Class -> 1 for Scene in:
  re, road, urban road   (road edge)
  fe, ub                 (building-adjacent / urban scenes labeled as 2)

Methods:
  Baseline
  BAMS150            (original human)
  BAMS150_proto1     (revised human)
  PE_Boundary_proto1 (PE from revised human)

Eval: OA / Boundary Error vs original WorldCover Class at the sample point.
No discrete-sample neighborhood OA (not true 3x3 pixels).

Outputs -> data2/nanjing_proto1_eval/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_pkgs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA2 = ROOT / "data2"
OUT_DIR = DATA2 / "nanjing_proto1_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY = "Nanjing"
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
SPLIT = dict(test_size=0.3, random_state=42)
SEEDS = [0, 1, 2, 3, 4]
K_LIST = [5, 10, 20, 30]
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256
PROTO1_SCENES = {"re", "road", "urban road", "fe", "ub"}
METHODS = ["Baseline", "BAMS150", "BAMS150_proto1", "PE_Boundary_proto1"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
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
    if "Scene" not in manual.columns:
        manual["Scene"] = ""
    manual["Scene"] = manual["Scene"].astype(str)
    return manual


def apply_proto1(manual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = manual.copy()
    scene = out["Scene"].str.lower().str.strip()
    mask = scene.isin(PROTO1_SCENES)
    changed = out.loc[mask].copy()
    changed["Human_Class_old"] = changed["Human_Class"]
    out.loc[mask, "Human_Class"] = 1
    changed["Human_Class_new"] = 1
    return out, changed


def train_mlp(X, y, sample_weight=None, device="cpu", seed: int = 0, epochs: int = EPOCHS):
    set_all_seeds(seed)
    if sample_weight is None:
        sample_weight = np.ones(len(y), dtype=np.float32)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        WeightedDataset(X, y, sample_weight),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )
    model = MLP(X.shape[1], 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = torch.softmax(model(xt), dim=1).cpu().numpy()
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    conf = probs[np.arange(len(probs)), top1]
    margin = conf - probs[np.arange(len(probs)), top2]
    return top1.astype(np.int64), conf, margin


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


def full_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.nan_to_num(np.diag(cm) / cm.sum(axis=1), nan=0.0)
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(per.mean()),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Boundary Error": int(cm[0, 1] + cm[1, 0]),
        "C1_to_C2": int(cm[0, 1]),
        "C2_to_C1": int(cm[1, 0]),
    }


def load_nanjing() -> dict:
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    manual = get_labeled_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    manual_p1, changed = apply_proto1(manual)

    manual_p1.to_csv(OUT_DIR / "Nanjing_BAMS150_Manual_proto1.csv", index=False)
    changed.to_csv(OUT_DIR / "Nanjing_proto1_changes.csv", index=False)

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_all
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    def labels_from_manual(man: pd.DataFrame):
        pos = np.asarray([id_to_pos[int(r.Original_ID)] for _, r in man.iterrows()], dtype=int)
        lab = np.asarray([int(r.Human_Class) - 1 for _, r in man.iterrows()], dtype=np.int64)
        y_full = y_all.copy()
        y_full[pos] = lab
        return pos, lab, y_full

    bams_pos, bams_label, y_bams = labels_from_manual(manual)
    bams_pos_p1, bams_label_p1, y_bams_p1 = labels_from_manual(manual_p1)

    scene = manual["Scene"].str.lower().str.strip()
    by_scene = (
        changed.assign(Scene_norm=changed["Scene"].str.lower().str.strip())
        .groupby("Scene_norm")
        .size()
        .to_dict()
    )
    n_changed = int((manual["Human_Class"].to_numpy() != manual_p1["Human_Class"].to_numpy()).sum())

    print(
        f"proto1 scenes={sorted(PROTO1_SCENES)}; "
        f"n_changed={n_changed}; by_scene={by_scene}; "
        f"orig_scene_counts="
        f"{scene[scene.isin(PROTO1_SCENES)].value_counts().to_dict()}"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "bams_pos": bams_pos,
        "bams_label": bams_label,
        "y_bams": y_bams,
        "bams_pos_p1": bams_pos_p1,
        "bams_label_p1": bams_label_p1,
        "y_bams_p1": y_bams_p1,
        "n_changed": n_changed,
        "by_scene": by_scene,
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def run(device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    data = load_nanjing()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    y_bams = data["y_bams"]
    bams_pos_p1 = data["bams_pos_p1"]
    bams_label_p1 = data["bams_label_p1"]
    y_bams_p1 = data["y_bams_p1"]

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos_p1] = True

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
        X_bams_s = X_all_s[bams_pos]
        X_bams_p1_s = X_all_s[bams_pos_p1]
        y_true = y_all[idx_test]

        rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_base.fit(X_train, y_all[idx_train])
        m_base = full_metrics(y_true, rf_base.predict(X_test))

        rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_bams.fit(X_train, y_bams[idx_train])
        m_bams = full_metrics(y_true, rf_bams.predict(X_test))

        rf_p1 = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_p1.fit(X_train, y_bams_p1[idx_train])
        m_p1 = full_metrics(y_true, rf_p1.predict(X_test))

        teacher = train_mlp(
            np.concatenate([X_train_s, X_bams_p1_s], axis=0),
            np.concatenate([y_all[idx_train], bams_label_p1], axis=0),
            device=device,
            seed=seed,
            epochs=epochs,
        )
        _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

        pe_cands = []
        for k in K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos_p1, bams_label_p1, k)
            keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            X_pe = np.concatenate([X_train_s, X_bams_p1_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_all[idx_train], bams_label_p1, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label_p1), REAL_WEIGHT, np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(
                X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
            )
            pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            m_pe = full_metrics(y_true, pred)
            pe_cands.append({"k": k, "n_exp": int(len(exp_idx)), "m": m_pe})
            print(
                f"  [seed={seed}] PE_proto1 K={k:<2d} exp={len(exp_idx):<3d} "
                f"OA={m_pe['OA']:.4f} BE={m_pe['Boundary Error']}"
            )
        best = pick_best(pe_cands)

        pack = [
            ("Baseline", m_base, {}),
            ("BAMS150", m_bams, {}),
            ("BAMS150_proto1", m_p1, {}),
            (
                "PE_Boundary_proto1",
                best["m"],
                {"BestK": best["k"], "n_expanded": best["n_exp"]},
            ),
        ]
        for method, m, extra in pack:
            rows.append(
                {
                    "City": CITY,
                    "Method": method,
                    "Seed": seed,
                    **m,
                    "BestK": extra.get("BestK", np.nan),
                    "n_expanded": extra.get("n_expanded", np.nan),
                }
            )
            extra_s = ""
            if method == "PE_Boundary_proto1":
                extra_s = f" K={extra['BestK']} exp={extra['n_expanded']}"
            print(f"  [seed={seed}] {method:20s} OA={m['OA']:.4f} BE={m['Boundary Error']}{extra_s}")

    return pd.DataFrame(rows), data


def summarize(df: pd.DataFrame, n_changed: int, by_scene: dict) -> str:
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    lines = [
        "# Nanjing proto1 human protocol (re/road/fe/ub → Class1)\n",
        f"- Forced Scene→Class1 changes: {n_changed}\n",
        f"- By scene: {by_scene}\n",
        "- Eval: pointwise vs original WorldCover Class only\n",
        "\n## Mean over seeds\n",
        overall.to_string(index=False),
        "\n\n## Reading guide\n",
        "- BAMS150_proto1 / PE_Boundary_proto1 use revised human labels\n",
        "- Compare to Baseline and original BAMS150 under the same point OA\n",
    ]
    return "\n".join(lines), overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print("device:", device, "seeds:", seeds, "epochs:", epochs)

    df, data = run(device=device, seeds=seeds, epochs=epochs)
    md, overall = summarize(df, data["n_changed"], data["by_scene"])
    df.to_csv(OUT_DIR / "seed_results.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "proto1_scenes": sorted(PROTO1_SCENES),
                "n_changed": data["n_changed"],
                "by_scene": data["by_scene"],
                "methods": METHODS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
