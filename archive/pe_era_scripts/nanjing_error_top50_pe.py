#!/usr/bin/env python3
"""
Nanjing ErrorTop150 partial PE test (PointID 1..N via --max-point-id).

Methods vs Baseline_RF:
  Baseline_RF / Baseline_MLP
  PE_BAMS_high
  PE_Err_Human_all / PE_Err_Human_high
  PE_Err_WC
  PE_Err_agreeHigh

Default out: data2/nanjing_error_top50_pe/ (override with --out-dir)
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
OUT_DIR = DATA2 / "nanjing_error_top50_pe"
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

METHODS = [
    "Baseline_RF",
    "Baseline_MLP",
    "PE_BAMS_high",
    "PE_Err_Human_all",
    "PE_Err_Human_high",
    "PE_Err_WC",
    "PE_Err_agreeHigh",
]


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


def norm_conf(x):
    if pd.isna(x) or str(x).strip() == "":
        return None
    s = str(x).strip().lower()
    if s in ("h", "high"):
        return "high"
    if s in ("m", "med", "medium"):
        return "med"
    if s in ("l", "low"):
        return "low"
    return None


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
        "Rec_C2": float(per[1]),
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def load_packs(max_point_id: int = 50):
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
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

    # BAMS high
    bams = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    bams["Human_Class"] = pd.to_numeric(bams["Human_Class"], errors="coerce")
    bams["Conf"] = bams["Confidence"].map(norm_conf)
    bams_h = bams[bams["Conf"] == "high"].copy()
    pos_bams_h = np.asarray([id_to_pos[int(x)] for x in bams_h.Original_ID], dtype=int)
    lab_bams_h = (bams_h["Human_Class"].astype(int) - 1).to_numpy()

    # ErrorTop 1..max_point_id labeled
    err = pd.read_csv(city_root / "04_manual" / "Nanjing_ErrorTop150.csv")
    err["Human_Class"] = pd.to_numeric(err["Human_Class"], errors="coerce")
    err["Conf"] = err["Confidence"].map(norm_conf)
    err_n = err[(err["PointID"] <= max_point_id) & err["Human_Class"].notna()].copy()
    err_n["Human_Class"] = err_n["Human_Class"].astype(int)
    err_n["WC_Class"] = err_n["WC_Class"].astype(int)

    pos = np.asarray([id_to_pos[int(x)] for x in err_n.Original_ID], dtype=int)
    lab_h = (err_n["Human_Class"].to_numpy() - 1).astype(np.int64)
    lab_w = (err_n["WC_Class"].to_numpy() - 1).astype(np.int64)
    conf = err_n["Conf"].tolist()
    agree = err_n["Human_Class"].to_numpy() == err_n["WC_Class"].to_numpy()
    high = np.asarray([c == "high" for c in conf], dtype=bool)
    agree_high = high & agree

    meta = {
        "max_point_id": int(max_point_id),
        "n_err": int(len(err_n)),
        "n_high": int(high.sum()),
        "n_agree": int(agree.sum()),
        "n_agree_high": int(agree_high.sum()),
        "agree_pct": float(agree.mean()) if len(err_n) else 0.0,
        "conf_counts": {str(k): int(v) for k, v in Counter(conf).items()},
        "error_type_counts": err_n["error_type"].value_counts().to_dict(),
        "n_bams_high": int(len(pos_bams_h)),
    }
    print("ErrorTop meta:", meta)

    packs = {
        "BAMS_high": {"pos": pos_bams_h, "lab": lab_bams_h},
        "Err_Human_all": {"pos": pos, "lab": lab_h},
        "Err_Human_high": {"pos": pos[high], "lab": lab_h[high]},
        "Err_WC": {"pos": pos, "lab": lab_w},
        "Err_agreeHigh": {"pos": pos[agree_high], "lab": lab_h[agree_high]},
    }
    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "packs": packs,
        "meta": meta,
    }


def run_pe(data, pos, lab, tag, device, seed, epochs):
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    if len(pos) == 0:
        return {"k": 0, "n_exp": 0, "m": full_metrics(y_all[idx_test], y_all[idx_test])}

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pos] = True
    X_train_s = X_all_s[idx_train]
    X_seed_s = X_all_s[pos]
    X_test_s = X_all_s[idx_test]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], lab], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, pos, lab, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], lab, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(lab), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(
            X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
        )
        pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        cands.append({"k": k, "n_exp": int(len(exp_idx)), "m": m})
        print(
            f"  [seed={seed}] {tag:22s} K={k:<2d} n_seed={len(pos):<3d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']} RecC2={m['Rec_C2']:.3f}"
        )
    return pick_best(cands)


def run(device, seeds, epochs, max_point_id: int = 50):
    data = load_packs(max_point_id=max_point_id)
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    packs = data["packs"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        y_true = y_all[idx_test]
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_all[idx_train], y_all[idx_train])
        m = full_metrics(y_true, rf.predict(X_all[idx_test]))
        rows.append({"Method": "Baseline_RF", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_RF            OA={m['OA']:.4f} BE={m['Boundary Error']}")

        mlp = train_mlp(X_all_s[idx_train], y_all[idx_train], device=device, seed=seed, epochs=epochs)
        pred, _, _ = predict_proba_mlp(mlp, X_all_s[idx_test], device=device)
        m = full_metrics(y_true, pred)
        rows.append({"Method": "Baseline_MLP", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_MLP           OA={m['OA']:.4f} BE={m['Boundary Error']}")

        for name, key in [
            ("PE_BAMS_high", "BAMS_high"),
            ("PE_Err_Human_all", "Err_Human_all"),
            ("PE_Err_Human_high", "Err_Human_high"),
            ("PE_Err_WC", "Err_WC"),
            ("PE_Err_agreeHigh", "Err_agreeHigh"),
        ]:
            pack = packs[key]
            best = run_pe(data, pack["pos"], pack["lab"], name, device, seed, epochs)
            rows.append(
                {
                    "Method": name,
                    "Seed": seed,
                    **best["m"],
                    "BestK": best["k"],
                    "n_expanded": best["n_exp"],
                    "n_seed": int(len(pack["pos"])),
                }
            )
            print(
                f"  [seed={seed}] {name:22s} OA={best['m']['OA']:.4f} "
                f"BE={best['m']['Boundary Error']} K={best['k']} exp={best['n_exp']}"
            )

    return pd.DataFrame(rows), data["meta"]


def summarize(df, meta):
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Mean_Rec_C2=("Rec_C2", "mean"),
            Mean_n_seed=("n_seed", "mean"),
            Mean_n_exp=("n_expanded", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    base = overall.loc[overall["Method"] == "Baseline_RF", "Mean_OA"].iloc[0]
    overall["dOA_vs_RF_pp"] = (overall["Mean_OA"] - base) * 100
    base_mlp = overall.loc[overall["Method"] == "Baseline_MLP", "Mean_OA"].iloc[0]
    overall["dOA_vs_MLP_pp"] = (overall["Mean_OA"] - base_mlp) * 100

    md = (
        f"# Nanjing ErrorTop150 (PointID 1–{meta['max_point_id']}) PE test\n\n"
        f"- Labeled n={meta['n_err']}; high={meta['n_high']}; "
        f"agree_WC={meta['n_agree']} ({meta['agree_pct']:.1%}); "
        f"agree_high={meta['n_agree_high']}\n"
        f"- conf={meta['conf_counts']}; error_types={meta['error_type_counts']}\n\n"
        "## Mean over seeds\n\n"
        + overall.to_string(index=False)
        + "\n"
    )
    return md, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--max-point-id", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print(
        "device:", device, "seeds:", seeds, "epochs:", epochs,
        "max_point_id:", args.max_point_id, "out:", out,
    )

    df, meta = run(device, seeds, epochs, max_point_id=args.max_point_id)
    md, overall = summarize(df, meta)
    df.to_csv(out / "seed_results.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "VERDICT.md").write_text(md, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps({"meta": meta, "seeds": seeds, "epochs": epochs}, indent=2),
        encoding="utf-8",
    )
    print("\n" + md)
    print("Wrote", out)


if __name__ == "__main__":
    main()
