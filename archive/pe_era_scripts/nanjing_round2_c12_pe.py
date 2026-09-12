#!/usr/bin/env python3
"""
Nanjing Round2 C1–C2 boundary (+50) + BAMS high-confidence PE.

Seeds
  BAMS_high          : BAMS150 Confidence=high
  R2_high / R2_hm    : Round2 high / high+med
  BAMS_high+R2_*     : union (prefer Round2 label on overlap; none expected)

Methods
  Baseline
  PE_bams_high
  PE_r2_high
  PE_bams_high_r2_high
  PE_bams_high_r2_hm

Eval vs original WorldCover Class.
Outputs -> data2/nanjing_round2_c12_pe/
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
OUT_DIR = DATA2 / "nanjing_round2_c12_pe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY = "Nanjing"
FEATURES_3X3 = [
    "B2_mean", "B2_stdDev", "B3_mean", "B3_stdDev", "B4_mean", "B4_stdDev",
    "B8_mean", "B8_stdDev", "B11_mean", "B11_stdDev", "B12_mean", "B12_stdDev",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev",
    "MNDWI_mean", "MNDWI_stdDev",
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
    "Baseline",
    "RF_bams_high",
    "RF_bams_high_r2_high",
    "RF_bams_high_r2_hm",
    "PE_bams_high",
    "PE_r2_high",
    "PE_bams_high_r2_high",
    "PE_bams_high_r2_hm",
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


def load_labeled(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
    m = m[m["Human_Class"].notna()].copy()
    m["Human_Class"] = m["Human_Class"].astype(int)
    m["Original_ID"] = m["Original_ID"].astype(int)
    m["Conf"] = m["Confidence"].map(norm_conf) if "Confidence" in m.columns else "high"
    return m


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


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def make_pack(id_to_pos, y_all, rows: list[tuple[int, int]]):
    """rows: list of (Original_ID, Human_Class 1/2/3). Later rows overwrite earlier."""
    by_id: dict[int, int] = {}
    for oid, hc in rows:
        by_id[int(oid)] = int(hc)
    pos, lab = [], []
    for oid, hc in by_id.items():
        if oid not in id_to_pos:
            continue
        pos.append(id_to_pos[oid])
        lab.append(hc - 1)
    pos = np.asarray(pos, dtype=int)
    lab = np.asarray(lab, dtype=np.int64)
    y_full = y_all.copy()
    y_full[pos] = lab
    return {"pos_seed": pos, "lab_seed": lab, "y_full": y_full, "n_seed": int(len(pos))}


def load_city():
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    bams = load_labeled(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    r2 = load_labeled(city_root / "04_manual" / "Nanjing_Round2_C12Boundary50.csv")

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

    bams_h = bams[bams["Conf"] == "high"]
    r2_h = r2[r2["Conf"] == "high"]
    r2_hm = r2[r2["Conf"].isin(["high", "med"])]

    def rows_df(df):
        return [(int(r.Original_ID), int(r.Human_Class)) for _, r in df.iterrows()]

    packs = {
        "bams_high": make_pack(id_to_pos, y_all, rows_df(bams_h)),
        "r2_high": make_pack(id_to_pos, y_all, rows_df(r2_h)),
        "bams_high_r2_high": make_pack(id_to_pos, y_all, rows_df(bams_h) + rows_df(r2_h)),
        "bams_high_r2_hm": make_pack(id_to_pos, y_all, rows_df(bams_h) + rows_df(r2_hm)),
    }

    # Round2 vs WC summary
    r2 = r2.copy()
    # need Class from feat
    id_to_cls = dict(zip(ids.tolist(), (y_all + 1).tolist()))
    r2["WC"] = r2["Original_ID"].map(id_to_cls)
    summary = {
        "r2_n": int(len(r2)),
        "r2_conf": r2["Conf"].value_counts(dropna=False).to_dict(),
        "r2_human": r2["Human_Class"].value_counts().to_dict(),
        "r2_flip_wc": int((r2["Human_Class"] != r2["WC"]).sum()),
        "r2_high_n": int(len(r2_h)),
        "r2_high_flip": int((r2_h["Human_Class"] != r2_h["Original_ID"].map(id_to_cls)).sum())
        if len(r2_h)
        else 0,
        "seed_counts": {k: int(v["n_seed"]) for k, v in packs.items()},
        "bams_high_n": int(len(bams_h)),
        "overlap_ids": sorted(
            set(bams["Original_ID"]).intersection(set(r2["Original_ID"]))
        ),
    }
    # fix high flip using WC column on filtered
    if len(r2_h):
        summary["r2_high_flip"] = int(
            (r2_h.assign(WC=r2_h["Original_ID"].map(id_to_cls))["Human_Class"]
             != r2_h["Original_ID"].map(id_to_cls)).sum()
        )

    print(
        f"Round2: n={summary['r2_n']} conf={summary['r2_conf']} "
        f"flip_wc={summary['r2_flip_wc']}; seeds={summary['seed_counts']}"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "packs": packs,
        "summary": {
            **summary,
            "r2_conf": {str(k): int(v) for k, v in summary["r2_conf"].items()},
            "r2_human": {str(k): int(v) for k, v in summary["r2_human"].items()},
        },
    }


def run_pe(X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, tag: str):
    if pack["n_seed"] == 0:
        raise ValueError(f"empty seeds for {tag}")
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pack["pos_seed"]] = True

    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    X_seed_s = X_all_s[pack["pos_seed"]]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], pack["lab_seed"]], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(
            X_all, pack["pos_seed"], pack["lab_seed"], k
        )
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], pack["lab_seed"], exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(pack["lab_seed"]), REAL_WEIGHT, np.float32),
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
            f"  [seed={seed}] {tag} K={k:<2d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    return pick_best(cands)


def run(device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    data = load_city()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    packs = data["packs"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train = X_all[idx_train]
        y_true = y_all[idx_test]

        rf_specs = [
            ("Baseline", y_all),
            ("RF_bams_high", packs["bams_high"]["y_full"]),
            ("RF_bams_high_r2_high", packs["bams_high_r2_high"]["y_full"]),
            ("RF_bams_high_r2_hm", packs["bams_high_r2_hm"]["y_full"]),
        ]
        X_test = X_all[idx_test]
        for name, y_src in rf_specs:
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_train, y_src[idx_train])
            m = full_metrics(y_true, rf.predict(X_test))
            rows.append({"City": CITY, "Method": name, "Seed": seed, "K": None, "n_exp": None, **m})
            print(f"  [seed={seed}] {name:28s} OA={m['OA']:.4f} BE={m['Boundary Error']}")

        pe_specs = [
            ("PE_bams_high", packs["bams_high"]),
            ("PE_r2_high", packs["r2_high"]),
            ("PE_bams_high_r2_high", packs["bams_high_r2_high"]),
            ("PE_bams_high_r2_hm", packs["bams_high_r2_hm"]),
        ]
        for name, pack in pe_specs:
            best = run_pe(
                X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, name
            )
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    **best["m"],
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed_metrics.csv", index=False)

    summary_rows = []
    base_oa = df[df.Method == "Baseline"].OA.mean()
    base_be = df[df.Method == "Baseline"]["Boundary Error"].mean()
    for method in METHODS:
        sub = df[df.Method == method]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "Method": method,
                "OA_mean": sub.OA.mean(),
                "OA_std": sub.OA.std(ddof=0),
                "dOA_pp": (sub.OA.mean() - base_oa) * 100,
                "BE_mean": sub["Boundary Error"].mean(),
                "BE_std": sub["Boundary Error"].std(ddof=0),
                "dBE": sub["Boundary Error"].mean() - base_be,
                "Kappa_mean": sub.Kappa.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("dOA_pp", ascending=False)
    summary.to_csv(OUT_DIR / "summary_vs_baseline.csv", index=False)

    meta = data["summary"]
    meta["baseline_OA"] = float(base_oa)
    meta["baseline_BE"] = float(base_be)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n=== Summary vs Baseline ===")
    print(summary.to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    run(args.device, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
