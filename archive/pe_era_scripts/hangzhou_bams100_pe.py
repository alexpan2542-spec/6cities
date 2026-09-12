#!/usr/bin/env python3
"""
Hangzhou BAMS PointID 1..N PE smoke / partial test (default N=100).

Methods vs Baseline_RF / Baseline_MLP:
  Direct_all / Direct_high / Direct_hm
  PE_all / PE_high / PE_hm

Eval: pointwise vs original WorldCover Class.
Outputs -> data2/hangzhou_bams100_pe/
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
CITY = "Hangzhou"
OUT_DIR = DATA2 / "hangzhou_bams100_pe"

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
    "Direct_all",
    "Direct_high",
    "Direct_hm",
    "PE_all",
    "PE_high",
    "PE_hm",
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


def get_manual(path: Path, max_point_id: int) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["PointID"] = pd.to_numeric(m["PointID"], errors="coerce")
    m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
    m = m[m["Human_Class"].notna()].copy()
    m = m[m["PointID"] <= max_point_id].copy()
    m["Human_Class"] = m["Human_Class"].astype(int)
    m["Original_ID"] = m["Original_ID"].astype(int)
    m["PointID"] = m["PointID"].astype(int)
    m["Conf"] = m["Confidence"].map(norm_conf) if "Confidence" in m.columns else "high"
    return m.reset_index(drop=True)


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


def pack_manual(man: pd.DataFrame, y_all: np.ndarray, id_to_pos: dict, conf_keep: set[str] | None):
    pos_all = np.asarray([id_to_pos[int(r.Original_ID)] for _, r in man.iterrows()], dtype=int)
    lab_all = np.asarray([int(r.Human_Class) - 1 for _, r in man.iterrows()], dtype=np.int64)
    conf = man["Conf"].tolist()
    if conf_keep is None:
        use = np.ones(len(man), dtype=bool)
    else:
        use = np.asarray([(c in conf_keep) for c in conf], dtype=bool)
    pos = pos_all[use]
    lab = lab_all[use]
    y_full = y_all.copy()
    y_full[pos] = lab
    return {
        "pos_seed": pos,
        "lab_seed": lab,
        "y_full": y_full,
        "n_seed": int(use.sum()),
    }


def run_pe(X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, tag: str):
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


def main() -> None:
    global OUT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--max-point-id", type=int, default=100)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    max_pid = args.max_point_id
    OUT_DIR = (
        Path(args.out_dir)
        if args.out_dir is not None
        else DATA2 / f"hangzhou_bams{max_pid}_pe"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs

    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = get_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv", max_pid)
    if len(man) == 0:
        raise SystemExit(f"No labeled rows with PointID<= {max_pid}")

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

    packs = {
        "all": pack_manual(man, y_all, id_to_pos, None),
        "high": pack_manual(man, y_all, id_to_pos, {"high"}),
        "hm": pack_manual(man, y_all, id_to_pos, {"high", "med"}),
    }
    conf_counts = man["Conf"].value_counts(dropna=False).to_dict()
    # WC agree on labeled
    wc_by_id = {int(r.Original_ID): int(r.Class) for _, r in feat3.iterrows()}
    agree = float(
        np.mean([int(r.Human_Class) == wc_by_id[int(r.Original_ID)] for _, r in man.iterrows()])
    )

    print(
        f"Hangzhou BAMS1-{max_pid}: n={len(man)} conf={conf_counts} "
        f"agree_WC={agree:.3f} seeds high={packs['high']['n_seed']} "
        f"hm={packs['hm']['n_seed']} all={packs['all']['n_seed']}"
    )
    print("device:", device, "seeds:", seeds, "epochs:", epochs, "out:", OUT_DIR)

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
        y_true = y_all[idx_test]

        # Baseline RF
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_all[idx_train])
        m = full_metrics(y_true, rf.predict(X_test))
        rows.append({"City": CITY, "Method": "Baseline_RF", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_RF          OA={m['OA']:.4f} BE={m['Boundary Error']}")

        # Baseline MLP
        mlp = train_mlp(X_train_s, y_all[idx_train], device=device, seed=seed, epochs=epochs)
        pred, _, _ = predict_proba_mlp(mlp, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        rows.append({"City": CITY, "Method": "Baseline_MLP", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_MLP         OA={m['OA']:.4f} BE={m['Boundary Error']}")

        # Direct RF inject
        for name, key in [
            ("Direct_all", "all"),
            ("Direct_high", "high"),
            ("Direct_hm", "hm"),
        ]:
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_train, packs[key]["y_full"][idx_train])
            m = full_metrics(y_true, rf.predict(X_test))
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    **m,
                    "n_seed": packs[key]["n_seed"],
                }
            )
            print(f"  [seed={seed}] {name:20s} OA={m['OA']:.4f} BE={m['Boundary Error']}")

        # PE
        for name, key in [("PE_all", "all"), ("PE_high", "high"), ("PE_hm", "hm")]:
            best = run_pe(
                X_all,
                X_all_s,
                y_all,
                idx_train,
                idx_test,
                packs[key],
                device,
                seed,
                epochs,
                name,
            )
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    **best["m"],
                    "BestK": best["k"],
                    "n_expanded": best["n_exp"],
                    "n_seed": packs[key]["n_seed"],
                }
            )
            print(
                f"  [seed={seed}] {name:20s} OA={best['m']['OA']:.4f} "
                f"BE={best['m']['Boundary Error']} K={best['k']} exp={best['n_exp']}"
            )

    df = pd.DataFrame(rows)
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_n_seed=("n_seed", "mean"),
            Mean_n_exp=("n_expanded", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    base_rf = overall.loc[overall["Method"] == "Baseline_RF", "Mean_OA"].iloc[0]
    base_mlp = overall.loc[overall["Method"] == "Baseline_MLP", "Mean_OA"].iloc[0]
    overall["dOA_vs_RF_pp"] = (overall["Mean_OA"] - base_rf) * 100
    overall["dOA_vs_MLP_pp"] = (overall["Mean_OA"] - base_mlp) * 100

    md = "\n".join(
        [
            f"# Hangzhou BAMS1-{max_pid} PE test\n",
            f"- Labeled: {len(man)} (PointID 1..{max_pid})\n",
            f"- Confidence: { {str(k): int(v) for k, v in conf_counts.items()} }\n",
            f"- Human≈WC: {agree:.3f}\n",
            "\n## Mean over seeds\n",
            overall.to_string(index=False),
            "\n\n## Reading\n",
            "- Direct_*: RF with human overwrite on train (no PE)\n",
            "- PE_high: only Confidence=high as hard PE seeds\n",
            "- dOA vs RF / MLP in percentage points\n",
        ]
    )

    df.to_csv(OUT_DIR / "seed_results.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "max_point_id": max_pid,
                "n_labeled": len(man),
                "conf_counts": {str(k): int(v) for k, v in conf_counts.items()},
                "agree_wc": agree,
                "seeds": seeds,
                "epochs": epochs,
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
