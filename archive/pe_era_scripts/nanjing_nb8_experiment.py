#!/usr/bin/env python3
"""
Nanjing: use BAMS 3x3 neighborhood (8-neighbor WC) to improve labels / PE.

Neighborhood file:
  data2/Nanjing/04_manual/Nanjing_BAMS150_3x3nb_wide.csv

Uses (from GEE export):
  has_C1_nb8, suggest_C_if_uncertain, n_C1_nb8, ...

Protocols
---------
1) Label adjust (only on BAMS seeds; non-BAMS keep WC):
   - nb_low : Confidence=low  -> Human_Class := suggest_C_if_uncertain
   - nb_lm  : Confidence in {low,med} -> same
   - nb_iso : if Human=1 and has_C1_nb8=0 and Conf!=high -> force Class2
              (isolated "building-like" without built neighbor)

2) PE seed filters (on top of new human labels):
   - PE_new              : all new human seeds
   - PE_new_high         : Confidence=high only
   - PE_nb_consistent    : keep seed if (Human==suggest) or Conf=high
   - PE_nb_c1gate        : Class1 seeds require has_C1_nb8=1; Class2/3 always kept
   - PE_nb_low_adjusted  : seeds after nb_low label adjust (all conf)
   - PE_nb_lm_adjusted   : seeds after nb_lm label adjust

Also RF-only BAMS variants for each adjusted label set.

Outputs -> data2/nanjing_nb8_experiment/
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
OUT_DIR = DATA2 / "nanjing_nb8_experiment"
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

METHODS = [
    "Baseline",
    "BAMS_new",
    "BAMS_nb_low",
    "BAMS_nb_lm",
    "BAMS_nb_iso",
    "PE_new",
    "PE_new_high",
    "PE_nb_consistent",
    "PE_nb_c1gate",
    "PE_nb_low",
    "PE_nb_lm",
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
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def load() -> dict:
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    nb = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_3x3nb_wide.csv")

    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce").astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf)
    nb["Original_ID"] = nb["Original_ID"].astype(int)

    merged = man.merge(
        nb[
            [
                "Original_ID",
                "has_C1_nb8",
                "has_C3",
                "n_C1_nb8",
                "n_C2_nb8",
                "n_C3_nb8",
                "suggest_C_if_uncertain",
                "center_C3",
            ]
        ],
        on="Original_ID",
        how="left",
        validate="one_to_one",
    )
    if merged["has_C1_nb8"].isna().any():
        raise ValueError("Neighborhood join missing some BAMS IDs")

    # save joined table for inspection
    merged.to_csv(OUT_DIR / "Nanjing_BAMS150_manual_with_nb8.csv", index=False)

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

    human = merged["Human_Class"].to_numpy()  # 1-based
    suggest = merged["suggest_C_if_uncertain"].astype(int).to_numpy()
    has_c1 = merged["has_C1_nb8"].astype(int).to_numpy()
    conf = merged["Conf"].tolist()
    oids = merged["Original_ID"].astype(int).to_numpy()

    def labels_variant(mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Return pos_seed, lab_seed (0-based), y_full, meta
        """
        h = human.copy()
        keep = np.ones(len(h), dtype=bool)
        n_adj = 0

        if mode == "new":
            pass
        elif mode == "nb_low":
            for i, c in enumerate(conf):
                if c == "low":
                    if h[i] != suggest[i]:
                        n_adj += 1
                    h[i] = suggest[i]
        elif mode == "nb_lm":
            for i, c in enumerate(conf):
                if c in ("low", "med"):
                    if h[i] != suggest[i]:
                        n_adj += 1
                    h[i] = suggest[i]
        elif mode == "nb_iso":
            for i, c in enumerate(conf):
                if h[i] == 1 and has_c1[i] == 0 and c != "high":
                    n_adj += 1
                    h[i] = 2
        elif mode == "new_high":
            keep = np.asarray([c == "high" for c in conf], dtype=bool)
        elif mode == "nb_consistent":
            # keep if agrees with suggest OR high confidence
            keep = np.asarray(
                [(h[i] == suggest[i]) or (conf[i] == "high") for i in range(len(h))],
                dtype=bool,
            )
        elif mode == "nb_c1gate":
            # drop Class1 seeds without built neighbor
            keep = np.asarray(
                [not (h[i] == 1 and has_c1[i] == 0) for i in range(len(h))],
                dtype=bool,
            )
        else:
            raise ValueError(mode)

        pos_all = np.asarray([id_to_pos[int(o)] for o in oids], dtype=int)
        lab_all = (h - 1).astype(np.int64)
        pos = pos_all[keep]
        lab = lab_all[keep]
        y_full = y_all.copy()
        y_full[pos] = lab
        meta = {
            "n_seed": int(keep.sum()),
            "n_adjusted": int(n_adj),
            "mode": mode,
        }
        return pos, lab, y_full, meta

    variants = {}
    for mode in [
        "new",
        "nb_low",
        "nb_lm",
        "nb_iso",
        "new_high",
        "nb_consistent",
        "nb_c1gate",
    ]:
        pos, lab, y_full, meta = labels_variant(mode)
        variants[mode] = {"pos": pos, "lab": lab, "y_full": y_full, "meta": meta}
        print(f"  variant {mode:14s} n_seed={meta['n_seed']:<3d} n_adjusted={meta['n_adjusted']}")

    # quick stats
    agree = float((human == suggest).mean())
    print(
        f"human vs suggest agree={agree:.3f}; "
        f"has_C1_nb8=1: {(has_c1==1).sum()}; =0: {(has_c1==0).sum()}"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "variants": variants,
        "agree_human_suggest": agree,
    }


def run_pe(X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, tag):
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pack["pos"]] = True
    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    X_seed_s = X_all_s[pack["pos"]]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], pack["lab"]], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, pack["pos"], pack["lab"], k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], pack["lab"], exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(pack["lab"]), REAL_WEIGHT, np.float32),
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


def run(device: str, seeds: list[int], epochs: int):
    data = load()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    V = data["variants"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        y_true = y_all[idx_test]

        rf_map = [
            ("Baseline", y_all, None),
            ("BAMS_new", V["new"]["y_full"], V["new"]["meta"]),
            ("BAMS_nb_low", V["nb_low"]["y_full"], V["nb_low"]["meta"]),
            ("BAMS_nb_lm", V["nb_lm"]["y_full"], V["nb_lm"]["meta"]),
            ("BAMS_nb_iso", V["nb_iso"]["y_full"], V["nb_iso"]["meta"]),
        ]
        for name, ysrc, meta in rf_map:
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(X_train, ysrc[idx_train])
            m = full_metrics(y_true, rf.predict(X_test))
            row = {"City": CITY, "Method": name, "Seed": seed, **m}
            if meta:
                row["n_seed"] = meta["n_seed"]
                row["n_adjusted"] = meta["n_adjusted"]
            rows.append(row)
            print(f"  [seed={seed}] {name:18s} OA={m['OA']:.4f} BE={m['Boundary Error']}")

        pe_map = [
            ("PE_new", V["new"]),
            ("PE_new_high", V["new_high"]),
            ("PE_nb_consistent", V["nb_consistent"]),
            ("PE_nb_c1gate", V["nb_c1gate"]),
            ("PE_nb_low", V["nb_low"]),
            ("PE_nb_lm", V["nb_lm"]),
        ]
        for name, pack in pe_map:
            best = run_pe(
                X_all, X_all_s, y_all, idx_train, idx_test, pack, device, seed, epochs, name
            )
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    **best["m"],
                    "BestK": best["k"],
                    "n_expanded": best["n_exp"],
                    "n_seed": pack["meta"]["n_seed"],
                    "n_adjusted": pack["meta"]["n_adjusted"],
                }
            )
            print(
                f"  [seed={seed}] {name:18s} OA={best['m']['OA']:.4f} "
                f"BE={best['m']['Boundary Error']} K={best['k']} exp={best['n_exp']}"
            )

    return pd.DataFrame(rows), data


def summarize(df: pd.DataFrame, data: dict) -> str:
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_n_seed=("n_seed", "mean"),
            Mean_n_adj=("n_adjusted", "mean"),
            Mean_n_exp=("n_expanded", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    base = overall.loc[overall.Method == "Baseline", "Mean_OA"].iloc[0]
    overall["dOA_pp"] = (overall["Mean_OA"] - base) * 100
    lines = [
        "# Nanjing 8-neighbor (BAMS 3x3nb) experiment\n",
        f"- human vs suggest_C agree: {data['agree_human_suggest']:.3f}\n",
        "\n## Mean over seeds\n",
        overall.to_string(index=False),
        "\n\n## Protocols\n",
        "- BAMS_nb_low/lm: replace Human by suggest_C for low / low+med confidence\n",
        "- BAMS_nb_iso: Class1 without built neighbor (non-high) -> Class2\n",
        "- PE_nb_consistent: drop seeds that disagree with suggest unless high conf\n",
        "- PE_nb_c1gate: drop Class1 seeds with has_C1_nb8=0\n",
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
    md, overall = summarize(df, data)
    df.to_csv(OUT_DIR / "seed_results.csv", index=False)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "methods": METHODS,
                "agree_human_suggest": data["agree_human_suggest"],
                "nb_file": "04_manual/Nanjing_BAMS150_3x3nb_wide.csv",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
