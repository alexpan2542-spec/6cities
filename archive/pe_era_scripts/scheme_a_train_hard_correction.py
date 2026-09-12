#!/usr/bin/env python3
"""
Scheme A: Train-set Hard Correction from BAMS prototypes.

Idea
----
BAMS human corrections are high-precision boundary fixes (~91% hit rate), but
150 labels are diluted by ~15k (noisy) original labels. Instead of only adding
soft pseudo-labels (PE), rewrite *training-set* labels that look like the same
error the human already corrected.

Algorithm
---------
1. Take corrected BAMS seeds: Human_Class != original Class.
2. For each seed (old_class -> new_class), find K nearest train neighbors
   in standardized feature space (cosine).
3. Hard-flip a neighbor iff:
     - neighbor is in the training split
     - neighbor original label == old_class  (same error type)
     - cosine similarity >= MIN_SIM
4. Always apply direct BAMS human labels on any BAMS points in train.
5. Train RF (and optional MLP) on corrected train labels; evaluate on test
   against original Class (same protocol as other experiments).

Compared methods
----------------
  Baseline
  BAMS150                 (only rewrite BAMS locations in train)
  SchemeA_HardCorrect     (BAMS + neighbor hard flips; best-K by BE)
  PE-Boundary             (existing soft expansion; reference)
  SchemeA+PE              (hard-correct train, then PE soft expand)

Outputs -> data2/scheme_a_hard_correction/
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
OUT_DIR = DATA2 / "scheme_a_hard_correction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"]
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
K_LIST = [3, 5, 10, 20]
MIN_SIM = 0.90  # stricter cosine similarity floor
MAX_FLIPS = 400  # safety cap per city (besides direct BAMS writes)
# Only propagate Class1↔Class2 corrections (dominant urban error type)
BOUNDARY_ONLY = True
PE_K_LIST = [5, 10, 20, 30]
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256

METHODS = [
    "Baseline",
    "BAMS150",
    "SchemeA_HardCorrect",
    "PE_Boundary",
    "SchemeA+PE",
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


def get_labeled_manual(path: Path) -> pd.DataFrame:
    manual = pd.read_csv(path)
    manual["Human_Class"] = pd.to_numeric(manual["Human_Class"], errors="coerce")
    manual = manual[manual["Human_Class"].notna()].copy()
    manual["Human_Class"] = manual["Human_Class"].astype(int)
    manual["Original_ID"] = manual["Original_ID"].astype(int)
    return manual


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


def load_city(city: str) -> dict:
    city_root = DATA2 / city
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")
    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_all
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    bams_pos = np.asarray(
        [id_to_pos[int(r.Original_ID)] for _, r in bams_manual.iterrows()], dtype=int
    )
    bams_label = np.asarray(
        [int(r.Human_Class) - 1 for _, r in bams_manual.iterrows()], dtype=np.int64
    )
    corrected_mask = bams_label != y_all[bams_pos]

    y_bams_full = y_all.copy()
    y_bams_full[bams_pos] = bams_label

    return {
        "city": city,
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "y_bams_full": y_bams_full,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "bams_pos": bams_pos,
        "bams_label": bams_label,
        "corrected_mask": corrected_mask,
        "ids": ids,
    }


def hard_correct_train_labels(
    X_all_s: np.ndarray,
    y_all: np.ndarray,
    idx_train: np.ndarray,
    bams_pos: np.ndarray,
    bams_label: np.ndarray,
    corrected_mask: np.ndarray,
    k: int,
    min_sim: float = MIN_SIM,
    max_flips: int = MAX_FLIPS,
) -> tuple[np.ndarray, dict]:
    """
    Return full-length label array with train indices hard-corrected.
    Test indices keep original labels (unused for training).
    """
    y_corr = y_all.copy()
    # Direct BAMS writes (train + anywhere; safe for train rewrite)
    y_corr[bams_pos] = bams_label

    train_set = set(idx_train.tolist())
    # Candidates for flipping: train points that are not BAMS seeds
    bams_set = set(bams_pos.tolist())
    train_candidates = np.asarray(
        [i for i in idx_train.tolist() if i not in bams_set], dtype=int
    )
    if len(train_candidates) == 0 or corrected_mask.sum() == 0:
        meta = {
            "n_seeds": int(corrected_mask.sum()),
            "n_flips": 0,
            "n_direct_bams_in_train": int(sum(1 for p in bams_pos if p in train_set)),
            "k": k,
            "min_sim": min_sim,
        }
        return y_corr, meta

    seed_mask = corrected_mask.copy()
    if BOUNDARY_ONLY:
        # Keep only Class1↔Class2 human corrections as propagation seeds
        old = y_all[bams_pos]
        new = bams_label
        seed_mask = corrected_mask & (
            ((old == 0) & (new == 1)) | ((old == 1) & (new == 0))
        )

    proto_pos = bams_pos[seed_mask]
    proto_new = bams_label[seed_mask]
    proto_old = y_all[proto_pos]

    if len(proto_pos) == 0:
        meta = {
            "n_seeds": 0,
            "n_flips": 0,
            "n_direct_bams_in_train": int(sum(1 for p in bams_pos if p in train_set)),
            "k": k,
            "min_sim": min_sim,
            "boundary_only": BOUNDARY_ONLY,
        }
        return y_corr, meta

    n_nn = min(k + 1, len(train_candidates))
    nn_model = NearestNeighbors(n_neighbors=n_nn, metric="cosine")
    nn_model.fit(X_all_s[train_candidates])
    dists, neigh_local = nn_model.kneighbors(X_all_s[proto_pos], return_distance=True)

    # Collect flip votes: index -> list of proposed new labels (from matching old_class seeds)
    flip_votes: dict[int, list[int]] = defaultdict(list)
    for pi in range(len(proto_pos)):
        old_c = int(proto_old[pi])
        new_c = int(proto_new[pi])
        for loc, dist in zip(neigh_local[pi].tolist(), dists[pi].tolist()):
            j = int(train_candidates[int(loc)])
            if j == int(proto_pos[pi]):
                continue
            sim = 1.0 - float(dist)
            if sim < min_sim:
                continue
            if int(y_all[j]) != old_c:
                continue
            # Only flip among Class1/2 mass for boundary-only mode
            if BOUNDARY_ONLY and int(y_all[j]) not in (0, 1):
                continue
            flip_votes[j].append(new_c)

    # Rank candidates by vote agreement * n_votes, apply up to max_flips
    scored = []
    for j, votes in flip_votes.items():
        lab = majority_label(votes)
        agree = sum(1 for v in votes if v == lab) / len(votes)
        scored.append((agree * len(votes), agree, len(votes), j, lab))
    scored.sort(reverse=True)

    n_flips = 0
    flipped_idx = []
    for _, agree, nv, j, lab in scored:
        if n_flips >= max_flips:
            break
        if int(y_corr[j]) == int(lab):
            continue
        y_corr[j] = int(lab)
        n_flips += 1
        flipped_idx.append(j)

    meta = {
        "n_seeds": int(len(proto_pos)),
        "n_flips": int(n_flips),
        "n_flip_candidates": int(len(flip_votes)),
        "n_direct_bams_in_train": int(sum(1 for p in bams_pos if p in train_set)),
        "k": k,
        "min_sim": min_sim,
        "boundary_only": BOUNDARY_ONLY,
        "mean_agree_applied": float(
            np.mean([a for _, a, _, j, _ in scored if j in set(flipped_idx)])
        )
        if flipped_idx
        else float("nan"),
    }
    return y_corr, meta


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def run_city(city: str, device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print(f"CITY: {city}")
    print("=" * 72)
    data = load_city(city)
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    y_bams_full = data["y_bams_full"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    corrected_mask = data["corrected_mask"]

    print(
        f"train={len(idx_train)} test={len(idx_test)} "
        f"BAMS={len(bams_pos)} corrected_seeds={int(corrected_mask.sum())}"
    )

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        y_true = y_all[idx_test]
        X_train = X_all[idx_train]
        X_test = X_all[idx_test]
        X_train_s = X_all_s[idx_train]
        X_test_s = X_all_s[idx_test]
        X_bams_s = X_all_s[bams_pos]

        # ---- Baseline / BAMS RF ----
        rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_base.fit(X_train, y_all[idx_train])
        m_base = full_metrics(y_true, rf_base.predict(X_test))

        rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_bams.fit(X_train, y_bams_full[idx_train])
        m_bams = full_metrics(y_true, rf_bams.predict(X_test))

        # ---- Scheme A: sweep K, pick best by BE ----
        a_cands = []
        for k in K_LIST:
            y_corr, meta = hard_correct_train_labels(
                X_all_s,
                y_all,
                idx_train,
                bams_pos,
                bams_label,
                corrected_mask,
                k=k,
                min_sim=MIN_SIM,
                max_flips=MAX_FLIPS,
            )
            rf_a = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf_a.fit(X_train, y_corr[idx_train])
            m_a = full_metrics(y_true, rf_a.predict(X_test))
            a_cands.append({"k": k, "m": m_a, "meta": meta, "y_corr": y_corr, "pred": rf_a.predict(X_test)})
            print(
                f"  [seed={seed}] SchemeA K={k:<2d} flips={meta['n_flips']:<4d} "
                f"OA={m_a['OA']:.4f} BE={m_a['Boundary Error']}"
            )
        best_a = pick_best(a_cands)

        # ---- PE-Boundary (MLP soft expand; same as prior protocol) ----
        teacher = train_mlp(
            np.concatenate([X_train_s, X_bams_s], axis=0),
            np.concatenate([y_all[idx_train], bams_label], axis=0),
            device=device,
            seed=seed,
            epochs=epochs,
        )
        _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)
        used = np.zeros(len(X_all), dtype=bool)
        used[idx_train] = True
        used[bams_pos] = True

        pe_cands = []
        for k in PE_K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
            keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_all[idx_train], bams_label, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs)
            pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            m_pe = full_metrics(y_true, pred)
            pe_cands.append({"k": k, "m": m_pe, "n_exp": int(len(exp_idx)), "pred": pred})
        best_pe = pick_best(pe_cands)

        # ---- SchemeA + PE: hard-correct train labels, then PE soft expand ----
        y_a = best_a["y_corr"]
        teacher_a = train_mlp(
            np.concatenate([X_train_s, X_bams_s], axis=0),
            np.concatenate([y_a[idx_train], bams_label], axis=0),
            device=device,
            seed=seed + 7,
            epochs=epochs,
        )
        _, _, margin_a = predict_proba_mlp(teacher_a, X_all_s, device=device)
        ape_cands = []
        for k in PE_K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
            keep = (~used[exp_idx_all]) & (margin_a[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_a[idx_train], bams_label, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 200 + k, epochs=epochs)
            pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            m_ape = full_metrics(y_true, pred)
            ape_cands.append({"k": k, "m": m_ape, "n_exp": int(len(exp_idx)), "pred": pred})
        best_ape = pick_best(ape_cands)

        pack = [
            ("Baseline", m_base, {}),
            ("BAMS150", m_bams, {}),
            (
                "SchemeA_HardCorrect",
                best_a["m"],
                {
                    "BestK": best_a["k"],
                    "n_flips": best_a["meta"]["n_flips"],
                    "n_seeds": best_a["meta"]["n_seeds"],
                },
            ),
            (
                "PE_Boundary",
                best_pe["m"],
                {"BestK": best_pe["k"], "n_expanded": best_pe["n_exp"]},
            ),
            (
                "SchemeA+PE",
                best_ape["m"],
                {
                    "BestK": best_ape["k"],
                    "n_flips": best_a["meta"]["n_flips"],
                    "n_expanded": best_ape["n_exp"],
                },
            ),
        ]

        for method, m, extra in pack:
            row = {
                "City": city,
                "Method": method,
                "Seed": seed,
                **m,
                "BestK": extra.get("BestK", np.nan),
                "n_flips": extra.get("n_flips", np.nan),
                "n_seeds": extra.get("n_seeds", np.nan),
                "n_expanded": extra.get("n_expanded", np.nan),
            }
            rows.append(row)
            extra_s = ""
            if method == "SchemeA_HardCorrect":
                extra_s = f" flips={extra['n_flips']} K={extra['BestK']}"
            elif method in ("PE_Boundary", "SchemeA+PE"):
                extra_s = f" exp={extra.get('n_expanded')} K={extra['BestK']}"
            print(
                f"  [seed={seed}] {method:22s} OA={m['OA']:.4f} BE={m['Boundary Error']}{extra_s}"
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_Flips=("n_flips", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    by_city = (
        df.groupby(["City", "Method"], sort=False)
        .agg(
            OA_Mean=("OA", "mean"),
            BE_Mean=("Boundary Error", "mean"),
            Flips_Mean=("n_flips", "mean"),
        )
        .reset_index()
    )

    lines = []
    lines.append("# Scheme A: Train-set Hard Correction — Summary\n")
    lines.append("## Overall (cities × seeds)\n")
    lines.append(overall.to_string(index=False))
    lines.append("\n\n## Per city\n")
    # pivot OA
    oa = by_city.pivot(index="City", columns="Method", values="OA_Mean")[METHODS]
    be = by_city.pivot(index="City", columns="Method", values="BE_Mean")[METHODS]
    lines.append("OA:\n" + oa.to_string())
    lines.append("\n\nBoundary Error:\n" + be.to_string())
    lines.append(
        "\n\n## Reading guide\n"
        "- SchemeA_HardCorrect should beat BAMS150 if neighbor flips reduce dilution.\n"
        "- Compare to PE_Boundary: different mechanism (hard train rewrite vs soft expand).\n"
        "- SchemeA+PE tests whether hard correction + soft expansion stacks.\n"
    )
    return "\n".join(lines), overall, by_city


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=CITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true", help="1 seed, fewer epochs")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds
    epochs = args.epochs
    if args.quick:
        seeds = [0]
        epochs = min(epochs, 40)
        print("[quick] seeds=", seeds, "epochs=", epochs)
    print("device:", device)
    print("cities:", args.cities)

    all_rows = []
    for city in args.cities:
        df_c = run_city(city, device=device, seeds=seeds, epochs=epochs)
        (OUT_DIR / city).mkdir(parents=True, exist_ok=True)
        df_c.to_csv(OUT_DIR / city / "results.csv", index=False)
        all_rows.append(df_c)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(OUT_DIR / "all_results.csv", index=False)
    md, overall, by_city = summarize(df)
    overall.to_csv(OUT_DIR / "overall_summary.csv", index=False)
    by_city.to_csv(OUT_DIR / "city_summary.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "cities": args.cities,
                "seeds": seeds,
                "epochs": epochs,
                "k_list": K_LIST,
                "min_sim": MIN_SIM,
                "max_flips": MAX_FLIPS,
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
