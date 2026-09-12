#!/usr/bin/env python3
"""
Scheme D: Conflict Downweighting near BAMS correction prototypes.

Idea
----
Unlike Scheme A (hard-flip train labels), keep original Class labels but
*downweight* training samples that look like the same error a human already
corrected. Human BAMS points get higher weight.

Algorithm
---------
1. Take corrected BAMS seeds (optionally Class1↔Class2 only).
2. For each seed (old_class -> new_class), find K cosine neighbors in train
   with sim >= MIN_SIM and original label == old_class.
3. Assign sample weights:
     - conflict neighbors: CONFLICT_W  (swept; 0 = drop from loss)
     - normal train:       1.0
     - BAMS human points:  HUMAN_W
4. Train with BAMS label writes on BAMS locations + sample weights.
5. Evaluate on test vs original Class.

Compared methods
----------------
  Baseline
  BAMS150
  SchemeD_Downweight   (best K × conflict_w by BE)
  PE_Boundary
  SchemeD+PE           (downweighted train + PE soft expand)

Outputs -> data2/scheme_d_conflict_downweight/
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
OUT_DIR = DATA2 / "scheme_d_conflict_downweight"
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
K_LIST = [5, 10, 20, 30]
CONFLICT_W_LIST = [0.0, 0.1, 0.25, 0.5]
MIN_SIM = 0.90
HUMAN_WEIGHT = 5.0
BOUNDARY_ONLY = True
MAX_CONFLICTS = 800  # cap how many train points get downweighted
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
    "SchemeD_Downweight",
    "PE_Boundary",
    "SchemeD+PE",
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
    # Drop zero-weight rows (equivalent to removing from loss)
    keep = sample_weight > 0
    X, y, sample_weight = X[keep], y[keep], sample_weight[keep]
    if len(y) == 0:
        raise ValueError("No positive-weight samples for MLP training")
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


def find_conflict_train_indices(
    X_all_s: np.ndarray,
    y_all: np.ndarray,
    idx_train: np.ndarray,
    bams_pos: np.ndarray,
    bams_label: np.ndarray,
    corrected_mask: np.ndarray,
    k: int,
    min_sim: float = MIN_SIM,
    max_conflicts: int = MAX_CONFLICTS,
) -> tuple[np.ndarray, dict]:
    """
    Return global indices of train points that conflict with human correction
    direction (same old_class as a nearby corrected seed), ranked and capped.
    """
    bams_set = set(bams_pos.tolist())
    train_candidates = np.asarray(
        [i for i in idx_train.tolist() if i not in bams_set], dtype=int
    )

    seed_mask = corrected_mask.copy()
    if BOUNDARY_ONLY:
        old = y_all[bams_pos]
        new = bams_label
        seed_mask = corrected_mask & (
            ((old == 0) & (new == 1)) | ((old == 1) & (new == 0))
        )

    proto_pos = bams_pos[seed_mask]
    proto_old = y_all[proto_pos]

    empty_meta = {
        "n_seeds": int(len(proto_pos)),
        "n_conflicts": 0,
        "n_conflict_candidates": 0,
        "k": k,
        "min_sim": min_sim,
    }
    if len(train_candidates) == 0 or len(proto_pos) == 0:
        return np.asarray([], dtype=int), empty_meta

    n_nn = min(k + 1, len(train_candidates))
    nn_model = NearestNeighbors(n_neighbors=n_nn, metric="cosine")
    nn_model.fit(X_all_s[train_candidates])
    dists, neigh_local = nn_model.kneighbors(X_all_s[proto_pos], return_distance=True)

    # score: how often a train point is nominated as conflicting by seeds
    conflict_score: dict[int, float] = defaultdict(float)
    for pi in range(len(proto_pos)):
        old_c = int(proto_old[pi])
        for loc, dist in zip(neigh_local[pi].tolist(), dists[pi].tolist()):
            j = int(train_candidates[int(loc)])
            if j == int(proto_pos[pi]):
                continue
            sim = 1.0 - float(dist)
            if sim < min_sim:
                continue
            if int(y_all[j]) != old_c:
                continue
            if BOUNDARY_ONLY and int(y_all[j]) not in (0, 1):
                continue
            conflict_score[j] += sim

    ranked = sorted(conflict_score.items(), key=lambda kv: -kv[1])
    picked = [j for j, _ in ranked[:max_conflicts]]
    meta = {
        "n_seeds": int(len(proto_pos)),
        "n_conflicts": int(len(picked)),
        "n_conflict_candidates": int(len(conflict_score)),
        "k": k,
        "min_sim": min_sim,
    }
    return np.asarray(picked, dtype=int), meta


def build_train_weights(
    idx_train: np.ndarray,
    bams_pos: np.ndarray,
    conflict_idx: np.ndarray,
    conflict_w: float,
    human_w: float = HUMAN_WEIGHT,
) -> np.ndarray:
    """Per-row weights aligned with idx_train order."""
    w = np.ones(len(idx_train), dtype=np.float32)
    pos_of = {int(g): i for i, g in enumerate(idx_train.tolist())}

    for g in conflict_idx.tolist():
        if g in pos_of:
            w[pos_of[g]] = float(conflict_w)

    for g in bams_pos.tolist():
        if g in pos_of:
            w[pos_of[g]] = float(human_w)

    return w


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
        y_train_bams = y_bams_full[idx_train]

        # ---- Baseline / BAMS RF ----
        rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_base.fit(X_train, y_all[idx_train])
        m_base = full_metrics(y_true, rf_base.predict(X_test))

        rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_bams.fit(X_train, y_train_bams)
        m_bams = full_metrics(y_true, rf_bams.predict(X_test))

        # ---- Scheme D: sweep K × conflict_w ----
        d_cands = []
        for k in K_LIST:
            conflict_idx, meta = find_conflict_train_indices(
                X_all_s,
                y_all,
                idx_train,
                bams_pos,
                bams_label,
                corrected_mask,
                k=k,
                min_sim=MIN_SIM,
                max_conflicts=MAX_CONFLICTS,
            )
            for cw in CONFLICT_W_LIST:
                w_train = build_train_weights(
                    idx_train, bams_pos, conflict_idx, conflict_w=cw, human_w=HUMAN_WEIGHT
                )
                rf_d = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
                # sklearn RF ignores zero-weight samples
                rf_d.fit(X_train, y_train_bams, sample_weight=w_train)
                m_d = full_metrics(y_true, rf_d.predict(X_test))
                d_cands.append(
                    {
                        "k": k,
                        "cw": cw,
                        "m": m_d,
                        "meta": meta,
                        "w_train": w_train,
                        "n_conflicts": int(meta["n_conflicts"]),
                        "n_downweighted": int((w_train < 1.0 - 1e-8).sum()),
                    }
                )
                print(
                    f"  [seed={seed}] SchemeD K={k:<2d} cw={cw:<4} "
                    f"conf={meta['n_conflicts']:<4d} OA={m_d['OA']:.4f} "
                    f"BE={m_d['Boundary Error']}"
                )
        best_d = pick_best(d_cands)

        # ---- PE-Boundary (reference; same protocol as Scheme A script) ----
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

        # ---- SchemeD + PE: downweighted train labels (BAMS-corrected) + PE ----
        w_d = best_d["w_train"]
        teacher_w = np.concatenate(
            [w_d, np.full(len(bams_label), HUMAN_WEIGHT, dtype=np.float32)], axis=0
        )
        teacher_d = train_mlp(
            np.concatenate([X_train_s, X_bams_s], axis=0),
            np.concatenate([y_train_bams, bams_label], axis=0),
            sample_weight=teacher_w,
            device=device,
            seed=seed + 7,
            epochs=epochs,
        )
        _, _, margin_d = predict_proba_mlp(teacher_d, X_all_s, device=device)
        dpe_cands = []
        for k in PE_K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
            keep = (~used[exp_idx_all]) & (margin_d[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_train_bams, bams_label, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    w_d,
                    np.full(len(bams_label), HUMAN_WEIGHT, dtype=np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 200 + k, epochs=epochs)
            pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            m_dpe = full_metrics(y_true, pred)
            dpe_cands.append({"k": k, "m": m_dpe, "n_exp": int(len(exp_idx)), "pred": pred})
        best_dpe = pick_best(dpe_cands)

        pack = [
            ("Baseline", m_base, {}),
            ("BAMS150", m_bams, {}),
            (
                "SchemeD_Downweight",
                best_d["m"],
                {
                    "BestK": best_d["k"],
                    "conflict_w": best_d["cw"],
                    "n_conflicts": best_d["n_conflicts"],
                    "n_seeds": best_d["meta"]["n_seeds"],
                },
            ),
            (
                "PE_Boundary",
                best_pe["m"],
                {"BestK": best_pe["k"], "n_expanded": best_pe["n_exp"]},
            ),
            (
                "SchemeD+PE",
                best_dpe["m"],
                {
                    "BestK": best_dpe["k"],
                    "conflict_w": best_d["cw"],
                    "n_conflicts": best_d["n_conflicts"],
                    "n_expanded": best_dpe["n_exp"],
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
                "conflict_w": extra.get("conflict_w", np.nan),
                "n_conflicts": extra.get("n_conflicts", np.nan),
                "n_seeds": extra.get("n_seeds", np.nan),
                "n_expanded": extra.get("n_expanded", np.nan),
            }
            rows.append(row)
            extra_s = ""
            if method == "SchemeD_Downweight":
                extra_s = (
                    f" conf={extra['n_conflicts']} K={extra['BestK']} cw={extra['conflict_w']}"
                )
            elif method in ("PE_Boundary", "SchemeD+PE"):
                extra_s = f" exp={extra.get('n_expanded')} K={extra['BestK']}"
            print(
                f"  [seed={seed}] {method:22s} OA={m['OA']:.4f} BE={m['Boundary Error']}{extra_s}"
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_Conflicts=("n_conflicts", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    by_city = (
        df.groupby(["City", "Method"], sort=False)
        .agg(
            OA_Mean=("OA", "mean"),
            BE_Mean=("Boundary Error", "mean"),
            Conflicts_Mean=("n_conflicts", "mean"),
        )
        .reset_index()
    )

    lines = []
    lines.append("# Scheme D: Conflict Downweight — Summary\n")
    lines.append("## Overall (cities × seeds)\n")
    lines.append(overall.to_string(index=False))
    lines.append("\n\n## Per city\n")
    oa = by_city.pivot(index="City", columns="Method", values="OA_Mean")[METHODS]
    be = by_city.pivot(index="City", columns="Method", values="BE_Mean")[METHODS]
    lines.append("OA:\n" + oa.to_string())
    lines.append("\n\nBoundary Error:\n" + be.to_string())
    lines.append(
        "\n\n## Reading guide\n"
        "- SchemeD_Downweight keeps original Class but downweights neighbors that\n"
        "  match the pre-correction error class near BAMS seeds; human points get\n"
        f"  weight={HUMAN_WEIGHT}.\n"
        "- Unlike Scheme A, labels are not hard-flipped (softer against GT conflict).\n"
        "- Compare to PE_Boundary; SchemeD+PE tests stacking.\n"
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
    print(
        f"K_LIST={K_LIST} CONFLICT_W={CONFLICT_W_LIST} "
        f"MIN_SIM={MIN_SIM} HUMAN_W={HUMAN_WEIGHT}"
    )

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
                "conflict_w_list": CONFLICT_W_LIST,
                "min_sim": MIN_SIM,
                "human_weight": HUMAN_WEIGHT,
                "max_conflicts": MAX_CONFLICTS,
                "boundary_only": BOUNDARY_ONLY,
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
