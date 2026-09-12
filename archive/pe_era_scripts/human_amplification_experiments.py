#!/usr/bin/env python3
"""
Human Amplification (HA): maximize reliable propagation of BAMS human corrections.

Core question
-------------
Can ~150 high-quality human labels (esp. true corrections) selectively update a much
larger fraction of the ~15k candidate pool without destroying label quality?

Design
------
1. Sample-centric pool (main): every eligible candidate queries nearest human
   prototypes and receives a scored pseudo-label. This can cover thousands of
   points, unlike classic prototype-centric PE (tens/hundreds).
2. Quality gates: margin thresholds, vote agreement, optional sim>=0.80 subset.
3. Sweep top-N by score -> coverage vs OA / Boundary Error curve.
4. Prototype sources: all_bams | corrected | corrected_12.

Proxy quality (no extra labeling)
---------------------------------
Leave-one-out propagation precision (LOO-PP) on human prototypes.

Outputs -> data2/human_amplification/

Example
-------
  python scripts/human_amplification_experiments.py --quick --cities Wuhan
  python scripts/human_amplification_experiments.py \\
      --modes corrected all_bams --gates boundary_025 unconstrained
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
OUT_ROOT = DATA2 / "human_amplification"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

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
MARGIN_THRESH = 0.15
K_POOL = 40
# Sample-centric: each eligible point looks at this many nearest prototypes
N_PROTO_NEIGHBORS = 5
MIN_VOTES = 1
MIN_AGREEMENT = 0.5
MIN_SIM = 0.0  # cosine similarity floor (1 - distance)
# Sample-centric similarity floors used when building large pools
SIM_FLOORS = (0.0, 0.70, 0.80, 0.90)
REAL_WEIGHT = 1.0
SOFT_WEIGHT_BASE = 0.5
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256
SEED = 42

# Expansion scale sweep: how many pseudo-labels to keep (ranked by score)
TOP_N_LIST = [0, 200, 500, 1000, 2000, 4000, 8000, 10_000]

PROTO_MODES = ("all_bams", "corrected", "corrected_12")
# Gate modes for sample-centric amplification
GATE_MODES = (
    ("boundary_015", True, 0.15),
    ("boundary_025", True, 0.25),
    ("boundary_040", True, 0.40),
    ("unconstrained", False, 1.0),
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    model = MLP(in_dim=X.shape[1], n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss_each = crit(model(xb), yb)
            loss = (loss_each * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba_mlp(model, X, device="cpu"):
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32)).to(device)
    probs = torch.softmax(model(xt), dim=1).cpu().numpy().astype(np.float64)
    order = np.argsort(probs, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2]
    top1 = probs[np.arange(len(probs)), top1_idx]
    top2 = probs[np.arange(len(probs)), top2_idx]
    margin = top1 - top2
    return top1_idx.astype(np.int64), top1, margin, probs


def full_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.nan_to_num(np.diag(cm) / cm.sum(axis=1), nan=0.0)
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "AA": float(per_class.mean()),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Boundary Error": int(cm[0, 1] + cm[1, 0]),
        "C1_to_C2": int(cm[0, 1]),
        "C2_to_C1": int(cm[1, 0]),
    }


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def select_prototypes(
    mode: str,
    bams_pos: np.ndarray,
    bams_label: np.ndarray,
    y_original: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return prototype positions and 0-based human labels."""
    if mode == "all_bams":
        mask = np.ones(len(bams_pos), dtype=bool)
    elif mode == "corrected":
        mask = bams_label != y_original[bams_pos]
    elif mode == "corrected_12":
        mask = (bams_label != y_original[bams_pos]) & np.isin(bams_label, [0, 1])
    else:
        raise ValueError(mode)
    if mask.sum() == 0:
        # fallback: keep all to avoid empty run
        mask = np.ones(len(bams_pos), dtype=bool)
    return bams_pos[mask], bams_label[mask]


def _empty_pool() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "idx",
            "pseudo_label",
            "n_votes",
            "agreement",
            "mean_sim",
            "score",
            "margin",
        ]
    )


def build_prototype_centric_pool(
    feature: np.ndarray,
    proto_pos: np.ndarray,
    proto_lab: np.ndarray,
    eligible: np.ndarray,
    margin_pool: np.ndarray,
    k: int = K_POOL,
    margin_thresh: float = MARGIN_THRESH,
    require_margin: bool = True,
) -> pd.DataFrame:
    """
    Classic PE-style pool: each prototype contributes up to K neighbors.
    Coverage is inherently capped (~n_proto * K after overlaps).
    """
    if len(proto_pos) == 0:
        return _empty_pool()

    n_neighbors = min(k + 1, len(feature))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_model.fit(feature)
    dists, neigh_idx = nn_model.kneighbors(feature[proto_pos], return_distance=True)

    votes: dict[int, list[int]] = defaultdict(list)
    sims: dict[int, list[float]] = defaultdict(list)

    for pi, (neighbors, drow) in enumerate(zip(neigh_idx, dists)):
        plab = int(proto_lab[pi])
        ppos = int(proto_pos[pi])
        for nidx, dist in zip(neighbors.tolist(), drow.tolist()):
            nidx = int(nidx)
            if nidx == ppos or not eligible[nidx]:
                continue
            if require_margin and margin_pool[nidx] >= margin_thresh:
                continue
            sim = 1.0 - float(dist)
            if sim < MIN_SIM:
                continue
            votes[nidx].append(plab)
            sims[nidx].append(sim)

    rows = []
    for idx, vlist in votes.items():
        lab = majority_label(vlist)
        n_votes = len(vlist)
        agreement = sum(1 for v in vlist if v == lab) / n_votes
        mean_sim = float(np.mean(sims[idx]))
        if n_votes < MIN_VOTES or agreement < MIN_AGREEMENT:
            continue
        score = float(agreement * mean_sim * np.log1p(n_votes))
        rows.append(
            {
                "idx": int(idx),
                "pseudo_label": int(lab),
                "n_votes": int(n_votes),
                "agreement": float(agreement),
                "mean_sim": mean_sim,
                "score": score,
                "margin": float(margin_pool[idx]),
            }
        )
    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    return pool.sort_values(["score", "mean_sim", "agreement"], ascending=False).reset_index(
        drop=True
    )


# Back-compat alias used by LOO helper
build_scored_pool = build_prototype_centric_pool


def build_sample_centric_pool(
    feature: np.ndarray,
    proto_pos: np.ndarray,
    proto_lab: np.ndarray,
    eligible: np.ndarray,
    margin_pool: np.ndarray,
    n_proto_neighbors: int = N_PROTO_NEIGHBORS,
    margin_thresh: float = MARGIN_THRESH,
    require_margin: bool = True,
    min_sim: float = 0.0,
) -> pd.DataFrame:
    """
    Max-amplification pool: EVERY eligible sample queries nearest human prototypes.

    This can cover thousands of candidates (up to all eligible points), unlike
    prototype-centric KNN which saturates at a few hundred.
    """
    if len(proto_pos) == 0:
        return _empty_pool()

    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) == 0:
        return _empty_pool()

    n_nn = min(n_proto_neighbors, len(proto_pos))
    nn_model = NearestNeighbors(n_neighbors=n_nn, metric="cosine")
    nn_model.fit(feature[proto_pos])
    dists, neigh_local = nn_model.kneighbors(feature[eligible_idx], return_distance=True)

    rows = []
    for i, eidx in enumerate(eligible_idx):
        eidx = int(eidx)
        if require_margin and margin_pool[eidx] >= margin_thresh:
            continue
        local = neigh_local[i]
        drow = dists[i]
        vlist = []
        slist = []
        for li, dist in zip(local.tolist(), drow.tolist()):
            sim = 1.0 - float(dist)
            if sim < min_sim:
                continue
            vlist.append(int(proto_lab[int(li)]))
            slist.append(sim)
        if not vlist:
            continue
        lab = majority_label(vlist)
        n_votes = len(vlist)
        agreement = sum(1 for v in vlist if v == lab) / n_votes
        mean_sim = float(np.mean(slist))
        if agreement < MIN_AGREEMENT:
            continue
        score = float(agreement * mean_sim * np.log1p(n_votes))
        rows.append(
            {
                "idx": eidx,
                "pseudo_label": int(lab),
                "n_votes": int(n_votes),
                "agreement": float(agreement),
                "mean_sim": mean_sim,
                "score": score,
                "margin": float(margin_pool[eidx]),
            }
        )

    pool = pd.DataFrame(rows)
    if pool.empty:
        return pool
    return pool.sort_values(["score", "mean_sim", "agreement"], ascending=False).reset_index(
        drop=True
    )


def loo_propagation_precision(
    feature: np.ndarray,
    proto_pos: np.ndarray,
    proto_lab: np.ndarray,
    margin_pool: np.ndarray,
    k: int = K_POOL,
    margin_thresh: float = MARGIN_THRESH,
) -> dict:
    """
    Leave-one-out: would held-out corrected prototypes be correctly labeled by peers?
    Only meaningful when |proto| >= 3.
    """
    n = len(proto_pos)
    if n < 3:
        return {"loo_n": 0, "loo_correct": 0, "loo_precision": float("nan")}

    correct = 0
    evaluated = 0
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        # eligible: only the held-out point
        eligible = np.zeros(len(feature), dtype=bool)
        eligible[int(proto_pos[i])] = True
        # temporarily relax margin for LOO on known human points
        pool = build_scored_pool(
            feature,
            proto_pos[keep],
            proto_lab[keep],
            eligible,
            margin_pool,
            k=k,
            margin_thresh=margin_thresh,
            require_margin=False,
        )
        if pool.empty:
            continue
        evaluated += 1
        if int(pool.iloc[0]["pseudo_label"]) == int(proto_lab[i]):
            correct += 1

    prec = correct / evaluated if evaluated else float("nan")
    return {"loo_n": evaluated, "loo_correct": correct, "loo_precision": float(prec)}


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

    # Labels after applying all BAMS human corrections on the full table (for RF-BAMS)
    y_bams_full = y_all.copy()
    y_bams_full[bams_pos] = bams_label

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True

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
        "used": used,
        "n_total": len(X_all),
        "n_corrected": int(corrected_mask.sum()),
    }


def train_eval_ha(
    data: dict,
    pool: pd.DataFrame,
    top_n: int,
    device: str,
    seed: int,
    epochs: int,
) -> tuple[dict, int, float]:
    """Train student with top-N amplified labels; return metrics, n_used, mean_score."""
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]

    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    y_train = y_all[idx_train]
    y_true = y_all[idx_test]
    X_bams_s = X_all_s[bams_pos]

    if top_n <= 0 or pool.empty:
        X_tr = np.concatenate([X_train_s, X_bams_s], axis=0)
        y_tr = np.concatenate([y_train, bams_label], axis=0)
        w_tr = np.full(len(y_tr), REAL_WEIGHT, dtype=np.float32)
        n_used = 0
        mean_score = float("nan")
        mean_agree = float("nan")
    else:
        take = pool.head(min(top_n, len(pool)))
        exp_idx = take["idx"].to_numpy(dtype=int)
        exp_lab = take["pseudo_label"].to_numpy(dtype=np.int64)
        # Tiered weights: score-normalized soft weights
        scores = take["score"].to_numpy(dtype=np.float64)
        smax = scores.max() if len(scores) else 1.0
        soft_w = (SOFT_WEIGHT_BASE * (0.4 + 0.6 * (scores / (smax + 1e-8)))).astype(np.float32)

        X_tr = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
        y_tr = np.concatenate([y_train, bams_label, exp_lab], axis=0)
        w_tr = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                soft_w,
            ]
        )
        n_used = len(exp_idx)
        mean_score = float(scores.mean())
        mean_agree = float(take["agreement"].mean())

    model = train_mlp(X_tr, y_tr, sample_weight=w_tr, device=device, seed=seed, epochs=epochs)
    y_pred, _, _, _ = predict_proba_mlp(model, X_test_s, device=device)
    m = full_metrics(y_true, y_pred)
    m["n_amplified"] = n_used
    m["mean_score"] = mean_score
    m["mean_agreement"] = mean_agree
    m["pool_size"] = int(len(pool))
    return m, n_used, mean_score


def run_city(
    city: str,
    device: str,
    epochs: int,
    top_n_list: list[int],
    proto_modes: tuple[str, ...],
    seed: int = SEED,
    gate_modes: tuple = GATE_MODES,
) -> pd.DataFrame:
    print("\n" + "=" * 72)
    print(f"CITY: {city}")
    print("=" * 72)
    data = load_city(city)
    print(
        f"n_total={data['n_total']}  BAMS={len(data['bams_pos'])}  "
        f"corrected={data['n_corrected']}"
    )

    set_all_seeds(seed)
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    used = data["used"]

    # Baselines (RF) for reference
    rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf_base.fit(X_all[idx_train], y_all[idx_train])
    m_base = full_metrics(y_all[idx_test], rf_base.predict(X_all[idx_test]))

    rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf_bams.fit(X_all[idx_train], data["y_bams_full"][idx_train])
    m_bams = full_metrics(y_all[idx_test], rf_bams.predict(X_all[idx_test]))

    # Teacher for margin gate (trained on train + BAMS human)
    teacher = train_mlp(
        np.concatenate([X_all_s[idx_train], X_all_s[bams_pos]], axis=0),
        np.concatenate([y_all[idx_train], bams_label], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool, _ = predict_proba_mlp(teacher, X_all_s, device=device)
    eligible = ~used
    print(f"eligible_for_amplification={int(eligible.sum())}")

    rows: list[dict] = []

    def add_row(method: str, proto_mode: str, top_n: int, metrics: dict, extra: dict | None = None):
        row = {
            "City": city,
            "Method": method,
            "ProtoMode": proto_mode,
            "TopN": top_n,
            "n_total": data["n_total"],
            "n_corrected_prototypes_available": data["n_corrected"],
            **metrics,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    add_row("Baseline_RF", "none", 0, m_base)
    add_row("BAMS150_RF", "all_bams", 0, m_bams)

    # Same-family MLP baseline (BAMS only, no amplification)
    m_mlp0, _, _ = train_eval_ha(data, _empty_pool(), 0, device=device, seed=seed, epochs=epochs)
    add_row("BAMS150_MLP", "all_bams", 0, m_mlp0, extra={"n_prototypes": len(bams_pos)})

    city_dir = OUT_ROOT / city
    city_dir.mkdir(parents=True, exist_ok=True)

    for mode in proto_modes:
        proto_pos, proto_lab = select_prototypes(mode, bams_pos, bams_label, y_all)
        print(f"\n[ProtoMode={mode}] n_prototypes={len(proto_pos)}")

        loo = loo_propagation_precision(
            X_all_s, proto_pos, proto_lab, margin_pool, k=K_POOL, margin_thresh=MARGIN_THRESH
        )
        print(
            f"  LOO-PP: {loo['loo_correct']}/{loo['loo_n']} = {loo['loo_precision']:.3f}"
            if loo["loo_n"]
            else "  LOO-PP: n/a"
        )

        # Reference: old prototype-centric PE pool (coverage ceiling check)
        pe_pool = build_prototype_centric_pool(
            X_all_s,
            proto_pos,
            proto_lab,
            eligible,
            margin_pool,
            k=K_POOL,
            margin_thresh=MARGIN_THRESH,
            require_margin=True,
        )
        print(f"  proto-centric PE pool (margin<0.15): {len(pe_pool)}")
        pe_pool.to_csv(city_dir / f"pool_pe_{mode}.csv", index=False)

        for gate_name, require_margin, thr in gate_modes:
            # Primary: sample-centric max amplification
            pool = build_sample_centric_pool(
                X_all_s,
                proto_pos,
                proto_lab,
                eligible,
                margin_pool,
                n_proto_neighbors=N_PROTO_NEIGHBORS,
                margin_thresh=thr,
                require_margin=require_margin,
                min_sim=0.0,
            )
            # High-sim subset for quality-focused amplification
            pool_hi = pool[pool["mean_sim"] >= 0.80].reset_index(drop=True) if not pool.empty else pool

            print(
                f"  gate={gate_name:16s} sample-centric pool={len(pool):5d}  "
                f"sim>=0.80={len(pool_hi):5d}"
            )
            pool.to_csv(city_dir / f"pool_sc_{mode}_{gate_name}.csv", index=False)

            for pool_tag, use_pool in (
                (f"HA_SC_{gate_name}", pool),
                (f"HA_SC_{gate_name}_sim80", pool_hi),
            ):
                if use_pool.empty and pool_tag.endswith("sim80"):
                    continue
                for top_n in top_n_list:
                    m, n_used, _ = train_eval_ha(
                        data,
                        use_pool,
                        top_n,
                        device=device,
                        seed=seed + (hash(mode + gate_name) % 1000),
                        epochs=epochs,
                    )
                    coverage = n_used / data["n_total"]
                    print(
                        f"    {pool_tag:28s} topN={top_n:<5d} used={n_used:<5d} "
                        f"cov={coverage:.3f} OA={m['OA']:.4f} BE={m['Boundary Error']}"
                    )
                    add_row(
                        pool_tag,
                        mode,
                        top_n,
                        m,
                        extra={
                            "n_prototypes": int(len(proto_pos)),
                            "require_margin": require_margin,
                            "margin_thresh": thr if require_margin else None,
                            "coverage": coverage,
                            "loo_n": loo["loo_n"],
                            "loo_correct": loo["loo_correct"],
                            "loo_precision": loo["loo_precision"],
                            "pool_size": int(len(use_pool)),
                            "gate": gate_name,
                        },
                    )

        # Also evaluate classic PE-sized amplification once (top all PE pool)
        if not pe_pool.empty:
            m, n_used, _ = train_eval_ha(
                data, pe_pool, len(pe_pool), device=device, seed=seed + 7, epochs=epochs
            )
            add_row(
                "HA_PE_Boundary",
                mode,
                int(len(pe_pool)),
                m,
                extra={
                    "n_prototypes": int(len(proto_pos)),
                    "require_margin": True,
                    "margin_thresh": MARGIN_THRESH,
                    "coverage": n_used / data["n_total"],
                    "loo_n": loo["loo_n"],
                    "loo_correct": loo["loo_correct"],
                    "loo_precision": loo["loo_precision"],
                    "pool_size": int(len(pe_pool)),
                    "gate": "pe_centric",
                },
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Build city-best table, scale curve, and markdown discussion."""
    ha = df[df["Method"].astype(str).str.startswith("HA_SC_")].copy()
    if ha.empty:
        return pd.DataFrame(), pd.DataFrame(), "No HA_SC rows."

    # Prefer non-sim80 methods for primary best table
    primary = ha[~ha["Method"].astype(str).str.contains("sim80")].copy()
    if primary.empty:
        primary = ha

    primary_sorted = primary.sort_values(
        ["City", "ProtoMode", "Boundary Error", "OA"],
        ascending=[True, True, True, False],
    )
    best = primary_sorted.groupby(["City", "ProtoMode"], as_index=False).first()

    curve = (
        ha.groupby(["Method", "ProtoMode", "TopN"], as_index=False)
        .agg(
            Mean_OA=("OA", "mean"),
            Std_OA=("OA", "std"),
            Mean_BE=("Boundary Error", "mean"),
            Std_BE=("Boundary Error", "std"),
            Mean_Amplified=("n_amplified", "mean"),
            Mean_Coverage=("coverage", "mean"),
            Mean_LOO=("loo_precision", "mean"),
            Mean_Agree=("mean_agreement", "mean"),
            Mean_Pool=("pool_size", "mean"),
        )
        .sort_values(["ProtoMode", "Method", "TopN"])
    )

    base = df[df["Method"].isin(["Baseline_RF", "BAMS150_RF", "BAMS150_MLP"])]
    base_mean = base.groupby("Method", as_index=False).agg(
        Mean_OA=("OA", "mean"), Mean_BE=("Boundary Error", "mean")
    )

    lines = []
    lines.append("# Human Amplification Experiment Summary\n")
    lines.append("## Question\n")
    lines.append(
        "Can high-quality BAMS human corrections (~150/city) be selectively amplified "
        "to update a large fraction of ~15k candidates via sample-centric prototype "
        "propagation (each candidate queries nearest human prototypes)?\n"
    )
    lines.append("## Baselines (mean over cities)\n")
    lines.append(base_mean.to_string(index=False))
    lines.append("\n\n## Best sample-centric HA per city / prototype mode\n")
    cols = [
        c
        for c in [
            "City",
            "ProtoMode",
            "Method",
            "TopN",
            "n_amplified",
            "coverage",
            "OA",
            "Boundary Error",
            "loo_precision",
            "mean_agreement",
            "pool_size",
        ]
        if c in best.columns
    ]
    lines.append(best[cols].to_string(index=False))
    lines.append("\n\n## Scale curve (mean over cities)\n")
    lines.append(curve.to_string(index=False))
    lines.append("\n\n## Interpretation checklist\n")
    lines.append(
        "- Compare against BAMS150_MLP (same model), not only RF.\n"
        "- Coverage can now reach thousands (sample-centric), unlike PE (~tens/hundreds).\n"
        "- Look for TopN sweet spot: OA/BE improve then plateau or degrade.\n"
        "- corrected vs all_bams: do true corrections make better seeds?\n"
        "- boundary_* vs unconstrained: is gating necessary at large coverage?\n"
        "- sim80 rows: quality-first subset of the same pool.\n"
    )
    return best, curve, "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Human Amplification experiments")
    p.add_argument("--cities", nargs="+", default=CITIES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--topn",
        nargs="+",
        type=int,
        default=TOP_N_LIST,
        help="Expansion sizes to sweep",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(PROTO_MODES),
        choices=list(PROTO_MODES),
    )
    p.add_argument(
        "--gates",
        nargs="+",
        default=["boundary_015", "boundary_025", "unconstrained"],
        choices=[g[0] for g in GATE_MODES],
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke: Wuhan-oriented settings, fewer configs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    top_n_list = args.topn
    modes = tuple(args.modes)
    epochs = args.epochs
    gate_lookup = {g[0]: g for g in GATE_MODES}
    gate_modes = tuple(gate_lookup[name] for name in args.gates)

    if args.quick:
        top_n_list = [0, 500, 2000, 8000]
        modes = ("corrected", "all_bams")
        gate_modes = tuple(gate_lookup[n] for n in ["boundary_025", "unconstrained"])
        epochs = min(epochs, 50)
        print(f"[quick] topn={top_n_list} modes={modes} gates={[g[0] for g in gate_modes]} epochs={epochs}")

    all_rows = []
    for city in args.cities:
        df_city = run_city(
            city,
            device=device,
            epochs=epochs,
            top_n_list=top_n_list,
            proto_modes=modes,
            seed=args.seed,
            gate_modes=gate_modes,
        )
        (OUT_ROOT / city).mkdir(parents=True, exist_ok=True)
        df_city.to_csv(OUT_ROOT / city / "results.csv", index=False)
        all_rows.append(df_city)

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(OUT_ROOT / "all_results.csv", index=False)

    best, curve, md = summarize(results)
    best.to_csv(OUT_ROOT / "best_per_city.csv", index=False)
    curve.to_csv(OUT_ROOT / "scale_curve.csv", index=False)
    (OUT_ROOT / "SUMMARY.md").write_text(md, encoding="utf-8")

    meta = {
        "cities": args.cities,
        "epochs": epochs,
        "seed": args.seed,
        "top_n_list": top_n_list,
        "modes": list(modes),
        "gates": [g[0] for g in gate_modes],
        "k_pool": K_POOL,
        "n_proto_neighbors": N_PROTO_NEIGHBORS,
        "margin_thresh": MARGIN_THRESH,
        "min_agreement": MIN_AGREEMENT,
    }
    (OUT_ROOT / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + md)
    print(f"\nWrote results to {OUT_ROOT}")


if __name__ == "__main__":
    main()
