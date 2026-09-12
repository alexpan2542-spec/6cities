#!/usr/bin/env python3
"""
Phase 1: Nanjing failure diagnostics (quantitative).

Produces cross-city fingerprints, BAMS/PE help-hurt flip autopsy, and
difficulty metrics. Outputs -> data2/nanjing_phase1_diagnostics/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_pkgs"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA2 = ROOT / "data2"
OUT_DIR = DATA2 / "nanjing_phase1_diagnostics"
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
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256


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


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.nan_to_num(np.diag(cm) / cm.sum(axis=1), nan=0.0)
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "BE": int(cm[0, 1] + cm[1, 0]),
        "C1_to_C2": int(cm[0, 1]),
        "C2_to_C1": int(cm[1, 0]),
        "C1_recall": float(recall[0]),
        "C2_recall": float(recall[1]),
        "C3_recall": float(recall[2]),
        "cm": cm,
    }


def flip_stats(y_true, pred_a, pred_b) -> dict:
    """Compare method B vs A on test: help/hurt relative to GT."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    flipped = pred_a != pred_b
    n_flip = int(flipped.sum())
    help_gt = int(((pred_b == y_true) & (pred_a != y_true) & flipped).sum())
    hurt_gt = int(((pred_b != y_true) & (pred_a == y_true) & flipped).sum())
    both_wrong = int(((pred_b != y_true) & (pred_a != y_true) & flipped).sum())
    # boundary-related flips (true label in {0,1} or either pred in C1/C2 swap)
    true_c12 = np.isin(y_true, [0, 1])
    help_c12 = int(((pred_b == y_true) & (pred_a != y_true) & flipped & true_c12).sum())
    hurt_c12 = int(((pred_b != y_true) & (pred_a == y_true) & flipped & true_c12).sum())
    ratio = help_gt / hurt_gt if hurt_gt > 0 else float("inf") if help_gt > 0 else float("nan")
    return {
        "n_flip": n_flip,
        "help_gt": help_gt,
        "hurt_gt": hurt_gt,
        "both_wrong_flip": both_wrong,
        "help_c12": help_c12,
        "hurt_c12": hurt_c12,
        "help_hurt_ratio": float(ratio) if math.isfinite(ratio) else ratio,
        "net_help": help_gt - hurt_gt,
    }


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371000.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bams_spatial_nn_m(lon: np.ndarray, lat: np.ndarray) -> dict:
    n = len(lon)
    if n < 2:
        return {"bams_mean_nn_m": float("nan"), "bams_median_nn_m": float("nan")}
    nn_d = []
    for i in range(n):
        d = haversine_m(lon[i], lat[i], lon, lat)
        d[i] = np.inf
        nn_d.append(float(np.min(d)))
    nn_d = np.asarray(nn_d)
    return {
        "bams_mean_nn_m": float(nn_d.mean()),
        "bams_median_nn_m": float(np.median(nn_d)),
    }


def class_overlap_metrics(X_s: np.ndarray, y: np.ndarray) -> dict:
    """C1 vs C2 separability in standardized feature space."""
    c1 = X_s[y == 0]
    c2 = X_s[y == 1]
    if len(c1) == 0 or len(c2) == 0:
        return {
            "c12_centroid_cos": float("nan"),
            "c12_mean_cross_cos": float("nan"),
            "c12_fisher_ndvi": float("nan"),
        }
    m1 = c1.mean(axis=0)
    m2 = c2.mean(axis=0)
    # cosine similarity of centroids (higher = more overlap / harder)
    cos_cent = float(
        np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-12)
    )
    # sample mean cross cosine (subsample for speed)
    rng = np.random.default_rng(0)
    s1 = c1[rng.choice(len(c1), size=min(500, len(c1)), replace=False)]
    s2 = c2[rng.choice(len(c2), size=min(500, len(c2)), replace=False)]
    s1n = s1 / (np.linalg.norm(s1, axis=1, keepdims=True) + 1e-12)
    s2n = s2 / (np.linalg.norm(s2, axis=1, keepdims=True) + 1e-12)
    # average max similarity to other class (proxy for overlap)
    cross = s1n @ s2n.T
    mean_cross = float(0.5 * (cross.max(axis=1).mean() + cross.max(axis=0).mean()))

    # Fisher on NDVI_mean index within FEATURES_3X3
    ndvi_i = FEATURES_3X3.index("NDVI_mean")
    v1, v2 = c1[:, ndvi_i], c2[:, ndvi_i]
    fisher = float(
        (v1.mean() - v2.mean()) ** 2 / (v1.var() + v2.var() + 1e-12)
    )
    return {
        "c12_centroid_cos": cos_cent,
        "c12_mean_cross_cos": mean_cross,
        "c12_fisher_ndvi": fisher,
    }


def load_city(city: str) -> dict:
    city_root = DATA2 / city
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")
    orig = pd.read_csv(city_root / "01_original" / f"{city}_WC_Samples_15000.csv")

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

    # coords
    orig_ids = orig["Original_ID"].astype(int).to_numpy()
    lon_map = dict(zip(orig_ids, orig["lon"].astype(float).to_numpy()))
    lat_map = dict(zip(orig_ids, orig["lat"].astype(float).to_numpy()))
    lon_all = np.asarray([lon_map[int(i)] for i in ids], dtype=float)
    lat_all = np.asarray([lat_map[int(i)] for i in ids], dtype=float)

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
        "lon_all": lon_all,
        "lat_all": lat_all,
    }


def static_city_stats(data: dict) -> dict:
    city = data["city"]
    y = data["y_all"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    corrected = data["corrected_mask"]
    old = y[bams_pos]
    new = bams_label

    n_corr = int(corrected.sum())
    c12_corr = int(
        (
            ((old == 0) & (new == 1) & corrected)
            | ((old == 1) & (new == 0) & corrected)
        ).sum()
    )
    overlap = class_overlap_metrics(data["X_all_s"], y)
    spat = bams_spatial_nn_m(data["lon_all"][bams_pos], data["lat_all"][bams_pos])

    # corrected BAMS spatial nn among corrected only
    if n_corr >= 2:
        spat_c_raw = bams_spatial_nn_m(
            data["lon_all"][bams_pos[corrected]],
            data["lat_all"][bams_pos[corrected]],
        )
        spat_c = {
            "bams_corr_mean_nn_m": spat_c_raw["bams_mean_nn_m"],
            "bams_corr_median_nn_m": spat_c_raw["bams_median_nn_m"],
        }
    else:
        spat_c = {
            "bams_corr_mean_nn_m": float("nan"),
            "bams_corr_median_nn_m": float("nan"),
        }

    counts = Counter(y.tolist())
    return {
        "City": city,
        "n_samples": int(len(y)),
        "n_C1": int(counts[0]),
        "n_C2": int(counts[1]),
        "n_C3": int(counts[2]),
        "pct_C2": float(counts[1] / len(y)),
        "bams_n": int(len(bams_pos)),
        "bams_n_corrected": n_corr,
        "bams_correction_rate": float(n_corr / len(bams_pos)),
        "bams_c12_correction_n": c12_corr,
        "bams_c12_among_corrected": float(c12_corr / n_corr) if n_corr else float("nan"),
        **overlap,
        **spat,
        **spat_c,
    }


def pick_best_pe(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["BE"], -c["m"]["OA"]))[0]


def run_city(city: str, device: str, seeds: list[int], epochs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    print("\n" + "=" * 72)
    print(f"CITY: {city}")
    print("=" * 72)
    data = load_city(city)
    static = static_city_stats(data)
    print(
        f"  corrected={static['bams_n_corrected']} "
        f"({static['bams_correction_rate']:.1%}) "
        f"C12frac={static['bams_c12_among_corrected']:.1%} "
        f"centroid_cos={static['c12_centroid_cos']:.3f}"
    )

    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    y_bams_full = data["y_bams_full"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    ids = data["ids"]

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True

    seed_rows = []
    flip_rows = []
    # detailed Nanjing flips for seed 0 best PE
    detail_rows = []

    for seed in seeds:
        set_all_seeds(seed)
        y_true = y_all[idx_test]
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
        X_bams_s = X_all_s[bams_pos]

        rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_base.fit(X_train, y_all[idx_train])
        pred_base = rf_base.predict(X_test).astype(np.int64)
        m_base = metrics(y_true, pred_base)

        # margin pool from RF (for pool-size fingerprint; PE still uses MLP teacher)
        proba_all = rf_base.predict_proba(X_all)
        # align columns to classes 0,1,2
        order = np.argsort(proba_all, axis=1)
        top1p = proba_all[np.arange(len(proba_all)), order[:, -1]]
        top2p = proba_all[np.arange(len(proba_all)), order[:, -2]]
        margin_rf = top1p - top2p
        low_margin_frac = float((margin_rf < MARGIN_THRESH).mean())
        low_margin_c12 = float(
            ((margin_rf < MARGIN_THRESH) & np.isin(rf_base.predict(X_all), [0, 1])).mean()
        )

        rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_bams.fit(X_train, y_bams_full[idx_train])
        pred_bams = rf_bams.predict(X_test).astype(np.int64)
        m_bams = metrics(y_true, pred_bams)
        flip_bams = flip_stats(y_true, pred_base, pred_bams)

        # PE teacher + expand
        teacher = train_mlp(
            np.concatenate([X_train_s, X_bams_s], axis=0),
            np.concatenate([y_all[idx_train], bams_label], axis=0),
            device=device,
            seed=seed,
            epochs=epochs,
        )
        _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)
        low_margin_mlp = float((margin_pool < MARGIN_THRESH).mean())

        pe_cands = []
        for k in K_LIST:
            exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
            keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
            exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
            agree_class = float((exp_lab == y_all[exp_idx]).mean()) if len(exp_idx) else float("nan")
            X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
            y_pe = np.concatenate([y_all[idx_train], bams_label, exp_lab], axis=0)
            w_pe = np.concatenate(
                [
                    np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, np.float32),
                    np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
                ]
            )
            st = train_mlp(
                X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k, epochs=epochs
            )
            pred_pe, _, _ = predict_proba_mlp(st, X_test_s, device=device)
            m_pe = metrics(y_true, pred_pe)
            pe_cands.append(
                {
                    "k": k,
                    "pred": pred_pe,
                    "m": m_pe,
                    "n_exp": int(len(exp_idx)),
                    "exp_agree_class": agree_class,
                    "exp_idx": exp_idx,
                    "exp_lab": exp_lab,
                }
            )
            print(
                f"  [seed={seed}] PE K={k:<2d} exp={len(exp_idx):<4d} "
                f"agreeClass={agree_class if len(exp_idx) else float('nan'):.3f} "
                f"OA={m_pe['OA']:.4f} BE={m_pe['BE']}"
            )

        best_pe = pick_best_pe(pe_cands)
        flip_pe = flip_stats(y_true, pred_base, best_pe["pred"])

        print(
            f"  [seed={seed}] Baseline OA={m_base['OA']:.4f} BE={m_base['BE']} | "
            f"BAMS OA={m_bams['OA']:.4f} BE={m_bams['BE']} "
            f"help/hurt={flip_bams['help_gt']}/{flip_bams['hurt_gt']} | "
            f"PE OA={best_pe['m']['OA']:.4f} BE={best_pe['m']['BE']} "
            f"help/hurt={flip_pe['help_gt']}/{flip_pe['hurt_gt']} "
            f"exp={best_pe['n_exp']} K={best_pe['k']}"
        )

        seed_rows.append(
            {
                "City": city,
                "Seed": seed,
                "Base_OA": m_base["OA"],
                "Base_BE": m_base["BE"],
                "Base_C2_recall": m_base["C2_recall"],
                "BAMS_OA": m_bams["OA"],
                "BAMS_BE": m_bams["BE"],
                "BAMS_dOA": m_bams["OA"] - m_base["OA"],
                "BAMS_dBE": m_bams["BE"] - m_base["BE"],
                "PE_OA": best_pe["m"]["OA"],
                "PE_BE": best_pe["m"]["BE"],
                "PE_dOA": best_pe["m"]["OA"] - m_base["OA"],
                "PE_dBE": best_pe["m"]["BE"] - m_base["BE"],
                "PE_BestK": best_pe["k"],
                "PE_n_exp": best_pe["n_exp"],
                "PE_exp_agree_class": best_pe["exp_agree_class"],
                "low_margin_frac_rf": low_margin_frac,
                "low_margin_c12_frac_rf": low_margin_c12,
                "low_margin_frac_mlp": low_margin_mlp,
            }
        )

        for method, fl, extra in [
            ("BAMS150", flip_bams, {}),
            (
                "PE_Boundary",
                flip_pe,
                {"BestK": best_pe["k"], "n_exp": best_pe["n_exp"]},
            ),
        ]:
            flip_rows.append(
                {
                    "City": city,
                    "Seed": seed,
                    "Method": method,
                    **fl,
                    **extra,
                }
            )

        # Export flip detail for Nanjing (all seeds) for Phase-2 sampling
        if city == "Nanjing":
            test_ids = ids[idx_test]
            test_lon = data["lon_all"][idx_test]
            test_lat = data["lat_all"][idx_test]
            for name, pred_new in [("BAMS150", pred_bams), ("PE_Boundary", best_pe["pred"])]:
                flipped = pred_base != pred_new
                for i in np.where(flipped)[0]:
                    pa, pb, yt = int(pred_base[i]), int(pred_new[i]), int(y_true[i])
                    if pb == yt and pa != yt:
                        tag = "help_gt"
                    elif pb != yt and pa == yt:
                        tag = "hurt_gt"
                    else:
                        tag = "both_wrong"
                    detail_rows.append(
                        {
                            "City": city,
                            "Seed": seed,
                            "Method": name,
                            "Original_ID": int(test_ids[i]),
                            "lon": float(test_lon[i]),
                            "lat": float(test_lat[i]),
                            "GT_Class": yt + 1,
                            "pred_base": pa + 1,
                            "pred_new": pb + 1,
                            "flip_tag": tag,
                            "BestK": best_pe["k"] if name == "PE_Boundary" else np.nan,
                            "n_exp": best_pe["n_exp"] if name == "PE_Boundary" else np.nan,
                        }
                    )

            # also dump PE expanded points for seed (best K)
            for j, lab in zip(best_pe["exp_idx"].tolist(), best_pe["exp_lab"].tolist()):
                detail_rows.append(
                    {
                        "City": city,
                        "Seed": seed,
                        "Method": "PE_ExpandedPool",
                        "Original_ID": int(ids[j]),
                        "lon": float(data["lon_all"][j]),
                        "lat": float(data["lat_all"][j]),
                        "GT_Class": int(y_all[j]) + 1,
                        "pred_base": np.nan,
                        "pred_new": int(lab) + 1,
                        "flip_tag": "exp_agree"
                        if int(lab) == int(y_all[j])
                        else "exp_conflict_class",
                        "BestK": best_pe["k"],
                        "n_exp": best_pe["n_exp"],
                    }
                )

    return pd.DataFrame(seed_rows), pd.DataFrame(flip_rows), static, pd.DataFrame(detail_rows)


def write_summary(
    static_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    flip_df: pd.DataFrame,
) -> str:
    city_mean = (
        seed_df.groupby("City", sort=False)
        .agg(
            Base_OA=("Base_OA", "mean"),
            Base_BE=("Base_BE", "mean"),
            Base_C2_recall=("Base_C2_recall", "mean"),
            BAMS_OA=("BAMS_OA", "mean"),
            BAMS_dOA=("BAMS_dOA", "mean"),
            BAMS_dBE=("BAMS_dBE", "mean"),
            PE_OA=("PE_OA", "mean"),
            PE_dOA=("PE_dOA", "mean"),
            PE_dBE=("PE_dBE", "mean"),
            PE_n_exp=("PE_n_exp", "mean"),
            PE_exp_agree_class=("PE_exp_agree_class", "mean"),
            low_margin_frac_mlp=("low_margin_frac_mlp", "mean"),
            low_margin_frac_rf=("low_margin_frac_rf", "mean"),
        )
        .reset_index()
    )
    merged = static_df.merge(city_mean, on="City", how="left")

    flip_mean = (
        flip_df.groupby(["City", "Method"], sort=False)
        .agg(
            help_gt=("help_gt", "mean"),
            hurt_gt=("hurt_gt", "mean"),
            net_help=("net_help", "mean"),
            help_hurt_ratio=("help_hurt_ratio", "mean"),
            n_flip=("n_flip", "mean"),
            help_c12=("help_c12", "mean"),
            hurt_c12=("hurt_c12", "mean"),
        )
        .reset_index()
    )

    # rank helpers
    def rank_desc(s):
        return s.rank(ascending=False, method="min").astype(int)

    def rank_asc(s):
        return s.rank(ascending=True, method="min").astype(int)

    ranks = pd.DataFrame({"City": merged["City"]})
    ranks["rank_hardest_BaseOA"] = rank_asc(merged["Base_OA"])
    ranks["rank_worst_BaseBE"] = rank_desc(merged["Base_BE"])
    ranks["rank_worst_BAMS_dOA"] = rank_asc(merged["BAMS_dOA"])
    ranks["rank_largest_low_margin"] = rank_desc(merged["low_margin_frac_mlp"])
    ranks["rank_most_PE_exp"] = rank_desc(merged["PE_n_exp"])
    ranks["rank_lowest_exp_agree"] = rank_asc(merged["PE_exp_agree_class"])
    ranks["rank_highest_c12_overlap"] = rank_desc(merged["c12_centroid_cos"])
    ranks["rank_lowest_fisher"] = rank_asc(merged["c12_fisher_ndvi"])

    lines = []
    lines.append("# Phase 1: Nanjing Failure Diagnostics\n")
    lines.append("## 1. Cross-city fingerprint (means over seeds)\n")
    show_cols = [
        "City",
        "Base_OA",
        "Base_BE",
        "Base_C2_recall",
        "bams_n_corrected",
        "bams_correction_rate",
        "BAMS_dOA",
        "BAMS_dBE",
        "PE_dOA",
        "PE_dBE",
        "PE_n_exp",
        "PE_exp_agree_class",
        "low_margin_frac_mlp",
        "c12_centroid_cos",
        "c12_fisher_ndvi",
        "bams_mean_nn_m",
    ]
    lines.append(merged[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lines.append("\n\n## 2. Where Nanjing ranks (1 = most extreme in the bad direction)\n")
    lines.append(ranks.to_string(index=False))

    lines.append("\n\n## 3. Help / Hurt flips vs Baseline (test set, mean over seeds)\n")
    lines.append(flip_mean.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Nanjing callout
    nj = merged[merged["City"] == "Nanjing"].iloc[0]
    nj_bams = flip_mean[(flip_mean["City"] == "Nanjing") & (flip_mean["Method"] == "BAMS150")].iloc[0]
    nj_pe = flip_mean[(flip_mean["City"] == "Nanjing") & (flip_mean["Method"] == "PE_Boundary")].iloc[0]
    lines.append("\n\n## 4. Nanjing callout\n")
    lines.append(
        f"- Baseline OA={nj['Base_OA']:.4f}, BE={nj['Base_BE']:.1f}, "
        f"C2 recall={nj['Base_C2_recall']:.3f}\n"
        f"- BAMS ΔOA={nj['BAMS_dOA']*100:.2f} pp, ΔBE={nj['BAMS_dBE']:.1f}; "
        f"help/hurt={nj_bams['help_gt']:.1f}/{nj_bams['hurt_gt']:.1f} "
        f"(ratio={nj_bams['help_hurt_ratio']:.2f})\n"
        f"- PE ΔOA={nj['PE_dOA']*100:.2f} pp, ΔBE={nj['PE_dBE']:.1f}; "
        f"help/hurt={nj_pe['help_gt']:.1f}/{nj_pe['hurt_gt']:.1f} "
        f"(ratio={nj_pe['help_hurt_ratio']:.2f}); "
        f"mean expanded={nj['PE_n_exp']:.1f}, "
        f"exp↔Class agree={nj['PE_exp_agree_class']:.3f}\n"
        f"- Low-margin pool (MLP)={nj['low_margin_frac_mlp']:.3f}; "
        f"C1–C2 centroid cos={nj['c12_centroid_cos']:.3f}; "
        f"NDVI Fisher={nj['c12_fisher_ndvi']:.3f}\n"
    )
    lines.append(
        "\n## 5. Reading guide\n"
        "- If Nanjing ranks #1 on worst BAMS_dOA and hurt≫help → H1 (GT conflict) supported.\n"
        "- If also #1 on c12 overlap / low Fisher / large low-margin pool → H3 (hardness) supported.\n"
        "- PE_exp_agree_class low means expansion disagrees with eval Class (conflict amplification).\n"
        "- Phase 2 should manually audit hurt_gt and exp_conflict_class points.\n"
    )
    return "\n".join(lines), merged, flip_mean, ranks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", default=CITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds
    epochs = args.epochs
    if args.quick:
        seeds = [0]
        epochs = min(epochs, 40)
        print("[quick]", seeds, epochs)
    print("device:", device)
    print("cities:", args.cities)

    static_rows = []
    seed_parts = []
    flip_parts = []
    detail_parts = []

    for city in args.cities:
        seed_df, flip_df, static, detail_df = run_city(
            city, device=device, seeds=seeds, epochs=epochs
        )
        static_rows.append(static)
        seed_parts.append(seed_df)
        flip_parts.append(flip_df)
        if len(detail_df):
            detail_parts.append(detail_df)
        (OUT_DIR / city).mkdir(parents=True, exist_ok=True)
        seed_df.to_csv(OUT_DIR / city / "seed_metrics.csv", index=False)
        flip_df.to_csv(OUT_DIR / city / "flip_stats.csv", index=False)

    static_df = pd.DataFrame(static_rows)
    seed_df = pd.concat(seed_parts, ignore_index=True)
    flip_df = pd.concat(flip_parts, ignore_index=True)
    static_df.to_csv(OUT_DIR / "static_fingerprint.csv", index=False)
    seed_df.to_csv(OUT_DIR / "seed_metrics_all.csv", index=False)
    flip_df.to_csv(OUT_DIR / "flip_stats_all.csv", index=False)

    if detail_parts:
        detail = pd.concat(detail_parts, ignore_index=True)
        detail.to_csv(OUT_DIR / "Nanjing_flip_and_expand_detail.csv", index=False)
        # Phase-2 sampling sheets
        hurt = detail[
            (detail["Method"].isin(["BAMS150", "PE_Boundary"]))
            & (detail["flip_tag"] == "hurt_gt")
        ]
        help_ = detail[
            (detail["Method"].isin(["BAMS150", "PE_Boundary"]))
            & (detail["flip_tag"] == "help_gt")
        ]
        exp_conflict = detail[
            (detail["Method"] == "PE_ExpandedPool")
            & (detail["flip_tag"] == "exp_conflict_class")
        ]
        hurt.to_csv(OUT_DIR / "phase2_candidates_hurt_gt.csv", index=False)
        help_.to_csv(OUT_DIR / "phase2_candidates_help_gt.csv", index=False)
        exp_conflict.to_csv(OUT_DIR / "phase2_candidates_exp_conflict.csv", index=False)

    md, merged, flip_mean, ranks = write_summary(static_df, seed_df, flip_df)
    merged.to_csv(OUT_DIR / "city_comparison.csv", index=False)
    flip_mean.to_csv(OUT_DIR / "flip_summary.csv", index=False)
    ranks.to_csv(OUT_DIR / "nanjing_ranks.csv", index=False)
    (OUT_DIR / "SUMMARY.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "cities": args.cities,
                "seeds": seeds,
                "epochs": epochs,
                "k_list": K_LIST,
                "margin_thresh": MARGIN_THRESH,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
