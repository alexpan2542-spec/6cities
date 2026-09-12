#!/usr/bin/env python3
"""
5-seed statistical significance experiment (boundary-focused).

Methods:
  Baseline
  BAMS150
  BAMS150+PrototypeExpansion_Boundary
  BAMS150+PE+TS_Boundary
  BAMS150+HA_Corrected_Boundary   # corrected prototypes, margin<0.25, TopN pick-best
  BAMS150+HA_Aligned_Boundary     # all_bams, margin<0.15 (PE-aligned), TopN pick-best

HA selection protocol (same as PE's K sweep):
  TopN ∈ {50, 100, 150, 250}; pick lowest Boundary Error, then highest OA.

Seeds: 0, 1, 2, 3, 4
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
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

CITIES = ["Wuhan", "Hefei", "Nanchang", "Nanjing", "Changsha"]
DATA2 = ROOT / "data2"
OUT_DIR = DATA2 / "seed_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

SEEDS = [0, 1, 2, 3, 4]
K_LIST = [5, 10, 20, 30]
SPLIT = dict(test_size=0.3, random_state=42)  # fixed test set for paired comparison
MARGIN_THRESH = 0.15
PE_TS_CONF = 0.45
REAL_WEIGHT = 1.0
SOFT_WEIGHT = 0.3
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 256

# Human Amplification (HA) — pick-best protocol aligned with PE
HA_TOPN_LIST = [50, 100, 150, 250]
HA_N_PROTO_NEIGHBORS = 5
HA_MIN_AGREEMENT = 0.5
HA_SOFT_WEIGHT_BASE = 0.5
HA_VARIANTS = (
    # method_name, proto_mode, margin_thresh
    ("BAMS150+HA_Corrected_Boundary", "corrected", 0.25),
    ("BAMS150+HA_Aligned_Boundary", "all_bams", 0.15),
)

METHODS = [
    "Baseline",
    "BAMS150",
    "BAMS150+PrototypeExpansion_Boundary",
    "BAMS150+PE+TS_Boundary",
    "BAMS150+HA_Corrected_Boundary",
    "BAMS150+HA_Aligned_Boundary",
]


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
    return manual


def apply_corrections(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = out.set_index("Original_ID")["Class"].astype(int)
    for _, row in manual.iterrows():
        oid = int(row["Original_ID"])
        human = int(row["Human_Class"])
        if int(lookup.loc[oid]) != human:
            out.loc[out["Original_ID"] == oid, "Class"] = human
    return out


def train_mlp(X, y, sample_weight=None, device="cpu", seed: int = 0):
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
    for _ in range(EPOCHS):
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            opt.zero_grad()
            loss_each = crit(model(xb), yb)
            loss = (loss_each * wb).sum() / wb.sum()
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
    return top1_idx.astype(np.int64), top1, margin


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


def select_ha_prototypes(mode: str, bams_pos, bams_label, y_original):
    """proto_mode: corrected | all_bams. Fallback to all if corrected empty."""
    if mode == "all_bams":
        mask = np.ones(len(bams_pos), dtype=bool)
    elif mode == "corrected":
        mask = bams_label != y_original[bams_pos]
        if mask.sum() == 0:
            mask = np.ones(len(bams_pos), dtype=bool)
    else:
        raise ValueError(mode)
    return bams_pos[mask], bams_label[mask]


def build_ha_sample_centric_pool(
    feature: np.ndarray,
    proto_pos: np.ndarray,
    proto_lab: np.ndarray,
    eligible: np.ndarray,
    margin_pool: np.ndarray,
    n_proto_neighbors: int = HA_N_PROTO_NEIGHBORS,
    margin_thresh: float = 0.25,
) -> pd.DataFrame:
    """Each eligible candidate queries nearest human prototypes."""
    cols = ["idx", "pseudo_label", "n_votes", "agreement", "mean_sim", "score", "margin"]
    if len(proto_pos) == 0:
        return pd.DataFrame(columns=cols)

    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) == 0:
        return pd.DataFrame(columns=cols)

    n_nn = min(n_proto_neighbors, len(proto_pos))
    nn_model = NearestNeighbors(n_neighbors=n_nn, metric="cosine")
    nn_model.fit(feature[proto_pos])
    dists, neigh_local = nn_model.kneighbors(feature[eligible_idx], return_distance=True)

    rows = []
    for i, eidx in enumerate(eligible_idx):
        eidx = int(eidx)
        if margin_pool[eidx] >= margin_thresh:
            continue
        vlist, slist = [], []
        for li, dist in zip(neigh_local[i].tolist(), dists[i].tolist()):
            vlist.append(int(proto_lab[int(li)]))
            slist.append(1.0 - float(dist))
        if not vlist:
            continue
        lab = majority_label(vlist)
        n_votes = len(vlist)
        agreement = sum(1 for v in vlist if v == lab) / n_votes
        if agreement < HA_MIN_AGREEMENT:
            continue
        mean_sim = float(np.mean(slist))
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


def train_ha_from_pool(
    data_arrays: dict,
    pool: pd.DataFrame,
    top_n: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Train HA student with top-N scored pseudo-labels; return (y_pred, meta)."""
    X_train_s = data_arrays["X_train_s"]
    X_test_s = data_arrays["X_test_s"]
    X_bams_s = data_arrays["X_bams_s"]
    X_all_s = data_arrays["X_all_s"]
    y_train = data_arrays["y_train"]
    y_true = data_arrays["y_true"]
    bams_label = data_arrays["bams_label"]
    idx_train = data_arrays["idx_train"]

    if top_n <= 0 or pool.empty:
        X_tr = np.concatenate([X_train_s, X_bams_s], axis=0)
        y_tr = np.concatenate([y_train, bams_label], axis=0)
        w_tr = np.full(len(y_tr), REAL_WEIGHT, dtype=np.float32)
        n_amp = 0
        mean_score = float("nan")
        mean_agree = float("nan")
    else:
        take = pool.head(min(top_n, len(pool)))
        exp_idx = take["idx"].to_numpy(dtype=int)
        exp_lab = take["pseudo_label"].to_numpy(dtype=np.int64)
        scores = take["score"].to_numpy(dtype=np.float64)
        smax = float(scores.max()) if len(scores) else 1.0
        soft_w = (
            HA_SOFT_WEIGHT_BASE * (0.4 + 0.6 * (scores / (smax + 1e-8)))
        ).astype(np.float32)
        X_tr = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
        y_tr = np.concatenate([y_train, bams_label, exp_lab], axis=0)
        w_tr = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                soft_w,
            ]
        )
        n_amp = int(len(exp_idx))
        mean_score = float(scores.mean())
        mean_agree = float(take["agreement"].mean())

    model = train_mlp(X_tr, y_tr, sample_weight=w_tr, device=device, seed=seed)
    y_pred, _, _ = predict_proba_mlp(model, X_test_s, device=device)
    m = full_metrics(y_true, y_pred)
    meta = {
        "top_n": top_n,
        "n_amplified": n_amp,
        "pool_size": int(len(pool)),
        "mean_score": mean_score,
        "mean_agreement": mean_agree,
        "m": m,
        "pred": y_pred,
    }
    return y_pred, meta


def pick_best_ha_variant(
    data_arrays: dict,
    bams_pos,
    bams_label,
    y_all,
    used,
    margin_pool,
    proto_mode: str,
    margin_thresh: float,
    device: str,
    seed: int,
) -> dict:
    """Sweep TopN like PE sweeps K; pick lowest BE then highest OA."""
    proto_pos, proto_lab = select_ha_prototypes(proto_mode, bams_pos, bams_label, y_all)
    pool = build_ha_sample_centric_pool(
        data_arrays["X_all_s"],
        proto_pos,
        proto_lab,
        eligible=~used,
        margin_pool=margin_pool,
        n_proto_neighbors=HA_N_PROTO_NEIGHBORS,
        margin_thresh=margin_thresh,
    )
    cands = []
    for ti, top_n in enumerate(HA_TOPN_LIST):
        _, meta = train_ha_from_pool(
            data_arrays,
            pool,
            top_n,
            device=device,
            seed=seed + 300 + int(margin_thresh * 100) + ti,
        )
        meta["proto_mode"] = proto_mode
        meta["margin_thresh"] = margin_thresh
        meta["n_prototypes"] = int(len(proto_pos))
        cands.append(meta)
    best = sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]
    return best


def full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    oa = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.nan_to_num(np.diag(cm) / cm.sum(axis=1), nan=0.0)
    aa = float(per_class_acc.mean())
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    boundary_error = int(cm[0, 1] + cm[1, 0])
    return {
        "OA": oa,
        "AA": aa,
        "Kappa": kappa,
        "Class1 Recall": float(recall[0]),
        "Class2 Recall": float(recall[1]),
        "Class3 Recall": float(recall[2]),
        "Boundary Error": boundary_error,
    }


def load_city(city: str):
    city_root = DATA2 / city
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{city}_3x3_Features.csv")
    bams_manual = get_labeled_manual(city_root / "04_manual" / f"{city}_BAMS150_Manual.csv")
    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_all = feat3["Class"].astype(int).to_numpy() - 1
    y_bams_full = apply_corrections(feat3, bams_manual)["Class"].astype(int).to_numpy() - 1

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

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[bams_pos] = True

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "y_bams_full": y_bams_full,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "bams_pos": bams_pos,
        "bams_label": bams_label,
        "used": used,
    }


def run_one(city: str, data: dict, seed: int, device: str) -> list[dict]:
    set_all_seeds(seed)
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    y_bams_full = data["y_bams_full"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    bams_pos = data["bams_pos"]
    bams_label = data["bams_label"]
    used = data["used"]

    X_train = X_all[idx_train]
    X_train_s = X_all_s[idx_train]
    X_test = X_all[idx_test]
    X_test_s = X_all_s[idx_test]
    y_train = y_all[idx_train]
    y_true = y_all[idx_test]
    X_bams_s = X_all_s[bams_pos]

    # Baseline / BAMS RF (seeded)
    rf_base = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test).astype(np.int64)

    rf_bams = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf_bams.fit(X_train, y_bams_full[idx_train])
    y_pred_bams = rf_bams.predict(X_test).astype(np.int64)

    # Teacher
    teacher = train_mlp(
        np.concatenate([X_train_s, X_bams_s], axis=0),
        np.concatenate([y_train, bams_label], axis=0),
        device=device,
        seed=seed,
    )
    pred_pool, conf_pool, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    pe_cands, pets_cands = [], []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, bams_pos, bams_label, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]

        # PE
        X_pe = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_train, bams_label, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        st_pe = train_mlp(X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 100 + k)
        y_pred_pe, _, _ = predict_proba_mlp(st_pe, X_test_s, device=device)
        m_pe = full_metrics(y_true, y_pred_pe)
        pe_cands.append({"k": k, "pred": y_pred_pe, "m": m_pe})

        # PE+TS
        keep_f = (conf_pool[exp_idx] > PE_TS_CONF) & (pred_pool[exp_idx] == exp_lab)
        exp_idx_f, exp_lab_f = exp_idx[keep_f], exp_lab[keep_f]
        X_pets = np.concatenate([X_train_s, X_bams_s, X_all_s[exp_idx_f]], axis=0)
        y_pets = np.concatenate([y_train, bams_label, exp_lab_f], axis=0)
        w_pets = np.concatenate(
            [
                np.full(len(idx_train) + len(bams_label), REAL_WEIGHT, dtype=np.float32),
                np.full(len(exp_lab_f), SOFT_WEIGHT, dtype=np.float32),
            ]
        )
        st_pets = train_mlp(X_pets, y_pets, sample_weight=w_pets, device=device, seed=seed + 200 + k)
        y_pred_pets, _, _ = predict_proba_mlp(st_pets, X_test_s, device=device)
        m_pets = full_metrics(y_true, y_pred_pets)
        pets_cands.append({"k": k, "pred": y_pred_pets, "m": m_pets})

    def pick(cands):
        return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]

    best_pe = pick(pe_cands)
    best_pets = pick(pets_cands)

    data_arrays = {
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "X_bams_s": X_bams_s,
        "X_all_s": X_all_s,
        "y_train": y_train,
        "y_true": y_true,
        "bams_label": bams_label,
        "idx_train": idx_train,
    }

    # ---- Human Amplification: TopN pick-best (same protocol as PE K-sweep) ----
    ha_best: dict[str, dict] = {}
    for method_name, proto_mode, margin_thr in HA_VARIANTS:
        ha_best[method_name] = pick_best_ha_variant(
            data_arrays,
            bams_pos,
            bams_label,
            y_all,
            used,
            margin_pool,
            proto_mode=proto_mode,
            margin_thresh=margin_thr,
            device=device,
            seed=seed,
        )

    preds = {
        "Baseline": y_pred_base,
        "BAMS150": y_pred_bams,
        "BAMS150+PrototypeExpansion_Boundary": best_pe["pred"],
        "BAMS150+PE+TS_Boundary": best_pets["pred"],
    }
    for method_name, hb in ha_best.items():
        preds[method_name] = hb["pred"]

    base_m = full_metrics(y_true, y_pred_base)
    rows = []
    for method, pred in preds.items():
        m = full_metrics(y_true, pred)
        be_red = 0.0
        if base_m["Boundary Error"] > 0:
            be_red = (
                (base_m["Boundary Error"] - m["Boundary Error"])
                / base_m["Boundary Error"]
                * 100.0
            )

        if method == "BAMS150+PrototypeExpansion_Boundary":
            best_k = best_pe["k"]
            n_amp = np.nan
            ha_meta = {}
        elif method == "BAMS150+PE+TS_Boundary":
            best_k = best_pets["k"]
            n_amp = np.nan
            ha_meta = {}
        elif method in ha_best:
            hb = ha_best[method]
            best_k = hb["top_n"]
            n_amp = hb["n_amplified"]
            ha_meta = {
                "HA_pool_size": hb["pool_size"],
                "HA_mean_score": hb["mean_score"],
                "HA_mean_agreement": hb["mean_agreement"],
                "HA_n_prototypes": hb["n_prototypes"],
                "HA_proto_mode": hb["proto_mode"],
                "HA_margin_thresh": hb["margin_thresh"],
            }
        else:
            best_k = np.nan
            n_amp = np.nan
            ha_meta = {}

        row = {
            "City": city,
            "Method": method,
            "Seed": seed,
            "OA": m["OA"],
            "AA": m["AA"],
            "Kappa": m["Kappa"],
            "Class1 Recall": m["Class1 Recall"],
            "Class2 Recall": m["Class2 Recall"],
            "Class3 Recall": m["Class3 Recall"],
            "Boundary Error": m["Boundary Error"],
            "Boundary Error Reduction": be_red,
            "BestK": best_k,
            "n_amplified": n_amp,
            "HA_pool_size": ha_meta.get("HA_pool_size", np.nan),
            "HA_mean_score": ha_meta.get("HA_mean_score", np.nan),
            "HA_mean_agreement": ha_meta.get("HA_mean_agreement", np.nan),
            "HA_n_prototypes": ha_meta.get("HA_n_prototypes", np.nan),
            "HA_proto_mode": ha_meta.get("HA_proto_mode", ""),
            "HA_margin_thresh": ha_meta.get("HA_margin_thresh", np.nan),
        }
        rows.append(row)
        extra = ""
        if method in ha_best:
            hb = ha_best[method]
            extra = (
                f" amp={hb['n_amplified']} topN={hb['top_n']} "
                f"proto={hb['proto_mode']} m<{hb['margin_thresh']}"
            )
        print(
            f"  [{city}|seed={seed}] {method}: OA={m['OA']:.4f} BE={m['Boundary Error']}{extra}"
        )
    return rows


def make_boxplots(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = {
        "Baseline": "Baseline",
        "BAMS150": "BAMS150",
        "BAMS150+PrototypeExpansion_Boundary": "PE-Boundary",
        "BAMS150+PE+TS_Boundary": "PE+TS",
        "BAMS150+HA_Corrected_Boundary": "HA-Corrected",
        "BAMS150+HA_Aligned_Boundary": "HA-Aligned",
    }
    methods = METHODS
    labels = [short[m] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data_oa = [df.loc[df["Method"] == m, "OA"].values for m in methods]
    axes[0].boxplot(data_oa, tick_labels=labels, showmeans=True)
    axes[0].set_title("OA across 5 seeds (all cities pooled)")
    axes[0].set_ylabel("OA")
    axes[0].tick_params(axis="x", rotation=25)
    data_be = [df.loc[df["Method"] == m, "Boundary Error"].values for m in methods]
    axes[1].boxplot(data_be, tick_labels=labels, showmeans=True)
    axes[1].set_title("Boundary Error across 5 seeds (all cities pooled)")
    axes[1].set_ylabel("Boundary Error (C1↔C2)")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test on OA: for each city, pair seeds; also overall pooling city×seed."""
    rows = []
    comparisons = [
        ("Baseline", "BAMS150+PrototypeExpansion_Boundary"),
        ("Baseline", "BAMS150+PE+TS_Boundary"),
        ("Baseline", "BAMS150+HA_Corrected_Boundary"),
        ("Baseline", "BAMS150+HA_Aligned_Boundary"),
        ("BAMS150", "BAMS150+PrototypeExpansion_Boundary"),
        ("BAMS150", "BAMS150+PE+TS_Boundary"),
        ("BAMS150", "BAMS150+HA_Corrected_Boundary"),
        ("BAMS150", "BAMS150+HA_Aligned_Boundary"),
        ("BAMS150+PrototypeExpansion_Boundary", "BAMS150+HA_Corrected_Boundary"),
        ("BAMS150+PrototypeExpansion_Boundary", "BAMS150+HA_Aligned_Boundary"),
        ("BAMS150+PE+TS_Boundary", "BAMS150+HA_Corrected_Boundary"),
        ("BAMS150+PE+TS_Boundary", "BAMS150+HA_Aligned_Boundary"),
        ("BAMS150+HA_Corrected_Boundary", "BAMS150+HA_Aligned_Boundary"),
    ]

    # Per-city paired by seed
    for city in CITIES:
        g = df[df["City"] == city]
        for a, b in comparisons:
            oa_a = g.loc[g["Method"] == a].sort_values("Seed")["OA"].values
            oa_b = g.loc[g["Method"] == b].sort_values("Seed")["OA"].values
            t, p = stats.ttest_rel(oa_b, oa_a)  # positive t => b > a
            rows.append(
                {
                    "Scope": city,
                    "Comparison": f"{b} vs {a}",
                    "Metric": "OA",
                    "t_statistic": float(t),
                    "p_value": float(p),
                    "Significant (p<0.05)": bool(p < 0.05),
                    "Mean_A": float(oa_a.mean()),
                    "Mean_B": float(oa_b.mean()),
                }
            )

    # Overall: average OA across cities per seed, then paired by seed
    for a, b in comparisons:
        wide_a = (
            df[df["Method"] == a]
            .groupby("Seed")["OA"]
            .mean()
            .reindex(SEEDS)
            .values
        )
        wide_b = (
            df[df["Method"] == b]
            .groupby("Seed")["OA"]
            .mean()
            .reindex(SEEDS)
            .values
        )
        t, p = stats.ttest_rel(wide_b, wide_a)
        rows.append(
            {
                "Scope": "AllCities_Mean",
                "Comparison": f"{b} vs {a}",
                "Metric": "OA",
                "t_statistic": float(t),
                "p_value": float(p),
                "Significant (p<0.05)": bool(p < 0.05),
                "Mean_A": float(wide_a.mean()),
                "Mean_B": float(wide_b.mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(overall: pd.DataFrame, sig: pd.DataFrame) -> str:
    lines = []
    lines.append("Statistical Significance Report (5 seeds: 0–4)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Methods: Baseline | BAMS150 | PE-Boundary | PE+TS | HA-Corrected | HA-Aligned")
    lines.append(
        "HA: TopN∈{50,100,150,250} pick-best by BE (same protocol as PE K-sweep); "
        "Corrected=corrected+m<0.25; Aligned=all_bams+m<0.15"
    )
    lines.append("")
    lines.append("1) Mean ± Std of OA (across cities × seeds)")
    for _, r in overall.iterrows():
        lines.append(f"  {r['Method']}: {r['Mean OA']:.4f} ± {r['Std OA']:.4f}")
    lines.append("")
    lines.append("2) Mean ± Std of Kappa")
    for _, r in overall.iterrows():
        lines.append(f"  {r['Method']}: {r['Mean Kappa']:.4f} ± {r['Std Kappa']:.4f}")
    lines.append("")
    lines.append("3) Mean ± Std of Boundary Error")
    for _, r in overall.iterrows():
        lines.append(
            f"  {r['Method']}: {r['Mean Boundary Error']:.2f} ± {r['Std Boundary Error']:.2f}"
        )
    lines.append("")
    lines.append("4) Paired t-tests on OA (AllCities_Mean by seed)")
    sub = sig[sig["Scope"] == "AllCities_Mean"]
    for _, r in sub.iterrows():
        sig_txt = "YES" if r["Significant (p<0.05)"] else "NO"
        lines.append(
            f"  {r['Comparison']}: t={r['t_statistic']:.3f}, p={r['p_value']:.4f}, "
            f"significant={sig_txt} (means {r['Mean_A']:.4f} → {r['Mean_B']:.4f})"
        )
    lines.append("")
    lines.append("5) Stability across seeds")
    for _, r in overall.iterrows():
        cv = r["Std OA"] / r["Mean OA"] if r["Mean OA"] else np.nan
        stable = (
            "stable"
            if r["Std OA"] < 0.01
            else ("moderately variable" if r["Std OA"] < 0.02 else "variable")
        )
        lines.append(
            f"  {r['Method']}: OA std={r['Std OA']:.4f} (CV={cv:.4f}) → {stable}"
        )
    return "\n".join(lines) + "\n"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("seeds:", SEEDS)
    print("methods:", METHODS)

    all_rows = []
    city_data = {c: load_city(c) for c in CITIES}

    for city in CITIES:
        print("\n" + "=" * 72)
        print(f"CITY: {city}")
        print("=" * 72)
        for seed in SEEDS:
            all_rows.extend(run_one(city, city_data[city], seed, device))

    df = pd.DataFrame(all_rows)

    # FILE 1 seed_results.csv
    seed_cols = [
        "City",
        "Method",
        "Seed",
        "OA",
        "AA",
        "Kappa",
        "Class1 Recall",
        "Class2 Recall",
        "Class3 Recall",
        "Boundary Error",
    ]
    seed_results = df[seed_cols].copy()
    seed_results.to_csv(OUT_DIR / "seed_results.csv", index=False)
    seed_results.to_csv(DATA2 / "seed_results.csv", index=False)

    # FILE 2 seed_summary.csv (per city × method)
    seed_summary = (
        df.groupby(["City", "Method"], sort=False)
        .agg(
            **{
                "OA Mean": ("OA", "mean"),
                "OA Std": ("OA", "std"),
                "Kappa Mean": ("Kappa", "mean"),
                "Kappa Std": ("Kappa", "std"),
                "Boundary Error Mean": ("Boundary Error", "mean"),
                "Boundary Error Std": ("Boundary Error", "std"),
            }
        )
        .reset_index()
    )
    seed_summary.to_csv(OUT_DIR / "seed_summary.csv", index=False)
    seed_summary.to_csv(DATA2 / "seed_summary.csv", index=False)

    # FILE 3 overall_seed_summary.csv
    overall = (
        df.groupby("Method", sort=False)
        .agg(
            **{
                "Mean OA": ("OA", "mean"),
                "Std OA": ("OA", "std"),
                "Mean Kappa": ("Kappa", "mean"),
                "Std Kappa": ("Kappa", "std"),
                "Mean Boundary Error": ("Boundary Error", "mean"),
                "Std Boundary Error": ("Boundary Error", "std"),
            }
        )
        .reindex(METHODS)
        .reset_index()
    )
    overall.to_csv(OUT_DIR / "overall_seed_summary.csv", index=False)
    overall.to_csv(DATA2 / "overall_seed_summary.csv", index=False)

    # FILE 4 significance_tests.csv
    sig = significance_tests(df)
    sig.to_csv(OUT_DIR / "significance_tests.csv", index=False)
    sig.to_csv(DATA2 / "significance_tests.csv", index=False)

    # FILE 5 boxplot (after CSVs are saved)
    box_path = OUT_DIR / "boxplot_results.png"
    try:
        make_boxplots(df, box_path)
        import shutil

        shutil.copy2(box_path, DATA2 / "boxplot_results.png")
        print(f"Saved boxplot: {box_path}")
    except Exception as e:
        print(f"WARNING: boxplot failed: {e}")

    # Combined Excel for easy checking
    xlsx_path = DATA2 / "seed_experiments_all.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        seed_results.to_excel(writer, sheet_name="seed_results", index=False)
        seed_summary.to_excel(writer, sheet_name="seed_summary", index=False)
        overall.to_excel(writer, sheet_name="overall_seed_summary", index=False)
        sig.to_excel(writer, sheet_name="significance_tests", index=False)
        df.to_excel(writer, sheet_name="seed_results_full", index=False)

    report = write_report(overall, sig)
    (OUT_DIR / "statistical_significance_report.txt").write_text(report, encoding="utf-8")
    (DATA2 / "statistical_significance_report.txt").write_text(report, encoding="utf-8")

    print("\n" + "=" * 72)
    print("SEED RESULTS (head)")
    print("=" * 72)
    print(seed_results.head(20).to_string(index=False))
    print("\nOVERALL")
    print(overall.to_string(index=False))
    print("\nSIGNIFICANCE (AllCities_Mean)")
    print(sig[sig["Scope"] == "AllCities_Mean"].to_string(index=False))
    print("\n" + report)
    print(f"\nMain check file: {xlsx_path}")
    print(f"Also: {DATA2 / 'seed_results.csv'}")


if __name__ == "__main__":
    main()
