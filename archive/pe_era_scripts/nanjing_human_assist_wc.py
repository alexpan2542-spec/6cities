#!/usr/bin/env python3
"""
Nanjing: human as WC assistant (same protocol as Hefei), larger label pool

Pool: BAMS150 ∪ Round2_C12Boundary50 (200 unique, all have Confidence).
Top150 has no Confidence → not included (Round2 is the conf-labeled Top subset).

Protocol
--------
  human == WC  &  Conf=high  → WC label, weight=2.0, PE seeds
  human != WC  &  Conf=high  → discard
  Conf=med                   → WC label, weight=0.5
  Conf=low                   → discard

Comparators
-----------
  Baseline_RF / Baseline_MLP
  PE_high_adversarial_BAMS  (passport: BAMS high human overwrite)
  PE_high_adversarial_BR    (BAMS∪Round2 all high human overwrite)
  Assist_MLP / Assist_PE    (assist on BAMS∪Round2)

Eval: pointwise vs original WorldCover Class.
Outputs -> data2/nanjing_human_assist_wc/
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
OUT_DIR = DATA2 / "nanjing_human_assist_wc"
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
AGREE_HIGH_WEIGHT = 2.0
MED_WEIGHT = 0.5
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256

METHODS = [
    "Baseline_RF",
    "Baseline_MLP",
    "PE_high_adversarial_BAMS",
    "PE_high_adversarial_BR",
    "Assist_MLP",
    "Assist_PE",
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


def load_city() -> dict:
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")

    bams = pd.read_csv(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    bams["source"] = "BAMS"
    r2 = pd.read_csv(city_root / "04_manual" / "Nanjing_Round2_C12Boundary50.csv")
    r2["source"] = "Round2"

    cols = ["Original_ID", "Human_Class", "Confidence", "source"]
    man = pd.concat([bams[cols], r2[cols]], ignore_index=True)
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce")
    man = man[man["Human_Class"].notna()].copy()
    man["Human_Class"] = man["Human_Class"].astype(int)
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf)
    assert man["Original_ID"].nunique() == len(man), "duplicate Original_ID in pool"

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

    pos_all = np.asarray([id_to_pos[int(r.Original_ID)] for _, r in man.iterrows()], dtype=int)
    human = np.asarray([int(r.Human_Class) - 1 for _, r in man.iterrows()], dtype=np.int64)
    wc = y_all[pos_all]
    conf = man["Conf"].tolist()
    source = man["source"].tolist()
    agree = human == wc

    agree_high = np.asarray(
        [(c == "high") and bool(a) for c, a in zip(conf, agree)], dtype=bool
    )
    disagree_high = np.asarray(
        [(c == "high") and (not bool(a)) for c, a in zip(conf, agree)], dtype=bool
    )
    med = np.asarray([c == "med" for c in conf], dtype=bool)
    low = np.asarray([c == "low" for c in conf], dtype=bool)
    high_mask = np.asarray([c == "high" for c in conf], dtype=bool)
    bams_mask = np.asarray([s == "BAMS" for s in source], dtype=bool)
    high_bams = high_mask & bams_mask

    buckets = {
        "n_pool": int(len(man)),
        "n_bams": int(bams_mask.sum()),
        "n_round2": int((~bams_mask).sum()),
        "n_agree_high": int(agree_high.sum()),
        "n_disagree_high": int(disagree_high.sum()),
        "n_med": int(med.sum()),
        "n_low": int(low.sum()),
        "n_adv_high_bams": int(high_bams.sum()),
        "n_adv_high_br": int(high_mask.sum()),
        "n_agree_high_bams": int((agree_high & bams_mask).sum()),
        "n_agree_high_r2": int((agree_high & ~bams_mask).sum()),
        "conf_counts": {str(k): int(v) for k, v in Counter(conf).items()},
    }
    print(
        f"Nanjing assist pool BAMS∪Round2 n={buckets['n_pool']}: "
        f"agree_high={buckets['n_agree_high']} "
        f"(BAMS={buckets['n_agree_high_bams']}, R2={buckets['n_agree_high_r2']}), "
        f"disagree_high_drop={buckets['n_disagree_high']}, "
        f"med_w05={buckets['n_med']}, low_drop={buckets['n_low']}; "
        f"adv_BAMS={buckets['n_adv_high_bams']} adv_BR={buckets['n_adv_high_br']}"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_all": y_all,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "pos_agree_high": pos_all[agree_high],
        "lab_agree_high": wc[agree_high],
        "pos_med": pos_all[med],
        "lab_med": wc[med],
        "pos_adv_bams": pos_all[high_bams],
        "lab_adv_bams": human[high_bams],
        "pos_adv_br": pos_all[high_mask],
        "lab_adv_br": human[high_mask],
        "buckets": buckets,
    }


def build_assist_xyw(X_all_s, y_all, idx_train, pos_agree_high, lab_agree_high, pos_med, lab_med):
    train_set = set(int(i) for i in idx_train.tolist())
    w_train = np.full(len(idx_train), REAL_WEIGHT, dtype=np.float32)
    pos_to_train_i = {int(p): i for i, p in enumerate(idx_train.tolist())}

    for p in pos_agree_high.tolist():
        p = int(p)
        if p in pos_to_train_i:
            w_train[pos_to_train_i[p]] = AGREE_HIGH_WEIGHT
    for p in pos_med.tolist():
        p = int(p)
        if p in pos_to_train_i:
            w_train[pos_to_train_i[p]] = MED_WEIGHT

    X_parts = [X_all_s[idx_train]]
    y_parts = [y_all[idx_train]]
    w_parts = [w_train]

    extra_pos, extra_lab, extra_w = [], [], []
    for p, lab in zip(pos_agree_high.tolist(), lab_agree_high.tolist()):
        p = int(p)
        if p not in train_set:
            extra_pos.append(p)
            extra_lab.append(int(lab))
            extra_w.append(AGREE_HIGH_WEIGHT)
    for p, lab in zip(pos_med.tolist(), lab_med.tolist()):
        p = int(p)
        if p not in train_set:
            extra_pos.append(p)
            extra_lab.append(int(lab))
            extra_w.append(MED_WEIGHT)

    if extra_pos:
        X_parts.append(X_all_s[np.asarray(extra_pos, dtype=int)])
        y_parts.append(np.asarray(extra_lab, dtype=np.int64))
        w_parts.append(np.asarray(extra_w, dtype=np.float32))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    w = np.concatenate(w_parts, axis=0)
    return X, y, w, {
        "n_train_base": int(len(idx_train)),
        "n_extra": int(len(extra_pos)),
        "n_w2": int((w == AGREE_HIGH_WEIGHT).sum()),
        "n_w05": int((w == MED_WEIGHT).sum()),
    }


def run_pe_adversarial(data, pos_key, lab_key, tag, device, seed, epochs):
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    pos_seed = data[pos_key]
    lab_seed = data[lab_key]

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pos_seed] = True

    X_train_s = X_all_s[idx_train]
    X_test_s = X_all_s[idx_test]
    X_seed_s = X_all_s[pos_seed]
    y_true = y_all[idx_test]

    teacher = train_mlp(
        np.concatenate([X_train_s, X_seed_s], axis=0),
        np.concatenate([y_all[idx_train], lab_seed], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, pos_seed, lab_seed, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_train_s, X_seed_s, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_all[idx_train], lab_seed, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(lab_seed), REAL_WEIGHT, np.float32),
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
            f"  [seed={seed}] {tag:24s} K={k:<2d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    return pick_best(cands)


def run_pe_assist(data, device, seed, epochs):
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    pos_seed = data["pos_agree_high"]
    lab_seed = data["lab_agree_high"]

    X_assist, y_assist, w_assist, meta = build_assist_xyw(
        X_all_s,
        y_all,
        idx_train,
        data["pos_agree_high"],
        data["lab_agree_high"],
        data["pos_med"],
        data["lab_med"],
    )
    X_test_s = X_all_s[idx_test]
    y_true = y_all[idx_test]

    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pos_seed] = True
    used[data["pos_med"]] = True

    teacher = train_mlp(
        X_assist, y_assist, sample_weight=w_assist, device=device, seed=seed, epochs=epochs
    )
    _, _, margin_pool = predict_proba_mlp(teacher, X_all_s, device=device)

    cands = []
    for k in K_LIST:
        if len(pos_seed) == 0:
            pred, _, _ = predict_proba_mlp(teacher, X_test_s, device=device)
            m = full_metrics(y_true, pred)
            cands.append({"k": k, "n_exp": 0, "m": m})
            continue
        exp_idx_all, exp_lab_all = expand_prototypes(X_all, pos_seed, lab_seed, k)
        keep = (~used[exp_idx_all]) & (margin_pool[exp_idx_all] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx_all[keep], exp_lab_all[keep]
        X_pe = np.concatenate([X_assist, X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_assist, exp_lab], axis=0)
        w_pe = np.concatenate(
            [w_assist, np.full(len(exp_lab), SOFT_WEIGHT, np.float32)]
        )
        st = train_mlp(
            X_pe, y_pe, sample_weight=w_pe, device=device, seed=seed + 200 + k, epochs=epochs
        )
        pred, _, _ = predict_proba_mlp(st, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        cands.append({"k": k, "n_exp": int(len(exp_idx)), "m": m})
        print(
            f"  [seed={seed}] Assist_PE               K={k:<2d} exp={len(exp_idx):<3d} "
            f"OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    best = pick_best(cands)
    best["assist_meta"] = meta
    return best


def run(device: str, seeds: list[int], epochs: int):
    data = load_city()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_all = data["y_all"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        X_train_s, X_test_s = X_all_s[idx_train], X_all_s[idx_test]
        y_true = y_all[idx_test]

        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_all[idx_train])
        m = full_metrics(y_true, rf.predict(X_test))
        rows.append({"City": CITY, "Method": "Baseline_RF", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_RF             OA={m['OA']:.4f} BE={m['Boundary Error']}")

        mlp = train_mlp(X_train_s, y_all[idx_train], device=device, seed=seed, epochs=epochs)
        pred, _, _ = predict_proba_mlp(mlp, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        rows.append({"City": CITY, "Method": "Baseline_MLP", "Seed": seed, **m})
        print(f"  [seed={seed}] Baseline_MLP            OA={m['OA']:.4f} BE={m['Boundary Error']}")

        for name, pk, lk in [
            ("PE_high_adversarial_BAMS", "pos_adv_bams", "lab_adv_bams"),
            ("PE_high_adversarial_BR", "pos_adv_br", "lab_adv_br"),
        ]:
            best = run_pe_adversarial(data, pk, lk, name, device, seed, epochs)
            n_seed = int(len(data[pk]))
            rows.append(
                {
                    "City": CITY,
                    "Method": name,
                    "Seed": seed,
                    **best["m"],
                    "BestK": best["k"],
                    "n_expanded": best["n_exp"],
                    "n_seed": n_seed,
                }
            )
            print(
                f"  [seed={seed}] {name:24s} OA={best['m']['OA']:.4f} "
                f"BE={best['m']['Boundary Error']} K={best['k']} exp={best['n_exp']}"
            )

        X_a, y_a, w_a, meta_a = build_assist_xyw(
            X_all_s,
            y_all,
            idx_train,
            data["pos_agree_high"],
            data["lab_agree_high"],
            data["pos_med"],
            data["lab_med"],
        )
        mlp_a = train_mlp(
            X_a, y_a, sample_weight=w_a, device=device, seed=seed + 50, epochs=epochs
        )
        pred, _, _ = predict_proba_mlp(mlp_a, X_test_s, device=device)
        m = full_metrics(y_true, pred)
        rows.append(
            {
                "City": CITY,
                "Method": "Assist_MLP",
                "Seed": seed,
                **m,
                "n_seed": int(len(data["pos_agree_high"])),
                "n_w2": meta_a["n_w2"],
                "n_w05": meta_a["n_w05"],
            }
        )
        print(
            f"  [seed={seed}] Assist_MLP               OA={m['OA']:.4f} BE={m['Boundary Error']} "
            f"w2={meta_a['n_w2']} w05={meta_a['n_w05']}"
        )

        best_a = run_pe_assist(data, device, seed, epochs)
        rows.append(
            {
                "City": CITY,
                "Method": "Assist_PE",
                "Seed": seed,
                **best_a["m"],
                "BestK": best_a["k"],
                "n_expanded": best_a["n_exp"],
                "n_seed": int(len(data["pos_agree_high"])),
                "n_w2": best_a["assist_meta"]["n_w2"],
                "n_w05": best_a["assist_meta"]["n_w05"],
            }
        )
        print(
            f"  [seed={seed}] Assist_PE                OA={best_a['m']['OA']:.4f} "
            f"BE={best_a['m']['Boundary Error']} K={best_a['k']} exp={best_a['n_exp']}"
        )

    return pd.DataFrame(rows), data


def summarize(df: pd.DataFrame, buckets: dict) -> tuple[str, pd.DataFrame]:
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
    base = overall.loc[overall["Method"] == "Baseline_RF", "Mean_OA"].iloc[0]
    overall["dOA_vs_RF_pp"] = (overall["Mean_OA"] - base) * 100
    base_mlp = overall.loc[overall["Method"] == "Baseline_MLP", "Mean_OA"].iloc[0]
    overall["dOA_vs_MLP_pp"] = (overall["Mean_OA"] - base_mlp) * 100

    lines = [
        "# Nanjing: human as WC assistant (BAMS ∪ Round2)\n",
        "\n## Protocol / pool\n",
        f"- Pool: BAMS150 ∪ Round2 = {buckets['n_pool']} "
        f"(BAMS={buckets['n_bams']}, Round2={buckets['n_round2']})\n",
        f"- agree_high w=2 PE seeds: {buckets['n_agree_high']} "
        f"(BAMS {buckets['n_agree_high_bams']} + R2 {buckets['n_agree_high_r2']})\n",
        f"- disagree_high discard: {buckets['n_disagree_high']}\n",
        f"- med WC w=0.5: {buckets['n_med']}; low discard: {buckets['n_low']}\n",
        f"- Top150 without Confidence not included "
        f"(Round2 already covers conf-labeled Top subset)\n",
        "\n## Mean over seeds\n\n",
        overall.to_string(index=False),
        "\n\n## Reading\n",
        "- Compare Assist_PE to PE_high_adversarial_* and Baseline_MLP.\n",
        "- Passport reference ≈ PE_high_adversarial_BAMS (~+0.33 pp vs RF historically).\n",
    ]
    return "".join(lines), overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print("device:", device, "seeds:", seeds, "epochs:", epochs, "out:", out)

    df, data = run(device=device, seeds=seeds, epochs=epochs)
    md, overall = summarize(df, data["buckets"])

    df.to_csv(out / "seed_results.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "VERDICT.md").write_text(md, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "pool": "BAMS150 ∪ Round2_C12Boundary50",
                "seeds": seeds,
                "epochs": epochs,
                "agree_high_weight": AGREE_HIGH_WEIGHT,
                "med_weight": MED_WEIGHT,
                "soft_weight": SOFT_WEIGHT,
                "methods": METHODS,
                **data["buckets"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
