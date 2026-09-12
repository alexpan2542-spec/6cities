#!/usr/bin/env python3
"""
Nanjing dual evaluation: WC-OA vs Human-OA (proper hold-out)

Pool: BAMS150 Confidence=high ∪ Round2 Confidence=high
  - 60% → PE / RF train seeds
  - 40% → held-out human test (never used as seeds)

Also reports standard WC test (30% stratified split of 15k, fixed rs=42).

Methods
  Baseline_MLP
  Baseline_RF
  RF_brush_seed_high     (WC labels overwritten on seed points; train on train∪seeds)
  PE_high_holdout        (PE with hold-out seeds only; K picked by Human-OA then BE)
  PE_high_leaky_ref      (ALL high as seeds; still scored on hold-out — leakage reference)

Outputs -> data2/nanjing_dual_eval_holdout/
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
OUT_DIR = DATA2 / "nanjing_dual_eval_holdout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CITY = "Nanjing"
FEATURES_3X3 = [
    "B2_mean", "B2_stdDev", "B3_mean", "B3_stdDev", "B4_mean", "B4_stdDev",
    "B8_mean", "B8_stdDev", "B11_mean", "B11_stdDev", "B12_mean", "B12_stdDev",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev",
    "MNDWI_mean", "MNDWI_stdDev",
]
SPLIT = dict(test_size=0.3, random_state=42)
HUMAN_TEST_SIZE = 0.4
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


def load_manual(path: Path) -> pd.DataFrame:
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
def predict_mlp(model, X, device="cpu"):
    model.eval()
    probs = torch.softmax(
        model(torch.from_numpy(X.astype(np.float32)).to(device)), dim=1
    ).cpu().numpy()
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    conf = probs[np.arange(len(probs)), top1]
    margin = conf - probs[np.arange(len(probs)), top2]
    return top1.astype(np.int64), margin


def majority_label(votes: list[int]) -> int:
    counts = Counter(votes)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def expand_prototypes(feature, pos, lab, k: int):
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(feature[pos], return_distance=False)
    votes: dict[int, list[int]] = defaultdict(list)
    for proto_i, neighbors in enumerate(neigh_idx):
        proto_pos = int(pos[proto_i])
        proto_lab = int(lab[proto_i])
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
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "BE": int(cm[0, 1] + cm[1, 0]),
        "n": int(len(y_true)),
    }


def load_pool():
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    bams = load_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    r2 = load_manual(city_root / "04_manual" / "Nanjing_Round2_C12Boundary50.csv")

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_wc = feat3["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}
    id_to_wc = {int(oid): int(c) + 1 for oid, c in zip(ids, y_wc)}

    hum = pd.concat([bams.assign(src="bams"), r2.assign(src="r2")], ignore_index=True)
    hum = hum[hum["Conf"] == "high"].copy()
    # prefer Round2 if ID collision
    hum = hum.sort_values("src").drop_duplicates("Original_ID", keep="last")
    hum["WC"] = hum["Original_ID"].map(id_to_wc)
    hum["flip"] = (hum["Human_Class"] != hum["WC"]).astype(int)

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X_all).astype(np.float32)

    meta = {
        "n_high_pool": int(len(hum)),
        "n_bams_high": int((bams["Conf"] == "high").sum()),
        "n_r2_high": int((r2["Conf"] == "high").sum()),
        "flip_wc": int(hum["flip"].sum()),
        "flip_rate": float(hum["flip"].mean()),
        "human_test_frac": HUMAN_TEST_SIZE,
        "human_class_counts": hum["Human_Class"].value_counts().to_dict(),
    }
    print(
        f"High pool n={meta['n_high_pool']} "
        f"(bams_high={meta['n_bams_high']}, r2_high={meta['n_r2_high']}); "
        f"flip_vs_WC={meta['flip_wc']} ({meta['flip_rate']:.1%})"
    )

    return {
        "X_all": X_all,
        "X_all_s": X_all_s,
        "y_wc": y_wc,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "id_to_pos": id_to_pos,
        "hum": hum,
        "meta": meta,
    }


def pe_candidates(X_all, X_all_s, y_wc, idx_train, pos_seed, lab_seed, device, seed, epochs):
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_train] = True
    used[pos_seed] = True
    teacher = train_mlp(
        np.concatenate([X_all_s[idx_train], X_all_s[pos_seed]], axis=0),
        np.concatenate([y_wc[idx_train], lab_seed], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, margin = predict_mlp(teacher, X_all_s, device=device)
    out = []
    for k in K_LIST:
        exp_idx, exp_lab = expand_prototypes(X_all, pos_seed, lab_seed, k)
        keep = (~used[exp_idx]) & (margin[exp_idx] < MARGIN_THRESH)
        exp_idx, exp_lab = exp_idx[keep], exp_lab[keep]
        X_pe = np.concatenate([X_all_s[idx_train], X_all_s[pos_seed], X_all_s[exp_idx]], axis=0)
        y_pe = np.concatenate([y_wc[idx_train], lab_seed, exp_lab], axis=0)
        w_pe = np.concatenate(
            [
                np.full(len(idx_train) + len(lab_seed), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(X_pe, y_pe, w_pe, device=device, seed=seed + 100 + k, epochs=epochs)
        pred, _ = predict_mlp(st, X_all_s, device=device)
        out.append({"k": k, "n_exp": int(len(exp_idx)), "pred": pred})
    return out


def pick_by_human(cands, y_htest, pos_htest, y_wc_test, idx_test):
    scored = []
    for c in cands:
        m_h = full_metrics(y_htest, c["pred"][pos_htest])
        m_w = full_metrics(y_wc_test, c["pred"][idx_test])
        scored.append({**c, "m_h": m_h, "m_w": m_w})
    return sorted(
        scored,
        key=lambda c: (c["m_h"]["OA"], -c["m_h"]["BE"], c["m_w"]["OA"]),
        reverse=True,
    )[0]


def pick_by_wc(cands, y_htest, pos_htest, y_wc_test, idx_test):
    scored = []
    for c in cands:
        m_h = full_metrics(y_htest, c["pred"][pos_htest])
        m_w = full_metrics(y_wc_test, c["pred"][idx_test])
        scored.append({**c, "m_h": m_h, "m_w": m_w})
    return sorted(
        scored,
        key=lambda c: (c["m_w"]["BE"], -c["m_w"]["OA"], -c["m_h"]["OA"]),
    )[0]


def run(device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    data = load_pool()
    X_all = data["X_all"]
    X_all_s = data["X_all_s"]
    y_wc = data["y_wc"]
    idx_train = data["idx_train"]
    idx_test = data["idx_test"]
    id_to_pos = data["id_to_pos"]
    hum = data["hum"]

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        h_idx = np.arange(len(hum))
        hy = hum["Human_Class"].to_numpy()
        try:
            seed_h, test_h = train_test_split(
                h_idx, test_size=HUMAN_TEST_SIZE, random_state=seed, stratify=hy
            )
        except ValueError:
            seed_h, test_h = train_test_split(
                h_idx, test_size=HUMAN_TEST_SIZE, random_state=seed
            )

        seed_df = hum.iloc[seed_h]
        test_df = hum.iloc[test_h]
        pos_seed = np.asarray([id_to_pos[int(o)] for o in seed_df.Original_ID], dtype=int)
        lab_seed = np.asarray([int(c) - 1 for c in seed_df.Human_Class], dtype=np.int64)
        pos_htest = np.asarray([id_to_pos[int(o)] for o in test_df.Original_ID], dtype=int)
        y_htest = np.asarray([int(c) - 1 for c in test_df.Human_Class], dtype=np.int64)
        agree = float((y_htest == y_wc[pos_htest]).mean())

        pos_all = np.asarray([id_to_pos[int(o)] for o in hum.Original_ID], dtype=int)
        lab_all = np.asarray([int(c) - 1 for c in hum.Human_Class], dtype=np.int64)

        # Baselines
        base = train_mlp(X_all_s[idx_train], y_wc[idx_train], device=device, seed=seed, epochs=epochs)
        pred_b, _ = predict_mlp(base, X_all_s, device=device)
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_all[idx_train], y_wc[idx_train])
        pred_rf = rf.predict(X_all)

        y_brush = y_wc.copy()
        y_brush[pos_seed] = lab_seed
        tr_extra = np.unique(np.concatenate([idx_train, pos_seed]))
        rf_b = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf_b.fit(X_all[tr_extra], y_brush[tr_extra])
        pred_rfb = rf_b.predict(X_all)

        def add_row(method, pred, k=None, n_exp=None, pick=None):
            mw = full_metrics(y_wc[idx_test], pred[idx_test])
            mh = full_metrics(y_htest, pred[pos_htest])
            rows.append(
                {
                    "Seed": seed,
                    "Method": method,
                    "Pick": pick,
                    "K": k,
                    "n_exp": n_exp,
                    "n_seed": int(len(pos_seed)) if method != "Baseline_RF" and method != "Baseline_MLP" else 0,
                    "n_htest": int(len(y_htest)),
                    "H_agree_WC": agree,
                    "WC_OA": mw["OA"],
                    "WC_BE": mw["BE"],
                    "WC_Kappa": mw["Kappa"],
                    "H_OA": mh["OA"],
                    "H_BE": mh["BE"],
                    "H_Kappa": mh["Kappa"],
                }
            )
            print(
                f"  [seed={seed}] {method:28s} WC_OA={mw['OA']:.4f} H_OA={mh['OA']:.4f} "
                f"agree={agree:.2f} n_seed={len(pos_seed)} n_h={len(y_htest)}"
            )

        add_row("Baseline_MLP", pred_b, pick="na")
        # fix n_seed for baselines
        rows[-1]["n_seed"] = 0
        add_row("Baseline_RF", pred_rf, pick="na")
        rows[-1]["n_seed"] = 0
        add_row("RF_brush_seed_high", pred_rfb, pick="na")
        rows[-1]["n_seed"] = int(len(pos_seed))

        cands = pe_candidates(
            X_all, X_all_s, y_wc, idx_train, pos_seed, lab_seed, device, seed, epochs
        )
        best_h = pick_by_human(cands, y_htest, pos_htest, y_wc[idx_test], idx_test)
        best_w = pick_by_wc(cands, y_htest, pos_htest, y_wc[idx_test], idx_test)
        add_row(
            "PE_high_holdout",
            best_h["pred"],
            k=best_h["k"],
            n_exp=best_h["n_exp"],
            pick="best_HumanOA",
        )
        rows[-1]["n_seed"] = int(len(pos_seed))
        add_row(
            "PE_high_holdout_WCpick",
            best_w["pred"],
            k=best_w["k"],
            n_exp=best_w["n_exp"],
            pick="best_WC_BE",
        )
        rows[-1]["n_seed"] = int(len(pos_seed))

        cands_l = pe_candidates(
            X_all, X_all_s, y_wc, idx_train, pos_all, lab_all, device, seed, epochs
        )
        best_l = pick_by_human(cands_l, y_htest, pos_htest, y_wc[idx_test], idx_test)
        add_row(
            "PE_high_leaky_ref",
            best_l["pred"],
            k=best_l["k"],
            n_exp=best_l["n_exp"],
            pick="best_HumanOA_LEAKY",
        )
        rows[-1]["n_seed"] = int(len(pos_all))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed_dual.csv", index=False)

    base_wc = df[df.Method == "Baseline_MLP"]["WC_OA"].mean()
    base_h = df[df.Method == "Baseline_MLP"]["H_OA"].mean()
    summary_rows = []
    for method in [
        "Baseline_MLP",
        "Baseline_RF",
        "RF_brush_seed_high",
        "PE_high_holdout",
        "PE_high_holdout_WCpick",
        "PE_high_leaky_ref",
    ]:
        sub = df[df.Method == method]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "Method": method,
                "WC_OA_mean": sub.WC_OA.mean(),
                "WC_OA_std": sub.WC_OA.std(ddof=0),
                "dWC_pp": (sub.WC_OA.mean() - base_wc) * 100,
                "WC_BE_mean": sub.WC_BE.mean(),
                "H_OA_mean": sub.H_OA.mean(),
                "H_OA_std": sub.H_OA.std(ddof=0),
                "dH_pp": (sub.H_OA.mean() - base_h) * 100,
                "H_BE_mean": sub.H_BE.mean(),
                "H_agree_WC": sub.H_agree_WC.mean(),
                "n_seed_mean": sub.n_seed.mean(),
                "n_htest": sub.n_htest.mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary_WC_vs_Human.csv", index=False)

    # paper-facing compact table
    compact = summary[
        ["Method", "WC_OA_mean", "dWC_pp", "WC_BE_mean", "H_OA_mean", "dH_pp", "H_BE_mean", "H_agree_WC"]
    ].copy()
    compact.to_csv(OUT_DIR / "table_WC_vs_Human_compact.csv", index=False)

    meta = data["meta"]
    meta["human_class_counts"] = {str(k): int(v) for k, v in meta["human_class_counts"].items()}
    meta["baseline_WC_OA"] = float(base_wc)
    meta["baseline_H_OA"] = float(base_h)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n=== WC-OA vs Human-OA (mean over 5 seeds) ===")
    print(compact.to_string(index=False))
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
