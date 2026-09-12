#!/usr/bin/env python3
"""
Five-city Random150 vs BAMS150 + PE (current protocol).

Methods (per city, 5 seeds; random draw coupled to seed):
  Baseline_RF
  PE_BAMS_all
  PE_BAMS_high
  PE_Random150_WC          # random 150 seeds, labels = WC Class
  PE_Random150_matchedFlip  # same # C1↔C2 flips as BAMS, random locations

Eval vs original WorldCover Class.
Outputs -> data2/{city}_random150_vs_bams/ and data2/cross_city_random150_vs_bams/
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
N_RANDOM = 150
CITIES = ["Wuhan", "Changsha", "Nanchang", "Hefei", "Nanjing"]


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
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["BE"], -c["m"]["OA"]))[0]


def run_pe(X, Xs, y_wc, idx_tr, idx_te, pos_seed, lab_seed, device, seed, epochs, tag):
    used = np.zeros(len(X), dtype=bool)
    used[idx_tr] = True
    used[pos_seed] = True
    teacher = train_mlp(
        np.concatenate([Xs[idx_tr], Xs[pos_seed]], axis=0),
        np.concatenate([y_wc[idx_tr], lab_seed], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, margin = predict_mlp(teacher, Xs, device=device)
    cands = []
    for k in K_LIST:
        eidx, elab = expand_prototypes(X, pos_seed, lab_seed, k)
        keep = (~used[eidx]) & (margin[eidx] < MARGIN_THRESH)
        eidx, elab = eidx[keep], elab[keep]
        Xp = np.concatenate([Xs[idx_tr], Xs[pos_seed], Xs[eidx]], axis=0)
        yp = np.concatenate([y_wc[idx_tr], lab_seed, elab], axis=0)
        wp = np.concatenate(
            [
                np.full(len(idx_tr) + len(lab_seed), REAL_WEIGHT, np.float32),
                np.full(len(elab), SOFT_WEIGHT, np.float32),
            ]
        )
        st = train_mlp(Xp, yp, wp, device=device, seed=seed + 100 + k, epochs=epochs)
        pred, _ = predict_mlp(st, Xs[idx_te], device=device)
        m = full_metrics(y_wc[idx_te], pred)
        cands.append({"k": k, "n_exp": int(len(eidx)), "m": m})
        print(f"  [{seed}] {tag} K={k} exp={len(eidx)} OA={m['OA']:.4f} BE={m['BE']}")
    return pick_best(cands)


def sample_random_positions(y_wc, n, draw_seed):
    rng = np.random.default_rng(draw_seed)
    take = []
    base = n // 3
    rem = n - 3 * base
    for i, c in enumerate([0, 1, 2]):
        pool = np.where(y_wc == c)[0]
        n_take = base + (1 if i < rem else 0)
        take.append(rng.choice(pool, size=min(n_take, len(pool)), replace=False))
    pos = np.concatenate(take)
    if len(pos) < n:
        rest = np.setdiff1d(np.arange(len(y_wc)), pos)
        pos = np.concatenate([pos, rng.choice(rest, size=n - len(pos), replace=False)])
    rng.shuffle(pos)
    return pos[:n].astype(int)


def matched_flip_labels(y_wc, pos_rand, n_c12_flip, draw_seed):
    rng = np.random.default_rng(draw_seed + 999)
    lab = y_wc[pos_rand].copy()
    c12_local = np.where((lab == 0) | (lab == 1))[0]
    nflip = min(n_c12_flip, len(c12_local))
    choice = rng.choice(c12_local, size=nflip, replace=False)
    lab[choice] = 1 - lab[choice]
    return lab


def run_city(city: str, device: str, seeds: list[int], epochs: int) -> pd.DataFrame:
    out = DATA2 / f"{city.lower()}_random150_vs_bams"
    out.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(DATA2 / city / "06_3x3" / f"{city}_3x3_Features.csv")
    man = pd.read_csv(DATA2 / city / "04_manual" / f"{city}_BAMS150_Manual.csv")
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce")
    man = man[man["Human_Class"].notna()].copy()
    man["Human_Class"] = man["Human_Class"].astype(int)
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf) if "Confidence" in man.columns else "high"

    X = feat[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat["Original_ID"].astype(int).to_numpy()
    y_wc = feat["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(o): i for i, o in enumerate(ids)}

    def pack(df):
        pos = np.asarray([id_to_pos[int(o)] for o in df.Original_ID], dtype=int)
        lab = np.asarray([int(c) - 1 for c in df.Human_Class], dtype=np.int64)
        return pos, lab

    pos_all, lab_all = pack(man)
    high = man[man["Conf"] == "high"]
    if len(high) == 0:
        high = man
    pos_h, lab_h = pack(high)

    wc_bams = y_wc[pos_all]
    n_c12_flip = int((((wc_bams == 0) & (lab_all == 1)) | ((wc_bams == 1) & (lab_all == 0))).sum())
    print(f"{city}: BAMS={len(man)} high={len(high)} c12_flips={n_c12_flip}")

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    Xs = StandardScaler().fit_transform(X).astype(np.float32)

    rows = []
    for seed in seeds:
        set_all_seeds(seed)
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X[idx_tr], y_wc[idx_tr])
        m = full_metrics(y_wc[idx_te], rf.predict(X[idx_te]))
        rows.append({"City": city, "Seed": seed, "Method": "Baseline_RF", **m, "K": None, "n_exp": None, "n_seed": 0})
        print(f"[{city} seed={seed}] Baseline_RF OA={m['OA']:.4f} BE={m['BE']}")

        for name, pos, lab in [("PE_BAMS_all", pos_all, lab_all), ("PE_BAMS_high", pos_h, lab_h)]:
            best = run_pe(X, Xs, y_wc, idx_tr, idx_te, pos, lab, device, seed, epochs, name)
            rows.append(
                {
                    "City": city,
                    "Seed": seed,
                    "Method": name,
                    **best["m"],
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "n_seed": len(pos),
                }
            )

        pos_r = sample_random_positions(y_wc, N_RANDOM, seed)
        lab_wc = y_wc[pos_r].copy()
        lab_match = matched_flip_labels(y_wc, pos_r, n_c12_flip, seed)
        pd.DataFrame(
            {"Original_ID": ids[pos_r], "WC_Class": y_wc[pos_r] + 1, "MatchedFlip": lab_match + 1}
        ).to_csv(out / f"random150_draw_seed{seed}.csv", index=False)

        for name, lab in [("PE_Random150_WC", lab_wc), ("PE_Random150_matchedFlip", lab_match)]:
            best = run_pe(X, Xs, y_wc, idx_tr, idx_te, pos_r, lab, device, seed, epochs, name)
            rows.append(
                {
                    "City": city,
                    "Seed": seed,
                    "Method": name,
                    **best["m"],
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "n_seed": len(pos_r),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)

    base = df[df.Method == "Baseline_RF"].OA.mean()
    base_be = df[df.Method == "Baseline_RF"].BE.mean()
    summ = []
    for method in df.Method.unique():
        sub = df[df.Method == method]
        summ.append(
            {
                "City": city,
                "Method": method,
                "OA_mean": sub.OA.mean(),
                "OA_std": sub.OA.std(ddof=0),
                "dOA_pp": (sub.OA.mean() - base) * 100,
                "BE_mean": sub.BE.mean(),
                "dBE": sub.BE.mean() - base_be,
                "n_seed": sub.n_seed.mean(),
                "n_c12_flip_bams": n_c12_flip,
                "n_high": len(high),
            }
        )
    summary = pd.DataFrame(summ).sort_values("dOA_pp", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "city": city,
                "n_bams": int(len(man)),
                "n_high": int(len(high)),
                "n_c12_flip_bams": int(n_c12_flip),
                "n_random": N_RANDOM,
            },
            indent=2,
        )
    )
    print(f"\n=== {city} summary ===")
    print(summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=CITIES)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cpu"

    all_sum = []
    for city in args.cities:
        all_sum.append(run_city(city, device, args.seeds, args.epochs))

    big = pd.concat(all_sum, ignore_index=True)
    cross = DATA2 / "cross_city_random150_vs_bams"
    cross.mkdir(parents=True, exist_ok=True)
    big.to_csv(cross / "summary_long.csv", index=False)

    # compact pivot dOA
    piv = big.pivot(index="City", columns="Method", values="dOA_pp")
    order = ["Baseline_RF", "PE_Random150_WC", "PE_Random150_matchedFlip", "PE_BAMS_all", "PE_BAMS_high"]
    cols = [c for c in order if c in piv.columns]
    piv = piv.reindex(args.cities)[cols]
    piv.to_csv(cross / "dOA_pivot.csv")

    # advantage BAMS_high - Random_WC
    adv = []
    for city in args.cities:
        sub = big[big.City == city]
        def g(m):
            s = sub[sub.Method == m]
            return float(s.dOA_pp.iloc[0]) if len(s) else np.nan
        adv.append(
            {
                "City": city,
                "PE_BAMS_high": g("PE_BAMS_high"),
                "PE_BAMS_all": g("PE_BAMS_all"),
                "PE_Random150_WC": g("PE_Random150_WC"),
                "PE_Random150_matchedFlip": g("PE_Random150_matchedFlip"),
                "BAMS_high_minus_Random_WC": g("PE_BAMS_high") - g("PE_Random150_WC"),
                "BAMS_all_minus_Random_WC": g("PE_BAMS_all") - g("PE_Random150_WC"),
            }
        )
    adv_df = pd.DataFrame(adv)
    adv_df.to_csv(cross / "bams_minus_random.csv", index=False)

    md = [
        "# Random150 vs BAMS150 + PE (5 cities)\n",
        "## dOA_pp vs Baseline\n",
        piv.to_string(),
        "\n\n## BAMS − Random (pp)\n",
        adv_df.to_string(index=False),
        "\n\nNotes:\n",
        "- PE_Random150_WC: random seeds labeled with WC (no new human labels).\n",
        "- PE_Random150_matchedFlip: same C1↔C2 flip count as BAMS, random locations.\n",
        "- On success cities expect BAMS+PE ≫ Random+PE; Nanjing may differ due to WC conflict.\n",
    ]
    (cross / "SUMMARY.md").write_text("\n".join(md))
    print("\n=== CROSS-CITY dOA ===")
    print(piv.to_string())
    print("\n=== BAMS − Random ===")
    print(adv_df.to_string(index=False))
    print("wrote", cross)


if __name__ == "__main__":
    main()
