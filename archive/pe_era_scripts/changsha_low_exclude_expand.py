#!/usr/bin/env python3
"""
Changsha: expand Confidence=low as interference zones and exclude from training.

Idea
  low = human-uncertain / noisy points
  1) do NOT use low as PE hard seeds (already true for PE_high)
  2) expand low into the 15k pool (cosine KNN)
  3) drop those expanded points (+ low seeds) from the TRAIN set
  Test set stays fixed (same WC eval) for comparable OA.

Methods
  Baseline
  PE_high                     # standard high-only PE
  PE_high_dropLowSeeds        # also remove low BAMS points from train (no expand)
  PE_high_exclLowK{k}         # remove low seeds + KNN(k) neighborhood from train
  Baseline_exclLowK{k}        # no PE; only interference exclusion (diagnostic)

Outputs -> data2/changsha_low_exclude_expand/
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
CITY = "Changsha"
FEATURES_3X3 = [
    "B2_mean", "B2_stdDev", "B3_mean", "B3_stdDev", "B4_mean", "B4_stdDev",
    "B8_mean", "B8_stdDev", "B11_mean", "B11_stdDev", "B12_mean", "B12_stdDev",
    "NDVI_mean", "NDVI_stdDev", "NDBI_mean", "NDBI_stdDev",
    "MNDWI_mean", "MNDWI_stdDev",
]
SPLIT = dict(test_size=0.3, random_state=42)
SEEDS = [0, 1, 2, 3, 4]
K_PE = [5, 10, 20, 30]
K_EXCL = [5, 10, 20, 30]
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
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
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


def get_manual(path: Path) -> pd.DataFrame:
    m = pd.read_csv(path)
    m["Human_Class"] = pd.to_numeric(m["Human_Class"], errors="coerce")
    m = m[m["Human_Class"].notna()].copy()
    m["Human_Class"] = m["Human_Class"].astype(int)
    m["Original_ID"] = m["Original_ID"].astype(int)
    if "Confidence" in m.columns:
        m["Conf"] = m["Confidence"].map(norm_conf)
    elif "Unnamed: 4" in m.columns:
        m["Conf"] = m["Unnamed: 4"].map(norm_conf)
    else:
        m["Conf"] = "high"
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
def predict_proba_mlp(model, X, device="cpu"):
    model.eval()
    probs = torch.softmax(
        model(torch.from_numpy(X.astype(np.float32)).to(device)), dim=1
    ).cpu().numpy()
    order = np.argsort(probs, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    conf = probs[np.arange(len(probs)), top1]
    margin = conf - probs[np.arange(len(probs)), top2]
    return top1.astype(np.int64), conf, margin


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


def expand_exclude_zone(feature, pos_low: np.ndarray, k: int) -> np.ndarray:
    """Neighborhood of low points (including the lows themselves)."""
    if len(pos_low) == 0:
        return np.asarray([], dtype=int)
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn_model.fit(feature)
    neigh_idx = nn_model.kneighbors(feature[pos_low], return_distance=False)
    zone = set(int(i) for i in pos_low.tolist())
    for neighbors in neigh_idx:
        for nidx in neighbors.tolist():
            zone.add(int(nidx))
    return np.asarray(sorted(zone), dtype=int)


def full_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "OA": float(accuracy_score(y_true, y_pred)),
        "Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Boundary Error": int(cm[0, 1] + cm[1, 0]),
    }


def pick_best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["m"]["Boundary Error"], -c["m"]["OA"]))[0]


def load_city():
    city_root = DATA2 / CITY
    feat3 = pd.read_csv(city_root / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = get_manual(city_root / "04_manual" / f"{CITY}_BAMS150_Manual.csv")

    X_all = feat3[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat3["Original_ID"].astype(int).to_numpy()
    y_wc = feat3["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(oid): i for i, oid in enumerate(ids)}

    man = man.copy()
    man["pos"] = man["Original_ID"].map(id_to_pos)
    man = man[man["pos"].notna()].copy()
    man["pos"] = man["pos"].astype(int)
    man["H0"] = man["Human_Class"].astype(int) - 1

    high = man[man["Conf"] == "high"]
    low = man[man["Conf"] == "low"]
    pos_high = high["pos"].to_numpy(dtype=int)
    lab_high = high["H0"].to_numpy(dtype=np.int64)
    pos_low = low["pos"].to_numpy(dtype=int)

    idx = np.arange(len(X_all))
    idx_train, idx_test = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    Xs = StandardScaler().fit_transform(X_all).astype(np.float32)

    conf_counts = man["Conf"].value_counts(dropna=False).to_dict()
    print(
        f"{CITY} low-exclude: conf={ {str(k): int(v) for k,v in conf_counts.items()} }; "
        f"high={len(pos_high)} low={len(pos_low)}"
    )
    return {
        "X_all": X_all,
        "Xs": Xs,
        "y_wc": y_wc,
        "ids": ids,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "pos_high": pos_high,
        "lab_high": lab_high,
        "pos_low": pos_low,
        "conf_counts": {str(k): int(v) for k, v in conf_counts.items()},
    }


def filter_train(idx_train, exclude_pos, protect_pos):
    excl = set(int(i) for i in exclude_pos.tolist()) if len(exclude_pos) else set()
    prot = set(int(i) for i in protect_pos.tolist()) if len(protect_pos) else set()
    # never drop high seeds / protect
    excl -= prot
    keep = np.asarray([i for i in idx_train if int(i) not in excl], dtype=int)
    n_dropped = int(len(idx_train) - len(keep))
    return keep, n_dropped, len(excl)


def run_pe_on_train(
    X_all, Xs, y_wc, idx_tr, idx_te, pos_seed, lab_seed, device, seed, epochs, tag
):
    if len(pos_seed) == 0:
        return None
    used = np.zeros(len(X_all), dtype=bool)
    used[idx_tr] = True
    used[pos_seed] = True

    teacher = train_mlp(
        np.concatenate([Xs[idx_tr], Xs[pos_seed]], axis=0),
        np.concatenate([y_wc[idx_tr], lab_seed], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, _, margin = predict_proba_mlp(teacher, Xs, device=device)

    cands = []
    for k in K_PE:
        eidx, elab = expand_prototypes(X_all, pos_seed, lab_seed, k)
        keep = (~used[eidx]) & (margin[eidx] < MARGIN_THRESH)
        # also do not expand onto excluded train holes — already not in idx_tr
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
        pred, _, _ = predict_proba_mlp(st, Xs[idx_te], device=device)
        m = full_metrics(y_wc[idx_te], pred)
        cands.append({"k": k, "n_exp": int(len(eidx)), "m": m})
        print(
            f"  [seed={seed}] {tag:28s} Kpe={k:<2d} n_tr={len(idx_tr):<5d} "
            f"exp={len(eidx):<3d} OA={m['OA']:.4f} BE={m['Boundary Error']}"
        )
    return pick_best(cands)


def run(device: str, seeds: list[int], epochs: int, out: Path):
    data = load_city()
    X_all, Xs, y_wc = data["X_all"], data["Xs"], data["y_wc"]
    idx_train, idx_test = data["idx_train"], data["idx_test"]
    pos_high, lab_high, pos_low = data["pos_high"], data["lab_high"], data["pos_low"]
    ids = data["ids"]

    rows = []
    zone_rows = []

    for seed in seeds:
        set_all_seeds(seed)

        # Baseline RF / MLP on full train
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X_all[idx_train], y_wc[idx_train])
        m_rf = full_metrics(y_wc[idx_test], rf.predict(X_all[idx_test]))
        rows.append(
            {
                "Seed": seed, "Method": "Baseline_RF", "K_excl": 0,
                "n_train": len(idx_train), "n_dropped": 0, "n_zone": 0,
                "n_seed": 0, "K_pe": None, "n_exp": None, **m_rf,
            }
        )
        print(f"  [seed={seed}] Baseline_RF OA={m_rf['OA']:.4f} BE={m_rf['Boundary Error']}")

        mlp = train_mlp(Xs[idx_train], y_wc[idx_train], device=device, seed=seed, epochs=epochs)
        pred, _, _ = predict_proba_mlp(mlp, Xs[idx_test], device=device)
        m_mlp = full_metrics(y_wc[idx_test], pred)
        rows.append(
            {
                "Seed": seed, "Method": "Baseline_MLP", "K_excl": 0,
                "n_train": len(idx_train), "n_dropped": 0, "n_zone": 0,
                "n_seed": 0, "K_pe": None, "n_exp": None, **m_mlp,
            }
        )

        # PE_high standard
        best = run_pe_on_train(
            X_all, Xs, y_wc, idx_train, idx_test, pos_high, lab_high,
            device, seed, epochs, "PE_high",
        )
        rows.append(
            {
                "Seed": seed, "Method": "PE_high", "K_excl": 0,
                "n_train": len(idx_train), "n_dropped": 0, "n_zone": 0,
                "n_seed": len(pos_high), "K_pe": best["k"], "n_exp": best["n_exp"], **best["m"],
            }
        )

        # drop low seeds from train only
        tr1, nd1, nz1 = filter_train(idx_train, pos_low, pos_high)
        best = run_pe_on_train(
            X_all, Xs, y_wc, tr1, idx_test, pos_high, lab_high,
            device, seed, epochs, "PE_high_dropLowSeeds",
        )
        rows.append(
            {
                "Seed": seed, "Method": "PE_high_dropLowSeeds", "K_excl": 0,
                "n_train": len(tr1), "n_dropped": nd1, "n_zone": int(len(pos_low)),
                "n_seed": len(pos_high), "K_pe": best["k"], "n_exp": best["n_exp"], **best["m"],
            }
        )

        for k_excl in K_EXCL:
            zone = expand_exclude_zone(X_all, pos_low, k_excl)
            # protect high seeds
            tr, n_dropped, n_zone = filter_train(idx_train, zone, pos_high)
            # overlap with test (info only)
            n_zone_in_test = int(np.isin(zone, idx_test).sum())
            n_zone_in_train = int(np.isin(zone, idx_train).sum())

            zone_rows.append(
                {
                    "Seed": seed, "K_excl": k_excl, "n_low": len(pos_low),
                    "n_zone": len(zone), "n_zone_train": n_zone_in_train,
                    "n_zone_test": n_zone_in_test, "n_dropped_train": n_dropped,
                    "n_train_kept": len(tr),
                }
            )
            pd.DataFrame(
                {
                    "Original_ID": ids[zone],
                    "pos": zone,
                    "in_train": np.isin(zone, idx_train).astype(int),
                    "in_test": np.isin(zone, idx_test).astype(int),
                    "is_low_seed": np.isin(zone, pos_low).astype(int),
                    "is_high_seed": np.isin(zone, pos_high).astype(int),
                }
            ).to_csv(out / f"exclude_zone_K{k_excl}_seed{seed}.csv", index=False)

            # Baseline with exclusion only
            mlp_e = train_mlp(Xs[tr], y_wc[tr], device=device, seed=seed, epochs=epochs)
            pred_e, _, _ = predict_proba_mlp(mlp_e, Xs[idx_test], device=device)
            m_e = full_metrics(y_wc[idx_test], pred_e)
            rows.append(
                {
                    "Seed": seed, "Method": f"Baseline_exclLowK{k_excl}", "K_excl": k_excl,
                    "n_train": len(tr), "n_dropped": n_dropped, "n_zone": len(zone),
                    "n_seed": 0, "K_pe": None, "n_exp": None, **m_e,
                }
            )
            print(
                f"  [seed={seed}] Baseline_exclLowK{k_excl:<2d} drop={n_dropped} "
                f"zone={len(zone)} OA={m_e['OA']:.4f}"
            )

            # PE_high + exclusion
            best = run_pe_on_train(
                X_all, Xs, y_wc, tr, idx_test, pos_high, lab_high,
                device, seed, epochs, f"PE_high_exclLowK{k_excl}",
            )
            rows.append(
                {
                    "Seed": seed, "Method": f"PE_high_exclLowK{k_excl}", "K_excl": k_excl,
                    "n_train": len(tr), "n_dropped": n_dropped, "n_zone": len(zone),
                    "n_seed": len(pos_high), "K_pe": best["k"], "n_exp": best["n_exp"], **best["m"],
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(zone_rows), data["conf_counts"]


def summarize(df: pd.DataFrame, conf_counts: dict) -> tuple[str, pd.DataFrame]:
    base = float(df[df.Method == "Baseline_RF"].OA.mean())
    base_be = float(df[df.Method == "Baseline_RF"]["Boundary Error"].mean())
    rows = []
    for method, g in df.groupby("Method", sort=False):
        rows.append(
            {
                "Method": method,
                "OA_mean": float(g.OA.mean()),
                "OA_std": float(g.OA.std(ddof=0)),
                "dOA_vs_RF_pp": (float(g.OA.mean()) - base) * 100,
                "BE_mean": float(g["Boundary Error"].mean()),
                "dBE": float(g["Boundary Error"].mean()) - base_be,
                "n_train_mean": float(g.n_train.mean()),
                "n_dropped_mean": float(g.n_dropped.mean()),
                "n_zone_mean": float(g.n_zone.mean()),
                "n_exp_mean": float(g.n_exp.dropna().mean()) if g.n_exp.notna().any() else np.nan,
                "n_seed_mean": float(g.n_seed.mean()),
            }
        )
    overall = pd.DataFrame(rows)
    lines = [
        "# Changsha: expand low-confidence as interference and exclude from train\n",
        f"- Confidence counts: {conf_counts}\n",
        "- Test set fixed; only training points in low-expand zones are dropped.\n",
        "- High seeds are protected (never dropped).\n",
        "\n## Mean over seeds\n\n",
        overall.to_string(index=False),
        "\n\n## Reading\n",
        "- If PE_high_exclLowK* ≫ PE_high → excluding expanded lows helps.\n",
        "- If ≈ PE_high → lows already unused as seeds; neighborhood exclusion adds little.\n",
        "- If worse → dropped too many useful WC-consistent train points.\n",
    ]
    return "\n".join(lines), overall


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DATA2 / "changsha_low_exclude_expand",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0] if args.quick else args.seeds
    epochs = min(args.epochs, 40) if args.quick else args.epochs
    print("device:", device, "seeds:", seeds, "epochs:", epochs, "out:", out)

    df, zones, conf_counts = run(device, seeds, epochs, out)
    md, overall = summarize(df, conf_counts)
    df.to_csv(out / "seed_results.csv", index=False)
    zones.to_csv(out / "zone_stats.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "k_pe": K_PE,
                "k_excl": K_EXCL,
                "margin_thresh": MARGIN_THRESH,
                "conf_counts": conf_counts,
                "out_dir": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
