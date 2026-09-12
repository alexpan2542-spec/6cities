#!/usr/bin/env python3
"""
Nanjing Phase D — is PE expansion wasted / noise-amplifying? (H3)

Fix seed geometry; ablate expansion.

Seed packs
  BAMS_Human_high
  BAMS_WC_high

Methods per pack
  SeedsOnly              # train+seeds, no prototype expansion
  PE_std                 # margin<0.15, best K by (BE, −OA) — Phase A protocol
  PE_tight               # margin<0.05
  PE_nolimit             # no margin filter on expansions
  PE_expAgreeWC          # keep expansions only if exp_lab == WC at that point

Also report per-K rows for PE_std (diagnostic).

H3 waste: SeedsOnly ≥ PE_std
H3 amplify: SeedsOnly ≫ PE_std (expansion hurts)
H3 helps: PE_std > SeedsOnly; PE_expAgreeWC best among PE

Outputs -> data2/nanjing_rootcause_v2/phase_d/
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


def load_packs():
    feat = pd.read_csv(DATA2 / CITY / "06_3x3" / f"{CITY}_3x3_Features.csv")
    man = pd.read_csv(DATA2 / CITY / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce")
    man = man[man["Human_Class"].notna()].copy()
    man["Human_Class"] = man["Human_Class"].astype(int)
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf) if "Confidence" in man.columns else "high"

    X = feat[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat["Original_ID"].astype(int).to_numpy()
    y_wc = feat["Class"].astype(int).to_numpy() - 1
    id_to_pos = {int(o): i for i, o in enumerate(ids)}
    man["pos"] = man["Original_ID"].map(id_to_pos)
    man = man[man["pos"].notna()].copy()
    man["pos"] = man["pos"].astype(int)
    high = man[man["Conf"] == "high"]
    pos = high["pos"].to_numpy(dtype=int)
    lab_h = (high["Human_Class"].to_numpy(dtype=int) - 1).astype(np.int64)
    lab_w = y_wc[pos].copy()

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    Xs = StandardScaler().fit_transform(X).astype(np.float32)

    packs = {
        "BAMS_Human_high": {"pos": pos, "lab": lab_h},
        "BAMS_WC_high": {"pos": pos, "lab": lab_w},
    }
    meta = {
        "n_high": int(len(pos)),
        "n_human_disagree_wc": int((lab_h != lab_w).sum()),
        "disagree_rate": float((lab_h != lab_w).mean()),
    }
    print(
        f"Phase D: high seeds={meta['n_high']}, "
        f"human≠WC={meta['n_human_disagree_wc']} ({meta['disagree_rate']:.1%})"
    )
    return X, Xs, y_wc, idx_tr, idx_te, packs, meta


def eval_student(Xs, y_wc, idx_tr, idx_te, pos, lab, exp_idx, exp_lab, device, seed, epochs):
    if len(exp_idx) == 0:
        Xp = np.concatenate([Xs[idx_tr], Xs[pos]], axis=0)
        yp = np.concatenate([y_wc[idx_tr], lab], axis=0)
        wp = np.full(len(yp), REAL_WEIGHT, np.float32)
    else:
        Xp = np.concatenate([Xs[idx_tr], Xs[pos], Xs[exp_idx]], axis=0)
        yp = np.concatenate([y_wc[idx_tr], lab, exp_lab], axis=0)
        wp = np.concatenate(
            [
                np.full(len(idx_tr) + len(lab), REAL_WEIGHT, np.float32),
                np.full(len(exp_lab), SOFT_WEIGHT, np.float32),
            ]
        )
    st = train_mlp(Xp, yp, wp, device=device, seed=seed, epochs=epochs)
    pred, _ = predict_mlp(st, Xs[idx_te], device=device)
    return full_metrics(y_wc[idx_te], pred)


def run_pe_mode(
    X,
    Xs,
    y_wc,
    idx_tr,
    idx_te,
    pos,
    lab,
    device,
    seed,
    epochs,
    tag: str,
    margin_thresh: float | None,
    agree_wc_only: bool,
    record_all_k: bool,
):
    """Returns best cand + optional all-K list."""
    used = np.zeros(len(X), dtype=bool)
    used[idx_tr] = True
    used[pos] = True
    teacher = train_mlp(
        np.concatenate([Xs[idx_tr], Xs[pos]], axis=0),
        np.concatenate([y_wc[idx_tr], lab], axis=0),
        device=device,
        seed=seed,
        epochs=epochs,
    )
    _, margin = predict_mlp(teacher, Xs, device=device)

    cands = []
    for k in K_LIST:
        eidx, elab = expand_prototypes(X, pos, lab, k)
        keep = ~used[eidx]
        if margin_thresh is not None:
            keep &= margin[eidx] < margin_thresh
        eidx, elab = eidx[keep], elab[keep]
        n_before_agree = len(eidx)
        n_agree = int((elab == y_wc[eidx]).sum()) if len(eidx) else 0
        if agree_wc_only and len(eidx):
            mask = elab == y_wc[eidx]
            eidx, elab = eidx[mask], elab[mask]
        m = eval_student(
            Xs, y_wc, idx_tr, idx_te, pos, lab, eidx, elab, device, seed + 100 + k, epochs
        )
        cands.append(
            {
                "k": k,
                "n_exp": int(len(eidx)),
                "n_exp_pre_agree": int(n_before_agree),
                "n_exp_agree_wc": n_agree,
                "m": m,
            }
        )
        print(
            f"  [seed={seed}] {tag:28s} K={k:<2d} exp={len(eidx):<3d} "
            f"(agreeWC={n_agree}/{n_before_agree}) OA={m['OA']:.4f} BE={m['BE']}"
        )
    best = sorted(cands, key=lambda c: (c["m"]["BE"], -c["m"]["OA"]))[0]
    return best, (cands if record_all_k else None)


def run(device: str, seeds: list[int], epochs: int) -> tuple[pd.DataFrame, dict]:
    X, Xs, y_wc, idx_tr, idx_te, packs, meta = load_packs()
    rows = []

    modes = [
        # name, margin_thresh, agree_wc_only, record_all_k
        ("PE_std", 0.15, False, True),
        ("PE_tight", 0.05, False, False),
        ("PE_nolimit", None, False, False),
        ("PE_expAgreeWC", 0.15, True, False),
    ]

    for seed in seeds:
        set_all_seeds(seed)

        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X[idx_tr], y_wc[idx_tr])
        m_rf = full_metrics(y_wc[idx_te], rf.predict(X[idx_te]))
        rows.append(
            {
                "Seed": seed,
                "Pack": "—",
                "Method": "Baseline_RF",
                "K": None,
                "n_seed": 0,
                "n_exp": None,
                "n_exp_pre_agree": None,
                "n_exp_agree_wc": None,
                **m_rf,
            }
        )
        print(f"  [seed={seed}] Baseline_RF OA={m_rf['OA']:.4f} BE={m_rf['BE']}")

        mlp = train_mlp(Xs[idx_tr], y_wc[idx_tr], device=device, seed=seed, epochs=epochs)
        pred, _ = predict_mlp(mlp, Xs[idx_te], device=device)
        m_mlp = full_metrics(y_wc[idx_te], pred)
        rows.append(
            {
                "Seed": seed,
                "Pack": "—",
                "Method": "Baseline_MLP",
                "K": None,
                "n_seed": 0,
                "n_exp": None,
                "n_exp_pre_agree": None,
                "n_exp_agree_wc": None,
                **m_mlp,
            }
        )
        print(f"  [seed={seed}] Baseline_MLP OA={m_mlp['OA']:.4f} BE={m_mlp['BE']}")

        for pack_name, pack in packs.items():
            pos = pack["pos"]
            lab = pack["lab"]

            # SeedsOnly
            m0 = eval_student(
                Xs,
                y_wc,
                idx_tr,
                idx_te,
                pos,
                lab,
                np.asarray([], dtype=int),
                np.asarray([], dtype=np.int64),
                device,
                seed + 50,
                epochs,
            )
            rows.append(
                {
                    "Seed": seed,
                    "Pack": pack_name,
                    "Method": "SeedsOnly",
                    "K": 0,
                    "n_seed": int(len(pos)),
                    "n_exp": 0,
                    "n_exp_pre_agree": 0,
                    "n_exp_agree_wc": 0,
                    **m0,
                }
            )
            print(
                f"  [seed={seed}] {pack_name:16s} SeedsOnly n_seed={len(pos)} "
                f"OA={m0['OA']:.4f} BE={m0['BE']}"
            )

            for mode_name, mthresh, agree_only, rec_k in modes:
                tag = f"{pack_name}/{mode_name}"
                best, allk = run_pe_mode(
                    X,
                    Xs,
                    y_wc,
                    idx_tr,
                    idx_te,
                    pos,
                    lab,
                    device,
                    seed,
                    epochs,
                    tag,
                    mthresh,
                    agree_only,
                    rec_k,
                )
                rows.append(
                    {
                        "Seed": seed,
                        "Pack": pack_name,
                        "Method": mode_name,
                        "K": best["k"],
                        "n_seed": int(len(pos)),
                        "n_exp": best["n_exp"],
                        "n_exp_pre_agree": best["n_exp_pre_agree"],
                        "n_exp_agree_wc": best["n_exp_agree_wc"],
                        **best["m"],
                    }
                )
                if allk is not None:
                    for c in allk:
                        rows.append(
                            {
                                "Seed": seed,
                                "Pack": pack_name,
                                "Method": f"PE_std_K{c['k']}",
                                "K": c["k"],
                                "n_seed": int(len(pos)),
                                "n_exp": c["n_exp"],
                                "n_exp_pre_agree": c["n_exp_pre_agree"],
                                "n_exp_agree_wc": c["n_exp_agree_wc"],
                                **c["m"],
                            }
                        )

    return pd.DataFrame(rows), meta


def summarize(df: pd.DataFrame, meta: dict) -> tuple[str, pd.DataFrame]:
    base_rf = float(df[df.Method == "Baseline_RF"].OA.mean())
    base_mlp = float(df[df.Method == "Baseline_MLP"].OA.mean())
    be_rf = float(df[df.Method == "Baseline_RF"].BE.mean())

    # main methods only (exclude per-K diagnostics from primary table, but keep in CSV)
    main_methods = {
        "Baseline_RF",
        "Baseline_MLP",
        "SeedsOnly",
        "PE_std",
        "PE_tight",
        "PE_nolimit",
        "PE_expAgreeWC",
    }
    sub = df[df.Method.isin(main_methods)].copy()

    rows = []
    for (pack, method), g in sub.groupby(["Pack", "Method"], sort=False):
        rows.append(
            {
                "Pack": pack,
                "Method": method,
                "OA_mean": float(g.OA.mean()),
                "OA_std": float(g.OA.std(ddof=0)),
                "dOA_vs_RF_pp": (float(g.OA.mean()) - base_rf) * 100,
                "dOA_vs_MLP_pp": (float(g.OA.mean()) - base_mlp) * 100,
                "BE_mean": float(g.BE.mean()),
                "dBE_vs_RF": float(g.BE.mean()) - be_rf,
                "n_seed_mean": float(g.n_seed.mean()),
                "n_exp_mean": float(g.n_exp.dropna().mean()) if g.n_exp.notna().any() else np.nan,
                "agree_frac_mean": (
                    float(
                        (
                            g["n_exp_agree_wc"].astype(float)
                            / g["n_exp_pre_agree"].astype(float).replace(0, np.nan)
                        ).mean()
                    )
                    if g["n_exp_pre_agree"].notna().any()
                    and float(g["n_exp_pre_agree"].fillna(0).sum()) > 0
                    else np.nan
                ),
            }
        )
    overall = pd.DataFrame(rows)

    lines = [
        "# Nanjing Phase D — expansion ablation (H3)\n",
        f"- High seeds n={meta['n_high']}; human≠WC={meta['n_human_disagree_wc']} "
        f"({meta['disagree_rate']:.1%})\n",
        "\n## Mean over seeds (main methods)\n\n",
        overall.to_string(index=False),
        "\n\n## Reading guide\n",
        "- SeedsOnly ≥ PE_std → expansion wasted.\n",
        "- SeedsOnly ≫ PE_std → expansion amplifies noise.\n",
        "- PE_expAgreeWC best → conflict amplification in expanded labels.\n",
        "- PE_tight vs PE_nolimit → margin gate usefulness.\n",
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
        default=DATA2 / "nanjing_rootcause_v2" / "phase_d",
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

    df, meta = run(device=device, seeds=seeds, epochs=epochs)
    md, overall = summarize(df, meta)
    df.to_csv(out / "seed_results.csv", index=False)
    overall.to_csv(out / "overall_summary.csv", index=False)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps(
            {
                "city": CITY,
                "seeds": seeds,
                "epochs": epochs,
                "out_dir": str(out),
                "k_list": K_LIST,
                **meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + md)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
