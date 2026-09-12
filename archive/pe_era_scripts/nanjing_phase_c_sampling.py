#!/usr/bin/env python3
"""
Nanjing Phase C — sampling元凶 (H2)

Fair contrast: same PE protocol, seed labels = WC (except one Human reference).
Vary only *where* the 150/73 seeds come from.

Methods
  Baseline_RF / Baseline_MLP
  PE_BAMS_Human_high      # passport reference (human labels)
  PE_BAMS_WC_all          # BAMS geometry, WC labels
  PE_BAMS_WC_high
  PE_Random150_WC         # class-balanced random, WC labels
  PE_Margin150_WC         # 150 lowest teacher-margin, WC labels
  PE_Easy150_WC           # 150 highest teacher-margin, WC labels

H2 supported if Random/Easy ≫ BAMS_WC on WC-OA (BAMS locations toxic),
or Margin≈BAMS_WC both fail (city-hard / PE weak regardless of AL).

Outputs -> data2/nanjing_rootcause_v2/phase_c/
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
MARGIN_THRESH = 0.15
SOFT_WEIGHT = 0.3
REAL_WEIGHT = 1.0
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 256
N_SEED = 150


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


def run_pe(X, Xs, y_wc, idx_tr, idx_te, pos_seed, lab_seed, device, seed, epochs, tag):
    pos_seed = np.asarray(pos_seed, dtype=int)
    lab_seed = np.asarray(lab_seed, dtype=np.int64)
    if len(pos_seed) == 0:
        return None
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
        print(
            f"  [seed={seed}] {tag:22s} K={k:<2d} n_seed={len(pos_seed):<3d} "
            f"exp={len(eidx):<3d} OA={m['OA']:.4f} BE={m['BE']}"
        )
    return sorted(cands, key=lambda c: (c["m"]["BE"], -c["m"]["OA"]))[0]


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


def sample_by_margin(margin, n, hardest: bool, draw_seed: int):
    """hardest=True -> lowest margin; False -> highest margin (easy)."""
    order = np.argsort(margin)
    if not hardest:
        order = order[::-1]
    # slight jitter for ties via seed-stable secondary key
    rng = np.random.default_rng(draw_seed)
    # take top candidates with small random among near-ties is unnecessary; take first n
    # but shuffle among equal margins
    pos = order[:n].astype(int)
    rng.shuffle(pos)
    return pos


def load_bams(ids, y_wc):
    man = pd.read_csv(DATA2 / CITY / "04_manual" / f"{CITY}_BAMS150_Manual.csv")
    man["Human_Class"] = pd.to_numeric(man["Human_Class"], errors="coerce")
    man = man[man["Human_Class"].notna()].copy()
    man["Human_Class"] = man["Human_Class"].astype(int)
    man["Original_ID"] = man["Original_ID"].astype(int)
    man["Conf"] = man["Confidence"].map(norm_conf) if "Confidence" in man.columns else "high"
    id_to_pos = {int(o): i for i, o in enumerate(ids)}
    man["pos"] = man["Original_ID"].map(id_to_pos)
    man = man[man["pos"].notna()].copy()
    man["pos"] = man["pos"].astype(int)
    man["H0"] = man["Human_Class"].astype(int) - 1
    man["WC0"] = y_wc[man["pos"].to_numpy()]
    return man


def run(device: str, seeds: list[int], epochs: int, out: Path) -> tuple[pd.DataFrame, dict]:
    feat = pd.read_csv(DATA2 / CITY / "06_3x3" / f"{CITY}_3x3_Features.csv")
    X = feat[FEATURES_3X3].to_numpy(dtype=np.float32)
    ids = feat["Original_ID"].astype(int).to_numpy()
    y_wc = feat["Class"].astype(int).to_numpy() - 1
    man = load_bams(ids, y_wc)

    pos_bams = man["pos"].to_numpy(dtype=int)
    lab_bams_h = man["H0"].to_numpy(dtype=np.int64)
    high = man["Conf"].to_numpy() == "high"
    pos_bams_high = pos_bams[high]
    lab_bams_human_high = lab_bams_h[high]
    lab_bams_wc_all = y_wc[pos_bams].copy()
    lab_bams_wc_high = y_wc[pos_bams_high].copy()

    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=SPLIT["test_size"], random_state=SPLIT["random_state"], stratify=y_wc
    )
    Xs = StandardScaler().fit_transform(X).astype(np.float32)

    meta = {
        "n_bams": int(len(man)),
        "n_bams_high": int(high.sum()),
        "n_seed_budget": N_SEED,
    }
    print(
        f"Phase C Nanjing: BAMS={meta['n_bams']} high={meta['n_bams_high']} "
        f"budget={N_SEED}"
    )

    rows = []
    for seed in seeds:
        set_all_seeds(seed)

        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(X[idx_tr], y_wc[idx_tr])
        m_rf = full_metrics(y_wc[idx_te], rf.predict(X[idx_te]))
        rows.append(
            {
                "Seed": seed,
                "Method": "Baseline_RF",
                **m_rf,
                "K": None,
                "n_exp": None,
                "n_seed": 0,
            }
        )
        print(f"  [seed={seed}] Baseline_RF OA={m_rf['OA']:.4f} BE={m_rf['BE']}")

        mlp = train_mlp(Xs[idx_tr], y_wc[idx_tr], device=device, seed=seed, epochs=epochs)
        pred_m, margin_pool = predict_mlp(mlp, Xs, device=device)
        m_mlp = full_metrics(y_wc[idx_te], pred_m[idx_te])
        rows.append(
            {
                "Seed": seed,
                "Method": "Baseline_MLP",
                **m_mlp,
                "K": None,
                "n_exp": None,
                "n_seed": 0,
            }
        )
        print(f"  [seed={seed}] Baseline_MLP OA={m_mlp['OA']:.4f} BE={m_mlp['BE']}")

        # sampling sets (labels = WC except Human reference)
        pos_rand = sample_random_positions(y_wc, N_SEED, seed)
        pos_margin = sample_by_margin(margin_pool, N_SEED, hardest=True, draw_seed=seed + 17)
        pos_easy = sample_by_margin(margin_pool, N_SEED, hardest=False, draw_seed=seed + 31)

        # save draws
        for tag, pos in [
            ("random150", pos_rand),
            ("margin150", pos_margin),
            ("easy150", pos_easy),
        ]:
            pd.DataFrame(
                {
                    "Original_ID": ids[pos],
                    "WC_Class": y_wc[pos] + 1,
                    "margin": margin_pool[pos],
                    "in_test": np.isin(pos, idx_te).astype(int),
                }
            ).to_csv(out / f"{tag}_draw_seed{seed}.csv", index=False)

        specs = [
            ("PE_BAMS_Human_high", pos_bams_high, lab_bams_human_high),
            ("PE_BAMS_WC_all", pos_bams, lab_bams_wc_all),
            ("PE_BAMS_WC_high", pos_bams_high, lab_bams_wc_high),
            ("PE_Random150_WC", pos_rand, y_wc[pos_rand].copy()),
            ("PE_Margin150_WC", pos_margin, y_wc[pos_margin].copy()),
            ("PE_Easy150_WC", pos_easy, y_wc[pos_easy].copy()),
        ]
        for name, pos, lab in specs:
            best = run_pe(X, Xs, y_wc, idx_tr, idx_te, pos, lab, device, seed, epochs, name)
            if best is None:
                continue
            rows.append(
                {
                    "Seed": seed,
                    "Method": name,
                    **best["m"],
                    "K": best["k"],
                    "n_exp": best["n_exp"],
                    "n_seed": int(len(pos)),
                }
            )

    return pd.DataFrame(rows), meta


def summarize(df: pd.DataFrame, meta: dict) -> tuple[str, pd.DataFrame]:
    base_rf = float(df[df.Method == "Baseline_RF"].OA.mean())
    base_mlp = float(df[df.Method == "Baseline_MLP"].OA.mean())
    be_rf = float(df[df.Method == "Baseline_RF"].BE.mean())
    be_mlp = float(df[df.Method == "Baseline_MLP"].BE.mean())

    rows = []
    for method, g in df.groupby("Method", sort=False):
        rows.append(
            {
                "Method": method,
                "OA_mean": float(g.OA.mean()),
                "OA_std": float(g.OA.std(ddof=0)),
                "dOA_vs_RF_pp": (float(g.OA.mean()) - base_rf) * 100,
                "dOA_vs_MLP_pp": (float(g.OA.mean()) - base_mlp) * 100,
                "BE_mean": float(g.BE.mean()),
                "dBE_vs_RF": float(g.BE.mean()) - be_rf,
                "dBE_vs_MLP": float(g.BE.mean()) - be_mlp,
                "n_seed_mean": float(g.n_seed.mean()),
                "n_exp_mean": float(g.n_exp.dropna().mean()) if g.n_exp.notna().any() else np.nan,
            }
        )
    overall = pd.DataFrame(rows)

    lines = [
        "# Nanjing Phase C — sampling contrast (H2)\n",
        f"- BAMS n={meta['n_bams']} high={meta['n_bams_high']}; budget={meta['n_seed_budget']}\n",
        "- All PE_*_WC use WorldCover labels on the selected positions.\n",
        "- PE_BAMS_Human_high is the passport reference only.\n",
        "\n## Mean over seeds\n\n",
        overall.to_string(index=False),
        "\n\n## Reading guide\n",
        "- H2 (BAMS locations toxic): Random_WC / Easy_WC ≫ BAMS_WC.\n",
        "- City-hard / PE weak: all WC samplers stuck near ~0–0.5 pp vs RF.\n",
        "- Margin≈BAMS: uncertainty sampling same regime as BAMS.\n",
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
        default=DATA2 / "nanjing_rootcause_v2" / "phase_c",
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

    df, meta = run(device=device, seeds=seeds, epochs=epochs, out=out)
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
                "margin_thresh": MARGIN_THRESH,
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
