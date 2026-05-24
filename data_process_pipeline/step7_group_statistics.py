import argparse
import logging
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step7")

FP_RADIUS = 2
FP_BITS = 2048
OVERFLOW = 20


def load_fingerprint(sdf_path: str):
    if not isinstance(sdf_path, str) or not sdf_path or not Path(sdf_path).exists():
        return None
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
        for mol in suppl:
            if mol is None:
                continue
            return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    except Exception:
        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            for mol in suppl:
                if mol is None:
                    continue
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    pass
                return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
        except Exception:
            return None
    return None


def avg_pairwise_tanimoto(fps: list) -> float:
    if len(fps) < 2:
        return float("nan")
    sims = [DataStructs.TanimotoSimilarity(a, b) for a, b in combinations(fps, 2)]
    return float(np.mean(sims))


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, sub in df.groupby("final_group_id", sort=False):
        n = len(sub)
        fps = []
        for _, r in sub.iterrows():
            fp = load_fingerprint(r.get("aligned_ligand_sdf", ""))
            if fp is not None:
                fps.append(fp)
        avg_t = avg_pairwise_tanimoto(fps) if len(fps) >= 2 else float("nan")
        rows.append({"final_group_id": gid, "n_ligands": n,
                     "n_valid_fps": len(fps), "avg_tanimoto": avg_t})
    return pd.DataFrame(rows)


def plot_figure(stats: pd.DataFrame, out_path: str):
    counts = stats["n_ligands"].to_numpy()
    capped = np.minimum(counts, OVERFLOW + 1)
    bins_a = np.arange(1, OVERFLOW + 3) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(capped, bins=bins_a, edgecolor="black", color="#3a7fbf")
    tick_positions = list(range(1, OVERFLOW + 1, 2)) + [OVERFLOW + 1]
    tick_labels = [str(t) for t in range(1, OVERFLOW + 1, 2)] + [f">{OVERFLOW}"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Number of Ligands")

    ax = axes[1]
    sim_vals = stats["avg_tanimoto"].dropna().to_numpy()
    bins_b = np.linspace(0.0, 1.0, 11)
    ax.hist(sim_vals, bins=bins_b, edgecolor="black", color="#3a7fbf")
    ax.set_xlabel("Average Tanimoto Similarity")
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))

    fig.text(0.27, 0.04, "(a) The number of ligands per pocket.", ha="center", fontsize=10)
    fig.text(0.77, 0.04, "(b) The Tanimoto similarity of ligands within each group.",
             ha="center", fontsize=10)
    fig.text(0.5, -0.01, "Figure 6: Group Ligands Statistics in the PDBBind dataset.",
             ha="center", fontsize=11)

    plt.subplots_adjust(bottom=0.18, top=0.95, wspace=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-png", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")

    stats = compute_group_stats(df)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output_csv, index=False)

    n_total_groups = len(stats)
    n_multi = int((stats["n_ligands"] >= 2).sum())
    n_with_sim = int(stats["avg_tanimoto"].notna().sum())
    log.info(f"total groups: {n_total_groups}")
    log.info(f"multi-ligand groups (>=2): {n_multi} ({100*n_multi/max(n_total_groups,1):.1f}%)")
    log.info(f"groups with computed similarity: {n_with_sim}")

    plot_figure(stats, args.output_png)
    log.info(f"wrote {args.output_csv} and {args.output_png}")


if __name__ == "__main__":
    main()
