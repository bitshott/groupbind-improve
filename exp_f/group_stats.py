#!/usr/bin/env python3
"""
Group-size and ligand-similarity statistics for the final groups produced by
step6_filter_by_pl_distance.py.

Expected group CSV:
    python step6_filter_by_pl_distance.py \
      --input pdbbind_pocket_aligned_complexes.csv \
      --output-csv pdbbind_final_groups.csv

This script consumes final_group_id directly. It does not rerun sequence-based
or ligand-centroid pocket clustering.

Outputs:
  out/group_summary.csv         - summary table
  out/group_size_histogram.csv  - group-size histogram
  out/group_tanimoto.csv        - per-pair Tanimoto values
  out/group_plots.png           - 2-panel figure

Run:
  python3 group_stats.py \
      --groups_csv pdbbind_final_groups.csv \
      --lp_pdbbind data/LP_PDBBind.csv \
      --out_dir out
"""
import argparse
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from step1_grouping import load_groups_from_step6_csv  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, DataStructs  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# --------------------------- statistics helpers -----------------------------


def size_histogram(groups: dict, bins=(1, 2, 3, 4, 5, (6, 10), (11, None))):
    """Group-size histogram with mixed scalar and (lo, hi) bin specs."""
    sizes = [len(v) for v in groups.values()]
    counts = {}
    for spec in bins:
        if isinstance(spec, int):
            lo, hi, label = spec, spec, str(spec)
        else:
            lo, hi = spec
            label = f"{lo}+" if hi is None else f"{lo}-{hi}"
        counts[label] = sum(1 for s in sizes if s >= lo and (hi is None or s <= hi))
    return counts, sizes


def get_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """Compute Morgan fingerprint from SMILES; None on failure."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def within_group_tanimoto(groups: dict, smiles_by_pdb: dict, max_pairs_per_group: int = 100):
    """Return per-pair Tanimoto values for multi-ligand groups."""
    fps_cache = {}

    def get_fp(pid):
        if pid not in fps_cache:
            fps_cache[pid] = get_morgan_fp(smiles_by_pdb.get(pid))
        return fps_cache[pid]

    all_tanimotos = []
    rng = np.random.default_rng(0)

    for pids in groups.values():
        if len(pids) < 2:
            continue
        fps = [(pid, get_fp(pid)) for pid in pids]
        fps = [(p, f) for p, f in fps if f is not None]
        if len(fps) < 2:
            continue

        pairs = list(combinations(range(len(fps)), 2))
        if len(pairs) > max_pairs_per_group:
            idx = rng.choice(len(pairs), size=max_pairs_per_group, replace=False)
            pairs = [pairs[i] for i in idx]

        for i, j in pairs:
            sim = DataStructs.TanimotoSimilarity(fps[i][1], fps[j][1])
            all_tanimotos.append(sim)

    return all_tanimotos


def smiles_lookup(lp_csv_path: Path) -> dict:
    """Read pdb_id -> SMILES from LP-PDBBind."""
    df = pd.read_csv(lp_csv_path, index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    if "smiles" not in df.columns:
        raise RuntimeError(f"'smiles' column missing from {lp_csv_path}")
    df.index = df.index.astype(str).str.lower().str.strip()
    return df["smiles"].to_dict()


# --------------------------- summary builder --------------------------------


def summarize(name: str, groups: dict, smiles_by_pdb: dict):
    """Compute group-size and within-group ligand-similarity statistics."""
    n_pockets = len(groups)
    sizes = [len(v) for v in groups.values()]
    n_multi = sum(1 for s in sizes if s > 1)
    n_single = n_pockets - n_multi
    n_complexes = sum(sizes)
    n_complexes_in_multi = sum(s for s in sizes if s > 1)

    summary = {
        "subset": name,
        "n_complexes": n_complexes,
        "n_pockets": n_pockets,
        "n_single_ligand_pockets": n_single,
        "n_multi_ligand_pockets": n_multi,
        "frac_multi_ligand_pockets": (n_multi / n_pockets) if n_pockets else 0.0,
        "frac_complexes_in_multi": (
            n_complexes_in_multi / n_complexes) if n_complexes else 0.0,
        "mean_group_size_overall": float(np.mean(sizes)) if sizes else 0.0,
        "mean_group_size_multi_only": (
            float(np.mean([s for s in sizes if s > 1])) if n_multi else 0.0
        ),
        "max_group_size": int(max(sizes)) if sizes else 0,
    }

    size_hist, _ = size_histogram(groups)

    print(f"[stats] computing Tanimoto for {n_multi} multi-ligand groups", flush=True)
    tanimotos = within_group_tanimoto(groups, smiles_by_pdb)
    if tanimotos:
        summary["median_tanimoto"] = float(np.median(tanimotos))
        summary["mean_tanimoto"] = float(np.mean(tanimotos))
        summary["n_tanimoto_pairs"] = len(tanimotos)
    else:
        summary["median_tanimoto"] = None
        summary["mean_tanimoto"] = None
        summary["n_tanimoto_pairs"] = 0

    return summary, size_hist, tanimotos


# --------------------------- plotting --------------------------------------


def make_plots(size_hist: dict, tanimotos: list, out_path: Path):
    """Two-panel figure: group-size bar chart + Tanimoto histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stats] matplotlib not available; skipping plots", file=sys.stderr)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = list(size_hist.keys())
    counts = [size_hist[lbl] for lbl in labels]
    x = np.arange(len(labels))
    ax1.bar(x, counts)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Group size (ligands per pocket)")
    ax1.set_ylabel("Number of pockets")
    ax1.set_title("Group-size distribution")
    ax1.set_yscale("log")

    if tanimotos:
        ax2.hist(tanimotos, bins=20, density=True)
    ax2.set_xlabel("Tanimoto similarity (within-group pairs)")
    ax2.set_ylabel("Density")
    ax2.set_title("Within-group ligand similarity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[stats] wrote {out_path}")


# --------------------------- main ------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups_csv", type=Path, default=Path("pdbbind_final_groups.csv"),
                    help="CSV produced by step6_filter_by_pl_distance.py")
    ap.add_argument("--lp_pdbbind", type=Path, default=Path("data/LP_PDBBind.csv"),
                    help="LP_PDBind/LP_PDBBind CSV with SMILES column")
    ap.add_argument("--out_dir", type=Path, default=Path("out"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    smiles_by_pdb = smiles_lookup(args.lp_pdbbind)
    print(f"[stats] loaded SMILES for {len(smiles_by_pdb)} PDB IDs")

    groups, _, df_groups = load_groups_from_step6_csv(args.groups_csv)
    print(f"[stats] loaded {len(df_groups)} rows from {args.groups_csv}")
    if "final_status" in df_groups.columns:
        print(f"[stats] final_status counts: {df_groups['final_status'].value_counts().to_dict()}")

    summary, size_hist, tanimotos = summarize("Final", groups, smiles_by_pdb)

    summary_df = pd.DataFrame([summary])
    print("\n[stats] ===== Summary =====")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(args.out_dir / "group_summary.csv", index=False)
    print(f"[stats] wrote {args.out_dir / 'group_summary.csv'}")

    size_hist_df = pd.DataFrame.from_dict(size_hist, orient="index", columns=["Final"])
    size_hist_df.index.name = "group_size"
    size_hist_df.to_csv(args.out_dir / "group_size_histogram.csv")
    print(f"[stats] wrote {args.out_dir / 'group_size_histogram.csv'}")
    print("\n[stats] group-size histogram:")
    print(size_hist_df.to_string())

    pd.DataFrame(
        [{"subset": "Final", "tanimoto": t} for t in tanimotos]
    ).to_csv(args.out_dir / "group_tanimoto.csv", index=False)
    print(f"[stats] wrote {args.out_dir / 'group_tanimoto.csv'}")

    make_plots(size_hist, tanimotos, args.out_dir / "group_plots.png")


if __name__ == "__main__":
    main()

