#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Summarize MCS triviality experiment.")
    p.add_argument("--pairwise-csv", required=True)
    p.add_argument("--group-summary-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--weak-frac-trivial", type=float, default=0.50)
    p.add_argument("--weak-median-frac", type=float, default=0.20)
    p.add_argument("--weak-median-atoms", type=float, default=2.0)
    return p.parse_args()


def save_hist(series, path, xlabel, title):
    s = series.dropna()
    if len(s) == 0:
        return
    plt.figure()
    plt.hist(s, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairwise = pd.read_csv(args.pairwise_csv)
    group = pd.read_csv(args.group_summary_csv)

    group["weakness3_flag"] = (
        (group["frac_trivial_mcs"] > args.weak_frac_trivial)
        | (group["median_mcs_atom_fraction_min"] < args.weak_median_frac)
        | (group["median_mcs_atoms"] <= args.weak_median_atoms)
    )

    group = group.sort_values(
        ["weakness3_flag", "frac_trivial_mcs", "median_mcs_atom_fraction_min"],
        ascending=[False, False, True],
    )

    top = group[group["weakness3_flag"]].copy()
    top.to_csv(out_dir / "mcs_top_trivial_groups.csv", index=False)

    group.to_csv(args.group_summary_csv, index=False)

    valid = pairwise[~pairwise["mcs_failed"]].copy()

    save_hist(
        valid["mcs_atom_fraction_min"],
        out_dir / "fig_mcs_fraction_hist.png",
        "MCS atoms / min ligand atoms",
        "Pairwise relative MCS size",
    )

    save_hist(
        group["frac_trivial_mcs"],
        out_dir / "fig_trivial_fraction_hist.png",
        "Fraction of trivial MCS pairs per final_group_id",
        "Group-level trivial MCS fraction",
    )

    n_pairs = len(pairwise)
    n_valid = int((~pairwise["mcs_failed"]).sum())
    n_failed = int(pairwise["mcs_failed"].sum())
    n_groups = group["final_group_id"].nunique()
    n_flagged = int(group["weakness3_flag"].sum())

    global_txt = f"""MCS triviality experiment summary

Pairwise comparisons:
  total_pairs: {n_pairs}
  valid_pairs: {n_valid}
  failed_pairs: {n_failed}

Groups:
  total_final_group_id: {n_groups}
  weakness3_flagged_groups: {n_flagged}
  weakness3_flagged_fraction: {n_flagged / max(1, n_groups):.4f}

Pairwise valid MCS:
  median_mcs_atoms: {valid["mcs_atoms"].median():.4f}
  median_mcs_atom_fraction_min: {valid["mcs_atom_fraction_min"].median():.4f}
  frac_single_atom_mcs: {valid["is_single_atom_mcs"].mean():.4f}
  frac_trivial_mcs: {valid["is_trivial_mcs"].mean():.4f}

Weakness 3 flag:
  frac_trivial_mcs > {args.weak_frac_trivial}
  OR median_mcs_atom_fraction_min < {args.weak_median_frac}
  OR median_mcs_atoms <= {args.weak_median_atoms}
"""
    (out_dir / "mcs_global_summary.txt").write_text(global_txt)

    report = f"""# MCS triviality report

## Aim

This experiment tests whether MCS-derived ligand-ligand correspondences inside `final_group_id` groups can become chemically trivial or noisy.

## Grouping variable

`final_group_id`

## Main result

- Total pairwise ligand comparisons: `{n_pairs}`
- Valid MCS comparisons: `{n_valid}`
- Failed MCS comparisons: `{n_failed}`
- Total groups: `{n_groups}`
- Weakness-3 flagged groups: `{n_flagged}`
- Flagged group fraction: `{n_flagged / max(1, n_groups):.4f}`

## Pairwise MCS statistics

- Median MCS atoms: `{valid["mcs_atoms"].median():.4f}`
- Median relative MCS size: `{valid["mcs_atom_fraction_min"].median():.4f}`
- Fraction of single-atom MCS pairs: `{valid["is_single_atom_mcs"].mean():.4f}`
- Fraction of trivial MCS pairs: `{valid["is_trivial_mcs"].mean():.4f}`

## Weakness-3 flag criterion

A group is flagged if at least one condition is true:

```text
frac_trivial_mcs > {args.weak_frac_trivial}
median_mcs_atom_fraction_min < {args.weak_median_frac}
median_mcs_atoms <= {args.weak_median_atoms}
```

## Interpretation

If many `final_group_id` groups are flagged, this supports the weakness that MCS-based ligand-ligand matching can become chemically weak for structurally diverse ligand groups. In that case, triangle attention may receive noisy ligand-ligand correspondences rather than conserved pharmacophoric anchors.
"""
    (out_dir / "mcs_report.md").write_text(report)

    print(global_txt)
    print(f"Wrote: {out_dir / 'mcs_top_trivial_groups.csv'}")
    print(f"Wrote: {out_dir / 'mcs_report.md'}")


if __name__ == "__main__":
    main()
