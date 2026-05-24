# exp_mcs_trivial

Computational experiment for testing whether MCS-based ligand-ligand correspondences inside `final_group_id` groups can become trivial or chemically weak.

The pipeline uses the CSV produced after reproducing Algorithm 1 and evaluates all pairwise ligand MCS comparisons inside each `final_group_id`.

## Input

Expected CSV columns:

```text
pdb_id
final_group_id
final_status
aligned_ligand_sdf
ligand_sdf
ligand_mol2
```

The pipeline uses `aligned_ligand_sdf` first, then falls back to `ligand_sdf`, then `ligand_mol2`.

## Install

```bash
conda env create -f environment.yml
conda activate exp-mcs-trivial
```

## Run

From inside `exp_mcs_trivial`:

```bash
bash run_mcs_trivial.sh ../pdbbind_final_groups.csv results 4
```

or:

```bash
make all CSV=../pdbbind_final_groups.csv OUT=results N_JOBS=4
```

## Main outputs

```text
results/mcs_pairwise_quality.csv
results/mcs_group_quality_summary.csv
results/mcs_global_summary.txt
results/mcs_top_trivial_groups.csv
results/mcs_report.md
results/fig_mcs_fraction_hist.png
results/fig_trivial_fraction_hist.png
```

## Interpretation

A `final_group_id` is suspicious if it has:

```text
frac_trivial_mcs > 0.5
median_mcs_atom_fraction_min < 0.2
median_mcs_atoms <= 2
```

This supports the weakness that MCS-derived ligand-ligand correspondences may become chemically trivial or noisy for structurally diverse ligand groups.
