#!/usr/bin/env python3
"""
Step 3 v2 — Per-bucket evaluation of DiffDock outputs using the GroupBind
authors' canonical 363-PDB CSV (pdbbind_benchmark_test.csv).

This replaces step3_evaluate.py. Key differences:
  - Reads ground-truth paths from the authors' CSV (matches paper Table 1 exactly).
  - Uses spyrmsd (symmetry-corrected) instead of hand-rolled RMSD.
  - Computes top-1, top-5, perfect (oracle top-40) RMSD per ligand.
  - Joins per-bucket assignments from step1's buckets.csv on complex_name.
  - Optionally runs PoseBusters on the top-1 pose.

Run:
    python3 step3_evaluate_v2.py \\
        --csv         pdbbind_benchmark_test.csv \\
        --buckets     out/buckets.csv \\
        --diffdock_out out/diffdock_results \\
        --data_root   .                          # so that 'data/PDBBind_processed/...' resolves
        --details     out/per_pdb_eval.csv \\
        --report      out/report.csv
"""
import argparse
import signal
import sys
import warnings
from contextlib import contextmanager
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# --------------------------- mol I/O (matches notebook) ---------------------


def load_mol(path: str):
    """Load a ligand from .sdf or .mol2, heavy atoms only.

    Crystal ligands from PDBBind sometimes include explicit hydrogens (or
    counter-ions in the same SDF entry); DiffDock predictions are heavy-atom
    only. We force heavy-atom-only loading on both sides so atom counts match.
    """
    if path.endswith(".sdf"):
        m = Chem.MolFromMolFile(path, removeHs=True, sanitize=True)
    elif path.endswith(".mol2"):
        m = Chem.MolFromMol2File(path, removeHs=True, sanitize=True)
    else:
        print(f"[step3] unsupported ligand format: {path}", file=sys.stderr)
        return None
    if m is None:
        # Retry without sanitization (some SDFs have non-standard valences)
        if path.endswith(".sdf"):
            m = Chem.MolFromMolFile(path, removeHs=True, sanitize=False)
        elif path.endswith(".mol2"):
            m = Chem.MolFromMol2File(path, removeHs=True, sanitize=False)
    if m is None:
        return None
    # If the file contains multiple disconnected fragments (counter-ions,
    # waters), keep only the largest by heavy-atom count.
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda f: f.GetNumHeavyAtoms(), reverse=True)
        return frags[0]
    return m


# --------------------------- sPyRMSD with timeout ---------------------------


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_symmetry_rmsd(mol, coords1, coords2_list, mol2=None):
    """Symmetry-corrected RMSD via sPyRMSD (Meli & Biggin 2020), with 10 s timeout.
    coords2_list can be a list of (N, 3) arrays; returns a vector of RMSDs.
    """
    from spyrmsd import rmsd as srmsd
    from spyrmsd import molecule as smol
    with time_limit(10):
        m = smol.Molecule.from_rdkit(mol)
        m2 = smol.Molecule.from_rdkit(mol2) if mol2 is not None else m
        return srmsd.symmrmsd(
            coords1,
            coords2_list,
            m.atomicnums,
            m2.atomicnums,
            m.adjacency_matrix,
            m2.adjacency_matrix,
        )


def naive_rmsd(gt_pos: np.ndarray, pred_pos_stack: np.ndarray) -> np.ndarray:
    """Fallback for when sPyRMSD times out or fails.
    Tolerates atom-count mismatch by truncating to the common length.
    """
    n_gt = gt_pos.shape[0]
    n_pred = pred_pos_stack.shape[1]
    n = min(n_gt, n_pred)
    if n_gt != n_pred:
        print(f"[naive_rmsd] atom count mismatch: gt={n_gt} pred={n_pred}, "
              f"truncating to {n}", file=sys.stderr)
    diffs = gt_pos[None, :n, :] - pred_pos_stack[:, :n, :]
    return np.sqrt((diffs ** 2).sum(axis=2).mean(axis=1))


# --------------------------- DiffDock output gathering ----------------------


def collect_predictions(result_root: Path, complex_name: str, max_rank: int = 40):
    """Return (list_of_pred_mols, list_of_confidences) sorted by rank.

    DiffDock writes outputs as <complex>/rank{n}_confidence{score}.sdf
    """
    pred_mols, pred_conf, pred_ranks = [], [], []
    for i in range(1, max_rank + 1):
        patt = str(result_root / complex_name / f"rank{i}_*.sdf")
        matches = glob(patt)
        if not matches:
            continue
        path = matches[0]
        mol = load_mol(path)
        if mol is None:
            continue
        # Parse "...confidence<float>.sdf"
        try:
            conf = float(path.split("confidence")[1][:-4])
        except (IndexError, ValueError):
            conf = float("nan")
        pred_mols.append(mol)
        pred_conf.append(conf)
        pred_ranks.append(i)
    return pred_mols, pred_conf, pred_ranks


def find_top1_pose_path(result_root: Path, complex_name: str):
    """Return the file path of the rank-1 prediction (for PoseBusters)."""
    patt = str(result_root / complex_name / "rank1_*.sdf")
    matches = glob(patt)
    return matches[0] if matches else None


# --------------------------- PoseBusters ------------------------------------


PB_CHECK_COLUMNS = [
    "mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded",
    "sanitization", "inchi_convertible", "all_atoms_connected",
    "no_radicals", "molecular_formula", "molecular_bonds",
    "double_bond_stereochemistry", "tetrahedral_chirality",
    "bond_lengths", "bond_angles", "internal_steric_clash",
    "aromatic_ring_flatness", "non-aromatic_ring_non-flatness",
    "double_bond_flatness", "internal_energy",
    "protein-ligand_maximum_distance", "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
]


def run_pb(pred_path: str, crystal_path: str, prot_path: str):
    try:
        from posebusters import PoseBusters
        pb = PoseBusters(config="redock")
        df = pb.bust(
            mol_pred=pred_path,
            mol_true=crystal_path,
            mol_cond=prot_path,
            full_report=True,
        )
        if df is None or len(df) == 0:
            return {}
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"[pb_error] {pred_path}: {e}", file=sys.stderr)
        return {}


def pb_valid_flag(row: dict):
    if not row:
        return None
    flags = [bool(row[c]) for c in PB_CHECK_COLUMNS
             if c in row and row[c] is not None]
    if not flags:
        return None
    return all(flags)


def resolve_protein_path(csv_path: str, data_root: Path,
                          protein_suffix: str, protein_dir_subst: str = None):
    """Try a few candidate paths for the protein file.

    1. The path exactly as written in the CSV.
    2. Substituting the directory if --protein_dir_subst is provided.
    3. Substituting the filename suffix if --protein_suffix differs.
    4. Both substitutions combined.

    Returns the first path that exists, or None.
    """
    candidates = [data_root / csv_path]
    csv_path_str = str(csv_path)

    # Suffix substitution
    if "_protein_processed.pdb" in csv_path_str:
        alt_csv = csv_path_str.replace("_protein_processed.pdb",
                                        protein_suffix)
        candidates.append(data_root / alt_csv)

    # Directory substitution
    if protein_dir_subst and "data/PDBBind_processed" in csv_path_str:
        alt_dir = csv_path_str.replace("data/PDBBind_processed",
                                        protein_dir_subst)
        candidates.append(data_root / alt_dir)
        # Combined
        if "_protein_processed.pdb" in alt_dir:
            alt_combo = alt_dir.replace("_protein_processed.pdb",
                                         protein_suffix)
            candidates.append(data_root / alt_combo)

    # Also try both PDBBind subdirs automatically
    pdb_id = Path(csv_path).parts[-2] if len(Path(csv_path).parts) >= 2 else ""
    if pdb_id:
        for sub in ("refined-set", "v2020-other-PL"):
            for suf in (protein_suffix, "_protein.pdb",
                        "_protein_processed.pdb"):
                candidates.append(data_root / "data" / sub / pdb_id /
                                  f"{pdb_id}{suf}")

    for c in candidates:
        if c.exists():
            return c
    return None


# --------------------------- main pipeline ----------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True,
                    help="Authors' pdbbind_benchmark_test.csv (363 rows)")
    ap.add_argument("--buckets", type=Path, required=True,
                    help="step1 output: out/buckets.csv (must contain "
                    "'pdb_id' and 'bucket' columns)")
    ap.add_argument("--diffdock_out", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=Path("."),
                    help="prefix for the CSV's relative paths (default: cwd)")
    ap.add_argument("--details", type=Path, default=Path("out/per_pdb_eval.csv"))
    ap.add_argument("--report", type=Path, default=Path("out/report.csv"))
    ap.add_argument("--protein_suffix", type=str, default="_protein_processed.pdb",
                    help="Filename suffix used in the authors' CSV "
                    "(default matches their 'data/PDBBind_processed' layout). "
                    "Pass '_protein.pdb' if you have raw PDBBind files.")
    ap.add_argument("--protein_dir_subst", type=str, default=None,
                    help="If set, replace 'data/PDBBind_processed' in the CSV "
                    "path with this string. E.g. 'data/refined-set' or "
                    "'data/v2020-other-PL' if you have the raw PDBBind layout. "
                    "Skipped (used as fallback) if the original path exists.")
    ap.add_argument("--no_pb", action="store_true",
                    help="skip PoseBusters (RMSD-only, much faster)")
    args = ap.parse_args()

    df_csv = pd.read_csv(args.csv)
    df_csv["complex_name"] = df_csv["complex_name"].astype(str).str.lower()
    print(f"[step3] authors' CSV: {len(df_csv)} complexes")

    df_buckets = pd.read_csv(args.buckets)
    df_buckets["pdb_id"] = df_buckets["pdb_id"].astype(str).str.lower()
    pid_to_bucket = dict(zip(df_buckets["pdb_id"], df_buckets["bucket"]))
    print(f"[step3] bucket assignments: {len(pid_to_bucket)} complexes")

    rows = []
    missing_bucket = 0
    missing_protein = 0
    for i, rec in df_csv.iterrows():
        cname = rec["complex_name"]
        prot_path = resolve_protein_path(
            rec["protein_path"], args.data_root,
            args.protein_suffix, args.protein_dir_subst,
        )
        crystal_path = args.data_root / rec["ligand_description"]
        bucket = pid_to_bucket.get(cname)
        if bucket is None:
            missing_bucket += 1
            bucket = "UNKNOWN"

        if prot_path is None:
            missing_protein += 1
            rows.append({"pdb_id": cname, "bucket": bucket,
                         "status": "protein_not_found"})
            continue

        # Load ground-truth ligand
        gt_mol = load_mol(str(crystal_path))
        if gt_mol is None:
            rows.append({"pdb_id": cname, "bucket": bucket,
                         "status": "gt_load_failed"})
            continue

        # Gather predictions
        pred_mols, pred_conf, pred_ranks = collect_predictions(
            args.diffdock_out, cname
        )
        if len(pred_mols) == 0:
            rows.append({"pdb_id": cname, "bucket": bucket,
                         "status": "no_predictions"})
            continue

        gt_pos = gt_mol.GetConformer().GetPositions()
        pred_pos_list = [m.GetConformer().GetPositions() for m in pred_mols]
        n_gt = gt_pos.shape[0]

        # Split predictions by atom-count match (sPyRMSD requires same N)
        matched_idx = [i for i, p in enumerate(pred_pos_list)
                        if p.shape[0] == n_gt]
        mismatched_idx = [i for i, p in enumerate(pred_pos_list)
                           if p.shape[0] != n_gt]
        if mismatched_idx:
            sizes = sorted({pred_pos_list[i].shape[0] for i in mismatched_idx})
            print(f"[step3] {cname}: {len(mismatched_idx)}/{len(pred_pos_list)}"
                  f" predictions have N≠{n_gt} (sizes: {sizes}); "
                  f"using naive RMSD for those", file=sys.stderr)

        rmsds = np.full(len(pred_pos_list), np.nan)

        # sPyRMSD on the matched subset
        if matched_idx:
            matched_stack = np.stack([pred_pos_list[i] for i in matched_idx])
            try:
                vals = np.array(get_symmetry_rmsd(
                    gt_mol, gt_pos, list(matched_stack)
                ))
                for k, i in enumerate(matched_idx):
                    rmsds[i] = vals[k]
            except Exception as e:
                vals = naive_rmsd(gt_pos, matched_stack)
                for k, i in enumerate(matched_idx):
                    rmsds[i] = vals[k]

        # Naive RMSD for size-mismatched ones (truncated to min length)
        for i in mismatched_idx:
            pp = pred_pos_list[i][None, ...]  # (1, N_pred, 3)
            rmsds[i] = naive_rmsd(gt_pos, pp)[0]

        # All-samples stack for centroid math below (centroids are independent
        # of atom count — we just need each sample's mean position)
        pred_centroids = np.array([p.mean(axis=0) for p in pred_pos_list])

        # Centroid distance per sample (Table 4 metric)
        gt_centroid = gt_pos.mean(axis=0)
        centroid_dists = np.linalg.norm(
            pred_centroids - gt_centroid[None, :], axis=1
        )

        # rank1 is at index 0 (sorted by rank in collect_predictions)
        top1_rmsd = float(rmsds[0])
        top5_rmsd = float(np.min(rmsds[:5]))
        perfect_rmsd = float(np.min(rmsds))

        top1_cd = float(centroid_dists[0])
        top5_cd = float(np.min(centroid_dists[:5]))
        perfect_cd = float(np.min(centroid_dists))

        # PoseBusters on top-1 only
        pb_flag = None
        pb_row = {}
        if not args.no_pb:
            pred_path = find_top1_pose_path(args.diffdock_out, cname)
            if pred_path:
                pb_row = run_pb(pred_path, str(crystal_path), str(prot_path))
                pb_flag = pb_valid_flag(pb_row)

        out = {
            "pdb_id": cname,
            "bucket": bucket,
            "status": "ok",
            "n_samples": len(pred_mols),
            "top1_rmsd": top1_rmsd,
            "top5_rmsd": top5_rmsd,
            "perfect_rmsd": perfect_rmsd,
            "top1_centroid": top1_cd,
            "top5_centroid": top5_cd,
            "perfect_centroid": perfect_cd,
            "pb_valid_top1": pb_flag,
        }
        for c in PB_CHECK_COLUMNS:
            if c in pb_row:
                out[f"pb__{c}"] = pb_row[c]
        rows.append(out)

        if (i + 1) % 25 == 0:
            print(f"[step3] {i+1}/{len(df_csv)} processed", flush=True)

    if missing_bucket:
        print(f"[step3] WARNING: {missing_bucket} complexes in CSV have no "
              f"bucket assignment (treated as UNKNOWN)")
    if missing_protein:
        print(f"[step3] WARNING: {missing_protein} complexes had no resolvable "
              f"protein file (status=protein_not_found, excluded from metrics). "
              f"Use --protein_suffix and --protein_dir_subst to point at your "
              f"actual layout.")

    details = pd.DataFrame(rows)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.details, index=False)
    print(f"[step3] wrote per-PDB details to {args.details}")

    # ----------------- per-bucket aggregation --------------------------------

    ok = details[details["status"] == "ok"].copy()

    # Threshold flags — paper uses < 2 Å for RMSD and < 2 Å / < 5 Å for centroid
    for col, val_col, thr in [
        ("top1_rmsd_lt_2",    "top1_rmsd",       2.0),
        ("top5_rmsd_lt_2",    "top5_rmsd",       2.0),
        ("perfect_rmsd_lt_2", "perfect_rmsd",    2.0),
        ("top1_cd_lt_2",      "top1_centroid",   2.0),
        ("top1_cd_lt_5",      "top1_centroid",   5.0),
        ("top5_cd_lt_2",      "top5_centroid",   2.0),
        ("top5_cd_lt_5",      "top5_centroid",   5.0),
    ]:
        ok[col] = ok[val_col] < thr

    def q(col):
        return lambda s: s.quantile(col)

    agg = ok.groupby("bucket").agg(
        n=("pdb_id", "count"),
        # ----- Ligand RMSD (Table 1 style) -----
        top1_rmsd_p25=("top1_rmsd", q(0.25)),
        top1_rmsd_p50=("top1_rmsd", q(0.50)),
        top1_rmsd_p75=("top1_rmsd", q(0.75)),
        top1_rmsd_lt_2=("top1_rmsd_lt_2", "mean"),
        top5_rmsd_p25=("top5_rmsd", q(0.25)),
        top5_rmsd_p50=("top5_rmsd", q(0.50)),
        top5_rmsd_p75=("top5_rmsd", q(0.75)),
        top5_rmsd_lt_2=("top5_rmsd_lt_2", "mean"),
        perfect_rmsd_p50=("perfect_rmsd", q(0.50)),
        perfect_rmsd_lt_2=("perfect_rmsd_lt_2", "mean"),
        # ----- Centroid distance (Table 4 style) -----
        top1_cd_p25=("top1_centroid", q(0.25)),
        top1_cd_p50=("top1_centroid", q(0.50)),
        top1_cd_p75=("top1_centroid", q(0.75)),
        top1_cd_lt_2=("top1_cd_lt_2", "mean"),
        top1_cd_lt_5=("top1_cd_lt_5", "mean"),
        top5_cd_p50=("top5_centroid", q(0.50)),
        top5_cd_lt_2=("top5_cd_lt_2", "mean"),
        top5_cd_lt_5=("top5_cd_lt_5", "mean"),
        # ----- PoseBuster -----
        frac_pb_valid_top1=("pb_valid_top1", "mean"),
    ).reset_index()

    # PB-valid restricted to top-1 accurate poses
    sub = ok[ok["top1_rmsd_lt_2"]].groupby("bucket").agg(
        n_top1_rmsd_lt_2=("pdb_id", "count"),
        frac_pb_valid_when_top1_lt_2=("pb_valid_top1", "mean"),
    ).reset_index()
    agg = agg.merge(sub, on="bucket", how="left")

    # ----- ALL row for direct comparison to paper Tables 1 and 4 -----
    all_row = {"bucket": "ALL", "n": len(ok)}
    quantile_defs = [
        ("top1_rmsd_p25",    "top1_rmsd",       0.25),
        ("top1_rmsd_p50",    "top1_rmsd",       0.50),
        ("top1_rmsd_p75",    "top1_rmsd",       0.75),
        ("top5_rmsd_p25",    "top5_rmsd",       0.25),
        ("top5_rmsd_p50",    "top5_rmsd",       0.50),
        ("top5_rmsd_p75",    "top5_rmsd",       0.75),
        ("perfect_rmsd_p50", "perfect_rmsd",    0.50),
        ("top1_cd_p25",      "top1_centroid",   0.25),
        ("top1_cd_p50",      "top1_centroid",   0.50),
        ("top1_cd_p75",      "top1_centroid",   0.75),
        ("top5_cd_p50",      "top5_centroid",   0.50),
    ]
    for out_col, src_col, q_val in quantile_defs:
        all_row[out_col] = ok[src_col].quantile(q_val)

    mean_cols = [
        "top1_rmsd_lt_2", "top5_rmsd_lt_2", "perfect_rmsd_lt_2",
        "top1_cd_lt_2", "top1_cd_lt_5", "top5_cd_lt_2", "top5_cd_lt_5",
        "frac_pb_valid_top1",
    ]
    # frac_pb_valid_top1 column in `ok` is `pb_valid_top1`
    rename_map = {"frac_pb_valid_top1": "pb_valid_top1"}
    for col in mean_cols:
        src = rename_map.get(col, col)
        if src in ok.columns:
            all_row[col] = ok[src].mean()

    all_row["n_top1_rmsd_lt_2"] = int(ok["top1_rmsd_lt_2"].sum())
    all_row["frac_pb_valid_when_top1_lt_2"] = ok.loc[
        ok["top1_rmsd_lt_2"], "pb_valid_top1"
    ].mean()
    agg = pd.concat([agg, pd.DataFrame([all_row])], ignore_index=True)

    agg.to_csv(args.report, index=False)
    print(f"[step3] wrote per-bucket report to {args.report}")
    print()
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"
                       if isinstance(x, float) else str(x)))


if __name__ == "__main__":
    main()
