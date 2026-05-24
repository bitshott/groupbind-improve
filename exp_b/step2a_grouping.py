#!/usr/bin/env python3
"""
Step 2a — Build the DiffDock input CSV listing all test-set complexes.

DiffDock's `inference.py` expects columns:
    complex_name, protein_path, ligand_description, protein_sequence

We populate:
    complex_name        = pdb_id
    protein_path        = absolute path to <pdb_id>_protein.pdb
    ligand_description  = absolute path to <pdb_id>_ligand.sdf (RDKit-readable)
    protein_sequence    = empty (since we provide the .pdb)

We skip any test PDB that fails to locate either file and write a warning.

Run:
    python3 step2a_make_csv.py --data_dir data --buckets out/buckets.csv \
                               --out diffdock_inputs.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def find_pair(data_dir: Path, pdb_id: str):
    for sub in ("refined-set", "v2020-other-PL"):
        d = data_dir / sub / pdb_id
        if not d.is_dir():
            continue
        prot = d / f"{pdb_id}_protein.pdb"
        sdf = d / f"{pdb_id}_ligand.sdf"
        if prot.exists() and sdf.exists():
            return prot.resolve(), sdf.resolve()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--buckets", type=Path, default=Path("out/buckets.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/diffdock_inputs.csv"))
    args = ap.parse_args()

    buckets = pd.read_csv(args.buckets)
    rows = []
    missing = []
    for pid in buckets["pdb_id"]:
        prot, sdf = find_pair(args.data_dir, pid)
        if prot is None:
            missing.append(pid)
            continue
        rows.append({
            "complex_name": pid,
            "protein_path": str(prot),
            "ligand_description": str(sdf),
            "protein_sequence": "",
        })
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[step2a] wrote {args.out} with {len(df)} rows")
    if missing:
        print(f"[step2a] WARNING: {len(missing)} test pdbs missing files: "
              f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
