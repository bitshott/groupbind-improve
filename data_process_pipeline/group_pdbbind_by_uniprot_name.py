#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def collect_complex_dirs(pdbbind_root: Path):
    rows = []

    complex_dirs = sorted([p for p in pdbbind_root.iterdir() if p.is_dir()])

    for complex_dir in tqdm(complex_dirs, desc="Reading complex dirs"):
        pdb_id = complex_dir.name.lower()

        protein_pdbs = sorted(complex_dir.glob("*_protein.pdb"))
        ligand_sdfs = sorted(complex_dir.glob("*_ligand.sdf"))
        ligand_mol2s = sorted(complex_dir.glob("*_ligand.mol2"))

        rows.append({
            "pdb_id": pdb_id,
            "complex_dir": str(complex_dir),
            "protein_pdb": str(protein_pdbs[0]) if protein_pdbs else "",
            "ligand_sdf": str(ligand_sdfs[0]) if ligand_sdfs else "",
            "ligand_mol2": str(ligand_mol2s[0]) if ligand_mol2s else "",
        })

    return pd.DataFrame(rows)


def parse_pdbbind_name_file(path: Path):
    rows = []

    with open(path, "r", errors="ignore") as f:
        for line in tqdm(f, desc=f"Parsing {path.name}"):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=3)

            if len(parts) < 4:
                continue

            pdb_id, year, uniprot_id, protein_name = parts

            rows.append({
                "pdb_id": pdb_id.lower(),
                "release_year": int(year),
                "uniprot_id": uniprot_id.strip(),
                "protein_name": protein_name.strip(),
                "metadata_source": str(path),
            })

    return pd.DataFrame(rows)


def make_initial_group(row):
    uniprot_id = str(row["uniprot_id"]).strip()
    protein_name = str(row["protein_name"]).strip().upper()

    if uniprot_id and uniprot_id.lower() != "nan":
        return f"UNIPROT:{uniprot_id}|NAME:{protein_name}"

    if protein_name and protein_name.lower() != "nan":
        return f"NAME:{protein_name}"

    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbbind-root", required=True)
    parser.add_argument("--metadata", nargs="+", required=True)
    parser.add_argument("--output", default="pdbbind_initial_protein_groups.csv")
    args = parser.parse_args()

    complexes = collect_complex_dirs(Path(args.pdbbind_root))

    metadata_tables = [
        parse_pdbbind_name_file(Path(p))
        for p in args.metadata
    ]

    meta = pd.concat(metadata_tables, ignore_index=True)

    # If the same PDB appears in both files, prefer refined metadata.
    meta["source_priority"] = meta["metadata_source"].apply(
        lambda x: 0 if "INDEX_refined_name" in x else 1
    )

    meta = (
        meta.sort_values(["pdb_id", "source_priority"])
        .drop_duplicates("pdb_id", keep="first")
        .drop(columns=["source_priority"])
    )

    out = complexes.merge(meta, on="pdb_id", how="left")

    out["uniprot_id"] = out["uniprot_id"].fillna("")
    out["protein_name"] = out["protein_name"].fillna("")
    out["metadata_source"] = out["metadata_source"].fillna("")
    out["release_year"] = out["release_year"].fillna(-1).astype(int)

    out["initial_protein_group"] = out.apply(make_initial_group, axis=1)

    out["status"] = "ok_uniprot_name"
    out.loc[out["initial_protein_group"].eq("UNKNOWN"), "status"] = "missing_metadata"

    out.to_csv(args.output, index=False)

    print(f"written: {args.output}")
    print(out["status"].value_counts())
    print(out["initial_protein_group"].value_counts().head(20))


if __name__ == "__main__":
    main()
