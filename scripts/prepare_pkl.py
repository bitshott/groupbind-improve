#!/usr/bin/env python3
"""
Prepare PDBBind refined/general-minus-refined complexes as input dataframe for
the GroupBind-style data preparation script.

Input:
  --refined-dir                  PDBBind refined set directory
  --general-minus-refined-dir    PDBBind general set minus refined set directory
  --lp-csv                       LP_PDBBIND.csv

Output:
  --out-pkl                      Pickle with required columns:
                                  complex_id, uniprot_id, protein_name,
                                  ligand_smiles, pocket_center,
                                  ligand_coords, protein_coords

Optional:
  --out-csv                      Metadata-only CSV for inspection
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem

import requests
import time

REQUIRED_INPUT_COLUMNS = [
    "complex_id",
    "uniprot_id",
    "protein_name",
    "ligand_smiles",
    "pocket_center",
    "ligand_coords",
    "protein_coords",
]


PDB_ID_ALIASES = [
    "pdb_id",
    "pdbid",
    "PDB_ID",
    "PDBID",
    "pdb",
    "PDB",
    "complex_id",
    "header",
]

UNIPROT_ALIASES = [
    "uniprot_id", "uniprot", "UniProt", "UNIPROT", "UniProt_ID",
    "uniprot_accession"
]

PROTEIN_NAME_ALIASES = [
    "protein_name", "Protein_Name", "protein", "Protein",
    "target", "Target", "name", "Name"
]

SMILES_ALIASES = [
    "ligand_smiles", "smiles", "SMILES", "canonical_smiles",
    "Canonical_SMILES", "ligand_smi"
]

def fetch_uniprot_from_pdbe(pdb_id: str, sleep_s: float = 0.05) -> tuple[str, str]:
    """
    Return (uniprot_id, protein_name) for a PDB ID using PDBe SIFTS.

    If several UniProt IDs are mapped, join them with ';'.
    """
    pdb_id = pdb_id.lower()
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return "", ""
        data = r.json()
    except Exception:
        return "", ""
    finally:
        time.sleep(sleep_s)

    entry = data.get(pdb_id, {})
    mappings = entry.get("UniProt", {})

    if not mappings:
        return "", ""

    uniprot_ids = sorted(mappings.keys())

    names = []
    for uid in uniprot_ids:
        name = mappings[uid].get("name", "")
        if name:
            names.append(name)

    return ";".join(uniprot_ids), ";".join(sorted(set(names)))

def find_column(df: pd.DataFrame, aliases: list[str], required: bool = False) -> Optional[str]:
    for col in aliases:
        if col in df.columns:
            return col

    lowered = {str(c).lower(): c for c in df.columns}
    for col in aliases:
        if col.lower() in lowered:
            return lowered[col.lower()]

    if required:
        raise ValueError(
            f"Required column not found. Tried aliases: {aliases}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_pdb_id(x: str) -> str:
    return str(x).strip().lower()


def read_lp_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    pdb_col = find_column(df, PDB_ID_ALIASES, required=True)
    uniprot_col = find_column(df, UNIPROT_ALIASES, required=False)
    protein_col = find_column(df, PROTEIN_NAME_ALIASES, required=False)
    smiles_col = find_column(df, SMILES_ALIASES, required=False)

    out = pd.DataFrame()
    out["pdb_id"] = df[pdb_col].map(normalize_pdb_id)

    if uniprot_col is not None:
        out["uniprot_id"] = df[uniprot_col].fillna("").astype(str)
    else:
        out["uniprot_id"] = ""

    if protein_col is not None:
        out["protein_name"] = df[protein_col].fillna("").astype(str)
    else:
        out["protein_name"] = ""

    if smiles_col is not None:
        out["csv_smiles"] = df[smiles_col].fillna("").astype(str)
    else:
        out["csv_smiles"] = ""

    out = out.drop_duplicates("pdb_id", keep="first")
    return out


def index_complex_dirs(root: Path, subset_name: str) -> pd.DataFrame:
    rows = []

    if root is None:
        return pd.DataFrame(columns=["pdb_id", "complex_dir", "subset"])

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    for d in root.iterdir():
        if not d.is_dir():
            continue

        pdb_id = normalize_pdb_id(d.name)

        protein = find_first_existing(d, [
            f"{pdb_id}_protein.pdb",
            "protein.pdb",
            f"{pdb_id}.pdb",
            f"pdb{pdb_id}.ent",
        ])

        ligand = find_first_existing(d, [
            f"{pdb_id}_ligand.sdf",
            f"{pdb_id}_ligand.mol2",
            f"{pdb_id}_ligand.pdb",
            "ligand.sdf",
            "ligand.mol2",
            "ligand.pdb",
        ])

        if protein is None or ligand is None:
            warnings.warn(
                f"Skipping {pdb_id}: missing protein or ligand file "
                f"(protein={protein}, ligand={ligand})"
            )
            continue

        rows.append({
            "pdb_id": pdb_id,
            "complex_dir": str(d),
            "subset": subset_name,
            "protein_file": str(protein),
            "ligand_file": str(ligand),
        })

    return pd.DataFrame(rows)


def find_first_existing(directory: Path, names: list[str]) -> Optional[Path]:
    for name in names:
        p = directory / name
        if p.exists():
            return p

    pdb_like = sorted(directory.glob("*_protein.pdb"))
    if pdb_like:
        return pdb_like[0]

    return None


def read_ligand_mol(path: Path) -> Optional[Chem.Mol]:
    suffix = path.suffix.lower()

    if suffix == ".sdf":
        suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return None
        mol = mols[0]

    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)

    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)

    else:
        return None

    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            return None

    return mol


def mol_heavy_atom_coords(mol: Chem.Mol) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        raise ValueError("Ligand has no conformer coordinates")

    conf = mol.GetConformer()
    coords = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(atom.GetIdx())
        coords.append([p.x, p.y, p.z])

    return np.asarray(coords, dtype=np.float32)


def mol_to_smiles(mol: Chem.Mol) -> str:
    """
    Convert RDKit mol to canonical SMILES with fallback for problematic
    aromatic/kekulization cases.
    """
    try:
        mol_no_h = Chem.RemoveHs(mol, sanitize=True)
        return Chem.MolToSmiles(mol_no_h, canonical=True)
    except Exception:
        pass

    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(
            mol_copy,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        mol_no_h = Chem.RemoveHs(mol_copy, sanitize=False)
        return Chem.MolToSmiles(mol_no_h, canonical=True, kekuleSmiles=False)
    except Exception:
        pass

    try:
        mol_no_h = Chem.RemoveHs(mol, sanitize=False)
        return Chem.MolToSmiles(mol_no_h, canonical=True, kekuleSmiles=False)
    except Exception as e:
        raise ValueError(f"Cannot convert ligand to SMILES: {e}")

def read_pdb_heavy_atom_coords(path: Path) -> np.ndarray:
    coords = []

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            element = line[76:78].strip()
            atom_name = line[12:16].strip()

            if element.upper() == "H" or atom_name.startswith("H"):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            coords.append([x, y, z])

    return np.asarray(coords, dtype=np.float32)


def prepare_dataframe(
    refined_dir: Path,
    general_minus_refined_dir: Path,
    lp_csv: Path,
) -> pd.DataFrame:
    lp = read_lp_csv(lp_csv)

    refined = index_complex_dirs(refined_dir, "refined")
    general = index_complex_dirs(general_minus_refined_dir, "general_minus_refined")

    complexes = pd.concat([refined, general], ignore_index=True)

    if complexes.empty:
        raise ValueError("No valid PDBBind complex directories found")

    merged = complexes.merge(lp, on="pdb_id", how="left")
    
    uniprot_cache = {}
    
    rows = []

    for _, r in merged.iterrows():
        pdb_id = r["pdb_id"]
        ligand_file = Path(r["ligand_file"])
        protein_file = Path(r["protein_file"])

        mol = read_ligand_mol(ligand_file)
        if mol is None:
            warnings.warn(f"Skipping {pdb_id}: cannot parse ligand {ligand_file}")
            continue

        try:
            ligand_coords = mol_heavy_atom_coords(mol)
        except Exception as e:
            warnings.warn(f"Skipping {pdb_id}: ligand coordinate error: {e}")
            continue

        if ligand_coords.size == 0:
            warnings.warn(f"Skipping {pdb_id}: ligand has no heavy atoms")
            continue

        protein_coords = read_pdb_heavy_atom_coords(protein_file)
        if protein_coords.size == 0:
            warnings.warn(f"Skipping {pdb_id}: protein has no readable heavy atoms")
            continue

        csv_smiles = str(r.get("csv_smiles", "")).strip()
        if csv_smiles and csv_smiles.lower() != "nan":
            ligand_smiles = csv_smiles
        else:
            ligand_smiles = mol_to_smiles(mol)

        uniprot_id = str(r.get("uniprot_id", "")).strip()
        protein_name = str(r.get("protein_name", "")).strip()

        if (
            not uniprot_id
            or uniprot_id.lower() == "nan"
            or uniprot_id.startswith("unknown")
        ):
            if pdb_id not in uniprot_cache:
                uniprot_cache[pdb_id] = fetch_uniprot_from_pdbe(pdb_id)

            fetched_uniprot, fetched_name = uniprot_cache[pdb_id]

            if fetched_uniprot:
                uniprot_id = fetched_uniprot
            else:
                uniprot_id = "unknown_uniprot"

            if fetched_name:
                protein_name = fetched_name
            elif not protein_name or protein_name.lower() == "nan":
                protein_name = "unknown_protein"

        if not protein_name or protein_name.lower() == "nan":
            protein_name = "unknown_protein"
            
        pocket_center = ligand_coords.mean(axis=0).astype(np.float32)

        rows.append({
            "complex_id": pdb_id,
            "uniprot_id": uniprot_id,
            "protein_name": protein_name,
            "ligand_smiles": ligand_smiles,
            "pocket_center": pocket_center,
            "ligand_coords": ligand_coords,
            "protein_coords": protein_coords,
            "subset": r["subset"],
            "complex_dir": r["complex_dir"],
            "protein_file": str(protein_file),
            "ligand_file": str(ligand_file),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("All complexes failed during parsing")

    if df["complex_id"].duplicated().any():
        dup = df.loc[df["complex_id"].duplicated(), "complex_id"].tolist()
        raise ValueError(f"Duplicate complex_id values after merge: {dup[:20]}")

    return df


def write_metadata_csv(df: pd.DataFrame, path: Path) -> None:
    meta_cols = [
        "complex_id",
        "uniprot_id",
        "protein_name",
        "ligand_smiles",
        "subset",
        "complex_dir",
        "protein_file",
        "ligand_file",
    ]

    meta = df[meta_cols].copy()
    meta["n_ligand_atoms"] = df["ligand_coords"].map(lambda x: int(len(x)))
    meta["n_protein_atoms"] = df["protein_coords"].map(lambda x: int(len(x)))
    meta.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined-dir", required=True, type=Path)
    ap.add_argument("--general-minus-refined-dir", required=True, type=Path)
    ap.add_argument("--lp-csv", required=True, type=Path)
    ap.add_argument("--out-pkl", default="pdbbind_groupbind_input.pkl", type=Path)
    ap.add_argument("--out-csv", default="pdbbind_groupbind_input_metadata.csv", type=Path)
    args = ap.parse_args()

    df = prepare_dataframe(
        refined_dir=args.refined_dir,
        general_minus_refined_dir=args.general_minus_refined_dir,
        lp_csv=args.lp_csv,
    )

    df[REQUIRED_INPUT_COLUMNS].to_pickle(args.out_pkl)
    write_metadata_csv(df, args.out_csv)

    print(f"Wrote pickle: {args.out_pkl}")
    print(f"Wrote metadata CSV: {args.out_csv}")
    print(f"Complexes kept: {len(df)}")
    print(f"Refined: {(df['subset'] == 'refined').sum()}")
    print(f"General minus refined: {(df['subset'] == 'general_minus_refined').sum()}")
    print(f"Required columns: {REQUIRED_INPUT_COLUMNS}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
