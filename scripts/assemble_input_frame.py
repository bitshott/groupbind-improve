"""
assemble_input_frame.py -- LP-PDBBind CSV + local PDBBind v2020 dir
                          -> input_frame.pkl  (consumed by data_prep_mcs.py)

This is the missing first stage of the pipeline. It is intentionally
*separate* from data_prep_mcs.py so that failures here (a malformed SDF, a
missing PDB file, an RDKit parse error on a single ligand) do not invalidate
a successful MCS preparation downstream.

Inputs:
  --lp-csv          public LP-PDBBind split CSV (from fetch_lp_pdbbind.sh)
  --pdbbind-dir     local PDBBind v2020 root containing one subdir per
                    complex, named by PDB ID, with:
                       {pdbid}/{pdbid}_ligand.sdf   (preferred; fallback mol2)
                       {pdbid}/{pdbid}_protein.pdb  (cleaned receptor)
                       {pdbid}/{pdbid}_pocket.pdb   (optional; used for
                                                     pocket centre if present)
  --split           which LP-PDBBind split to materialise (default: test)
  --split-column    column name in the CSV holding the split label
                    (auto-detected from {new_split, split} if not given)
  --uniprot-column  column name holding UniProt ID; default 'uniprot_id',
                    falls back to looking up the PDB header if absent
  --name-column     column name holding the protein name; default
                    'protein_name', falls back to the PDB HEADER record
  --out             output pickle path (default: input_frame.pkl)

The script's job is mechanical: for each row of the filtered LP-PDBBind CSV,
load the ligand and protein structural files, derive the fields the MCS
pipeline needs, and write the resulting frame. It does no MCS, no grouping,
no filtering against the paper's S5/S6 thresholds -- those belong to
data_prep_mcs.py.

Output frame columns (exactly matches data_prep_mcs.REQUIRED_INPUT_COLUMNS):
  complex_id, uniprot_id, protein_name, ligand_smiles,
  pocket_center, ligand_coords, protein_coords

Failure handling:
  - per-complex errors are caught, logged, and the complex is skipped;
  - the script exits 0 if at least one complex was successfully assembled,
    exits 2 if zero complexes succeeded (so Make can stop the pipeline);
  - a sidecar JSON listing all skipped complexes + reasons is written next
    to the pickle for forensic review.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

# Silence RDKit's verbose per-molecule warnings; we surface our own errors.
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")

try:
    from Bio.PDB import PDBParser
    _HAVE_BIOPYTHON = True
except Exception:
    _HAVE_BIOPYTHON = False


# --------------------------------------------------------------------------- #
# Schema and constants                                                        #
# --------------------------------------------------------------------------- #
OUTPUT_COLUMNS = [
    "complex_id",
    "uniprot_id",
    "protein_name",
    "ligand_smiles",
    "pocket_center",
    "ligand_coords",
    "protein_coords",
]

# Heuristic split-column auto-detection order.
SPLIT_COLUMN_CANDIDATES = ("new_split", "split", "Split", "set")
UNIPROT_COLUMN_CANDIDATES = ("uniprot_id", "uniprot", "UniProt", "uniProtID")
NAME_COLUMN_CANDIDATES = ("protein_name", "name", "protein", "Name")


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class LigandData:
    smiles: str
    heavy_coords: np.ndarray  # (n_heavy, 3)


def _sanitize_ligand_mol(mol: Chem.Mol) -> Chem.Mol:
    """
    Sanitize ligand with fallback for kekulization failures.
    Keeps molecule usable for coordinates + SMILES generation.
    """
    if mol is None:
        raise ValueError("mol is None")

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        pass

    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ),
        )
        return mol
    except Exception:
        pass

    try:
        Chem.UpdatePropertyCache(mol, strict=False)
        Chem.GetSymmSSSR(mol)
        return mol
    except Exception as e:
        raise ValueError(f"could not sanitize ligand: {e}")


def _mol_to_smiles_safe(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
    except Exception:
        try:
            mol2 = Chem.Mol(mol)
            Chem.UpdatePropertyCache(mol2, strict=False)
            Chem.GetSymmSSSR(mol2)
            return Chem.MolToSmiles(mol2, canonical=True, kekuleSmiles=False)
        except Exception as e:
            raise ValueError(f"could not generate SMILES: {e}")


def _load_ligand(sdf_path: Path, mol2_fallback: Optional[Path]) -> LigandData:
    """
    Parse ligand from SDF or MOL2.

    Uses sanitize=False first, then controlled sanitization, because some
    PDBBind ligands fail RDKit kekulization but still have valid coordinates.
    """
    mol = None

    if sdf_path.exists():
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
        for m in suppl:
            if m is not None:
                mol = m
                break

    if mol is None and mol2_fallback is not None and mol2_fallback.exists():
        mol = Chem.MolFromMol2File(
            str(mol2_fallback),
            removeHs=True,
            sanitize=False,
        )

    if mol is None:
        raise ValueError(f"could not parse ligand from {sdf_path} / {mol2_fallback}")

    if mol.GetNumConformers() == 0:
        raise ValueError(f"ligand has no 3D conformer: {sdf_path}")

    mol = _sanitize_ligand_mol(mol)

    conf = mol.GetConformer(0)
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )

    return LigandData(
        smiles=_mol_to_smiles_safe(mol),
        heavy_coords=coords,
    )

def _load_protein_ca(pdb_path: Path) -> np.ndarray:
    """Return (m, 3) array of Cα coordinates from the protein PDB. Used as
    the GroupBind 'protein_coords' field; the paper represents the pocket
    at the residue level via Cα positions (Section 3.1)."""
    if not _HAVE_BIOPYTHON:
        raise RuntimeError(
            "biopython required to parse PDB files (`pip install biopython`)"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PDBConstructionWarning is noisy
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("p", str(pdb_path))
    coords = []
    for model in struct:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    a = res["CA"].get_coord()
                    coords.append([float(a[0]), float(a[1]), float(a[2])])
        break  # first model only (PDBBind structures are single-model)
    if not coords:
        raise ValueError(f"no Cα atoms in {pdb_path}")
    return np.asarray(coords, dtype=float)


def _pocket_center(
    pocket_pdb: Optional[Path], ligand_coords: np.ndarray
) -> np.ndarray:
    """If a pocket PDB is provided by PDBBind, use the centroid of its
    heavy atoms; otherwise default to the ligand's centre of mass (the
    standard convention when no pocket file is supplied)."""
    if pocket_pdb is not None and pocket_pdb.exists() and _HAVE_BIOPYTHON:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parser = PDBParser(QUIET=True)
                s = parser.get_structure("pkt", str(pocket_pdb))
            coords = []
            for model in s:
                for chain in model:
                    for res in chain:
                        for a in res:
                            c = a.get_coord()
                            coords.append([float(c[0]), float(c[1]), float(c[2])])
                break
            if coords:
                return np.asarray(coords, dtype=float).mean(axis=0)
        except Exception:
            pass
    return ligand_coords.mean(axis=0)


def _pdb_header_metadata(pdb_path: Path) -> dict[str, str]:
    """Extract UniProt-ID and protein name fallbacks from PDB header records,
    used when the LP-PDBBind CSV doesn't carry those columns explicitly."""
    info: dict[str, str] = {}
    if not pdb_path.exists():
        return info
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("HEADER"):
                    info["pdb_header"] = line[10:50].strip()
                elif line.startswith("COMPND") and "MOLECULE:" in line:
                    info.setdefault(
                        "protein_name",
                        line.split("MOLECULE:")[-1].strip().rstrip(";"),
                    )
                elif line.startswith("DBREF") and "UNP" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "UNP" and i + 1 < len(parts):
                            info.setdefault("uniprot_id", parts[i + 1])
                if line.startswith("ATOM"):
                    break  # stop at coordinate block
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Column auto-detection                                                       #
# --------------------------------------------------------------------------- #
def _autodetect_column(
    df: pd.DataFrame, candidates: tuple[str, ...], required: bool, label: str
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(
            f"could not auto-detect a column for {label}; tried {candidates}. "
            f"CSV columns: {list(df.columns)}. Pass --{label.replace('_','-')}-column."
        )
    return None


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
def assemble(
    lp_csv: Path,
    pdbbind_dir: Path,
    split: str,
    split_column: Optional[str],
    uniprot_column: Optional[str],
    name_column: Optional[str],
    out_pkl: Path,
) -> tuple[int, int, list[dict]]:
    """Returns (n_succeeded, n_skipped, skipped_log)."""
    df = pd.read_csv(lp_csv)
    if "pdbid" not in df.columns:
        # LP-PDBBind sometimes capitalises; normalise.
        for cand in ("PDBID", "pdb_id", "PDB_ID", "header"):
            if cand in df.columns:
                df = df.rename(columns={cand: "pdbid"})
                break
        else:
            raise ValueError(
                f"LP-PDBBind CSV has no 'pdbid' column; got {list(df.columns)}"
            )

    sc = split_column or _autodetect_column(
        df, SPLIT_COLUMN_CANDIDATES, required=True, label="split"
    )
    uc = uniprot_column or _autodetect_column(
        df, UNIPROT_COLUMN_CANDIDATES, required=False, label="uniprot"
    )
    nc = name_column or _autodetect_column(
        df, NAME_COLUMN_CANDIDATES, required=False, label="name"
    )

    # Filter by split (case-insensitive, since LP-PDBBind has used both 'test'
    # and 'Test' in different releases).
    mask = df[sc].astype(str).str.lower() == split.lower()
    df = df.loc[mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"no rows in split={split!r} under column {sc!r}; "
            f"available values: {sorted(set(pd.read_csv(lp_csv)[sc].astype(str)))}"
        )

    rows: list[dict] = []
    skipped: list[dict] = []
    for _, row in df.iterrows():
        pdbid = str(row["pdbid"]).strip().lower()
        cdir = pdbbind_dir / pdbid
        if not cdir.exists():
            skipped.append({"pdbid": pdbid, "reason": "complex_dir_missing"})
            continue
        sdf = cdir / f"{pdbid}_ligand.sdf"
        mol2 = cdir / f"{pdbid}_ligand.mol2"
        protein_pdb = cdir / f"{pdbid}_protein.pdb"
        pocket_pdb = cdir / f"{pdbid}_pocket.pdb"

        try:
            lig = _load_ligand(sdf, mol2)
        except Exception as e:
            skipped.append({"pdbid": pdbid, "reason": f"ligand_parse:{e}"})
            continue
        try:
            prot_ca = _load_protein_ca(protein_pdb)
        except Exception as e:
            skipped.append({"pdbid": pdbid, "reason": f"protein_parse:{e}"})
            continue
        try:
            pkt = _pocket_center(pocket_pdb, lig.heavy_coords)
        except Exception as e:
            skipped.append({"pdbid": pdbid, "reason": f"pocket_center:{e}"})
            continue

        # Resolve metadata: CSV columns first, then PDB header fallback.
        upid = str(row[uc]).strip() if uc and pd.notna(row[uc]) else ""
        pname = str(row[nc]).strip() if nc and pd.notna(row[nc]) else ""
        if not upid or not pname:
            hdr = _pdb_header_metadata(protein_pdb)
            upid = upid or hdr.get("uniprot_id", f"UNK_{pdbid}")
            pname = pname or hdr.get("protein_name", pdbid)

        rows.append({
            "complex_id": pdbid,
            "uniprot_id": upid,
            "protein_name": pname,
            "ligand_smiles": lig.smiles,
            "pocket_center": pkt,
            "ligand_coords": lig.heavy_coords,
            "protein_coords": prot_ca,
        })

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(out_pkl)

    skip_log = out_pkl.with_suffix(".skipped.json")
    skip_log.write_text(json.dumps(skipped, indent=2))

    return len(rows), len(skipped), skipped


# --------------------------------------------------------------------------- #
# Self-test on a synthetic PDBBind directory                                  #
# --------------------------------------------------------------------------- #
def _build_synthetic_pdbbind(tmpdir: Path) -> tuple[Path, Path]:
    """Create a tiny on-disk fake PDBBind layout with three complexes plus a
    minimal LP-PDBBind-style CSV pointing at them. Lets the script exercise
    the SDF/PDB parsers end-to-end without network access."""
    from rdkit.Chem import AllChem
    pdbbind = tmpdir / "pdbbind_v2020"
    pdbbind.mkdir()

    def _write_complex(pdbid: str, smi: str, prot_atoms: int) -> None:
        cdir = pdbbind / pdbid
        cdir.mkdir()
        # Ligand: embed a 3D conformer with ETKDG.
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        w = Chem.SDWriter(str(cdir / f"{pdbid}_ligand.sdf"))
        w.write(mol); w.close()
        # Protein: minimal PDB with `prot_atoms` Cα atoms in a shell around
        # the ligand centre (so MCS pipeline's S5 [0.4, 3.0] window is
        # later satisfiable -- a contact atom at ~2 A).
        conf = mol.GetConformer(0)
        lig_centre = np.mean(
            [[conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())], axis=0,
        )
        rng = np.random.default_rng(hash(pdbid) % 2**31)
        lines = []
        # One explicit contact Cα at ~4 A from ligand centre. The ligand
        # atoms are spread (sigma ~ 1 A) around lig_centre, so 4 A from the
        # centre places the contact reliably in (0.4, 3.0) A of the nearest
        # ligand atom -- inside the paper's S5 window.
        lines.append(_atom_line(1, "A", 1, lig_centre + np.array([4.0, 0, 0])))
        for i in range(2, prot_atoms + 1):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            r = rng.uniform(6.5, 9.0)
            pos = lig_centre + d * r
            lines.append(_atom_line(i, "A", i, pos))
        (cdir / f"{pdbid}_protein.pdb").write_text(
            "HEADER    HYDROLASE                               01-JAN-20   "
            + pdbid.upper() + "\n"
            "COMPND    MOLECULE: TEST_PROTEIN_" + pdbid.upper() + ";\n"
            "DBREF  " + pdbid.upper() + " A    1    " + str(prot_atoms)
            + "  UNP    P0000" + pdbid[-1].upper()
            + "   TESTP_HUMAN     1     " + str(prot_atoms) + "\n"
            + "\n".join(lines) + "\nEND\n"
        )

    _write_complex("1abc", "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1", 30)
    _write_complex("2def", "Cc1ccc(Nc2ncnc3sccc23)cc1", 28)
    _write_complex("3ghi", "O=C(Nc1ccc(Cl)cc1)c1cccs1", 25)

    csv = tmpdir / "LP_PDBBind.csv"
    csv.write_text(
        "pdbid,new_split,uniprot_id,protein_name\n"
        "1abc,test,P00001,kinase_test\n"
        "2def,test,P00001,kinase_test\n"
        "3ghi,test,P00002,unrelated_test\n"
    )
    return csv, pdbbind


def _atom_line(serial: int, chain: str, resi: int, xyz: np.ndarray) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{resi:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"  1.00  0.00           C"
    )


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        csv, pdbbind = _build_synthetic_pdbbind(tdp)
        out = tdp / "input_frame.pkl"
        n_ok, n_skip, _ = assemble(
            lp_csv=csv,
            pdbbind_dir=pdbbind,
            split="test",
            split_column=None,
            uniprot_column=None,
            name_column=None,
            out_pkl=out,
        )
        if n_ok != 3:
            print(f"FAIL: expected 3 assembled rows, got {n_ok} (skipped {n_skip})")
            return 1
        df = pd.read_pickle(out)
        # Schema must match what data_prep_mcs consumes.
        import data_prep_mcs as DP
        missing = [c for c in DP.REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            print(f"FAIL: missing columns vs data_prep_mcs contract: {missing}")
            return 1
        # Round-trip through data_prep_mcs to confirm everything fits.
        groups = DP.prepare_groups(df, verbose=False)
        if not groups:
            print("FAIL: data_prep_mcs produced no groups from assembled frame")
            return 1
        total_kept = sum(len(g.ligand_ids) for g in groups)
        if total_kept < 2:
            print(f"FAIL: only {total_kept} ligands survived S5/S6 filters")
            return 1
        print("\nASSEMBLE SELFTEST PASSED")
        print(f"  rows assembled        : {n_ok}")
        print(f"  rows skipped          : {n_skip}")
        print(f"  groups after data prep: {len(groups)}")
        print(f"  ligands kept          : {total_kept}")
        return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lp-csv", type=Path)
    ap.add_argument("--pdbbind-dir", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-column", default=None)
    ap.add_argument("--uniprot-column", default=None)
    ap.add_argument("--name-column", default=None)
    ap.add_argument("--out", type=Path, default=Path("input_frame.pkl"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.lp_csv is None or args.pdbbind_dir is None:
        ap.error("--lp-csv and --pdbbind-dir are required (or use --selftest)")

    if not args.lp_csv.exists():
        print(f"ERROR: LP-PDBBind CSV not found: {args.lp_csv}", file=sys.stderr)
        return 2
    if not args.pdbbind_dir.exists():
        print(f"ERROR: PDBBind dir not found: {args.pdbbind_dir}", file=sys.stderr)
        return 2

    n_ok, n_skip, _ = assemble(
        lp_csv=args.lp_csv,
        pdbbind_dir=args.pdbbind_dir,
        split=args.split,
        split_column=args.split_column,
        uniprot_column=args.uniprot_column,
        name_column=args.name_column,
        out_pkl=args.out,
    )
    print(f"Assembled {n_ok} complexes, skipped {n_skip}. "
          f"Wrote {args.out} (+ skipped log next to it).")
    if n_ok == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
