import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from scipy.spatial.distance import cdist

warnings.simplefilter("ignore", BiopythonWarning)
RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step6")

MIN_DIST = 0.4
MAX_DIST = 3.0


def protein_heavy_coords(pdb_path: str) -> np.ndarray:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    coords = []
    for atom in structure.get_atoms():
        if atom.element.strip().upper() == "H":
            continue
        coords.append(atom.get_coord())
    return np.asarray(coords, dtype=float)


def ligand_heavy_coords(sdf_path: str) -> np.ndarray:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    for mol in suppl:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        coords = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
                continue
            p = conf.GetAtomPosition(i)
            coords.append([p.x, p.y, p.z])
        return np.asarray(coords, dtype=float)
    return np.empty((0, 3), dtype=float)


def min_pl_distance(prot_path: str, lig_path: str) -> float:
    pc = protein_heavy_coords(prot_path)
    lc = ligand_heavy_coords(lig_path)
    if pc.shape[0] == 0 or lc.shape[0] == 0:
        return float("nan")
    return float(cdist(pc, lc).min())


def classify(row):
    pid = row["pdb_id"]
    status = row.get("align_status", "")
    prot = row.get("aligned_protein_pdb", "")
    lig = row.get("aligned_ligand_sdf", "")
    pocket_id = row.get("pocket_cluster_id", "")

    if status in {"align_error", "ref_parse_error", "ref_write_error"}:
        return np.nan, "upstream_error", f"ISOLATED:{pid}"

    if not isinstance(lig, str) or not lig or not Path(lig).exists():
        return np.nan, status if status else "no_ligand", pocket_id

    if not isinstance(prot, str) or not prot or not Path(prot).exists():
        return np.nan, "no_protein_file", f"ISOLATED:{pid}"

    try:
        d = min_pl_distance(prot, lig)
    except Exception as e:
        log.warning(f"distance compute failed {pid}: {e}")
        return np.nan, "distance_error", f"ISOLATED:{pid}"

    if np.isnan(d):
        return np.nan, "empty_coords", f"ISOLATED:{pid}"

    if d < MIN_DIST:
        return d, "isolated_too_close", f"ISOLATED:{pid}"
    if d > MAX_DIST:
        return d, "isolated_too_far", f"ISOLATED:{pid}"
    return d, "ok", pocket_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")

    distances, statuses, group_ids = [], [], []
    for _, row in df.iterrows():
        d, st, gid = classify(row)
        distances.append(d)
        statuses.append(st)
        group_ids.append(gid)

    df["min_pl_distance"] = distances
    df["final_status"] = statuses
    df["final_group_id"] = group_ids

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    counts = pd.Series(statuses).value_counts().to_dict()
    log.info(f"status counts: {counts}")
    log.info(f"wrote {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
