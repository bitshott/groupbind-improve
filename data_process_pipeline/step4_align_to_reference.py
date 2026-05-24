import argparse
import logging
import re
import shutil
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, Superimposer
from Bio.PDB.Polypeptide import index_to_one, is_aa, three_to_index
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

warnings.simplefilter("ignore", BiopythonWarning)
RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step4")

MIN_KABSCH_POINTS = 3
SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(s: str) -> str:
    return SAFE_RE.sub("_", s)


def extract_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    seq_chars, ca_atoms = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True) or "CA" not in residue:
                    continue
                try:
                    aa = index_to_one(three_to_index(residue.get_resname()))
                except Exception:
                    continue
                seq_chars.append(aa)
                ca_atoms.append(residue["CA"])
        break
    return structure, "".join(seq_chars), ca_atoms


def lcs_pairs(a: str, b: str) -> list[tuple[int, int]]:
    sm = SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for ai, bi, size in sm.get_matching_blocks():
        for k in range(size):
            out.append((ai + k, bi + k))
    return out


def compute_transform(ref_atoms, ref_seq, mov_atoms, mov_seq):
    pairs = lcs_pairs(ref_seq, mov_seq)
    if len(pairs) < MIN_KABSCH_POINTS:
        return None, None, len(pairs), np.nan
    fixed = [ref_atoms[a] for a, _ in pairs]
    moving = [mov_atoms[b] for _, b in pairs]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    R, t = sup.rotran
    return np.asarray(R), np.asarray(t), len(pairs), float(sup.rms)


def apply_transform_to_structure(structure, R, t):
    for atom in structure.get_atoms():
        coord = np.asarray(atom.get_coord())
        new_coord = coord @ R + t
        atom.set_coord(new_coord.astype(np.float32))


def write_pdb(structure, out_path: str):
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path)


def iter_ligand_candidates(sdf_path, mol2_path):
    for path, fmt in [(sdf_path, "sdf"), (mol2_path, "mol2")]:
        if not isinstance(path, str) or not path or not Path(path).exists():
            continue
        try:
            if fmt == "sdf":
                suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
                for mol in suppl:
                    if mol is not None:
                        yield mol, fmt
                        break
            else:
                mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
                if mol is not None:
                    yield mol, fmt
        except Exception:
            continue


def apply_transform_to_ligand(mol, R, t):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z]) @ R + t
        conf.SetAtomPosition(i, Point3D(float(v[0]), float(v[1]), float(v[2])))


def write_ligand_sdf(mol, out_path: str):
    block = Chem.MolToMolBlock(mol, kekulize=False)
    with open(out_path, "w") as f:
        f.write(block)
        f.write("\n$$$$\n")


def select_reference(cluster_pdbs: list[str], rmsd_df: pd.DataFrame) -> str:
    sub = rmsd_df[rmsd_df["pdb_id_i"].isin(cluster_pdbs) & rmsd_df["pdb_id_j"].isin(cluster_pdbs)]
    means = {}
    for pid in cluster_pdbs:
        vals = sub[(sub["pdb_id_i"] == pid) | (sub["pdb_id_j"] == pid)]["rmsd"].dropna()
        means[pid] = float(vals.mean()) if len(vals) else np.inf
    min_val = min(means.values())
    candidates = sorted([p for p, v in means.items() if v == min_val])
    return candidates[0]


def process_cluster(cluster_id: str, cdf: pd.DataFrame, rmsd_df: pd.DataFrame, out_dir: Path) -> list[dict]:
    pdbs = list(cdf["pdb_id"])
    ref_pid = select_reference(pdbs, rmsd_df) if len(pdbs) >= 2 else pdbs[0]
    log.info(f"cluster {cluster_id}: ref={ref_pid}")

    cluster_dir = out_dir / safe_name(cluster_id)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    ref_row = cdf[cdf["pdb_id"] == ref_pid].iloc[0]
    try:
        ref_struct, ref_seq, ref_ca = extract_residues(ref_row["protein_pdb"])
    except Exception as e:
        log.warning(f"ref parse_error {ref_pid}: {e}")
        return [{**r, "aligned_protein_pdb": "", "aligned_ligand_sdf": "",
                 "is_reference": (r["pdb_id"] == ref_pid), "alignment_rmsd": np.nan,
                 "align_status": "ref_parse_error"} for r in cdf.to_dict("records")]

    out_rows = []
    for _, row in cdf.iterrows():
        pid = row["pdb_id"]
        is_ref = (pid == ref_pid)
        prot_out = cluster_dir / f"{pid}_protein_aligned.pdb"
        lig_out = cluster_dir / f"{pid}_ligand_aligned.sdf"

        if is_ref:
            try:
                shutil.copyfile(row["protein_pdb"], prot_out)
                status = "ok"
                rmsd_val = 0.0
            except Exception as e:
                log.warning(f"ref copy failed {pid}: {e}")
                status, rmsd_val, prot_out, lig_out = "ref_write_error", np.nan, "", ""

            if status == "ok":
                lig_written = False
                for mol, fmt in iter_ligand_candidates(row.get("ligand_sdf"), row.get("ligand_mol2")):
                    try:
                        write_ligand_sdf(mol, str(lig_out))
                        lig_written = True
                        break
                    except Exception as e:
                        log.warning(f"ref ligand write attempt failed {pid} ({fmt}): {e}")
                if not lig_written:
                    lig_out = ""
                    status = "ok_no_ligand"
        else:
            try:
                mov_struct, mov_seq, mov_ca = extract_residues(row["protein_pdb"])
                R, t, n_match, rmsd_val = compute_transform(ref_ca, ref_seq, mov_ca, mov_seq)
                if R is None:
                    raise ValueError(f"too_few_matches ({n_match})")
                apply_transform_to_structure(mov_struct, R, t)
                write_pdb(mov_struct, str(prot_out))
                status = "ok"
            except Exception as e:
                log.warning(f"align failed {pid}: {e}")
                status, rmsd_val, prot_out, lig_out = "align_error", np.nan, "", ""

            if status == "ok":
                lig_written = False
                for mol, fmt in iter_ligand_candidates(row.get("ligand_sdf"), row.get("ligand_mol2")):
                    try:
                        if mol.GetNumConformers() == 0:
                            continue
                        apply_transform_to_ligand(mol, R, t)
                        write_ligand_sdf(mol, str(lig_out))
                        lig_written = True
                        break
                    except Exception as e:
                        log.warning(f"ligand write attempt failed {pid} ({fmt}): {e}")
                if not lig_written:
                    lig_out = ""
                    status = "ok_no_ligand"

        out_rows.append({
            **row.to_dict(),
            "aligned_protein_pdb": str(prot_out) if prot_out else "",
            "aligned_ligand_sdf": str(lig_out) if lig_out else "",
            "is_reference": is_ref,
            "alignment_rmsd": rmsd_val,
            "align_status": status,
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-csv", required=True)
    ap.add_argument("--rmsd-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    cdf = pd.read_csv(args.cluster_csv)
    rdf = pd.read_csv(args.rmsd_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"loaded {len(cdf)} cluster rows, {len(rdf)} rmsd rows")

    all_rows = []
    for cid, sub in cdf.groupby("sequence_cluster_id", sort=False):
        if len(sub) < 2:
            for r in sub.to_dict("records"):
                all_rows.append({
                    **r, "aligned_protein_pdb": r["protein_pdb"],
                    "aligned_ligand_sdf": r.get("ligand_sdf", ""),
                    "is_reference": True, "alignment_rmsd": 0.0, "align_status": "singleton",
                })
            continue
        all_rows.extend(process_cluster(cid, sub, rdf, out_dir))

    out_df = pd.DataFrame(all_rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    log.info(f"wrote {len(out_df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
