import argparse
import logging
import warnings
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import index_to_one, is_aa, three_to_index

warnings.simplefilter("ignore", BiopythonWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step3")

MIN_KABSCH_POINTS = 3


def extract_residues(pdb_path: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    seq_chars, ca_atoms = [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                if "CA" not in residue:
                    continue
                try:
                    aa = index_to_one(three_to_index(residue.get_resname()))
                except Exception:
                    continue
                seq_chars.append(aa)
                ca_atoms.append(residue["CA"])
        break
    return "".join(seq_chars), ca_atoms


def lcs_index_pairs(seq_a: str, seq_b: str) -> list[tuple[int, int]]:
    sm = SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    pairs = []
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            pairs.append((a + k, b + k))
    return pairs


def pair_rmsd(atoms_i: list, seq_i: str, atoms_j: list, seq_j: str):
    pairs = lcs_index_pairs(seq_i, seq_j)
    if len(pairs) < MIN_KABSCH_POINTS:
        return None, len(pairs), "too_few_matches"
    fixed = [atoms_i[a] for a, _ in pairs]
    moving = [atoms_j[b] for _, b in pairs]
    sup = Superimposer()
    sup.set_atoms(fixed, moving)
    return float(sup.rms), len(pairs), "ok"


def process_cluster(cluster_id: str, cluster_df: pd.DataFrame) -> list[dict]:
    cache = {}
    for _, row in cluster_df.iterrows():
        pid = row["pdb_id"]
        try:
            seq, atoms = extract_residues(row["protein_pdb"])
            if len(seq) < 1:
                raise ValueError("no residues")
            cache[pid] = (seq, atoms)
        except Exception as e:
            log.warning(f"parse_error {pid}: {e}")
            cache[pid] = None

    out_rows = []
    ids = list(cluster_df["pdb_id"])
    for pid_i, pid_j in combinations(ids, 2):
        rec_i, rec_j = cache.get(pid_i), cache.get(pid_j)
        if rec_i is None or rec_j is None:
            out_rows.append({
                "sequence_cluster_id": cluster_id,
                "pdb_id_i": pid_i, "pdb_id_j": pid_j,
                "n_matched_residues": 0, "rmsd": np.nan, "status": "parse_error",
            })
            continue
        seq_i, atoms_i = rec_i
        seq_j, atoms_j = rec_j
        rmsd, n_match, status = pair_rmsd(atoms_i, seq_i, atoms_j, seq_j)
        out_rows.append({
            "sequence_cluster_id": cluster_id,
            "pdb_id_i": pid_i, "pdb_id_j": pid_j,
            "n_matched_residues": n_match,
            "rmsd": rmsd if rmsd is not None else np.nan,
            "status": status,
        })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")

    all_rows = []
    for cid, cdf in df.groupby("sequence_cluster_id", sort=False):
        if len(cdf) < 2:
            continue
        log.info(f"cluster {cid} (n={len(cdf)})")
        all_rows.extend(process_cluster(cid, cdf))

    out_df = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    log.info(f"wrote {len(out_df)} pair rows -> {args.output}")


if __name__ == "__main__":
    main()
