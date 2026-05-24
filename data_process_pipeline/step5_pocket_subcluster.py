import argparse
import logging
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from step4_align_to_reference import (
    apply_transform_to_ligand,
    apply_transform_to_structure,
    compute_transform,
    extract_residues,
    iter_ligand_candidates,
    safe_name,
    write_ligand_sdf,
    write_pdb,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step5")

MAX_K = 10
MIN_KABSCH_POINTS = 3


def ligand_com(sdf_path: str):
    if not isinstance(sdf_path, str) or not sdf_path or not Path(sdf_path).exists():
        return None
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            conf = mol.GetConformer()
            coords = np.array([[conf.GetAtomPosition(i).x,
                                conf.GetAtomPosition(i).y,
                                conf.GetAtomPosition(i).z]
                               for i in range(mol.GetNumAtoms())])
            return coords.mean(axis=0)
    except Exception:
        return None
    return None


def choose_k_on_coords(coords: np.ndarray, max_k: int):
    n = coords.shape[0]
    upper = min(max_k, n - 1)
    best_k, best_score, best_labels = 1, -1.0, np.zeros(n, dtype=int)
    for k in range(2, upper + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(coords)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_score <= 0:
        return 1, best_score, np.zeros(n, dtype=int)
    return best_k, best_score, best_labels


def pair_rmsd_for_subcluster(sub_df: pd.DataFrame, pocket_cluster_id: str) -> list[dict]:
    cache = {}
    for _, row in sub_df.iterrows():
        pid = row["pdb_id"]
        try:
            _, seq, ca = extract_residues(row["aligned_protein_pdb"])
            cache[pid] = (seq, ca)
        except Exception as e:
            log.warning(f"parse_error {pid}: {e}")
            cache[pid] = None

    out = []
    ids = list(sub_df["pdb_id"])
    for pid_i, pid_j in combinations(ids, 2):
        ri, rj = cache.get(pid_i), cache.get(pid_j)
        if ri is None or rj is None:
            out.append({"pocket_cluster_id": pocket_cluster_id, "pdb_id_i": pid_i, "pdb_id_j": pid_j,
                        "n_matched_residues": 0, "rmsd": np.nan, "status": "parse_error"})
            continue
        seq_i, atoms_i = ri
        seq_j, atoms_j = rj
        R, t, n_match, rmsd = compute_transform(atoms_i, seq_i, atoms_j, seq_j)
        status = "ok" if R is not None else "too_few_matches"
        out.append({"pocket_cluster_id": pocket_cluster_id, "pdb_id_i": pid_i, "pdb_id_j": pid_j,
                    "n_matched_residues": n_match, "rmsd": rmsd if rmsd is not None else np.nan,
                    "status": status})
    return out


def select_reference(pdbs: list[str], pair_rows: list[dict]) -> str:
    df = pd.DataFrame(pair_rows)
    means = {}
    for pid in pdbs:
        vals = df[(df["pdb_id_i"] == pid) | (df["pdb_id_j"] == pid)]["rmsd"].dropna()
        means[pid] = float(vals.mean()) if len(vals) else np.inf
    min_val = min(means.values())
    return sorted([p for p, v in means.items() if v == min_val])[0]


def realign_subcluster(sub_df: pd.DataFrame, pocket_cluster_id: str, out_dir: Path) -> list[dict]:
    pdbs = list(sub_df["pdb_id"])
    if len(pdbs) < 2:
        rows = []
        for r in sub_df.to_dict("records"):
            rows.append({**r, "pocket_cluster_id": pocket_cluster_id,
                         "is_reference": True, "alignment_rmsd": 0.0,
                         "align_status": "singleton_pocket"})
        return rows, []

    pair_rows = pair_rmsd_for_subcluster(sub_df, pocket_cluster_id)
    ref_pid = select_reference(pdbs, pair_rows)
    log.info(f"  pocket {pocket_cluster_id}: ref={ref_pid}")

    cluster_dir = out_dir / safe_name(pocket_cluster_id)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    ref_row = sub_df[sub_df["pdb_id"] == ref_pid].iloc[0]
    try:
        _, ref_seq, ref_ca = extract_residues(ref_row["aligned_protein_pdb"])
    except Exception as e:
        log.warning(f"ref parse_error {ref_pid}: {e}")
        return [{**r, "pocket_cluster_id": pocket_cluster_id,
                 "is_reference": (r["pdb_id"] == ref_pid),
                 "alignment_rmsd": np.nan, "align_status": "ref_parse_error"}
                for r in sub_df.to_dict("records")], pair_rows

    out_rows = []
    for _, row in sub_df.iterrows():
        pid = row["pdb_id"]
        is_ref = (pid == ref_pid)
        prot_out = cluster_dir / f"{pid}_protein_aligned.pdb"
        lig_out = cluster_dir / f"{pid}_ligand_aligned.sdf"
        rmsd_val = 0.0
        status = "ok"

        if is_ref:
            try:
                import shutil
                shutil.copyfile(row["aligned_protein_pdb"], prot_out)
            except Exception as e:
                log.warning(f"ref copy failed {pid}: {e}")
                status, prot_out, lig_out = "ref_write_error", "", ""
        else:
            try:
                mov_struct, mov_seq, mov_ca = extract_residues(row["protein_pdb"])
                R, t, n_match, rmsd_val = compute_transform(ref_ca, ref_seq, mov_ca, mov_seq)
                if R is None:
                    raise ValueError(f"too_few_matches ({n_match})")
                apply_transform_to_structure(mov_struct, R, t)
                write_pdb(mov_struct, str(prot_out))
            except Exception as e:
                log.warning(f"align failed {pid}: {e}")
                status, rmsd_val, prot_out, lig_out = "align_error", np.nan, "", ""

        if status == "ok":
            lig_written = False
            for mol, fmt in iter_ligand_candidates(row.get("ligand_sdf"), row.get("ligand_mol2")):
                try:
                    if mol.GetNumConformers() == 0:
                        continue
                    if not is_ref:
                        apply_transform_to_ligand(mol, R, t)
                    write_ligand_sdf(mol, str(lig_out))
                    lig_written = True
                    break
                except Exception as e:
                    log.warning(f"ligand write attempt failed {pid} ({fmt}): {e}")
            if not lig_written:
                lig_out = ""
                status = "ok_no_ligand"

        out_rows.append({**row.to_dict(),
                         "pocket_cluster_id": pocket_cluster_id,
                         "aligned_protein_pdb": str(prot_out) if prot_out else "",
                         "aligned_ligand_sdf": str(lig_out) if lig_out else "",
                         "is_reference": is_ref,
                         "alignment_rmsd": rmsd_val,
                         "align_status": status})
    return out_rows, pair_rows


def process_seq_cluster(seq_cid: str, sdf: pd.DataFrame, out_dir: Path):
    coms = {}
    no_lig_rows = []
    for _, row in sdf.iterrows():
        com = ligand_com(row.get("aligned_ligand_sdf"))
        if com is None:
            no_lig_rows.append(row)
        else:
            coms[row["pdb_id"]] = com

    manifest_rows = []
    pair_rows = []

    if no_lig_rows:
        nl_pocket_id = f"{seq_cid}|POCKETCLUST:nolig"
        for r in no_lig_rows:
            manifest_rows.append({**r.to_dict(),
                                  "pocket_cluster_id": nl_pocket_id,
                                  "is_reference": True,
                                  "alignment_rmsd": 0.0,
                                  "align_status": "no_ligand_passthrough"})

    if len(coms) == 0:
        return manifest_rows, pair_rows

    com_df = sdf[sdf["pdb_id"].isin(coms.keys())].reset_index(drop=True)
    coords = np.array([coms[pid] for pid in com_df["pdb_id"]])

    if len(coords) < 2:
        labels = np.zeros(len(coords), dtype=int)
    else:
        k, score, labels = choose_k_on_coords(coords, MAX_K)
        log.info(f"seq_cluster {seq_cid}: n={len(coords)}, k={k}, silhouette={score:.3f}")

    com_df["_pocket_label"] = labels
    for lbl, sub in com_df.groupby("_pocket_label", sort=True):
        pocket_id = f"{seq_cid}|POCKETCLUST:{int(lbl)}"
        sub = sub.drop(columns=["_pocket_label"])
        rows, pairs = realign_subcluster(sub, pocket_id, out_dir)
        manifest_rows.extend(rows)
        pair_rows.extend(pairs)

    return manifest_rows, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--rmsd-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_manifest, all_pairs = [], []
    for seq_cid, sdf in df.groupby("sequence_cluster_id", sort=False):
        if len(sdf) < 2:
            for r in sdf.to_dict("records"):
                all_manifest.append({**r,
                                     "pocket_cluster_id": f"{seq_cid}|POCKETCLUST:0",
                                     "is_reference": True,
                                     "alignment_rmsd": 0.0,
                                     "align_status": "singleton_seq"})
            continue
        log.info(f"seq_cluster {seq_cid} (n={len(sdf)})")
        m, p = process_seq_cluster(seq_cid, sdf, out_dir)
        all_manifest.extend(m)
        all_pairs.extend(p)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rmsd_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_manifest).to_csv(args.output_csv, index=False)
    pd.DataFrame(all_pairs).to_csv(args.rmsd_csv, index=False)
    log.info(f"wrote {len(all_manifest)} manifest rows -> {args.output_csv}")
    log.info(f"wrote {len(all_pairs)} pair rows -> {args.rmsd_csv}")


if __name__ == "__main__":
    main()
