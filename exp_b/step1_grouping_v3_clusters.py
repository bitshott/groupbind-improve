#!/usr/bin/env python3
"""
Step 1 v3 — Build PDBBind groups using MMseqs2 clusters_cluster.tsv.

This replaces exact-string sequence matching / hash(canonical_seq) grouping.

Grouping key = (mmseqs_cluster_id, pocket_cluster_index)
  - mmseqs_cluster_id: representative PDB ID from clusters_cluster.tsv,
    produced by `mmseqs easy-cluster` as:
        representative_pdb_id <TAB> member_pdb_id
  - pocket_cluster_index: single-linkage clustering of ligand centroids
    among complexes sharing the same MMseqs2 sequence cluster, with a
    --pocket_radius A threshold. This ensures we don't group ligands
    binding to different sites in the same protein family.

Bucket assignment is identical to v1:
    SG : test PDB has co-binders in the test set
    AG : co-binders only in train/val
    NG : no co-binders anywhere

Test/train PDB lists come from DiffDock plain-text files (one id per line),
NOT from LP_PDBBind's new_split (which is a different, similarity-based split).

Run:
    python3 step1_grouping_v2.py \\
        --test_list  $DIFFDOCK_DIR/data/splits/timesplit_test \\
        --train_list $DIFFDOCK_DIR/data/splits/timesplit_no_lig_overlap_train \\
        --train_list $DIFFDOCK_DIR/data/splits/timesplit_no_lig_overlap_val \\
        --lp_pdbbind data/LP_PDBBind.csv \\
        --data_dir   data \\
        --out_dir    out \\
        --pocket_radius 8.0
"""
import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# --------------------------- I/O helpers ------------------------------------


def find_complex_dirs(data_dir: Path):
    """Return dict pdb_id -> Path(complex_dir) covering refined + general sets."""
    out = {}
    for sub in ("refined-set", "v2020-other-PL"):
        root = data_dir / sub
        if not root.is_dir():
            print(f"[warn] missing {root}", file=sys.stderr)
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            pdb_id = d.name.lower()
            if len(pdb_id) != 4:
                continue
            out[pdb_id] = d
    return out


def ligand_centroid(sdf_path: Path):
    """Return (3,) heavy-atom centroid, or None on failure."""
    if not sdf_path.exists():
        return None
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=False)
    for mol in suppl:
        if mol is None:
            continue
        conf = mol.GetConformer()
        coords = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )
        if coords.size == 0:
            return None
        return coords.mean(axis=0)
    return None


# --------------------------- Sequence normalization -------------------------


def canonical_seq(seq: str):
    """Light normalization so near-identical sequences merge.

    Rules:
      - uppercase
      - strip whitespace, non-letter characters
      - drop the very first / very last residue if doing so would make
        the sequence match a strictly longer one already seen
        (handled later, at the group-merge step, not here)
    Returns: cleaned string, or '' on failure.
    """
    if seq is None or (isinstance(seq, float) and np.isnan(seq)):
        return ""
    s = re.sub(r"[^A-Za-z]", "", str(seq)).upper()
    return s


def merge_near_identical(seq_to_pdbs: dict) -> dict:
    """Merge canonical sequences that differ only by 1-2 residues at the ends.

    Builds a new dict where each key is the *longest* representative of a
    near-identical cluster, and the value is the union of all member PDBs.

    Strategy: sort unique sequences by length descending. For each shorter
    sequence, if it is a suffix or substring (within a small slack) of an
    already-seen longer one, merge it. This handles the "extra leading H"
    case from carbonic anhydrase II observed in the data.
    """
    items = sorted(seq_to_pdbs.items(), key=lambda kv: -len(kv[0]))
    canonical_to_pdbs = {}
    canonical_keys = []  # in insertion order for stable merging
    for seq, pdbs in items:
        if not seq:
            continue
        matched = None
        for existing in canonical_keys:
            # cheap containment check both directions
            if seq in existing or existing in seq:
                # require near-equal lengths (within 4 residues)
                if abs(len(seq) - len(existing)) <= 4:
                    matched = existing
                    break
        if matched is None:
            canonical_to_pdbs[seq] = list(pdbs)
            canonical_keys.append(seq)
        else:
            canonical_to_pdbs[matched].extend(pdbs)
    return canonical_to_pdbs


# --------------------------- Pocket clustering -----------------------------


def assign_pocket_clusters(complex_records, pocket_radius_A: float):
    """Single-linkage cluster ligand centroids inside one sequence group."""
    if not complex_records:
        return {}
    ids = [r["pdb_id"] for r in complex_records]
    centroids = np.array(
        [r["centroid"] if r["centroid"] is not None else [0, 0, 0]
         for r in complex_records]
    )
    valid = np.array([r["centroid"] is not None for r in complex_records])
    n = len(ids)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            if np.linalg.norm(centroids[i] - centroids[j]) <= pocket_radius_A:
                union(i, j)
    return {ids[i]: find(i) for i in range(n)}


# --------------------------- MMseqs2 cluster loading ------------------------


def read_mmseqs_clusters(clusters_tsv: Path, valid_pdb_ids: set[str] | None = None):
    """Return dict cluster_id -> [pdb_id, ...] from MMseqs2 clusters_cluster.tsv.

    Expected format from `mmseqs easy-cluster`:
        representative_pdb_id <TAB> member_pdb_id

    The representative PDB ID is used directly as the sequence-group key.
    """
    if not clusters_tsv.exists():
        raise FileNotFoundError(f"clusters TSV not found: {clusters_tsv}")

    cluster_to_pdbs = defaultdict(list)
    malformed = 0
    skipped_not_in_data = 0

    with open(clusters_tsv, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                malformed += 1
                continue

            rep = parts[0].strip().lower()[:4]
            member = parts[1].strip().lower()[:4]

            if valid_pdb_ids is not None and member not in valid_pdb_ids:
                skipped_not_in_data += 1
                continue

            cluster_to_pdbs[rep].append(member)

    # Deduplicate members while preserving order.
    out = {}
    for rep, members in cluster_to_pdbs.items():
        seen = set()
        dedup = []
        for m in members:
            if m not in seen:
                seen.add(m)
                dedup.append(m)
        if dedup:
            out[rep] = dedup

    print(f"[step1] MMseqs2 clusters loaded: {len(out)}")
    print(f"[step1] malformed cluster lines: {malformed}")
    print(f"[step1] cluster members skipped outside data_dir: {skipped_not_in_data}")
    return out


# --------------------------- Build groups ----------------------------------


def build_groups(
    data_dir: Path,
    lp_csv: Path,
    clusters_tsv: Path,
    pocket_radius_A: float,
):
    """Construct {group_key -> [pdb_id, ...]} keyed by MMseqs cluster + pocket."""
    # LP_PDBBind.csv is still read only for diagnostics / ID intersection.
    # Grouping itself comes from clusters_cluster.tsv.
    df = pd.read_csv(lp_csv, index_col=0)
    df.columns = [c.lower().strip() for c in df.columns]
    df["__pdbid"] = df.index.astype(str).str.lower().str.strip().str[:4]
    pdb_col = "__pdbid"

    complex_dirs = find_complex_dirs(data_dir)
    print(f"[step1] found {len(complex_dirs)} complex directories")
    print(f"[step1] LP_PDBBind rows: {len(df)}")

    sample_dirs = sorted(list(complex_dirs.keys())[:5])
    sample_csv = df[pdb_col].head().tolist()
    print(f"[step1] sample dir keys: {sample_dirs}")
    print(f"[step1] sample CSV pdb ids: {sample_csv}")
    intersection = set(complex_dirs.keys()) & set(df[pdb_col].tolist())
    print(f"[step1] dir/CSV intersection size: {len(intersection)}")
    if len(intersection) == 0:
        print(f"[step1] FATAL: no PDB ids in CSV match any directory.")
        print(f"[step1]   first complex_dirs keys (full): "
              f"{list(complex_dirs.keys())[:3]}")
        print(f"[step1]   first CSV pdb_col values (full): "
              f"{df[pdb_col].head(3).tolist()}")

    # MMseqs2 output is the new sequence-family grouping source.
    cluster_to_pdbs = read_mmseqs_clusters(
        clusters_tsv=clusters_tsv,
        valid_pdb_ids=set(complex_dirs.keys()),
    )

    print(f"[step1] MMseqs2 sequence groups: {len(cluster_to_pdbs)}")
    multi_seq = sum(1 for v in cluster_to_pdbs.values() if len(v) > 1)
    print(f"[step1] MMseqs2 groups with >1 complex: {multi_seq}")

    # Compute ligand centroids only for complexes in multi-member MMseqs groups.
    centroid_cache = {}
    needed_pids = set()
    for pids in cluster_to_pdbs.values():
        if len(pids) > 1:
            needed_pids.update(pids)
    print(f"[step1] computing centroids for {len(needed_pids)} multi-ligand complexes")
    for i, pid in enumerate(sorted(needed_pids)):
        if i % 500 == 0 and i > 0:
            print(f"[step1]   centroids {i}/{len(needed_pids)}", flush=True)
        cdir = complex_dirs[pid]
        sdf = cdir / f"{pid}_ligand.sdf"
        centroid_cache[pid] = ligand_centroid(sdf)

    # Per-MMseqs-cluster pocket clustering.
    groups = {}
    pdb_to_groups = defaultdict(list)

    for cluster_id, pids in cluster_to_pdbs.items():
        if len(pids) == 1:
            pid = pids[0]
            key = f"CLUSTER_{cluster_id}__0"
            groups[key] = [pid]
            pdb_to_groups[pid].append(key)
            continue

        recs = [{"pdb_id": p, "centroid": centroid_cache.get(p)} for p in pids]
        clusters = assign_pocket_clusters(recs, pocket_radius_A)

        for pid, cidx in clusters.items():
            key = f"CLUSTER_{cluster_id}__{cidx}"
            groups.setdefault(key, []).append(pid)
            pdb_to_groups[pid].append(key)

    return groups, pdb_to_groups


# --------------------------- Bucket assignment -----------------------------


def assign_buckets(test_ids, train_ids, groups, pdb_to_groups):
    test_set = set(test_ids)
    train_set = set(train_ids)
    rows = []
    for pid in sorted(test_set):
        peer_test, peer_train = set(), set()
        for gk in pdb_to_groups.get(pid, []):
            for other in groups.get(gk, []):
                if other == pid:
                    continue
                if other in test_set:
                    peer_test.add(other)
                if other in train_set:
                    peer_train.add(other)
        if peer_test:
            bucket = "SG"
        elif peer_train:
            bucket = "AG"
        else:
            bucket = "NG"
        rows.append({
            "pdb_id": pid,
            "bucket": bucket,
            "n_peers_test": len(peer_test),
            "n_peers_train": len(peer_train),
            "group_key": ";".join(pdb_to_groups.get(pid, [])),
        })
    return pd.DataFrame(rows)


# --------------------------- Split loading ---------------------------------


def load_split(test_path: Path, train_paths):
    """Load test+train PDB ids from DiffDock plain-text lists."""
    test_path = Path(test_path)
    test_ids = [
        ln.strip().lower() for ln in test_path.read_text().splitlines()
        if ln.strip()
    ]
    train_ids = []
    for tp in (train_paths or []):
        tp = Path(tp)
        if not tp.exists():
            print(f"[step1] WARNING: train list {tp} missing; skipping",
                  flush=True)
            continue
        train_ids.extend(
            ln.strip().lower() for ln in tp.read_text().splitlines()
            if ln.strip()
        )
    seen = set()
    train_ids = [x for x in train_ids if not (x in seen or seen.add(x))]
    return test_ids, train_ids


# --------------------------- main ------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list", type=Path, required=True,
                    help="DiffDock data/splits/timesplit_test (363 PDBs)")
    ap.add_argument("--train_list", type=Path, action="append", default=None,
                    help="DiffDock train (and val) plain-text PDB lists. "
                    "Pass twice to include both train and val.")
    ap.add_argument("--lp_pdbbind", type=Path, default=Path("data/LP_PDBBind.csv"),
                    help="LP_PDBBind.csv — used only for PDB ID diagnostics")
    ap.add_argument("--clusters_tsv", type=Path, default=Path("clusters_cluster.tsv"),
                    help="MMseqs2 easy-cluster output: representative_pdb_id<TAB>member_pdb_id")
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out_dir", type=Path, default=Path("out"))
    ap.add_argument("--pocket_radius", type=float, default=8.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_ids, train_ids = load_split(args.test_list, args.train_list)
    print(f"[step1] test set size:  {len(test_ids)}")
    print(f"[step1] train+val size: {len(train_ids)}")

    groups, pdb_to_groups = build_groups(
        args.data_dir, args.lp_pdbbind, args.clusters_tsv, args.pocket_radius
    )
    multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"[step1] total groups:           {len(groups)}")
    print(f"[step1] groups with >1 complex: {multi}")

    df = assign_buckets(test_ids, train_ids, groups, pdb_to_groups)
    counts = df["bucket"].value_counts().to_dict()
    print(f"[step1] bucket counts:  {counts}")

    df.to_csv(args.out_dir / "buckets.csv", index=False)
    with open(args.out_dir / "groups.json", "w") as f:
        json.dump(groups, f)
    print(f"[step1] wrote {args.out_dir/'buckets.csv'} and groups.json")


if __name__ == "__main__":
    main()

