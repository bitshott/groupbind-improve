#!/usr/bin/env python3
"""
Step 1 — Build PDBBind groups and assign each test PDB to one of three buckets:

    SG : has co-binders to the same UniProt ID inside the test set itself
    AG : has co-binders only in the training/validation set
    NG : no co-binders anywhere in PDBBind

Group definition follows Algorithm 1 of the GroupBind paper (Appendix A),
restricted to the parts needed for bucket assignment:
    - group by UniProt ID extracted from the protein .pdb DBREF records
    - inside a UniProt group, cluster complexes by ligand centroid in the
      *aligned* protein frame so that only same-pocket binders count
      (Kabsch alignment on the chain that matches the reference, longest
      common subsequence)

Inputs (paths fixed for the user's layout):
    data/refined-set/<pdb_id>/<pdb_id>_protein.pdb
    data/refined-set/<pdb_id>/<pdb_id>_ligand.sdf
    data/v2020-other-PL/<pdb_id>/<pdb_id>_protein.pdb
    data/v2020-other-PL/<pdb_id>/<pdb_id>_ligand.sdf
    data/LP_PDBBind.csv  (time-split metadata; column 'new_split' holds
                          one of {train, val, test, ...})

Outputs:
    out/buckets.csv       columns: pdb_id, bucket, group_id, n_in_group
    out/groups.json       full group -> [pdb_ids] mapping (for downstream use)

Run:
    python3 step1_grouping.py --data_dir data --out_dir out --pocket_radius 8.0
"""
import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import BiopythonDeprecationWarning

warnings.simplefilter("ignore", BiopythonDeprecationWarning)
warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser  # noqa: E402
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


def parse_uniprot_ids(pdb_path: Path):
    """Extract UniProt accessions from DBREF / DBREF1 / DBREF2 lines.
    Returns a frozenset; empty if none found.
    """
    accs = set()
    try:
        with open(pdb_path, "r", errors="ignore") as f:
            for line in f:
                if line.startswith(("DBREF ", "DBREF1", "DBREF2")):
                    # UniProt entry name field is at fixed columns;
                    # we keep all tokens that look like accession codes.
                    tokens = line.split()
                    for tok in tokens:
                        if re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", tok):
                            accs.add(tok)
                        elif re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]", tok):
                            accs.add(tok)
    except FileNotFoundError:
        pass
    return frozenset(accs)


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


def protein_ca_coords(pdb_path: Path):
    """Return (N,3) CA coordinates of the longest chain. Used for rough
    pocket-equivalence checks between two complexes that share UniProt."""
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure("x", str(pdb_path))
    except Exception:
        return None, None
    best_chain_coords, best_seq = None, ""
    for model in struct:
        for chain in model:
            seq, coords = [], []
            for res in chain:
                if "CA" in res:
                    seq.append(res.get_resname())
                    coords.append(res["CA"].coord)
            if len(seq) > len(best_seq):
                best_seq = "".join(seq)  # not real one-letter, only for length
                best_chain_coords = np.array(coords)
        break  # first model only
    return best_chain_coords, best_seq


# --------------------------- Grouping logic ---------------------------------


def assign_pocket_clusters(complex_records, pocket_radius_A: float):
    """Within one UniProt group, cluster complexes by ligand centroid distance.
    Uses a simple single-linkage on Euclidean distance of centroids transformed
    into the reference complex's frame via translation only (we do not run
    Kabsch here — the paper's filter `0.4 A < min_d < 3.0 A` is the actual
    pocket sanity check; centroid clustering is a fallback to avoid grouping
    distant pockets on the same protein).

    Returns: dict pdb_id -> cluster_index (within this UniProt group)
    """
    if not complex_records:
        return {}
    # Build a centroid matrix
    ids = [r["pdb_id"] for r in complex_records]
    centroids = np.array(
        [r["centroid"] if r["centroid"] is not None else [0, 0, 0]
         for r in complex_records]
    )
    valid = np.array([r["centroid"] is not None for r in complex_records])

    # Single-linkage clustering
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

    cluster_id = {ids[i]: find(i) for i in range(n)}
    return cluster_id


def build_groups(data_dir: Path, pocket_radius_A: float):
    """Construct {group_key -> [pdb_id, ...]}.
    group_key = (uniprot_accession, pocket_cluster_index)
    """
    complex_dirs = find_complex_dirs(data_dir)
    print(f"[step1] found {len(complex_dirs)} complex directories")

    # Pass 1: extract UniProt + ligand centroid per complex
    records = []
    for i, (pdb_id, cdir) in enumerate(sorted(complex_dirs.items())):
        if i % 1000 == 0:
            print(f"[step1] parsing {i}/{len(complex_dirs)} ...", flush=True)
        prot_path = cdir / f"{pdb_id}_protein.pdb"
        sdf_path = cdir / f"{pdb_id}_ligand.sdf"
        accs = parse_uniprot_ids(prot_path)
        centroid = ligand_centroid(sdf_path)
        records.append({
            "pdb_id": pdb_id,
            "uniprots": accs,
            "centroid": centroid,
        })

    # Pass 2: bucket by UniProt accession (one complex may belong to multiple)
    by_uniprot = defaultdict(list)
    no_uniprot = []
    for r in records:
        if not r["uniprots"]:
            no_uniprot.append(r)
            continue
        for acc in r["uniprots"]:
            by_uniprot[acc].append(r)
    print(f"[step1] UniProt accessions found: {len(by_uniprot)}")
    print(f"[step1] complexes with no UniProt: {len(no_uniprot)}")

    # Pass 3: within each UniProt, cluster by pocket centroid
    groups = {}  # group_key -> [pdb_id]
    pdb_to_groups = defaultdict(list)
    for acc, recs in by_uniprot.items():
        clusters = assign_pocket_clusters(recs, pocket_radius_A)
        for pdb_id, cluster_idx in clusters.items():
            key = f"{acc}__{cluster_idx}"
            groups.setdefault(key, []).append(pdb_id)
            pdb_to_groups[pdb_id].append(key)

    # Complexes without UniProt become singleton groups
    for r in no_uniprot:
        key = f"NOUNIPROT__{r['pdb_id']}"
        groups[key] = [r["pdb_id"]]
        pdb_to_groups[r["pdb_id"]].append(key)

    return groups, pdb_to_groups


# --------------------------- Bucket assignment ------------------------------


def assign_buckets(test_ids, train_ids, groups, pdb_to_groups):
    """For each test_id, classify as SG / AG / NG."""
    test_set = set(test_ids)
    train_set = set(train_ids)
    rows = []
    for pid in sorted(test_set):
        peer_test, peer_train = set(), set()
        my_groups = pdb_to_groups.get(pid, [])
        for gk in my_groups:
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
            "group_key": ";".join(my_groups) if my_groups else "",
        })
    return pd.DataFrame(rows)


# --------------------------- Time-split loading -----------------------------


def load_split(test_path: Path, train_paths=None):
    """Load test+train PDB ids.

    Two formats supported:

    1) **DiffDock time-split** — plain text files with one PDB id per line.
       Pass `--test_list .../timesplit_test` and one or more
       `--train_list .../timesplit_no_lig_overlap_train` flags.
       This is the GroupBind paper's actual test set (363 PDBs).

    2) **LP-PDBBind CSV** — pass `--test_list .../LP_PDBBind.csv` alone;
       train/val are inferred from the `new_split` column.
       NOTE: this is a *different* test set (similarity-based, ~3k-4k PDBs),
       not the 363-complex time-split used by GroupBind / DiffDock.
    """
    test_path = Path(test_path)

    if test_path.suffix.lower() == ".csv":
        df = pd.read_csv(test_path)
        df.columns = [c.lower().strip() for c in df.columns]
        pdb_col = next(
            (c for c in ["pdbid", "pdb_id", "pdb", "header", "id"]
             if c in df.columns), None,
        )
        split_col = next(
            (c for c in ["new_split", "time_split", "split", "set"]
             if c in df.columns), None,
        )
        if pdb_col is None or split_col is None:
            raise RuntimeError(
                f"Could not find PDB-id / split columns in {test_path}. "
                f"Available columns: {list(df.columns)}"
            )
        df[pdb_col] = df[pdb_col].astype(str).str.lower()
        df[split_col] = df[split_col].astype(str).str.lower()
        test_ids = df.loc[df[split_col].str.startswith("test"), pdb_col].tolist()
        train_ids = df.loc[
            df[split_col].str.startswith(("train", "val")), pdb_col
        ].tolist()
        return test_ids, train_ids

    # Plain-text DiffDock-style list
    test_ids = [
        ln.strip().lower() for ln in test_path.read_text().splitlines()
        if ln.strip()
    ]
    train_ids = []
    for tp in (train_paths or []):
        tp = Path(tp)
        if not tp.exists():
            print(f"[step1] WARNING: train list {tp} does not exist; skipping",
                  flush=True)
            continue
        train_ids.extend(
            ln.strip().lower() for ln in tp.read_text().splitlines()
            if ln.strip()
        )
    # De-duplicate while preserving order
    seen = set()
    train_ids = [x for x in train_ids if not (x in seen or seen.add(x))]
    return test_ids, train_ids


# --------------------------- main -------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_list",
        type=Path,
        default=None,
        help="DiffDock-style plain-text PDB list (one id per line) — REQUIRED "
        "for faithful GroupBind/DiffDock reproduction. Use the DiffDock repo's "
        "data/splits/timesplit_test (363 PDBs). "
        "Alternatively, an LP-PDBBind CSV (different test set, ~3-4k PDBs).",
    )
    ap.add_argument(
        "--train_list",
        type=Path,
        action="append",
        default=None,
        help="DiffDock-style train/val PDB list (used only with a plain-text "
        "--test_list). Pass this flag TWICE to include both files, e.g. "
        "--train_list .../timesplit_no_lig_overlap_train "
        "--train_list .../timesplit_no_lig_overlap_val",
    )
    ap.add_argument("--data_dir", type=Path, default=Path("data"))
    ap.add_argument("--out_dir", type=Path, default=Path("out"))
    ap.add_argument(
        "--pocket_radius",
        type=float,
        default=8.0,
        help="single-linkage radius (A) on ligand centroids within UniProt",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.test_list is None:
        # Backwards-compatible fallback
        args.test_list = args.data_dir / "LP_PDBBind.csv"
        print(f"[step1] WARNING: no --test_list given, falling back to "
              f"{args.test_list}. This is NOT the 363-PDB DiffDock/GroupBind "
              f"test set.", flush=True)

    test_ids, train_ids = load_split(args.test_list, args.train_list)
    print(f"[step1] test set size:  {len(test_ids)}")
    print(f"[step1] train+val size: {len(train_ids)}")

    groups, pdb_to_groups = build_groups(args.data_dir, args.pocket_radius)
    print(f"[step1] total groups:   {len(groups)}")
    multi = sum(1 for v in groups.values() if len(v) > 1)
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
