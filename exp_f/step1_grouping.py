#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Step 1 — Assign SG/AG/NG buckets from the final groups produced by
step6_filter_by_pl_distance.py.

Expected input CSV is produced as:
    python step6_filter_by_pl_distance.py \
      --input pdbbind_pocket_aligned_complexes.csv \
      --output-csv pdbbind_final_groups.csv

This script no longer recomputes groups from LP_PDBBind.csv, protein sequence,
or ligand-centroid clustering. It consumes the already validated/isolated group
IDs from step 6:

    grouping key = final_group_id

Bucket assignment:
    SG : test PDB has co-binders in the test set
    AG : co-binders only in train/val
    NG : no co-binders anywhere

Run:
    python3 step1_grouping.py \
        --test_list  $DIFFDOCK_DIR/data/splits/timesplit_test \
        --train_list $DIFFDOCK_DIR/data/splits/timesplit_no_lig_overlap_train \
        --train_list $DIFFDOCK_DIR/data/splits/timesplit_no_lig_overlap_val \
        --groups_csv pdbbind_final_groups.csv \
        --out_dir out
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


# --------------------------- Group loading ---------------------------------


def load_groups_from_step6_csv(
    groups_csv: Path,
    pdb_col: str = "pdb_id",
    group_col: str = "final_group_id",
):
    """Load {group_key -> [pdb_id, ...]} from step6 output CSV.

    step6 assigns problematic complexes to isolated groups such as
    ISOLATED:<pdb_id>. Therefore all rows with a non-empty final_group_id are
    retained. This keeps failed/outlier complexes available for NG assignment
    instead of silently dropping test IDs.
    """
    df = pd.read_csv(groups_csv)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in (pdb_col, group_col) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing required column(s) in {groups_csv}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df[pdb_col] = df[pdb_col].astype(str).str.lower().str.strip()
    df[group_col] = df[group_col].astype(str).str.strip()
    df = df[(df[pdb_col] != "") & (df[group_col] != "")]
    df = df[df[group_col].str.lower() != "nan"]

    groups = defaultdict(list)
    pdb_to_groups = defaultdict(list)

    for _, row in df.iterrows():
        pid = row[pdb_col]
        gk = row[group_col]
        if pid not in groups[gk]:
            groups[gk].append(pid)
        if gk not in pdb_to_groups[pid]:
            pdb_to_groups[pid].append(gk)

    return dict(groups), pdb_to_groups, df


# Backward-compatible name used by group_stats.py.
def build_groups_from_csv(groups_csv: Path):
    groups, pdb_to_groups, _ = load_groups_from_step6_csv(groups_csv)
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
    """Load test+train PDB ids from plain-text lists, one id per line."""
    test_path = Path(test_path)
    test_ids = [
        ln.strip().lower() for ln in test_path.read_text().splitlines()
        if ln.strip()
    ]

    train_ids = []
    for tp in (train_paths or []):
        tp = Path(tp)
        if not tp.exists():
            print(f"[step1] WARNING: train list {tp} missing; skipping", flush=True)
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
                    help="Plain-text test PDB list, one PDB ID per line.")
    ap.add_argument("--train_list", type=Path, action="append", default=None,
                    help="Plain-text train/val PDB list. Pass multiple times if needed.")
    ap.add_argument("--groups_csv", type=Path, default=Path("pdbbind_final_groups.csv"),
                    help="CSV produced by step6_filter_by_pl_distance.py")
    ap.add_argument("--out_dir", type=Path, default=Path("out"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_ids, train_ids = load_split(args.test_list, args.train_list)
    print(f"[step1] test set size:  {len(test_ids)}")
    print(f"[step1] train+val size: {len(train_ids)}")

    groups, pdb_to_groups, df_groups = load_groups_from_step6_csv(args.groups_csv)
    multi = sum(1 for v in groups.values() if len(v) > 1)
    print(f"[step1] loaded rows:            {len(df_groups)}")
    print(f"[step1] total groups:           {len(groups)}")
    print(f"[step1] groups with >1 complex: {multi}")

    if "final_status" in df_groups.columns:
        print(f"[step1] final_status counts: {df_groups['final_status'].value_counts().to_dict()}")

    df = assign_buckets(test_ids, train_ids, groups, pdb_to_groups)
    counts = df["bucket"].value_counts().to_dict()
    print(f"[step1] bucket counts: {counts}")

    df.to_csv(args.out_dir / "buckets.csv", index=False)
    with open(args.out_dir / "groups.json", "w") as f:
        json.dump(groups, f)
    print(f"[step1] wrote {args.out_dir / 'buckets.csv'} and groups.json")


if __name__ == "__main__":
    main()

