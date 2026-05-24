#!/usr/bin/env python

import argparse
import os
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
from tqdm import tqdm

import csv

from pebble import ProcessPool
from concurrent.futures import TimeoutError

def parse_args():
    p = argparse.ArgumentParser(
        description="Pairwise MCS quality analysis inside final_group_id groups."
    )
    p.add_argument("--input-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group-col", default="final_group_id")
    p.add_argument("--status-col", default="final_status")
    p.add_argument("--ok-status", default="ok")
    p.add_argument("--pdb-col", default="pdb_id")
    p.add_argument("--timeout", type=int, default=10)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument("--trivial-max-atoms", type=int, default=2)
    p.add_argument("--trivial-min-frac", type=float, default=0.20)
    p.add_argument("--path-root")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--hard-timeout", type=int, default=30)
    return p.parse_args()


def first_existing_path(row):
    for col in ("ligand_sdf", "ligand_mol2"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            path = str(row[col]).strip()
            if os.path.exists(path):
                return path
    return None


def load_mol(path):
    if path is None:
        return None, "missing_path"

    ext = Path(path).suffix.lower()

    try:
        if ext == ".sdf":
            suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
            mol = suppl[0] if suppl is not None and len(suppl) > 0 else None
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
        else:
            mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)

        if mol is None:
            return None, "rdkit_read_failed"

        if mol.GetNumAtoms() == 0:
            return None, "zero_atoms"

        try:
            Chem.SanitizeMol(mol)
            return mol, None
        except Exception as e_full:
            try:
                sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
                return mol, f"partial_sanitize_no_kekulize:{type(e_full).__name__}:{e_full}"
            except Exception as e_partial:
                if mol.GetNumAtoms() > 0 and mol.GetNumBonds() > 0:
                    return mol, f"unsanitized_kept:{type(e_partial).__name__}:{e_partial}"
                return None, f"sanitize_failed:{type(e_partial).__name__}:{e_partial}"

    except Exception as e:
        return None, f"load_exception:{type(e).__name__}:{e}"

def mol_to_bytes(mol):
    return mol.ToBinary()


def mol_from_bytes(blob):
    return Chem.Mol(blob)


def mcs_one_pair(task):
    (
        group_id,
        pdb1,
        pdb2,
        mol1_blob,
        mol2_blob,
        timeout,
        trivial_max_atoms,
        trivial_min_frac,
    ) = task

    mol1 = mol_from_bytes(mol1_blob)
    mol2 = mol_from_bytes(mol2_blob)

    base = {
        "final_group_id": group_id,
        "pdb_id_1": pdb1,
        "pdb_id_2": pdb2,
        "mcs_failed": False,
        "error": "",
    }

    try:
        n1 = int(mol1.GetNumAtoms())
        n2 = int(mol2.GetNumAtoms())
        b1 = int(mol1.GetNumBonds())
        b2 = int(mol2.GetNumBonds())

        mcs = rdFMCS.FindMCS(
            [mol1, mol2],
            timeout=timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareAny,
        )

        mcs_atoms = int(mcs.numAtoms)
        mcs_bonds = int(mcs.numBonds)

        frac_min = mcs_atoms / max(1, min(n1, n2))
        frac_mean = (2.0 * mcs_atoms) / max(1, n1 + n2)
        bond_frac_min = mcs_bonds / max(1, min(b1, b2))

        base.update({
            "n_atoms_1": n1,
            "n_atoms_2": n2,
            "n_bonds_1": b1,
            "n_bonds_2": b2,
            "mcs_atoms": mcs_atoms,
            "mcs_bonds": mcs_bonds,
            "mcs_atom_fraction_min": frac_min,
            "mcs_atom_fraction_mean": frac_mean,
            "mcs_bond_fraction_min": bond_frac_min,
            "mcs_smarts": mcs.smartsString,
            "is_single_atom_mcs": mcs_atoms == 1,
            "is_trivial_mcs": (mcs_atoms <= trivial_max_atoms) or (frac_min < trivial_min_frac),
            "mcs_canceled": bool(mcs.canceled),
        })
        return base

    except Exception as e:
        base.update({
            "mcs_failed": True,
            "error": f"{type(e).__name__}:{e}",
            "n_atoms_1": np.nan,
            "n_atoms_2": np.nan,
            "n_bonds_1": np.nan,
            "n_bonds_2": np.nan,
            "mcs_atoms": np.nan,
            "mcs_bonds": np.nan,
            "mcs_atom_fraction_min": np.nan,
            "mcs_atom_fraction_mean": np.nan,
            "mcs_bond_fraction_min": np.nan,
            "mcs_smarts": "",
            "is_single_atom_mcs": False,
            "is_trivial_mcs": False,
            "mcs_canceled": False,
        })
        return base


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    required = [args.group_col, args.pdb_col]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    if args.status_col in df.columns:
        df = df[df[args.status_col].astype(str).eq(args.ok_status)].copy()

    df = df[df[args.group_col].notna()].copy()

    mol_cache = {}
    load_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading ligands"):
        pdb_id = str(row[args.pdb_col])
        path = first_existing_path(row)
        mol, err = load_mol(path)

        load_records.append({
            "pdb_id": pdb_id,
            "final_group_id": row[args.group_col],
            "ligand_path": path if path else "",
            "load_ok": mol is not None,
            "load_error": "" if err is None else err,
        })

        if mol is not None:
            mol_cache[pdb_id] = mol_to_bytes(mol)

    pd.DataFrame(load_records).to_csv(out_dir / "ligand_load_status.csv", index=False)

    tasks = []

    for group_id, g in df.groupby(args.group_col):
        pdb_ids = [str(x) for x in g[args.pdb_col].tolist()]
        pdb_ids = [x for x in pdb_ids if x in mol_cache]

        if len(pdb_ids) < args.min_group_size:
            continue

        for pdb1, pdb2 in combinations(pdb_ids, 2):
            tasks.append((
                group_id,
                pdb1,
                pdb2,
                mol_cache[pdb1],
                mol_cache[pdb2],
                args.timeout,
                args.trivial_max_atoms,
                args.trivial_min_frac,
            ))

    if not tasks:
        raise RuntimeError("No ligand pairs found. Check final_group_id groups and ligand paths.")

    pairwise_path = out_dir / "mcs_pairwise_quality.csv"

    done_keys = set()
    existing_records = []

    if args.resume and pairwise_path.exists():
        old = pd.read_csv(pairwise_path)
        existing_records = old.to_dict("records")

        for _, r in old.iterrows():
            done_keys.add((
                str(r["final_group_id"]),
                str(r["pdb_id_1"]),
                str(r["pdb_id_2"]),
            ))

        print(f"[resume] loaded completed pairs: {len(done_keys)}")

    filtered_tasks = []
    for t in tasks:
        key = (str(t[0]), str(t[1]), str(t[2]))
        if key not in done_keys:
            filtered_tasks.append(t)

    print(f"[resume] total pairs: {len(tasks)}")
    print(f"[resume] remaining pairs: {len(filtered_tasks)}")

    new_records = []

    def timeout_record(task, err):
        return {
            "final_group_id": task[0],
            "pdb_id_1": task[1],
            "pdb_id_2": task[2],
            "mcs_failed": True,
            "error": err,
            "n_atoms_1": np.nan,
            "n_atoms_2": np.nan,
            "n_bonds_1": np.nan,
            "n_bonds_2": np.nan,
            "mcs_atoms": np.nan,
            "mcs_bonds": np.nan,
            "mcs_atom_fraction_min": np.nan,
            "mcs_atom_fraction_mean": np.nan,
            "mcs_bond_fraction_min": np.nan,
            "mcs_smarts": "",
            "is_single_atom_mcs": False,
            "is_trivial_mcs": False,
            "mcs_canceled": False,
        }

    def flush_checkpoint():
        all_records = existing_records + new_records
        pd.DataFrame(all_records).to_csv(pairwise_path, index=False)
        print(f"[checkpoint] wrote {len(all_records)} records", flush=True)

    if args.n_jobs > 1:
        with ProcessPool(max_workers=args.n_jobs, max_tasks=1) as pool:
            future = pool.map(mcs_one_pair, filtered_tasks, timeout=args.hard_timeout)
            iterator = future.result()

            pbar = tqdm(total=len(filtered_tasks), desc="MCS pairs")

            task_i = 0
            while True:
                try:
                    rec = next(iterator)
                    new_records.append(rec)

                except StopIteration:
                    break

                except TimeoutError:
                    task = filtered_tasks[task_i]
                    rec = timeout_record(task, f"hard_timeout>{args.hard_timeout}s")
                    new_records.append(rec)

                except Exception as e:
                    task = filtered_tasks[task_i]
                    rec = timeout_record(task, f"{type(e).__name__}:{e}")
                    new_records.append(rec)

                finally:
                    task_i += 1
                    pbar.update(1)

                    if len(new_records) % args.checkpoint_every == 0:
                        flush_checkpoint()

            pbar.close()

    else:
        for task in tqdm(filtered_tasks, desc="MCS pairs"):
            try:
                rec = mcs_one_pair(task)
            except Exception as e:
                rec = timeout_record(task, f"{type(e).__name__}:{e}")

            new_records.append(rec)

            if len(new_records) % args.checkpoint_every == 0:
                flush_checkpoint()

    flush_checkpoint()

    pairwise = pd.read_csv(pairwise_path)
    valid = pairwise[~pairwise["mcs_failed"]].copy()

    group_summary = (
        valid.groupby("final_group_id")
        .agg(
            n_pairs=("pdb_id_1", "count"),
            median_mcs_atoms=("mcs_atoms", "median"),
            mean_mcs_atoms=("mcs_atoms", "mean"),
            min_mcs_atoms=("mcs_atoms", "min"),
            median_mcs_bonds=("mcs_bonds", "median"),
            median_mcs_atom_fraction_min=("mcs_atom_fraction_min", "median"),
            mean_mcs_atom_fraction_min=("mcs_atom_fraction_min", "mean"),
            min_mcs_atom_fraction_min=("mcs_atom_fraction_min", "min"),
            frac_single_atom_mcs=("is_single_atom_mcs", "mean"),
            frac_trivial_mcs=("is_trivial_mcs", "mean"),
            frac_mcs_canceled=("mcs_canceled", "mean"),
        )
        .reset_index()
    )

    failed = (
        pairwise.groupby("final_group_id")
        .agg(n_total_pairs=("pdb_id_1", "count"), n_failed=("mcs_failed", "sum"))
        .reset_index()
    )
    failed["frac_failed"] = failed["n_failed"] / failed["n_total_pairs"]

    group_summary = group_summary.merge(failed, on="final_group_id", how="outer")
    group_summary.to_csv(out_dir / "mcs_group_quality_summary.csv", index=False)

    print(f"Wrote: {out_dir / 'mcs_pairwise_quality.csv'}")
    print(f"Wrote: {out_dir / 'mcs_group_quality_summary.csv'}")
    print(f"Wrote: {out_dir / 'ligand_load_status.csv'}")


if __name__ == "__main__":
    main()
