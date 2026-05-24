import argparse
import logging
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("step8")

FP_RADIUS = 2
FP_BITS = 2048
MAX_LIGS_PER_GROUP = 5


def load_fp(sdf_path):
    if not isinstance(sdf_path, str) or not sdf_path or not Path(sdf_path).exists():
        return None
    try:
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True):
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    except Exception:
        pass
    try:
        for mol in Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False):
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception:
                pass
            return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    except Exception:
        return None
    return None


def cap_group(fps, max_per_cluster):
    n = len(fps)
    if n <= max_per_cluster:
        return [n]
    dist = np.zeros((n, n), dtype=float)
    for i, j in combinations(range(n), 2):
        d = 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        dist[i, j] = dist[j, i] = d
    k = math.ceil(n / max_per_cluster)
    labels = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="complete"
    ).fit_predict(dist)
    sizes = pd.Series(labels).value_counts().to_list()
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    total_complexes = len(df)

    group_sizes = df.groupby("final_group_id", sort=False).size()
    total_groups = len(group_sizes)
    multi_groups = int((group_sizes >= 2).sum())
    multi_pct = 100.0 * multi_groups / max(total_groups, 1)

    capped_datapoints = 0
    capped_multi = 0
    for gid, sub in df.groupby("final_group_id", sort=False):
        n = len(sub)
        if n <= MAX_LIGS_PER_GROUP:
            sizes = [n]
        else:
            fps = []
            for _, r in sub.iterrows():
                fp = load_fp(r.get("aligned_ligand_sdf", ""))
                if fp is not None:
                    fps.append(fp)
            if len(fps) < 2:
                sizes = [n]
            else:
                sizes = cap_group(fps, MAX_LIGS_PER_GROUP)
                if len(fps) < n:
                    sizes.append(n - len(fps))
        capped_datapoints += len(sizes)
        capped_multi += sum(1 for s in sizes if s >= 2)
    capped_multi_pct = 100.0 * capped_multi / max(capped_datapoints, 1)

    print("=" * 60)
    print(f"Input: {args.input}")
    print("=" * 60)
    print(f"Total complexes:                  {total_complexes}")
    print(f"Total groups (pockets):           {total_groups}")
    print(f"Groups with > 1 ligand:           {multi_groups} ({multi_pct:.1f}%)")
    print(f"Groups with = 1 ligand:           {total_groups - multi_groups} ({100-multi_pct:.1f}%)")
    print("-" * 60)
    print(f"After capping max {MAX_LIGS_PER_GROUP} ligands/group (Tanimoto + complete-linkage):")
    print(f"Datapoints:                       {capped_datapoints}")
    print(f"Datapoints with > 1 ligand:       {capped_multi} ({capped_multi_pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
