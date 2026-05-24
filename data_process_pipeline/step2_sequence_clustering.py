import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import PDBParser, PPBuilder
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

warnings.simplefilter("ignore", BiopythonWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("step2")

MAX_K = 10
MDS_RANDOM_STATE = 42


def extract_sequence(pdb_path: str) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    ppb = PPBuilder()
    seq_parts = [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    return "".join(seq_parts)


def build_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def pairwise_identity_distance(sequences: list[str], aligner: PairwiseAligner) -> np.ndarray:
    n = len(sequences)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                aln = aligner.align(sequences[i], sequences[j])[0]
                a, b = aln[0], aln[1]
                matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
                length = max(len(sequences[i]), len(sequences[j]))
                ident = matches / length if length > 0 else 0.0
            except Exception as e:
                log.warning(f"alignment failed {i},{j}: {e}")
                ident = 0.0
            d[i, j] = d[j, i] = 1.0 - ident
    return d


def choose_k(embedding: np.ndarray, max_k: int) -> tuple[int, float, np.ndarray]:
    n = embedding.shape[0]
    upper = min(max_k, n - 1)
    best_k, best_score, best_labels = 1, -1.0, np.zeros(n, dtype=int)
    for k in range(2, upper + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embedding)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embedding, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if best_score <= 0:
        return 1, best_score, np.zeros(n, dtype=int)
    return best_k, best_score, best_labels


def cluster_group(group_df: pd.DataFrame, aligner: PairwiseAligner) -> pd.DataFrame:
    sequences, valid_idx = [], []
    for idx, row in group_df.iterrows():
        try:
            seq = extract_sequence(row["protein_pdb"])
            if len(seq) < 10:
                raise ValueError("sequence too short")
            sequences.append(seq)
            valid_idx.append(idx)
        except Exception as e:
            log.warning(f"failed to read {row['protein_pdb']}: {e}")

    out = group_df.copy()
    out["sequence_cluster"] = -1

    if len(sequences) < 2:
        out.loc[valid_idx, "sequence_cluster"] = 0
        return out

    dist = pairwise_identity_distance(sequences, aligner)
    n = len(sequences)

    if dist.max() < 1e-9:
        out.loc[valid_idx, "sequence_cluster"] = 0
        log.info(f"  -> n={n}, all sequences identical, k=1")
        return out

    n_components = min(max(2, n - 1), 10)
    try:
        embedding = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=MDS_RANDOM_STATE,
            normalized_stress="auto",
        ).fit_transform(dist)
    except Exception as e:
        log.warning(f"MDS failed: {e}")
        out.loc[valid_idx, "sequence_cluster"] = 0
        return out

    k, score, labels = choose_k(embedding, MAX_K)
    log.info(f"  -> n={n}, k={k}, silhouette={score:.3f}")
    for vi, lbl in zip(valid_idx, labels):
        out.at[vi, "sequence_cluster"] = int(lbl)
    return out


def make_cluster_id(initial_group: str, cluster_idx: int) -> str:
    return f"{initial_group}|SEQCLUST:{cluster_idx}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--singletons-output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")

    aligner = build_aligner()
    multi_results, singleton_rows = [], []

    for grp_name, grp_df in df.groupby("initial_protein_group", sort=False):
        if len(grp_df) < 2:
            log.info(f"singleton group: {grp_name}")
            singleton_rows.append(grp_df)
            continue
        log.info(f"clustering group: {grp_name} (n={len(grp_df)})")
        clustered = cluster_group(grp_df, aligner)
        clustered["sequence_cluster_id"] = clustered.apply(
            lambda r: make_cluster_id(grp_name, int(r["sequence_cluster"])), axis=1
        )
        clustered = clustered.drop(columns=["sequence_cluster"])
        multi_results.append(clustered)

    out_df = pd.concat(multi_results, ignore_index=True) if multi_results else pd.DataFrame(columns=df.columns)
    singleton_df = pd.concat(singleton_rows, ignore_index=True) if singleton_rows else pd.DataFrame(columns=df.columns)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.singletons_output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    singleton_df.to_csv(args.singletons_output, index=False)

    log.info(f"wrote {len(out_df)} clustered rows -> {args.output}")
    log.info(f"wrote {len(singleton_df)} singleton rows -> {args.singletons_output}")


if __name__ == "__main__":
    main()
