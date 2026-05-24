"""
Data preparation -- MCS-based ligand-group construction for GroupBind.

Reproduces the preprocessing in GroupBind (ICLR 2025), Algorithm 1 of the paper,
combined with the MCS edge-construction described in Section 3.3 ("Build Group
Ligands Graph"). Emits two artefacts:

  (1) group_manifest.csv   -- one row per (group, ligand) with metadata and the
                              MCS-derived inter-ligand edge set serialised.
  (2) records_template.csv -- a long-format scaffold compatible with the
                              `REQUIRED_COLUMNS` contract of
                              experiment5_stratified.py. The `rmsd` column is
                              left as NaN (it is the output of the docking
                              pipeline, not of data prep) and `intra_group_sim`
                              is populated here, since the paper itself
                              characterises groups by intra-group Tanimoto
                              (Fig. 6b).

The paper's preprocessing steps, in order, with the exact criteria it states:

  S1. Group complexes initially by UniProt ID + protein name (raw PDBBind
      metadata).
  S2. Cluster proteins further by amino-acid sequence alignment +
      agglomerative clustering (Ward), with k chosen via silhouette score.
  S3. Within each cluster, align proteins by longest common subsequence (LCS)
      of residues + Kabsch; pick the reference protein as the one with the
      minimum mean RMSD to the others; apply the same rigid transform to
      paired ligands.
  S4. Cluster grouped complexes further by ligand centre-of-mass, so that
      each final group binds the SAME pocket (not just the same protein).
  S5. Isolate as singletons any complex whose minimum protein-ligand distance
      is < 0.4 A (clash) or > 3.0 A (unbound).
  S6. Cap the maximum group size at 5 via agglomerative clustering on ligand
      Tanimoto similarity. Select ligands whose centre of mass is within 8 A
      of the native or P2Rank-predicted pocket centre.

Section 3.3 then constructs the inter-ligand edge set used by the MPNN:

  E1. 2D molecular-graph matching via Maximum Common Substructure (MCS) to
      establish a one-to-one atom mapping across ligands.
  E2. For unmatched atoms: connect to neighbours within 4 A.
        - At training time the 4 A radius is on TRUE coordinates.
        - At inference time, on coordinates from a generated conformer.
  E3. Edges are time-independent (do NOT depend on diffusion step).

What this script implements / does not implement:

  IMPLEMENTED in code below (the steps that are ligand-side and can be done
  with public Python tooling without re-running PDBBind structural alignment
  from scratch):
    - S1   grouping by UniProt-ID / name (consumed from an input manifest)
    - S4/S6 ligand-side filtering: 8 A pocket-centre cutoff, Tanimoto-based
            agglomerative capping at K_max ligands per group
    - S5   minimum-distance filtering, when atomic coordinates are supplied
    - E1   MCS atom-atom mapping via rdFMCS (RDKit's standard solver)
    - E2   4 A neighbour expansion at TRAIN time on supplied true coordinates
    - intra-group Tanimoto computation (Fig. 6b style) for stratification
    - serialisation of the (group, ligand) edge sets in a format that the
      docking pipeline can consume verbatim

  OUT OF SCOPE (require running structural-bioinformatics tools that are not
  available in this sandbox; the script defines the interfaces and consumes
  the products if you bring them):
    - S2 amino-acid clustering            (BLAST + sklearn agglomerative)
    - S3 Kabsch protein alignment + LCS  (e.g., Biopython / PyMOL)
    - E2 at INFERENCE time, the 3D matching on a generated conformer
         (depends on your conformer generator; the same code path applies
         with the conformer's coordinates substituted for the true ones)

The script is self-validating via a synthetic mini-dataset at the bottom (no
network, no PDBBind download), which lets the VM run end-to-end and confirm
the output schema matches experiment5_stratified.REQUIRED_COLUMNS.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


# --------------------------------------------------------------------------- #
# Paper-stated thresholds (Algorithm 1 + Section 3.3)                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PaperThresholds:
    pocket_centre_cutoff_A: float = 8.0   # S6: ligand CoM within 8 A of pocket
    minpl_clash_A: float = 0.4            # S5: minimum protein-ligand distance lower bound
    minpl_unbound_A: float = 3.0          # S5: minimum protein-ligand distance upper bound
    max_group_size: int = 5               # Training Consideration
    inter_ligand_radius_A: float = 4.0    # E2: 4 A unmatched-atom expansion


PAPER = PaperThresholds()


# --------------------------------------------------------------------------- #
# Input schema                                                                #
# --------------------------------------------------------------------------- #
# A complex is one (protein, ligand) pair. Minimum fields needed for the
# ligand-side of preprocessing:
#
#   complex_id       : str       -- unique key
#   uniprot_id       : str       -- from raw PDBBind metadata (S1)
#   protein_name     : str       -- from raw PDBBind metadata (S1)
#   ligand_smiles    : str       -- ligand SMILES
#   pocket_center    : (3,) float or None  -- native pocket centre coords; if
#                                             None, the S6 pocket-centre filter
#                                             is skipped for this complex
#   ligand_coords    : (n, 3) float or None -- true heavy-atom coords (training
#                                              time); needed for E2 and S5
#   protein_coords   : (m, 3) float or None -- C_alpha or heavy-atom coords for
#                                              S5 min-distance check
#
# A pandas frame with these columns (the coords-bearing fields stored as
# `numpy.ndarray` objects in cells) is the canonical input.

REQUIRED_INPUT_COLUMNS = [
    "complex_id",
    "uniprot_id",
    "protein_name",
    "ligand_smiles",
    "pocket_center",
    "ligand_coords",
    "protein_coords",
]


# --------------------------------------------------------------------------- #
# Building blocks                                                             #
# --------------------------------------------------------------------------- #
def _safe_mol(smi: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=(
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ),
            )
        except Exception:
            return None

    try:
        Chem.GetSymmSSSR(mol)
    except Exception:
        return None

    return mol

def _morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    try:
        Chem.GetSymmSSSR(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    except Exception:
        return None

def tanimoto(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    fp_a = _morgan_fp(mol_a)
    fp_b = _morgan_fp(mol_b)

    if fp_a is None or fp_b is None:
        return 0.0

    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

def mcs_atom_mapping(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    timeout_s: int = 5,
    match_valences: bool = True,
) -> list[tuple[int, int]]:
    """Paper step E1: one-to-one MCS atom mapping between two ligands.

    Returns a list of (atom_idx_in_a, atom_idx_in_b) tuples. Empty list if MCS
    is empty or solver times out. We use rdFMCS with the canonical settings the
    RDKit cookbook recommends for chemically-meaningful MCS (atom-compare by
    element, bond-compare by order; valence match optional). The paper does
    not pin down the exact MCS variant, so these defaults are the standard
    chemoinformatics choice."""
    res = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=timeout_s,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        matchValences=match_valences,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
    )
    if res.numAtoms == 0 or not res.smartsString:
        return []
    patt = Chem.MolFromSmarts(res.smartsString)
    match_a = mol_a.GetSubstructMatch(patt)
    match_b = mol_b.GetSubstructMatch(patt)
    if not match_a or not match_b:
        return []
    return list(zip(match_a, match_b))


def neighbours_within_radius(
    coords: np.ndarray, idx: int, radius_A: float
) -> list[int]:
    """List of atom indices whose distance to atom `idx` is < `radius_A`,
    excluding `idx` itself. Used to expand the MCS mapping to unmatched-atom
    neighbours (paper step E2).
    """
    if coords is None:
        return []

    if idx < 0 or idx >= len(coords):
        return []

    d = np.linalg.norm(coords - coords[idx], axis=1)
    nbrs = np.where((d < radius_A) & (d > 0))[0].tolist()
    return nbrs

# --------------------------------------------------------------------------- #
# Step S5 -- protein-ligand minimum-distance filter                           #
# --------------------------------------------------------------------------- #
def min_protein_ligand_distance(
    lig_coords: Optional[np.ndarray], prot_coords: Optional[np.ndarray]
) -> Optional[float]:
    if lig_coords is None or prot_coords is None:
        return None
    # Pairwise minimum without forming the full distance matrix when large.
    # For PDBBind pocket-level inputs the size is small; broadcast is fine.
    diff = lig_coords[:, None, :] - prot_coords[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).min())


def passes_minpl_filter(d_min: Optional[float]) -> bool:
    if d_min is None:
        return True  # cannot evaluate; keep
    return PAPER.minpl_clash_A <= d_min <= PAPER.minpl_unbound_A


# --------------------------------------------------------------------------- #
# Step S6 -- pocket-centre 8 A cutoff (ligand centre of mass)                 #
# --------------------------------------------------------------------------- #
def ligand_com(coords: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if coords is None:
        return None
    return coords.mean(axis=0)


def passes_pocket_cutoff(
    com: Optional[np.ndarray], pocket_center: Optional[np.ndarray]
) -> bool:
    if com is None or pocket_center is None:
        return True  # cannot evaluate; keep
    return float(np.linalg.norm(com - pocket_center)) <= PAPER.pocket_centre_cutoff_A


# --------------------------------------------------------------------------- #
# Step S6 -- agglomerative cap at K_max via Tanimoto                          #
# --------------------------------------------------------------------------- #
def cap_group_by_tanimoto(
    mols: list[Chem.Mol], k_max: int = PAPER.max_group_size
) -> list[list[int]]:
    """Return a list of clusters (indices into `mols`), each of size <= k_max.

    Mirrors the paper's "Training Consideration": when a group has more than
    K_max ligands, run agglomerative clustering on Tanimoto distances and cap
    each cluster at K_max. We use Ward linkage on (1 - Tanimoto), then keep
    walking up the cut-off until no cluster exceeds K_max.
    """
    n = len(mols)
    if n <= k_max:
        return [list(range(n))]
    # Pairwise Tanimoto distance.    
    fps = [_morgan_fp(m) for m in mols]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if fps[i] is None or fps[j] is None:
                sim = 0.0
            else:
                sim = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

            dist[i, j] = dist[j, i] = 1.0 - sim 
    
    Z = linkage(squareform(dist, checks=False), method="ward")
    # Increase number of clusters until each is <= k_max.
    k = max(2, n // k_max)
    while True:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = pd.Series(labels).value_counts()
        if sizes.max() <= k_max:
            break
        k += 1
        if k > n:
            break
    clusters = [np.where(labels == c)[0].tolist() for c in sorted(set(labels))]
    # Final hard cap (in case k > n bailout): if any cluster still > k_max,
    # truncate to k_max nearest centroids (most central ligands).
    capped = []
    for c in clusters:
        if len(c) <= k_max:
            capped.append(c)
            continue
        sub = dist[np.ix_(c, c)]
        centrality = sub.sum(axis=1)
        keep = np.argsort(centrality)[:k_max]
        capped.append([c[i] for i in keep])
    return capped


# --------------------------------------------------------------------------- #
# Intra-group Tanimoto (Fig. 6b in the paper; required for Experiment 5)      #
# --------------------------------------------------------------------------- #
def intra_group_tanimoto(mols: list[Chem.Mol]) -> float:
    if len(mols) < 2:
        return float("nan")

    fps = [_morgan_fp(m) for m in mols]
    sims = []

    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if fps[i] is None or fps[j] is None:
                continue
            sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

    if not sims:
        return float("nan")

    return float(np.mean(sims))

# --------------------------------------------------------------------------- #
# Build the inter-ligand edge set (paper Section 3.3, E1 + E2)                #
# --------------------------------------------------------------------------- #
@dataclass
class InterLigandEdges:
    """Edges between atoms of two different ligands in the same group.

    Each edge is a tuple (atom_i_in_ligand_a, atom_j_in_ligand_b). MCS edges
    are the chemical-correspondence pairs; neighbour edges are added for
    unmatched atoms by 4 A expansion (paper E2)."""
    ligand_a: str
    ligand_b: str
    mcs_edges: list[tuple[int, int]]
    neighbour_edges: list[tuple[int, int]]

    @property
    def n_mcs(self) -> int:
        return len(self.mcs_edges)

    @property
    def n_neighbour(self) -> int:
        return len(self.neighbour_edges)


def build_inter_ligand_edges(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    coords_a: Optional[np.ndarray],
    coords_b: Optional[np.ndarray],
    name_a: str,
    name_b: str,
    radius_A: float = PAPER.inter_ligand_radius_A,
) -> InterLigandEdges:
    """Implements paper Section 3.3 'Build Group Ligands Graph':

      1. MCS gives a one-to-one mapping for matched atoms.
      2. For atoms NOT covered by the MCS in ligand a, connect them to
         neighbours within 4 A on the ligand-b side -- via the MCS partners of
         a's heavy neighbours. Symmetric handling on the b side.
    """
    matches = mcs_atom_mapping(mol_a, mol_b)
    mcs_a = {ia for ia, _ in matches}
    mcs_b = {ib for _, ib in matches}
    map_a2b = dict(matches)
    map_b2a = {ib: ia for ia, ib in matches}

    neighbour_edges: list[tuple[int, int]] = []

    # E2 left-to-right: each unmatched atom in ligand a -> neighbour atoms
    # within 4 A of its mapped partner on ligand b. We approach this via
    # ligand-a topology: for each unmatched i in a, look at its bonded
    # neighbours j (in a) that ARE in the MCS; their partner map_a2b[j] in b
    # gives an anchor, then pick 4 A neighbours of that anchor on b's
    # coordinates that are NOT already MCS-matched.
    if coords_a is not None and coords_b is not None and matches:
        for atom in mol_a.GetAtoms():
            i = atom.GetIdx()
            if i in mcs_a:
                continue
            anchored_b: set[int] = set()
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if j in map_a2b:
                    anchored_b.add(map_a2b[j])
            for b_anchor in anchored_b:
                for k in neighbours_within_radius(coords_b, b_anchor, radius_A):
                    if k in mcs_b:
                        continue
                    neighbour_edges.append((i, k))

        # E2 right-to-left, symmetric.
        for atom in mol_b.GetAtoms():
            i = atom.GetIdx()
            if i in mcs_b:
                continue
            anchored_a: set[int] = set()
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if j in map_b2a:
                    anchored_a.add(map_b2a[j])
            for a_anchor in anchored_a:
                for k in neighbours_within_radius(coords_a, a_anchor, radius_A):
                    if k in mcs_a:
                        continue
                    # Store as (a_idx, b_idx) for consistent orientation.
                    neighbour_edges.append((k, i))

    # Deduplicate (an edge can be generated by both passes).
    neighbour_edges = sorted(set(neighbour_edges))
    return InterLigandEdges(
        ligand_a=name_a,
        ligand_b=name_b,
        mcs_edges=matches,
        neighbour_edges=neighbour_edges,
    )


# --------------------------------------------------------------------------- #
# Top-level pipeline                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class GroupRecord:
    group_id: str
    uniprot_id: str
    protein_name: str
    complex_ids: list[str]
    ligand_ids: list[str]
    ligand_smiles: list[str]
    intra_group_sim: float
    edges: list[InterLigandEdges]
    dropped: list[tuple[str, str]]  # (complex_id, reason)

    def to_manifest_rows(self) -> list[dict]:
        rows = []
        for lig in self.ligand_ids:
            rows.append({
                "group_id": self.group_id,
                "ligand_id": lig,
                "uniprot_id": self.uniprot_id,
                "protein_name": self.protein_name,
                "intra_group_sim": self.intra_group_sim,
                "group_size": len(self.ligand_ids),
            })
        return rows

    def to_edges_json(self) -> str:
        return json.dumps([
            {
                "a": e.ligand_a, "b": e.ligand_b,
                "mcs_edges": e.mcs_edges,
                "neighbour_edges": e.neighbour_edges,
            } for e in self.edges
        ])


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame missing columns: {missing}")
    if df["complex_id"].duplicated().any():
        dups = df["complex_id"][df["complex_id"].duplicated()].tolist()
        raise ValueError(f"duplicate complex_id values: {dups}")


def prepare_groups(
    df: pd.DataFrame,
    k_max: int = PAPER.max_group_size,
    mcs_timeout_s: int = 5,
    verbose: bool = False,
) -> list[GroupRecord]:
    """End-to-end ligand-side preparation:

       S1 grouping -> S5/S6 filtering -> S6 K_max cap -> E1/E2 edge building.

    Returns a list of GroupRecord, one per final group. Singletons are kept
    (the paper falls back to single-ligand docking when no co-binders exist)
    but carry an intra_group_sim of NaN and no inter-ligand edges.
    """
    _validate_input(df)
    groups: list[GroupRecord] = []
    # S1 -- group by UniProt ID + protein_name. (Steps S2/S3 -- amino-acid
    # clustering and Kabsch alignment -- are protein-side and are assumed
    # already applied to the structural coordinates of `df` upstream.)
    grouper = df.groupby(["uniprot_id", "protein_name"], sort=False)
    for (upid, pname), g in grouper:
        dropped: list[tuple[str, str]] = []
        # Parse molecules once.
        rows = g.to_dict("records")
        mols: list[Optional[Chem.Mol]] = [
            _safe_mol(r["ligand_smiles"]) for r in rows
        ]
        
        # S5 + S6 row-level filtering.
        keep_idx: list[int] = []
        for i, r in enumerate(rows):
            cid = r["complex_id"]
            if mols[i] is None:
                dropped.append((cid, "smiles_parse_failed"))
                continue
            d_min = min_protein_ligand_distance(
                r["ligand_coords"], r["protein_coords"]
            )
            if not passes_minpl_filter(d_min):
                dropped.append((cid, f"minpl_out_of_range:{d_min:.2f}"))
                continue
            com = ligand_com(r["ligand_coords"])
            if not passes_pocket_cutoff(com, r["pocket_center"]):
                dropped.append((cid, "ligand_com_>8A_from_pocket"))
                continue
            keep_idx.append(i)

        if not keep_idx:
            if verbose:
                print(f"[skip] {upid}/{pname}: all complexes filtered out")
            continue

        kept_rows = [rows[i] for i in keep_idx]
        kept_mols = [mols[i] for i in keep_idx]

        # S6 -- cap at K_max via Tanimoto clustering. May yield multiple final
        # groups from one (uniprot, name) pre-group; index them as suffixes.
        clusters = cap_group_by_tanimoto(kept_mols, k_max=k_max)

        for ci, cl in enumerate(clusters):
            cl_rows = [kept_rows[j] for j in cl]
            cl_mols = [kept_mols[j] for j in cl]
            group_id = (
                f"{upid}__{pname}__c{ci}".replace(" ", "_")
                if len(clusters) > 1
                else f"{upid}__{pname}".replace(" ", "_")
            )
            lig_ids = [r["complex_id"] for r in cl_rows]
            smis = [r["ligand_smiles"] for r in cl_rows]
            sim = intra_group_tanimoto(cl_mols)

            # E1 + E2: pairwise inter-ligand edge sets.
            edges: list[InterLigandEdges] = []
            if len(cl_mols) >= 2:
                for i in range(len(cl_mols)):
                    for j in range(i + 1, len(cl_mols)):
                        if cl_rows[i]["ligand_coords"] is not None:
                            if cl_mols[i].GetNumAtoms() != len(cl_rows[i]["ligand_coords"]):
                                dropped.append((
                                    cl_rows[i]["complex_id"],
                                    f"mol_coord_size_mismatch:"
                                    f"{cl_mols[i].GetNumAtoms()}!="
                                    f"{len(cl_rows[i]['ligand_coords'])}"
                                ))
                                continue

                        if cl_rows[j]["ligand_coords"] is not None:
                            if cl_mols[j].GetNumAtoms() != len(cl_rows[j]["ligand_coords"]):
                                dropped.append((
                                    cl_rows[j]["complex_id"],
                                    f"mol_coord_size_mismatch:"
                                    f"{cl_mols[j].GetNumAtoms()}!="
                                    f"{len(cl_rows[j]['ligand_coords'])}"
                                ))
                                continue

                        edges.append(build_inter_ligand_edges(
                            mol_a=cl_mols[i],
                            mol_b=cl_mols[j],
                            coords_a=cl_rows[i]["ligand_coords"],
                            coords_b=cl_rows[j]["ligand_coords"],
                            name_a=lig_ids[i],
                            name_b=lig_ids[j],
                        ))

            groups.append(GroupRecord(
                group_id=group_id,
                uniprot_id=upid,
                protein_name=pname,
                complex_ids=[r["complex_id"] for r in cl_rows],
                ligand_ids=lig_ids,
                ligand_smiles=smis,
                intra_group_sim=sim,
                edges=edges,
                dropped=dropped if ci == 0 else [],  # attribute dropped only once
            ))
            if verbose:
                print(
                    f"[group] {group_id}: |L|={len(lig_ids)} sim={sim:.3f} "
                    f"edges={[(e.n_mcs, e.n_neighbour) for e in edges]}"
                )
    return groups


# --------------------------------------------------------------------------- #
# Emit the two output artefacts                                               #
# --------------------------------------------------------------------------- #
def write_outputs(
    groups: list[GroupRecord],
    manifest_path: Path,
    records_path: Path,
    seeds: int = 5,
    regimes: tuple[str, ...] = ("mcs", "gin", "oracle"),
) -> None:
    """Write the group manifest (one row per (group, ligand) plus serialised
    edge JSON for the MCS regime) and the long-format `records_template.csv`
    consumed by experiment5_stratified.py.

    The records template has `rmsd` left as NaN -- it is the downstream
    docking pipeline's job to overwrite those cells with the actual measured
    RMSD values per (group, ligand, regime, seed). All other columns
    (group_id, ligand_id, regime, seed, intra_group_sim) are populated, so
    the file conforms to experiment5_stratified.REQUIRED_COLUMNS verbatim.
    """
    manifest_rows: list[dict] = []
    for grp in groups:
        rows = grp.to_manifest_rows()
        edges_json = grp.to_edges_json()
        for r in rows:
            r["mcs_edges_json"] = edges_json
        manifest_rows.extend(rows)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    rec_rows: list[dict] = []
    for grp in groups:
        for lig in grp.ligand_ids:
            for regime in regimes:
                for s in range(seeds):
                    rec_rows.append({
                        "group_id": grp.group_id,
                        "ligand_id": lig,
                        "regime": regime,
                        "seed": s,
                        "rmsd": float("nan"),
                        "intra_group_sim": grp.intra_group_sim,
                    })
    pd.DataFrame(rec_rows).to_csv(records_path, index=False)


# --------------------------------------------------------------------------- #
# Synthetic mini-dataset for VM self-validation                               #
# --------------------------------------------------------------------------- #
def _synth_input(rng_seed: int = 0) -> pd.DataFrame:
    """Five small artificial complexes across two pockets, with coordinates,
    so the whole pipeline (S5, S6, E1, E2) actually executes -- not just a
    smoke test of the schema."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    # Pocket A: three kinase-like inhibitors (high Tanimoto, similar size).
    smis_a = [
        "Cc1ccc(Nc2ncnc3[nH]ccc23)cc1",   # 7-aza-indol-anilino
        "Cc1ccc(Nc2ncnc3sccc23)cc1",      # thieno analogue
        "Cc1ccc(Nc2ncnc3occc23)cc1",      # furo analogue
    ]
    # Pocket B: two structurally diverse ligands (lower Tanimoto).
    smis_b = [
        "O=C(Nc1ccc(Cl)cc1)c1cccs1",
        "CC(=O)NC1CCN(Cc2ccccc2)CC1",
    ]
    pocket_a_centre = np.array([0.0, 0.0, 0.0])
    pocket_b_centre = np.array([30.0, 0.0, 0.0])
    # Build the protein clouds as a SHELL around the pocket (radius ~7 A) so
    # that minpl is naturally in the paper's [0.4, 3.0] A window rather than
    # the cloud overlapping the ligand atoms (which would trip S5).
    def _shell_cloud(centre, n=60, r_inner=6.5, r_outer=9.0, rng_=rng):
        dirs = rng_.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = rng_.uniform(r_inner, r_outer, size=n)
        return centre + dirs * radii[:, None]

    prot_a = _shell_cloud(pocket_a_centre)
    prot_b = _shell_cloud(pocket_b_centre)

    def _make_coords(smi: str, centre: np.ndarray, rng_) -> np.ndarray:
        n = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
        # Compact ligand cloud at the pocket centre (radius < 3 A).
        return centre + rng_.normal(scale=1.0, size=(n, 3))

    def _anchor_contact(prot_cloud: np.ndarray, ligand_coords: np.ndarray,
                        contact_A: float = 2.0) -> np.ndarray:
        """Append one protein atom that sits exactly `contact_A` from ligand
        atom 0, so the synthetic pocket guarantees a paper-valid minpl in
        [0.4, 3.0] regardless of the random clouds. Mirrors a real
        contact residue."""
        anchor = ligand_coords[0] + np.array([contact_A, 0.0, 0.0])
        return np.vstack([prot_cloud, anchor[None, :]])

    cid = 0
    for smi in smis_a:
        lc = _make_coords(smi, pocket_a_centre, rng)
        rows.append({
            "complex_id": f"c{cid:03d}",
            "uniprot_id": "P00000",
            "protein_name": "kinase_test",
            "ligand_smiles": smi,
            "pocket_center": pocket_a_centre,
            "ligand_coords": lc,
            "protein_coords": _anchor_contact(prot_a, lc),
        })
        cid += 1
    for smi in smis_b:
        lc = _make_coords(smi, pocket_b_centre, rng)
        rows.append({
            "complex_id": f"c{cid:03d}",
            "uniprot_id": "P11111",
            "protein_name": "other_test",
            "ligand_smiles": smi,
            "pocket_center": pocket_b_centre,
            "ligand_coords": lc,
            "protein_coords": _anchor_contact(prot_b, lc),
        })
        cid += 1
    # One deliberately-dropped clash case: a protein atom is placed 0.1 A from
    # a ligand atom -- below the paper's 0.4 A clash floor.
    bad_lig = _make_coords("CCO", pocket_a_centre, rng)
    bad_prot = np.vstack([prot_a, bad_lig[0:1] + np.array([0.1, 0.0, 0.0])])
    rows.append({
        "complex_id": f"c{cid:03d}",
        "uniprot_id": "P00000",
        "protein_name": "kinase_test",
        "ligand_smiles": "CCO",
        "pocket_center": pocket_a_centre,
        "ligand_coords": bad_lig,
        "protein_coords": bad_prot,
    })
    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _selftest() -> int:
    """End-to-end self-check on the synthetic dataset:
       * groups assembled
       * S5 clash case dropped
       * intra_group_sim populated for multi-ligand groups
       * MCS edges non-empty for similar ligands
       * output records template conforms to Experiment 5's column contract
    """
    df = _synth_input()
    groups = prepare_groups(df, verbose=True)
    if not groups:
        print("FAIL: no groups produced"); return 1

    # Find the kinase group; it should contain 3 ligands and have all rejected
    # the clash case via S5.
    kinase = [g for g in groups if g.uniprot_id == "P00000"]
    if not kinase:
        print("FAIL: kinase group missing"); return 1
    kg = kinase[0]
    if len(kg.ligand_ids) != 3:
        print(f"FAIL: expected 3 kinase ligands, got {len(kg.ligand_ids)}"); return 1
    if not any(reason.startswith("minpl") for _, reason in kg.dropped):
        print(f"FAIL: clash case not dropped via S5; dropped={kg.dropped}"); return 1
    if not (kg.intra_group_sim > 0.0):
        print(f"FAIL: intra_group_sim not computed: {kg.intra_group_sim}"); return 1
    if not any(e.n_mcs > 0 for e in kg.edges):
        print("FAIL: no MCS edges across similar ligands"); return 1

    # Other-pocket group with two dissimilar ligands -- MCS should still find
    # SOME atoms (a carbonyl or aromatic ring), but the count is low.
    other = [g for g in groups if g.uniprot_id == "P11111"][0]
    if len(other.edges) != 1:
        print(f"FAIL: expected 1 inter-ligand edge set, got {len(other.edges)}"); return 1

    # Schema check vs Experiment 5's contract.
    out_dir = Path("./test/")
    manifest = out_dir / "_selftest_manifest.csv"
    records = out_dir / "_selftest_records.csv"
    write_outputs(groups, manifest, records, seeds=3)
    rec = pd.read_csv(records)
    required = ["group_id", "ligand_id", "regime", "seed", "rmsd", "intra_group_sim"]
    missing = [c for c in required if c not in rec.columns]
    if missing:
        print(f"FAIL: records template missing columns: {missing}"); return 1
    if set(rec["regime"].unique()) != {"mcs", "gin", "oracle"}:
        print(f"FAIL: regimes wrong: {rec.regime.unique()}"); return 1
    if not rec["rmsd"].isna().all():
        print("FAIL: rmsd should be NaN until docking pipeline fills it"); return 1
    # Per-(ligand,regime,seed) cardinality
    expected = sum(len(g.ligand_ids) for g in groups) * 3 * 3
    if len(rec) != expected:
        print(f"FAIL: expected {expected} record rows, got {len(rec)}"); return 1

    # Round-trip compatibility with experiment5_stratified.py
    import experiment5_stratified as E
    missing = [c for c in E.REQUIRED_COLUMNS if c not in rec.columns]
    if missing:
        print(f"FAIL: not compatible with Experiment 5 contract: {missing}"); return 1

    print("\nALL DATA-PREP SELFTESTS PASSED")
    print(f"  groups produced       : {len(groups)}")
    print(f"  total ligands kept    : {sum(len(g.ligand_ids) for g in groups)}")
    print(f"  total dropped         : {sum(len(g.dropped) for g in groups)}")
    print(f"  manifest -> {manifest}")
    print(f"  records  -> {records}")
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        help="Pickle (.pkl) of the input frame; must contain columns "
        f"{REQUIRED_INPUT_COLUMNS}. Coordinate cells must be numpy arrays. "
        "If omitted, runs the synthetic self-test.",
    )
    ap.add_argument("--manifest-out", default="group_manifest.csv")
    ap.add_argument("--records-out", default="records_template.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=PAPER.max_group_size)
    ap.add_argument("--mcs-timeout", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest or args.input is None:
        return _selftest()

    df = pd.read_pickle(args.input)
    groups = prepare_groups(
        df, k_max=args.k_max,
        mcs_timeout_s=args.mcs_timeout, verbose=args.verbose,
    )
    write_outputs(
        groups,
        manifest_path=Path(args.manifest_out),
        records_path=Path(args.records_out),
        seeds=args.seeds,
    )
    print(f"Wrote {args.manifest_out} and {args.records_out} "
          f"({len(groups)} groups, "
          f"{sum(len(g.ligand_ids) for g in groups)} ligands kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
