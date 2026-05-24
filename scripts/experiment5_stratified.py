"""
Experiment 5 -- Stratified decisive test for the Option-C claim:

    "A WL/GIN learned correspondence defines better inter-ligand edges than MCS,
     and the benefit is concentrated where MCS is weak (low intra-group ligand
     similarity / MCS non-uniqueness)."

This harness implements the *analysis layer* of Experiment 5 as specified in the
experimental design:

  - Stratify GroupBind test groups by intra-group Tanimoto similarity.
  - Compare two correspondence regimes (MCS vs GIN) on docking RMSD per stratum.
  - Include the oracle (structural ground-truth correspondence) ceiling.
  - Test the *pre-registered* interaction prediction:
        improvement(GIN - MCS) is significantly larger in the LOW-similarity
        stratum than in the HIGH-similarity stratum, with NO degradation in
        the HIGH-similarity stratum.
  - Parameter-matched bookkeeping is enforced as a required input field
    (the harness refuses to render a verdict if it is not asserted).

What this file is NOT: it does not train GroupBind or DiffDock. Producing the
per-(group, ligand, seed) RMSD values is the expensive step and is the user's
docking pipeline. This harness consumes those RMSD records (real or, for a
dry run, synthetic) and produces the statistically correct verdict.

The synthetic generator at the bottom encodes the pre-registered hypothesis as
the *ground truth* of one scenario and the *null* (uniform / capacity-only
improvement) as another, so the statistical machinery can be validated to
accept the former and reject the latter before being run on real data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# RDKit only used by the (optional) similarity-stratification helper, so that a
# user supplying precomputed similarities does not need it.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    _HAVE_RDKIT = False


# --------------------------------------------------------------------------- #
# 1. Data contract                                                            #
# --------------------------------------------------------------------------- #
# The harness consumes a long-format DataFrame with exactly these columns. Each
# row is ONE docking outcome for ONE query ligand under ONE correspondence
# regime, at ONE random seed, evaluated with the paper's perfect-selection
# protocol (best of 40 sampled poses) so the confidence model is factored out.
#
#   group_id        : str  -- PDBBind group (ligands sharing one pocket)
#   ligand_id       : str  -- the query ligand within that group
#   regime          : str  -- one of {"mcs", "gin", "oracle"}
#   seed            : int  -- training/sampling seed
#   rmsd            : float -- permutation-corrected heavy-atom ligand RMSD (A)
#   intra_group_sim : float -- mean pairwise Tanimoto in the ligand's group
#                              (constant per group; carried per row for convenience)
#
# Plus a small sidecar dict asserting the parameter-matched control, without
# which the harness will refuse a verdict.

REQUIRED_COLUMNS = [
    "group_id",
    "ligand_id",
    "regime",
    "seed",
    "rmsd",
    "intra_group_sim",
]
VALID_REGIMES = {"mcs", "gin", "oracle"}


@dataclass
class ParamMatchAssertion:
    """Bookkeeping that MUST be supplied. The downstream comparison is
    uninterpretable if the GIN variant simply has more parameters than the MCS
    variant, so the harness treats an unmet assertion as a hard stop."""
    mcs_param_count: int
    gin_param_count: int
    tolerance_frac: float = 0.02  # GroupBind vs DiffDock were matched to ~ this

    @property
    def matched(self) -> bool:
        if self.mcs_param_count <= 0:
            return False
        rel = abs(self.gin_param_count - self.mcs_param_count) / self.mcs_param_count
        return rel <= self.tolerance_frac

    def explain(self) -> str:
        rel = abs(self.gin_param_count - self.mcs_param_count) / max(
            self.mcs_param_count, 1
        )
        return (
            f"MCS params={self.mcs_param_count:,} | GIN params={self.gin_param_count:,} | "
            f"relative diff={rel:.4f} | tolerance={self.tolerance_frac:.4f} | "
            f"matched={'YES' if self.matched else 'NO'}"
        )


# --------------------------------------------------------------------------- #
# 2. Stratification                                                           #
# --------------------------------------------------------------------------- #
def tanimoto_intra_group(smiles_by_group: dict[str, list[str]]) -> dict[str, float]:
    """Mean pairwise Morgan-fingerprint Tanimoto within each group.

    Provided so a user can derive `intra_group_sim` the same way the GroupBind
    paper characterises groups (Fig. 6b). Groups with a single ligand get NaN
    (they cannot exercise the group mechanism and are excluded from Exp 5).
    """
    if not _HAVE_RDKIT:
        raise RuntimeError("RDKit not available; supply intra_group_sim directly.")
    out: dict[str, float] = {}
    for gid, smis in smiles_by_group.items():
        mols = [Chem.MolFromSmiles(s) for s in smis]
        mols = [m for m in mols if m is not None]
        if len(mols) < 2:
            out[gid] = float("nan")
            continue
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
        sims = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        out[gid] = float(np.mean(sims)) if sims else float("nan")
    return out


def assign_strata(
    df: pd.DataFrame,
    method: str = "median",
    low_q: float = 0.5,
    high_q: float = 0.5,
) -> pd.DataFrame:
    """Add a `stratum` column in {"low", "high", "mid"}.

    `method="median"`  -> binary split at the median similarity (default; the
                           cleanest pre-registered design: LOW vs HIGH only).
    `method="tertile"` -> low_q / high_q quantile cuts, middle dropped, so the
                           contrast is sharpened (LOW = bottom tertile, HIGH =
                           top tertile). Use only if pre-registered.
    Stratum is assigned at the GROUP level (similarity is a group property) to
    avoid leaking the same group across strata.
    """
    g = df.drop_duplicates("group_id")[["group_id", "intra_group_sim"]].copy()
    g = g.dropna(subset=["intra_group_sim"])
    if method == "median":
        thr = g["intra_group_sim"].median()
        g["stratum"] = np.where(g["intra_group_sim"] <= thr, "low", "high")
    elif method == "tertile":
        lo = g["intra_group_sim"].quantile(low_q)
        hi = g["intra_group_sim"].quantile(high_q)
        g["stratum"] = np.select(
            [g["intra_group_sim"] <= lo, g["intra_group_sim"] >= hi],
            ["low", "high"],
            default="mid",
        )
    else:
        raise ValueError(f"unknown stratification method: {method}")
    return df.merge(g[["group_id", "stratum"]], on="group_id", how="inner")


# --------------------------------------------------------------------------- #
# 3. Core statistics                                                          #
# --------------------------------------------------------------------------- #
def _aggregate_per_ligand(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds -> one robust RMSD per (group, ligand, regime).

    Median over seeds is used (RMSD is heavy-tailed; DiffDock-family failures
    produce >10 A outliers). Keeps stratum + similarity.
    """
    agg = (
        df.groupby(["group_id", "ligand_id", "regime", "stratum"], as_index=False)
        .agg(rmsd=("rmsd", "median"), intra_group_sim=("intra_group_sim", "first"))
    )
    return agg


def _paired_frame(agg: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Pivot so each (group, ligand) row carries regime `a` and `b` RMSD,
    enabling a PAIRED test (same query ligand, same group, only the
    correspondence construction differs)."""
    piv = agg.pivot_table(
        index=["group_id", "ligand_id", "stratum"],
        columns="regime",
        values="rmsd",
    ).reset_index()
    need = {a, b}
    if not need.issubset(piv.columns):
        missing = need - set(piv.columns)
        raise ValueError(f"missing regimes for pairing: {missing}")
    piv = piv.dropna(subset=[a, b])
    return piv


def _success_rate(x: np.ndarray, thresh: float = 2.0) -> float:
    return float(np.mean(x < thresh)) if len(x) else float("nan")


def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.array([stat_fn(values[i]) for i in idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


@dataclass
class StratumResult:
    stratum: str
    n_pairs: int
    median_rmsd_mcs: float
    median_rmsd_gin: float
    median_rmsd_oracle: Optional[float]
    success_mcs: float
    success_gin: float
    success_oracle: Optional[float]
    delta_median: float            # median(MCS_rmsd - GIN_rmsd); >0 favours GIN
    delta_ci: tuple[float, float]
    wilcoxon_stat: float
    wilcoxon_p: float
    cliffs_delta: float            # paired effect size, GIN better => positive
    oracle_headroom: Optional[float]  # median(MCS - oracle); MCS distance to ceiling


def _cliffs_delta_paired(diff: np.ndarray) -> float:
    """Effect size on the paired differences d = MCS - GIN.
    Reported as P(d>0) - P(d<0); positive => GIN tends to beat MCS."""
    if len(diff) == 0:
        return float("nan")
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    return float((pos - neg) / len(diff))


def analyse_stratum(piv: pd.DataFrame, stratum: str,
                     rng: np.random.Generator) -> StratumResult:
    sub = piv[piv["stratum"] == stratum]
    mcs = sub["mcs"].to_numpy(dtype=float)
    gin = sub["gin"].to_numpy(dtype=float)
    has_oracle = "oracle" in sub.columns and sub["oracle"].notna().any()
    ora = sub["oracle"].to_numpy(dtype=float) if has_oracle else None

    diff = mcs - gin  # >0 means GIN produced the lower (better) RMSD
    if len(diff) >= 1 and np.any(diff != 0):
        try:
            w_stat, w_p = stats.wilcoxon(
                mcs, gin, zero_method="wilcox", alternative="two-sided"
            )
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")

    delta_ci = _bootstrap_ci(diff, np.median, rng=rng)

    return StratumResult(
        stratum=stratum,
        n_pairs=int(len(sub)),
        median_rmsd_mcs=float(np.median(mcs)) if len(mcs) else float("nan"),
        median_rmsd_gin=float(np.median(gin)) if len(gin) else float("nan"),
        median_rmsd_oracle=float(np.median(ora)) if has_oracle else None,
        success_mcs=_success_rate(mcs),
        success_gin=_success_rate(gin),
        success_oracle=_success_rate(ora) if has_oracle else None,
        delta_median=float(np.median(diff)) if len(diff) else float("nan"),
        delta_ci=delta_ci,
        wilcoxon_stat=float(w_stat),
        wilcoxon_p=float(w_p),
        cliffs_delta=_cliffs_delta_paired(diff),
        oracle_headroom=(float(np.median(mcs - ora)) if has_oracle else None),
    )


# --------------------------------------------------------------------------- #
# 4. The pre-registered interaction test (this is the decisive part)          #
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    param_matched: bool
    low: StratumResult
    high: StratumResult
    interaction_p: float          # is (GIN-MCS)_low > (GIN-MCS)_high ?
    interaction_effect: float     # difference of paired deltas, low minus high
    high_not_degraded: bool       # non-inferiority of GIN in HIGH stratum
    oracle_headroom_exists: Optional[bool]
    supported: bool
    reasoning: list[str] = field(default_factory=list)


def interaction_test(
    piv: pd.DataFrame,
    rng: np.random.Generator,
    non_inferiority_margin: float = 0.25,  # A; GIN may not be worse than this in HIGH
    alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Test the pre-registered prediction:

        delta_low  := median(MCS - GIN | low  stratum)
        delta_high := median(MCS - GIN | high stratum)
        H1 : delta_low > delta_high          (benefit concentrated where MCS weak)

    Implemented as a between-stratum comparison of the paired per-ligand
    differences with a Mann-Whitney U (one-sided), plus a non-inferiority check
    that GIN does not *hurt* in the HIGH stratum (lower CI of delta_high must be
    above -margin).
    """
    d_low = (
        piv.loc[piv["stratum"] == "low", "mcs"].to_numpy(float)
        - piv.loc[piv["stratum"] == "low", "gin"].to_numpy(float)
    )
    d_high = (
        piv.loc[piv["stratum"] == "high", "mcs"].to_numpy(float)
        - piv.loc[piv["stratum"] == "high", "gin"].to_numpy(float)
    )
    if len(d_low) == 0 or len(d_high) == 0:
        return float("nan"), float("nan"), False

    # One-sided: are LOW-stratum improvements stochastically larger than HIGH?
    try:
        _, p_inter = stats.mannwhitneyu(d_low, d_high, alternative="greater")
    except ValueError:
        p_inter = float("nan")
    effect = float(np.median(d_low) - np.median(d_high))

    # Non-inferiority of GIN in HIGH stratum: lower bootstrap CI of delta_high
    lo_high, _ = _bootstrap_ci(d_high, np.median, rng=rng, alpha=alpha)
    high_ok = lo_high > -abs(non_inferiority_margin)

    return p_inter, effect, high_ok


def render_verdict(
    df: pd.DataFrame,
    pmatch: ParamMatchAssertion,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Verdict:
    # Surface a clear, actionable error if the user passed the records
    # *template* (rmsd all-NaN). Without this guard, the subsequent pivot
    # drops every row and `_paired_frame` raises "missing regimes for
    # pairing" -- a misleading error that suggests the schema is wrong.
    if df["rmsd"].isna().all():
        raise ValueError(
            "Every 'rmsd' value is NaN -- this looks like the records "
            "template from data_prep_mcs.py, not a results file. The "
            "rmsd column must be populated by your docking pipeline "
            "(one RMSD per [group_id, ligand_id, regime, seed]) before "
            "Experiment 5 can compute a verdict."
        )
    n_total = len(df)
    n_with_rmsd = int(df["rmsd"].notna().sum())
    if n_with_rmsd < n_total:
        # Partial fill is allowed but flag it: it can silently bias the
        # stratified comparison if NaNs are not missing-at-random.
        print(
            f"[warn] {n_total - n_with_rmsd}/{n_total} rows have NaN rmsd "
            "and will be dropped by the pairing step.",
            file=sys.stderr,
        )

    agg = _aggregate_per_ligand(df)
    # _paired_frame pivots ALL regimes present, so an `oracle` column is
    # already carried here when oracle records exist (no separate merge,
    # which would collide into oracle_x/oracle_y and silently drop it).
    try:
        piv = _paired_frame(agg, "mcs", "gin")
    except ValueError as e:
        # Re-raise with a more helpful diagnosis if the underlying issue is
        # that *some* but not all RMSDs are missing for a regime.
        present_regimes = sorted(set(df.loc[df["rmsd"].notna(), "regime"]))
        raise ValueError(
            f"{e}. Regimes with at least one populated rmsd: "
            f"{present_regimes}. Both 'mcs' and 'gin' must have rmsd "
            "values from the docking pipeline."
        ) from e

    low = analyse_stratum(piv, "low", rng)
    high = analyse_stratum(piv, "high", rng)
    p_inter, eff, high_ok = interaction_test(piv, rng, alpha=alpha)

    reasoning: list[str] = []

    # Gate 0 -- parameter match is a HARD prerequisite.
    if not pmatch.matched:
        reasoning.append(
            "HARD STOP: parameter-matched control not satisfied "
            f"({pmatch.explain()}). Any RMSD difference is confounded with "
            "model capacity; verdict withheld."
        )
        return Verdict(
            param_matched=False, low=low, high=high,
            interaction_p=p_inter, interaction_effect=eff,
            high_not_degraded=high_ok, oracle_headroom_exists=None,
            supported=False, reasoning=reasoning,
        )

    # Gate 1 -- oracle headroom. If MCS is already ~ at the oracle ceiling,
    # the hypothesis has no room to be true regardless of GIN.
    headroom_exists: Optional[bool] = None
    if low.oracle_headroom is not None:
        headroom_exists = (low.oracle_headroom > 0.10) or (
            high.oracle_headroom is not None and high.oracle_headroom > 0.10
        )
        if not headroom_exists:
            reasoning.append(
                "Oracle ceiling reached: median(MCS - oracle) <= 0.10 A in both "
                "strata. MCS correspondence is already near-optimal; no headroom "
                "for a learned correspondence to exploit."
            )
    else:
        reasoning.append(
            "NOTE: oracle regime absent. Run the oracle-correspondence ceiling "
            "first; without it a null result cannot distinguish 'GIN no better' "
            "from 'no headroom for anyone'."
        )

    # Gate 2 -- the pre-registered interaction.
    inter_ok = (not math.isnan(p_inter)) and (p_inter < alpha) and (eff > 0)
    if inter_ok:
        reasoning.append(
            f"Interaction supported: improvement (MCS-GIN) is significantly "
            f"larger in LOW than HIGH stratum (one-sided MWU p={p_inter:.4g}, "
            f"median-of-deltas gap={eff:+.3f} A)."
        )
    else:
        reasoning.append(
            f"Interaction NOT supported: p={p_inter:.4g}, effect={eff:+.3f} A. "
            "If GIN improves RMSD uniformly across strata, the gain is most "
            "consistent with added capacity / regularisation, not the "
            "correspondence mechanism -- the Option-C claim is unsupported."
        )

    # Gate 3 -- GIN must not degrade the HIGH (similar-ligand) stratum.
    if high_ok:
        reasoning.append(
            "Non-inferiority in HIGH stratum holds: GIN does not significantly "
            "harm easy (high-similarity) groups."
        )
    else:
        reasoning.append(
            "Non-inferiority FAILS in HIGH stratum: GIN degrades easy groups; "
            "even a positive interaction does not justify replacing MCS."
        )

    supported = bool(
        inter_ok
        and high_ok
        and (headroom_exists is True if headroom_exists is not None else True)
    )
    if supported:
        reasoning.append(
            "VERDICT: Option-C claim SUPPORTED for this dataset -- learned "
            "correspondence beats MCS specifically where MCS is weak, with no "
            "harm where MCS is strong, under a parameter-matched comparison."
        )
    else:
        reasoning.append(
            "VERDICT: Option-C claim NOT SUPPORTED. This is a clean, publishable "
            "negative result: redefining inter-ligand edges via a learned "
            "encoder does not beat MCS for GroupBind docking on this split."
        )

    return Verdict(
        param_matched=True, low=low, high=high,
        interaction_p=p_inter, interaction_effect=eff,
        high_not_degraded=high_ok,
        oracle_headroom_exists=headroom_exists,
        supported=supported, reasoning=reasoning,
    )


# --------------------------------------------------------------------------- #
# 5. Reporting                                                                #
# --------------------------------------------------------------------------- #
def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def print_report(v: Verdict) -> None:
    line = "=" * 78
    print(line)
    print("EXPERIMENT 5 -- STRATIFIED GIN-vs-MCS CORRESPONDENCE TEST (GroupBind)")
    print(line)
    print(f"Parameter-matched control satisfied : {v.param_matched}")
    print()
    hdr = (
        f"{'stratum':>7} | {'n':>5} | {'med RMSD MCS':>12} | {'med RMSD GIN':>12} | "
        f"{'med RMSD ORA':>12} | {'%<2A MCS':>9} | {'%<2A GIN':>9} | "
        f"{'delta(med)':>11} | {'delta 95% CI':>18} | {'Wilcoxon p':>11} | "
        f"{'Cliff d':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in (v.low, v.high):
        ora = "n/a" if s.median_rmsd_oracle is None else f"{s.median_rmsd_oracle:12.3f}"
        sora = "n/a" if s.success_oracle is None else f"{s.success_oracle:9.3f}"
        print(
            f"{s.stratum:>7} | {s.n_pairs:5d} | {s.median_rmsd_mcs:12.3f} | "
            f"{s.median_rmsd_gin:12.3f} | {ora:>12} | {s.success_mcs:9.3f} | "
            f"{s.success_gin:9.3f} | {s.delta_median:+11.3f} | "
            f"{_fmt_ci(s.delta_ci):>18} | {s.wilcoxon_p:11.4g} | "
            f"{s.cliffs_delta:+8.3f}"
        )
        if s.oracle_headroom is not None:
            print(
                f"{'':>7} | oracle headroom (median MCS-oracle) = "
                f"{s.oracle_headroom:+.3f} A"
            )
    print()
    print(
        f"Pre-registered interaction (LOW improvement > HIGH improvement):\n"
        f"  one-sided MWU p = {v.interaction_p:.4g} | "
        f"median-of-deltas gap (low-high) = {v.interaction_effect:+.3f} A | "
        f"HIGH-stratum non-inferiority = {v.high_not_degraded}"
    )
    print()
    print("Reasoning:")
    for r in v.reasoning:
        print("  - " + r)
    print(line)
    print(f"SUPPORTED = {v.supported}")
    print(line)


# --------------------------------------------------------------------------- #
# 6. Synthetic generator -- validates the statistics on known ground truth    #
# --------------------------------------------------------------------------- #
def synthesize(
    scenario: str,
    n_groups: int = 120,
    ligands_per_group: int = 3,
    seeds: int = 5,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Generate per-(group, ligand, regime, seed) RMSD records.

    scenario = "hypothesis":
        encodes the pre-registered ground truth -- GIN helps in LOW-similarity
        groups (MCS weak) and is neutral in HIGH-similarity groups; oracle is
        the ceiling everywhere. The harness MUST return SUPPORTED.

    scenario = "null_uniform":
        GIN improves RMSD by the SAME amount in both strata (capacity-style
        gain, not mechanism). The harness MUST return NOT SUPPORTED via the
        interaction test even though raw RMSD improved.

    scenario = "null_noheadroom":
        MCS already at the oracle ceiling; GIN cannot help. MUST return
        NOT SUPPORTED via the oracle gate.
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for g in range(n_groups):
        sim = float(np.clip(rng.beta(2, 2), 0.02, 0.98))  # spread of similarities
        is_low = sim <= 0.5  # provisional; real split is by median later
        for L in range(ligands_per_group):
            # Base difficulty: harder (higher RMSD) for low-similarity groups,
            # mirroring the paper's finding that dissimilar groups are harder
            # to dock well but also where group info helps most.
            base = 2.0 + (1.6 if is_low else 0.4) + 0.4 * rng.standard_normal()
            base = max(base, 0.3)

            # Oracle: structural ground-truth correspondence -> the ceiling.
            oracle_gain = 0.9 if is_low else 0.35

            if scenario == "hypothesis":
                mcs_gain = 0.05 if is_low else 0.30      # MCS weak when dissimilar
                gin_gain = 0.70 if is_low else 0.33      # GIN recovers it in LOW
            elif scenario == "null_uniform":
                mcs_gain = 0.20
                gin_gain = 0.45                          # same lift both strata
            elif scenario == "null_noheadroom":
                mcs_gain = oracle_gain - 0.02            # MCS already at ceiling
                gin_gain = oracle_gain - 0.02
            else:
                raise ValueError(scenario)

            for s in range(seeds):
                noise = lambda: 0.35 * rng.standard_normal()
                rec_common = dict(
                    group_id=f"G{g:04d}",
                    ligand_id=f"G{g:04d}_L{L}",
                    seed=s,
                    intra_group_sim=sim,
                )
                rows.append({**rec_common, "regime": "mcs",
                             "rmsd": max(base - mcs_gain + noise(), 0.2)})
                rows.append({**rec_common, "regime": "gin",
                             "rmsd": max(base - gin_gain + noise(), 0.2)})
                rows.append({**rec_common, "regime": "oracle",
                             "rmsd": max(base - oracle_gain + noise(), 0.2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. CLI                                                                      #
# --------------------------------------------------------------------------- #
def load_records(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input missing required columns: {missing}")
    bad = set(df["regime"].unique()) - VALID_REGIMES
    if bad:
        raise ValueError(f"invalid regime values: {bad}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--records",
        help="CSV of per-(group,ligand,regime,seed) RMSD records "
        f"(columns: {REQUIRED_COLUMNS}). If omitted, runs the synthetic "
        "self-validation.",
    )
    ap.add_argument("--mcs-params", type=int, default=18_800_000)
    ap.add_argument("--gin-params", type=int, default=18_900_000)
    ap.add_argument("--param-tol", type=float, default=0.02)
    ap.add_argument(
        "--stratify", choices=["median", "tertile"], default="median"
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="Run all three synthetic scenarios and assert the verdicts.",
    )
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.rng)

    if args.selftest:
        ok = True
        expectations = {
            "hypothesis": True,
            "null_uniform": False,
            "null_noheadroom": False,
        }
        for scen, expect in expectations.items():
            df = synthesize(scen)
            df = assign_strata(df, method=args.stratify)
            pm = ParamMatchAssertion(args.mcs_params, args.gin_params,
                                     args.param_tol)
            v = render_verdict(df, pm, rng, alpha=args.alpha)
            print(f"\n##### SELFTEST scenario = {scen} "
                  f"(expect supported={expect}) #####")
            print_report(v)
            if v.supported != expect:
                ok = False
                print(f"!!! SELFTEST FAILED for {scen}: "
                      f"got {v.supported}, expected {expect}")
        print("\n" + ("ALL SELFTESTS PASSED" if ok else "SELFTESTS FAILED"))
        return 0 if ok else 1

    if args.records:
        df = load_records(args.records)
    else:
        print("No --records given: running synthetic 'hypothesis' scenario "
              "as a demonstration.\n")
        df = synthesize("hypothesis")

    df = assign_strata(df, method=args.stratify)
    pm = ParamMatchAssertion(args.mcs_params, args.gin_params, args.param_tol)
    v = render_verdict(df, pm, rng, alpha=args.alpha)
    print_report(v)
    print("\nMachine-readable verdict:")
    print(json.dumps(
        {
            "supported": v.supported,
            "param_matched": v.param_matched,
            "interaction_p": v.interaction_p,
            "interaction_effect_A": v.interaction_effect,
            "high_not_degraded": v.high_not_degraded,
            "oracle_headroom_exists": v.oracle_headroom_exists,
            "low_stratum": asdict(v.low),
            "high_stratum": asdict(v.high),
        },
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
