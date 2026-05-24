#!/usr/bin/env bash
set -euo pipefail

CSV="${1:-../pdbbind_final_groups.csv}"
OUT="${2:-results}"
N_JOBS="${3:-1}"

mkdir -p "${OUT}"

python scripts/compute_mcs_trivial.py \
  --input-csv "${CSV}" \
  --out-dir "${OUT}" \
  --group-col final_group_id \
  --n-jobs "${N_JOBS}"

python scripts/summarize_mcs_trivial.py \
  --pairwise-csv "${OUT}/mcs_pairwise_quality.csv" \
  --group-summary-csv "${OUT}/mcs_group_quality_summary.csv" \
  --out-dir "${OUT}"
