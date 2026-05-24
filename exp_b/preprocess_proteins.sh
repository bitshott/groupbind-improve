#!/bin/bash
# preprocess_proteins.sh
#
# For each complex_name listed in the authors' CSV (pdbbind_benchmark_test.csv),
# read <pdb_id>_protein.pdb from data/pdbbind_all/<pdb_id>/ (raw PDBBind),
# run `reduce` to add hydrogens, and write the result to
# data/PDBBind_processed/<pdb_id>/<pdb_id>_protein_processed.pdb
# so the authors' CSV paths resolve directly.
#
# Usage:
#   bash preprocess_proteins.sh [csv] [src_dir] [dst_dir] [num_parallel]
# Defaults:
#   csv          = pdbbind_benchmark_test.csv
#   src_dir      = data/pdbbind_all
#   dst_dir      = data/PDBBind_processed
#   num_parallel = 8

set -u

CSV="${1:-pdbbind_benchmark_test.csv}"
SRC_DIR="${2:-data/pdbbind_all}"
DST_DIR="${3:-data/PDBBind_processed}"
JOBS="${4:-8}"

if ! command -v reduce >/dev/null 2>&1; then
    echo "ERROR: 'reduce' not found in PATH." >&2
    echo "Install with: conda install -c bioconda reduce" >&2
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "ERROR: CSV $CSV not found." >&2
    exit 1
fi
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir $SRC_DIR not found." >&2
    exit 1
fi

mkdir -p "$DST_DIR"

# Extract PDB IDs from column 1 of the CSV (skip header), lowercase, dedupe
PDB_IDS=$(tail -n +2 "$CSV" | awk -F',' '{print tolower($1)}' | sort -u)
TOTAL=$(echo "$PDB_IDS" | wc -l)
echo "[info] $TOTAL unique PDB ids to process from $CSV"

process_one() {
    local pdb_id="$1"
    local src_dir="$2"
    local dst_dir="$3"

    local in="$src_dir/$pdb_id/${pdb_id}_protein.pdb"
    local out_subdir="$dst_dir/$pdb_id"
    local out="$out_subdir/${pdb_id}_protein_processed.pdb"

    if [ ! -f "$in" ]; then
        echo "[skip] no input file: $in" >&2
        return
    fi
    if [ -f "$out" ] && [ -s "$out" ]; then
        return  # already processed and non-empty
    fi

    mkdir -p "$out_subdir"

    # reduce: strip existing H, then add H with optimization
    reduce -Trim "$in" 2>/dev/null \
        | reduce -BUILD - 2>/dev/null \
        > "$out"

    if [ ! -s "$out" ]; then
        echo "[fail] empty output: $out" >&2
        rm -f "$out"
    fi
}
export -f process_one

# Parallelize via xargs
echo "$PDB_IDS" | xargs -P "$JOBS" -I {} bash -c \
    'process_one "$1" "$2" "$3"' _ {} "$SRC_DIR" "$DST_DIR"

DONE=$(find "$DST_DIR" -name "*_protein_processed.pdb" | wc -l)
echo "[summary] $DONE _protein_processed.pdb files exist in $DST_DIR"
