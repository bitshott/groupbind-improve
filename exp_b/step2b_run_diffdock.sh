#!/bin/bash
# Step 2b — Run DiffDock v1.0 inference on all 363 test PDBs.
#
# Prerequisites:
#   1. DiffDock checked out at v1.0:
#        git clone https://github.com/gcorso/DiffDock.git
#        cd DiffDock && git checkout v1.0
#        conda env create -f environment.yml && conda activate diffdock
#   2. ESM embeddings cached (see DiffDock README; one-time ~30 min):
#        python datasets/esm_embedding_preparation.py
#        # then run facebookresearch/esm scripts/extract.py as in README
#        python datasets/esm_embeddings_to_pt.py
#   3. Input CSV produced by step2a_make_csv.py
#
# Output: per-complex SDF files under $OUT_DIR/<complex_name>/rank{1..40}.sdf
#         and per-complex confidence scores in rank{n}_confidence.sdf names.
#
# Adjust DIFFDOCK_DIR and OUT_DIR for your machine.

set -euo pipefail

# ---- user-configurable paths ------------------------------------------------
DIFFDOCK_DIR="${DIFFDOCK_DIR:-$HOME/DiffDock}"
OUT_DIR="${OUT_DIR:-$PWD/out/diffdock_results}"
INPUT_CSV="${INPUT_CSV:-$PWD/out/diffdock_inputs.csv}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
BATCH_SIZE="${BATCH_SIZE:-10}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
# -----------------------------------------------------------------------------

if [ ! -d "$DIFFDOCK_DIR" ]; then
    echo "ERROR: DIFFDOCK_DIR=$DIFFDOCK_DIR does not exist." >&2
    echo "Clone https://github.com/gcorso/DiffDock.git and checkout v1.0." >&2
    exit 1
fi

if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: INPUT_CSV=$INPUT_CSV not found. Run step2a_make_csv.py first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

cd "$DIFFDOCK_DIR"

# v1.0 entry point. On main branch, add --old_score_model --old_confidence_model.
python -m inference \
    --config default_inference_args.yaml \
    --protein_ligand_csv "$INPUT_CSV" \
    --out_dir "$OUT_DIR" \
    --samples_per_complex "$SAMPLES_PER_COMPLEX" \
    --batch_size "$BATCH_SIZE" \
    --inference_steps "$INFERENCE_STEPS" \
    --no_final_step_noise

echo "[step2b] DiffDock inference complete. Outputs in $OUT_DIR"
