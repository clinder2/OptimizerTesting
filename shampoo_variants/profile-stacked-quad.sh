#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

NCU="${NCU:-ncu}"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/results/stacked-quad-${SLURM_JOB_ID:-local}}"
ITERS="${ITERS:-2000}"
PROFILE_ITERS="${PROFILE_ITERS:-1}"

command -v "$NCU" >/dev/null || { printf 'Nsight Compute (ncu) not found: %s\n' "$NCU" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

for variant in default cholesky; do
    "$PYTHON" shampoo_variants/evalStackedQuad.py \
        --variant "$variant" --iters "$ITERS" \
        --output "$OUTPUT_DIR/$variant.json" "$@"

    "$NCU" --set basic --target-processes all --nvtx \
        --nvtx-include 'optimizer_step/' --force-overwrite \
        --export "$OUTPUT_DIR/$variant" \
        "$PYTHON" shampoo_variants/evalStackedQuad.py \
        --variant "$variant" --iters "$PROFILE_ITERS" --nvtx \
        --output "$OUTPUT_DIR/$variant-profile.json" "$@"

    "$NCU" --import "$OUTPUT_DIR/$variant.ncu-rep" --page raw --csv \
        > "$OUTPUT_DIR/$variant-kernels.csv"
done

printf 'Results and Nsight Compute reports: %s\n' "$OUTPUT_DIR"