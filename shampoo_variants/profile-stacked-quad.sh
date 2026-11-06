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

find_ncu() {
    command -v "$NCU" >/dev/null && return 0
    # PACE provides Nsight Compute through the CUDA module
    if command -v module >/dev/null 2>&1 || [[ -f /etc/profile.d/modules.sh ]]; then
        command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
        module load cuda 2>/dev/null || true
        command -v "$NCU" >/dev/null && return 0
    fi
    local cand
    for cand in "${CUDA_HOME:-}"/bin/ncu /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu; do
        if [[ -x "$cand" ]]; then NCU="$cand"; return 0; fi
    done
    # Fallback: install into the active conda env (no root needed)
    if command -v conda >/dev/null 2>&1; then
        conda install -y -c nvidia nsight-compute || true
        cand="$(find "${CONDA_PREFIX:-}" -name ncu -type f -perm -u+x 2>/dev/null | head -n1)"
        if [[ -n "$cand" ]]; then NCU="$cand"; return 0; fi
    fi
    return 1
}

find_ncu || true
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