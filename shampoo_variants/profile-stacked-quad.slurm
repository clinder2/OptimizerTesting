#!/bin/bash
#SBATCH --job-name=stacked-quad
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=stacked-quad-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash "$SLURM_SUBMIT_DIR/shampoo_variants/profile-stacked-quad.sh" "$@"