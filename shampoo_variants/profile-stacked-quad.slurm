#!/bin/bash
#SBATCH --job-name=stacked-quad
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
## #SBATCH --partition=gpu:H100:1
#SBATCH --output=stacked-quad-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash "$SLURM_SUBMIT_DIR/profile-stacked-quad.sh" "$@"