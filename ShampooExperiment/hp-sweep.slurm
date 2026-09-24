#!/bin/bash
#SBATCH --job-name=hp
#SBATCH --account=gts-mtao8
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu
#SBATCH --output=./logs/hp.out
#SBATCH -oReport-%j.out   

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Run your bash script
bash sweep.sh