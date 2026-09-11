#!/bin/bash

# Build the environment as a job rather than on a login node. From the repo root:
#
#     sbatch scripts/run_setup.sh

#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --job-name=sapt-setup
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=sapt-setup-%j.out

set -euo pipefail

# Dice compiles on every core the job was given.
BUILD_JOBS="$SLURM_CPUS_PER_TASK" bash "$SLURM_SUBMIT_DIR/scripts/setup.sh"
