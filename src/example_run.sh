#!/bin/bash

# Example SLURM job

#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --job-name=<job_name>
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=<job_name>-%j.out

set -euo pipefail

# Keep Python's stdout unbuffered so progress lands in the log as it happens,
# rather than being lost in a buffer if the job is killed.
export PYTHONUNBUFFERED=1

EXP_DIR="$DATA/thesis-experiments/<dir>"

module purge
module load Anaconda3/2025.06-1
conda activate "$DATA/<env_name>"

nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

...
