#!/bin/bash

# Solve RHF and run Dice over the hpc-marked tests.
#
# Run setup.sh once from a login node first, then: sbatch test.sh

#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --job-name=sapt-rhf
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# No GPU: PySCF's RHF here is CPU-only, and an idle GPU only makes the job harder to schedule.
#SBATCH --array=0-2
#SBATCH --output=sapt-rhf-%A_%a.out

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where setup.sh put the environment}/sapt-preproc}"

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

if [ ! -d "$REPO/src" ]; then
    echo "REPO is $REPO, which has no src/. Submit from the repo root, or set REPO explicitly:" \
         "REPO=\"\$DATA/thesis-experiments/sapt-tests\" sbatch test.sh" >&2
    exit 1
fi

# Report progress
export PYTHONUNBUFFERED=1

module purge
module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"

cd "$REPO/src"

# The bin is defined in the test module. Read it from there.
COMPLEXES=$(python -c "import sys; sys.path[:0] = ['.', 'tests', 'tests/encode']; from cutouts import SUBSET; print(' '.join(SUBSET))")
read -r -a SUBSET <<< "$COMPLEXES"
if [ "${#SUBSET[@]}" -ne "$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))" ]; then
    echo "The array is ${SLURM_ARRAY_TASK_MIN}-${SLURM_ARRAY_TASK_MAX} but the bin holds ${#SUBSET[@]}" \
         "complexes. Update --array in this file." >&2
    exit 1
fi
NAME="${SUBSET[$SLURM_ARRAY_TASK_ID]}"

# PySCF reads both of these. Every thread it uses comes from OMP.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PYSCF_MAX_MEMORY="$(( ${SLURM_MEM_PER_NODE:-32768} * 3 / 4 ))"

export PYSCF_TMPDIR="${TMPDIR:-/tmp}"

# How Dice is launched. `setup.sh` put it on the PATH, and empty runs it on this task's one rank,
# where it takes the same sixteen cores through OpenMP that PySCF does. To give it more than one
# rank, ask SLURM for the tasks and set this to "srun". Its scratch goes to TMPDIR, node-local.
export MPIPREFIX="${MPIPREFIX:-}"

echo "[$(date +%T)] Complex   $NAME  (task $SLURM_ARRAY_TASK_ID of ${#SUBSET[@]})"
echo "[$(date +%T)] Threads   $OMP_NUM_THREADS"
echo "[$(date +%T)] Memory    ${PYSCF_MAX_MEMORY} MB of ${SLURM_MEM_PER_NODE:-?} MB"
echo "[$(date +%T)] Scratch   $PYSCF_TMPDIR"
echo "[$(date +%T)] Dice      $(command -v Dice || echo 'not on the PATH')  ${MPIPREFIX:+under $MPIPREFIX}"
echo "[$(date +%T)] Host      $(hostname)"

# --encode narrows to the encoding tests and --hpc opts into the ones that solve. -k also picks up
# this complex's contract tests, which are seconds and run first, so a broken environment or a
# cutout that has moved out of the bin fails here
#
# -x because the seven hpc tests run in sequence on one cutout, and everything after the first
# failure is either the same failure again or a step that cannot run without it. A solve that dies
# is not repeated from scratch any more, since it leaves a checkpoint the next attempt resumes
# from, but there is still no reason to spend the rest of the wall clock finding that out.
echo "[$(date +%T)] Running the tests for $NAME"
pytest tests --long-protonate --hpc -k "$NAME" -v -x --durations=0

echo "[$(date +%T)] Done"
