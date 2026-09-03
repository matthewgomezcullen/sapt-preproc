#!/bin/bash

# Carry each complex of the bin through the whole pipeline and keep the active space it produces.
#
# Run setup.sh once from a login node first, then: sbatch run.sh
#
# 48 hours, because RHF alone was 11h42m for 7W06_ITN and a fifty-orbital Dice run beside it is
# the unknown this job exists to measure. If one does run out anyway, the SCF checkpoint it banked
# means the resubmission starts at the correlated steps rather than from the beginning.

#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --job-name=sapt-encode
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# No GPU: none of PySCF, Dice or the SCF underneath them uses one, and asking for an idle GPU only
# makes the job harder to schedule.
#SBATCH --array=0-2
#SBATCH --output=sapt-encode-%A_%a.out

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where setup.sh put the environment}/sapt-preproc}"

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

if [ ! -d "$REPO/src" ]; then
    echo "REPO is $REPO, which has no src/. Submit from the repo root, or set REPO explicitly:" \
         "REPO=\"\$DATA/thesis-experiments/sapt-tests\" sbatch run.sh" >&2
    exit 1
fi

# Report progress
export PYTHONUNBUFFERED=1

module purge
module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"

cd "$REPO/src"

# The bin under test, which is the three complexes tracked as fixtures. `filter.py --reuse` names
# the other nine of Q1; to run those instead, pass them in rather than editing this:
#
#   COMPLEXES="7BJJ_TVW 6YQW_82I" sbatch --array=0-1 run.sh
if [ -z "${COMPLEXES:-}" ]; then
    COMPLEXES=$(python -c "import sys; sys.path[:0] = ['.', 'tests', 'tests/encode']; from cutouts import SUBSET; print(' '.join(SUBSET))")
fi
read -r -a SUBSET <<< "$COMPLEXES"
if [ "${#SUBSET[@]}" -ne "$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))" ]; then
    echo "The array is ${SLURM_ARRAY_TASK_MIN}-${SLURM_ARRAY_TASK_MAX} but there are" \
         "${#SUBSET[@]} complexes. Pass --array to match, or set COMPLEXES." >&2
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

# Where the SCF checkpoints and the finished active spaces go. Both default into $DATA.
# Exported for `utils.encode.store`
export SCF_CHECKPOINTS="${SCF_CHECKPOINTS:-$DATA/scf}"
SPACES="${SPACES:-$DATA/spaces}"

echo "[$(date +%T)] Complex   $NAME  (task $SLURM_ARRAY_TASK_ID of ${#SUBSET[@]})"
echo "[$(date +%T)] Threads   $OMP_NUM_THREADS"
echo "[$(date +%T)] Memory    ${PYSCF_MAX_MEMORY} MB of ${SLURM_MEM_PER_NODE:-?} MB"
echo "[$(date +%T)] Scratch   $PYSCF_TMPDIR"
echo "[$(date +%T)] Dice      $(command -v Dice || echo 'not on the PATH')  ${MPIPREFIX:+under $MPIPREFIX}"

BANKED=0
if [ -d "$SCF_CHECKPOINTS" ]; then
    BANKED=$(find "$SCF_CHECKPOINTS" -maxdepth 1 -name '*.chk' | wc -l | tr -d ' ')
fi
echo "[$(date +%T)] Store     $SCF_CHECKPOINTS  ($BANKED banked)"
echo "[$(date +%T)] Spaces    $SPACES"
echo "[$(date +%T)] Host      $(hostname)"

if [ -f "$SPACES/$NAME.npz" ]; then
    echo "[$(date +%T)] $NAME already has an active space; this job will do nothing"
else
    echo "[$(date +%T)] Running the pipeline for $NAME"
fi

# --verbose 4 puts one line per SCF cycle in the log. Without it nothing is printed between the
# banner and the RHF step line.
python run.py --complex "$NAME" --out "$SPACES" --verbose 4

echo "[$(date +%T)] Done"
