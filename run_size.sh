#!/bin/bash

# Carry the eight neutral cutouts through the whole pipeline, to measure how the cost of it
# scales with size.
#
# Run setup.sh once from a login node first, then: sbatch run_size.sh

#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --job-name=sapt-encode-size
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# No GPU: none of PySCF, Dice or the SCF underneath them uses one, and asking for an idle GPU only
# makes the job harder to schedule.
#SBATCH --array=0-7
#SBATCH --output=sapt-encode-size-%A_%a.out

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where setup.sh put the environment}/sapt-preproc}"

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"

if [ ! -d "$REPO/src" ]; then
    echo "REPO is $REPO, which has no src/. Submit from the repo root, or set REPO explicitly:" \
         "REPO=\"\$DATA/thesis-experiments/sapt-tests\" sbatch run_size.sh" >&2
    exit 1
fi

# Report progress
export PYTHONUNBUFFERED=1

module purge
module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"

cd "$REPO/src"

# The eight eligible cutouts of zero charge, in ascending size.
#
#   COMPLEXES="7BJJ_TVW 6YQW_82I" sbatch --array=0-1 run_size.sh
if [ -z "${COMPLEXES:-}" ]; then
    COMPLEXES="6YQW_82I 7R59_I5F 7LOE_Y84 7NSW_HC4 7XFA_D9J 6ZC3_JOR 7XBV_APC 7ZHP_IQY"
fi
read -r -a NAMES <<< "$COMPLEXES"
# `sbatch --array=5,6,7 run_size.sh`, addresses the same list this one did.
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#NAMES[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID is past the end of ${#NAMES[@]} complexes. Pass --array" \
         "within 0-$(( ${#NAMES[@]} - 1 )), or set COMPLEXES." >&2
    exit 1
fi
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"

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

echo "[$(date +%T)] Complex   $NAME  (task $SLURM_ARRAY_TASK_ID of ${#NAMES[@]})"
# From the screen filter.py already wrote
echo "[$(date +%T)] Size      $(python -c "import filter; r = {x['name']: x for x in filter.read()}.get('$NAME'); print(f\"{r['heavy_atoms']} heavy atoms, {r['electrons']} electrons\" if r else 'not in out/filter.csv')")"
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
    echo "[$(date +%T)] $NAME already has a result; the run will finish it, or do nothing if it is complete"
else
    echo "[$(date +%T)] Running the pipeline for $NAME"
fi

# GNU time reports peak resident memory. Its report goes to stderr, merged into this log.
TIME=""
if [ -x /usr/bin/time ]; then
    TIME="/usr/bin/time -v"
fi

# --verbose 4 puts one line per SCF cycle in the log. Without it nothing is printed between the
# banner and the RHF step line.
$TIME python run.py --complex "$NAME" --out "$SPACES" --verbose 4

echo "[$(date +%T)] Done"
