#!/bin/bash

# The full pipeline over the complexes chosen from a screen, one array task a complex.
#
# Each complex is read back from the preparation the screen kept in src/out/<job>/<complex>. run.py 
# solves every pose at RHF, carries the protein through RHF, AVAS, MP2 and Dice, encodes the 
# Hamiltonian and solves it by CASCI, then scores every pose against the protein by SAPT, keeping each
# stage beside the preparation. A task that reaches the time limit keeps every pose SAPT scored, and
# submitting it again resumes after them. Once every task has finished and confidence.py has ranked
# the screen, `python sapt.py --name v1_1_mm_unsize` from src/ reranks it.
#
# Run setup.sh once first, then, from the repo root: sbatch scripts/run.sh
# --array on the command line runs part of it, e.g. --array=11 for the last complex of chosen.csv.

#SBATCH --clusters=htc
#SBATCH --partition=medium
#SBATCH --job-name=sapt-run
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
# No GPU: PySCF's RHF here is CPU-only, and an idle GPU only makes the job harder to schedule.
# One task a row of chosen.csv.
#SBATCH --array=0-11
#SBATCH --output=sapt-run-%A_%a.out

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where setup.sh put the environment}/sapt-preproc}"

# The screen, as its directory under src/out is named. Its chosen.csv names the complexes.
JOB="${JOB:-filter_v1_1_mm_unsize}"

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

if [ ! -d "$REPO/src" ]; then
    echo "REPO is $REPO, which has no src/. Submit from the repo root, or set REPO explicitly:" \
         "REPO=\"\$DATA/thesis-experiments/sapt-tests\" sbatch scripts/run.sh" >&2
    exit 1
fi

CHOSEN="$REPO/src/out/$JOB/chosen.csv"
if [ ! -f "$CHOSEN" ]; then
    echo "$CHOSEN is missing. Pull the screen's artefacts into src/out/$JOB first." >&2
    exit 1
fi

# Report progress
export PYTHONUNBUFFERED=1

module purge
module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PREFIX"

export PATH="$PREFIX/bin:$PATH"
if ! python -c "import gemmi, pyscf"; then
    echo "$(command -v python) cannot import the pipeline's packages; $PREFIX should be the" \
         "environment setup.sh built." >&2
    exit 1
fi

cd "$REPO/src"

# The complexes are the rows of chosen.csv, in order. Read them from there.
COMPLEXES=$(python -c "import csv, sys; print(' '.join(row['name'] for row in csv.DictReader(open(sys.argv[1], newline=''))))" "$CHOSEN")
read -r -a CHOSEN_NAMES <<< "$COMPLEXES"
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#CHOSEN_NAMES[@]}" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID has nothing to run: chosen.csv holds ${#CHOSEN_NAMES[@]}" \
         "complexes. Update --array in this file." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_MAX" -lt "$(( ${#CHOSEN_NAMES[@]} - 1 ))" ]; then
    echo "The array stops at task $SLURM_ARRAY_TASK_MAX, which leaves" \
         "${CHOSEN_NAMES[*]:$(( SLURM_ARRAY_TASK_MAX + 1 ))} unrun."
fi
NAME="${CHOSEN_NAMES[$SLURM_ARRAY_TASK_ID]}"
DIRECTORY="out/$JOB/$NAME"

# run.py prepares whatever it cannot read back, which needs the benchmark set and would not reproduce
# the screen's preparation, so a complex without a readable one is refused here instead.
if ! python -c "import sys; from prepare import PrepareComplex; sys.exit(not PrepareComplex('', [], sys.argv[1]).prepared())" "$DIRECTORY"; then
    echo "$DIRECTORY holds no preparation of $NAME that can be read back." >&2
    exit 1
fi

if ! command -v Dice >/dev/null; then
    echo "Dice is not on the PATH, and the protein's space cannot be solved without it. See setup.sh." >&2
    exit 1
fi

# PySCF reads both of these. Every thread it uses comes from OMP.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PYSCF_MAX_MEMORY="$(( ${SLURM_MEM_PER_NODE:-32768} * 3 / 4 ))"

# CASCI transforms the integrals through a swap file, every pair of the window's orbitals against
# every pair of basis functions: 5 GB on the largest cutout at the fourteen orbitals a window leaves,
# and 58 GB at the fifty Dice solves. It goes to the job's $SCRATCH, on the shared filesystem, rather
# than a node-local disk of unknown size. Both are removed when the job ends.
export PYSCF_TMPDIR="${SCRATCH:-${TMPDIR:-/tmp}}"

# How Dice is launched. `setup.sh` put it on the PATH, and empty runs it on this task's one rank.
# To give it more than one rank, ask SLURM for the tasks and set this to "srun".
export MPIPREFIX="${MPIPREFIX:-}"

# What the complex's directory holds: the stage a task starts from, and the one it reached.
kept() {
    local files poses
    files=$(find "$DIRECTORY" -maxdepth 1 -type f -name "${NAME}[._]*" -exec basename {} \; | sort | tr '\n' ' ')
    poses=$(find "$DIRECTORY" -path "*/pose_scf/*_rhf.chk" 2>/dev/null | wc -l | tr -d ' ')
    echo "[$(date +%T)] Kept      ${files}and $poses pose SCFs"
}

echo "[$(date +%T)] Complex   $NAME  (task $SLURM_ARRAY_TASK_ID of ${#CHOSEN_NAMES[@]})"
echo "[$(date +%T)] Threads   $OMP_NUM_THREADS"
echo "[$(date +%T)] Memory    ${PYSCF_MAX_MEMORY} MB of ${SLURM_MEM_PER_NODE:-?} MB"
echo "[$(date +%T)] Scratch   $PYSCF_TMPDIR"
echo "[$(date +%T)] Dice      $(command -v Dice)  ${MPIPREFIX:+under $MPIPREFIX}"
echo "[$(date +%T)] Host      $(hostname)"
kept

ASKED=("$@")

echo "[$(date +%T)] Running the pipeline for $NAME ${ASKED[*]:+with ${ASKED[*]}}"
# Expanded so that an empty ASKED passes nothing rather than an empty argument.
python run.py "$JOB" --complexes "$NAME" ${ASKED[@]+"${ASKED[@]}"}

kept
echo "[$(date +%T)] Done"
