#!/bin/bash

# The whole test suite, as one array job.
#
# Task 0 runs every test that does not solve a cutout of the subset: the quick ones, the preparation
# pipelines, Dice over the fragment, DiffDock's scoring pass, and the SAPT terms over 7LOE_Y84 once
# run.sh has correlated it. Each task after it carries one complex of the subset through RHF, AVAS,
# MP2, Dice and the stability analysis, all sharing the one SCF.
#
# Run setup.sh once first, then, from the repo root: sbatch scripts/test.sh
# --array on the command line runs part of it, e.g. --array=0 for everything but the subset's solves.

#SBATCH --clusters=htc
# The stability analysis is six to ten hours after a solve of one or two
#SBATCH --partition=medium
#SBATCH --job-name=sapt-tests
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# No GPU: PySCF's RHF here is CPU-only, and an idle GPU only makes the job harder to schedule.
# Task 0, then one task a complex of the subset.
#SBATCH --array=0-2
#SBATCH --output=sapt-tests-%A_%a.out

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where setup.sh put the environment}/sapt-preproc}"

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

if [ ! -d "$REPO/src" ]; then
    echo "REPO is $REPO, which has no src/. Submit from the repo root, or set REPO explicitly:" \
         "REPO=\"\$DATA/thesis-experiments/sapt-tests\" sbatch scripts/test.sh" >&2
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

# The subset is defined in the test module.
COMPLEXES=$(python -c "import sys; sys.path[:0] = ['.', 'tests', 'tests/encode']; from cutouts import SUBSET; print(' '.join(SUBSET))")
read -r -a SUBSET <<< "$COMPLEXES"
TASKS=$(( ${#SUBSET[@]} + 1 ))
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TASKS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID has nothing to run: the suite is $TASKS tasks, the rest and" \
         "${#SUBSET[@]} complexes. Update --array in this file." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_MAX" -lt "$(( TASKS - 1 ))" ]; then
    echo "The array stops at task $SLURM_ARRAY_TASK_MAX, which leaves ${SUBSET[*]:$SLURM_ARRAY_TASK_MAX} untested."
fi

# PySCF reads both of these. Every thread it uses comes from OMP.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export PYSCF_MAX_MEMORY="$(( ${SLURM_MEM_PER_NODE:-32768} * 3 / 4 ))"

export PYSCF_TMPDIR="${TMPDIR:-/tmp}"

# The preparation tests minimise with OpenMM
export OPENMM_CPU_THREADS="$OMP_NUM_THREADS"

# How Dice is launched. `setup.sh` put it on the PATH, and empty runs it on this task's one rank,
# where it takes the same sixteen cores through OpenMP that PySCF does. To give it more than one
# rank, ask SLURM for the tasks and set this to "srun". Its scratch goes to TMPDIR, node-local.
export MPIPREFIX="${MPIPREFIX:-}"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    PART="everything but the subset's solves"

    # DiffDock scores poses in an interpreter of its own. exp_1 built one on this cluster
    DIFFDOCK_PYTHON="${DIFFDOCK_PYTHON:-$DATA/diffdock/bin/python}"
    export DIFFDOCK_MODELS="${DIFFDOCK_MODELS:-$DATA/thesis-experiments/exp_1/DiffDock/workdir/v1.1}"
    export DIFFDOCK_ESM="${DIFFDOCK_ESM:-${TORCH_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/torch}/hub/checkpoints/esm2_t33_650M_UR50D.pt}"
    if [ -x "$DIFFDOCK_PYTHON" ] && [ -d "$DIFFDOCK_MODELS" ] && [ -f "$DIFFDOCK_ESM" ] \
        && [ -f diffdock/inference.py ]; then
        export DIFFDOCK_PYTHON
    else
        unset DIFFDOCK_PYTHON
    fi

    # The SAPT tests read 7LOE_Y84 (tests/sapt/monomers.py's COMPLEX) as run.sh left it in out,
    # through the link tests/data/encoded/7LOE_Y84, and fail rather than skip without it. They run
    # only once CASCI has correlated it there.
    FLAGS=(--prepare-long)
    CORRELATED="tests/data/encoded/7LOE_Y84/7LOE_Y84_casci.npz"
    if [ -f "$CORRELATED" ]; then
        FLAGS+=(--sapt-long)
        SAPT="$CORRELATED, so its tests run"
    else
        SAPT="no $CORRELATED, so its tests are skipped"
    fi
else
    NAME="${SUBSET[$(( SLURM_ARRAY_TASK_ID - 1 ))]}"
    PART="$NAME"
fi

echo "[$(date +%T)] Task      $SLURM_ARRAY_TASK_ID of $TASKS, $PART"
echo "[$(date +%T)] Threads   $OMP_NUM_THREADS"
echo "[$(date +%T)] Memory    ${PYSCF_MAX_MEMORY} MB of ${SLURM_MEM_PER_NODE:-?} MB"
echo "[$(date +%T)] Scratch   $PYSCF_TMPDIR"
echo "[$(date +%T)] Dice      $(command -v Dice || echo 'not on the PATH')  ${MPIPREFIX:+under $MPIPREFIX}"
echo "[$(date +%T)] Host      $(hostname)"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[$(date +%T)] DiffDock  ${DIFFDOCK_PYTHON:-not found, so its tests are skipped}"
    echo "[$(date +%T)] SAPT      $SAPT"

    # No -x: these tests are independent
    echo "[$(date +%T)] Running every test but the subset's solves"
    pytest tests "${FLAGS[@]}" -m "not (hpc or hpc_long_stab or hpc_long_dice)" -v --durations=0
else
    # The contract tests run first and take seconds, so a broken environment or a cutout that has
    # moved out of the subset fails before the solve. AVAS solves the SCF everything after it shares,
    # then MP2 and Dice, and test_rhf.py last, since it holds the stability analysis.
    #
    # -x because a solve that fails is not cached: every test after it would solve again, only to
    # fail too. Nothing here is checkpointed, so a rerun starts the solve over.
    echo "[$(date +%T)] Running the tests for $NAME"
    pytest tests/encode/test_molecular.py tests/encode/test_avas.py tests/encode/test_mp2.py \
        tests/encode/test_shci.py tests/encode/test_rhf.py \
        --hpc --hpc-long-dice --hpc-long-stab -k "$NAME" -v -x --durations=0
fi

echo "[$(date +%T)] Done"
