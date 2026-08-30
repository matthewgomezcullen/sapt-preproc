#!/bin/bash

# Build the sapt-preproc environment on a Linux HPC node.
#
# Run once from a login node, then submit test.sh. Safe to re-run: it replaces the environment.
#
# pyscf comes from PyPI rather than conda-forge, which is the one deviation from environment.yml
# and is forced. conda-forge's only linux-64 builds of pyscf 2.14.0 are python 3.10, and both
# require _x86_64-microarch-level >=4, meaning AVX-512. Dropping to python 3.10 to reach them then
# breaks scipy 1.17.1, which needs 3.11 or newer, so the conda route cannot honour environment.yml
# on Linux at all. PyPI's wheel is py3-none-manylinux_2_17_x86_64: no python pin, no AVX-512, and
# every other version in environment.yml is kept exactly.

set -euo pipefail

MODULE="${MODULE:-Anaconda3/2025.06-1}"
PREFIX="${PREFIX:-${DATA:?DATA is not set; it is where the environment goes}/sapt-preproc}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="$REPO/src/environment.yml"

# One source of truth for the version, so this cannot drift from environment.yml.
PYSCF="$(sed -n 's/^[[:space:]]*-[[:space:]]*pyscf=\([0-9.]*\).*/\1/p' "$ENVIRONMENT")"
if [ -z "$PYSCF" ]; then
    echo "Could not read the pyscf version out of $ENVIRONMENT" >&2
    exit 1
fi

echo "[$(date +%T)] Environment  $PREFIX"
echo "[$(date +%T)] pyscf        $PYSCF, from PyPI"

# Reported rather than acted on. Level 4 means the conda-forge build is reachable, and being
# compiled for this machine's vector width it is likely faster than the generic wheel. Taking it
# costs python 3.10 and a scipy downgrade.
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' || true
try:
    from archspec.cpu import host
    features = host().features
    level = 4 if "avx512f" in features else 3 if "avx2" in features else 1
    print(f"[node] {host().name}, x86_64_v{level}"
          + ("" if level >= 4 else "  (no AVX-512: the conda-forge pyscf build is out of reach)"))
except Exception:
    pass
PY
fi

module purge
module load "$MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
WITHOUT_PYSCF="$WORK/environment.yml"
sed -e '/^[[:space:]]*-[[:space:]]*pyscf=/d' -e '/^name:/d' "$ENVIRONMENT" > "$WITHOUT_PYSCF"

echo "[$(date +%T)] Solving the environment"
conda env create --yes --prefix "$PREFIX" --file "$WITHOUT_PYSCF"

conda activate "$PREFIX"

echo "[$(date +%T)] Installing pyscf $PYSCF"
pip install --no-cache-dir "pyscf==$PYSCF"

echo "[$(date +%T)] Checking the environment"
python - <<'PY'
import numpy, scipy, gemmi, openmm, pyscf, rdkit
from pyscf import gto, scf

print(f"  python {__import__('sys').version.split()[0]}")
for module in (numpy, scipy, gemmi, pyscf, rdkit):
    print(f"  {module.__name__:8s} {module.__version__}")
print(f"  openmm   {openmm.__version__}")

# Small enough to be instant, real enough to prove the integrals and the SCF work.
mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24", basis="6-31g", verbose=0)
mean_field = scf.RHF(mol)
mean_field.kernel()
assert mean_field.converged, "the check SCF did not converge"
print(f"  water RHF/6-31G {mean_field.e_tot:.6f} Ha, converged")
PY

echo "[$(date +%T)] Done. Submit with: sbatch $REPO/test.sh"
