#!/bin/bash

# What /tmp has, on whichever machine runs this.
#
# MP2's integral transformation spools its half-transformed integrals to PYSCF_TMPDIR, which
# `run.sh` sets from TMPDIR and which the cluster hands out as /tmp. A conventional transform over
# a cutout of the bin wrote 62 GB there before the node ran out of disk.
#
# The login node's /tmp is a different disk from a compute node's, so submit it to see what a job
# would actually get:
#
#   sbatch temp.sh
#
# or run it here to see the login node's:
#
#   bash temp.sh

#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --job-name=sapt-tmp
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=sapt-tmp-%j.out

set -euo pipefail

cd /tmp

echo "Host      $(hostname)"
echo "TMPDIR    ${TMPDIR:-<unset, so PySCF would use /tmp>}"
echo

echo "Room in /tmp"
df -h /tmp

# Only when TMPDIR names somewhere else, since that is the disk PySCF would actually write to.
if [ -n "${TMPDIR:-}" ] && [ "$TMPDIR" != "/tmp" ] && [ -d "$TMPDIR" ]; then
    echo
    echo "Room in \$TMPDIR"
    df -h "$TMPDIR"
fi

echo
echo "What is in /tmp"
# Guarded: du exits non-zero over any directory it cannot read, which under `set -e` would end the
# script on a shared /tmp holding other people's files.
du -sh /tmp 2>/dev/null || echo "  not fully readable; the df line above is the one that matters"
