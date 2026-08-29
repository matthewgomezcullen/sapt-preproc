import os
import re
from collections import Counter

from prepare import PrepareComplex, OutOfScopeError, PrepareError

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
DIFFDOCK = os.path.join(DATA, "diffdock")
POSEBUSTERS = os.path.join(DATA, "posebusters")

# DiffDock names each pose it kept rank<N>_confidence<X>.sdf. Alongside those it writes a bare
# rank1.sdf copy of the top-ranked pose and, for some complexes, an energy-minimised
# rank<N>_confidence<X>_ensemble_relaxed.sdf.
POSE = re.compile(r"^rank\d+_confidence-?\d+\.\d+\.sdf$")

FAIL = "confidence-1000"


def _protein(name):
    """
    The single deposited structure of a PoseBusters complex.
    """
    directory = os.path.join(POSEBUSTERS, name)
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".pdb")
    ]


def _poses(name):
    """
    The candidate poses DiffDock produced for a complex, one per rank.
    """
    directory = os.path.join(DIFFDOCK, name)
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if POSE.match(f) and FAIL not in f
    ]


# A complex is screened only where both halves are on disk: DIFFDOCK also holds bust_results.csv,
# and POSEBUSTERS holds complexes DiffDock was never run on.
names = sorted(
    name
    for name in set(os.listdir(POSEBUSTERS)) & set(os.listdir(DIFFDOCK))
    if os.path.isdir(os.path.join(POSEBUSTERS, name))
    and os.path.isdir(os.path.join(DIFFDOCK, name))
)

complexes = []
incomplete = []
for name in names:
    proteins, poses = _protein(name), _poses(name)
    if len(proteins) != 1 or not poses:
        incomplete.append(name)
        continue
    complexes.append((name, proteins[0], poses))


eligible = []
rejected = []
failed = []

for i, (name, protein, poses) in enumerate(complexes, start=1):
    prepared = PrepareComplex(protein, poses)
    try:
        prepared._fetch()
        prepared._verify()
    except OutOfScopeError as error:
        # The complex parsed but falls outside the scope of the method.
        rejected.append((name, error.error_type))
    except PrepareError as error:
        # The complex could not be read, so the scope says nothing about it either way.
        failed.append((name, error))
    else:
        eligible.append(name)
    if i % 50 == 0:
        print(f'File {i} screened')

print('Screened', len(complexes))
print('Eligible', len(eligible))
print('Rejected', len(rejected))
for error_type, count in Counter(t for _, t in rejected).most_common():
    print(f'  {count:4d}  {error_type.value}')
print('Failed', len(failed))
for name, error in failed:
    print(f'  {name}: {error}')
if incomplete:
    print('Incomplete', len(incomplete), sorted(incomplete))
