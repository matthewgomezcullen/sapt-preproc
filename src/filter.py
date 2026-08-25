import os
from encode import EncodeProtein, OutOfScopeError

ROOT = os.path.dirname(os.path.abspath(__file__))
print(ROOT)
DATA = os.path.join(ROOT, 'data')
DIFFDOCK = os.path.join(DATA, "diffdock")
POSEBUSTERS = os.path.join(DATA, "posebusters")

diffdock_names = set(os.listdir(DIFFDOCK))

pdb_files = [
        os.path.join(POSEBUSTERS, name, f)
        for name in os.listdir(POSEBUSTERS)
        if name in diffdock_names
        for f in os.listdir(os.path.join(POSEBUSTERS, name))
        if f.endswith(".pdb")
        ]
pose_files = [
        os.path.join(DIFFDOCK, name, f)
        for name in os.listdir(POSEBUSTERS)
        if name in diffdock_names
        for f in os.listdir(os.path.join(DIFFDOCK, name))
        if f.startswith("rank") and f.endswith(".sdf") 
        # TODO: Deduplicate rank 1 and ignore confidence-1000.sdf
        ]


eligible = []
rejected = []

for i, pdb_file in enumerate(pdb_files):
    encoding = EncodeProtein(pdb_file, pose_files)
    try:
        encoding._fetch()
        encoding._verify()
        eligible.append(pdb_file)
    except OutOfScopeError:
        rejected.append(pdb_file)
    if i % 50 == 0:
        print(f'File {i+1} screened')

print('Eligible', len(eligible))
print('Rejected', len(rejected))
