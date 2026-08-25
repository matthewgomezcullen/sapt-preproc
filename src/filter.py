import os
import gemmi

ROOT = os.path.dirname(os.path.abspath(__file__))
print(ROOT)
DATA = os.path.join(ROOT, 'data')
DIFFDOCK = os.path.join(DATA, "diffdock")
POSEBUSTERS = os.path.join(DATA, "posebusters")

diffdock_names = set(os.listdir(DIFFDOCK))

pdb_files = sorted(
        os.path.join(POSEBUSTERS, name, f)
        for name in os.listdir(POSEBUSTERS)
        if name in diffdock_names
        for f in os.listdir(os.path.join(POSEBUSTERS, name))
        if f.endswith(".pdb")
        )

def is_eligible(pdb_file):
    model = gemmi.read_pdb(pdb_file)[0] # pyright: ignore[reportAttributeAccessIssue]
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.altloc and atom.altloc != "\00":
                    print(f'Altloc found. Rejecting {pdb_file}')
                    return False
                ...
    return True


eligible = []
rejected = []

for i, pdb_file in enumerate(pdb_files):
    if is_eligible(pdb_file):
        eligible.append(pdb_file)
    else:
        rejected.append(pdb_file)
    if i % 50 == 0:
        print(f'File {i+1} screened')

print('Eligible', len(eligible))
print('Rejected', len(rejected))
