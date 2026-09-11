import os
from prepare import PrepareComplex
from encode import EncodeProtein


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
DATA = os.path.join(ROOT, "data")


def _load(data, out, complexes) -> list[PrepareComplex]:
    """
    TODO: return list of PrepareComplexes
    """
    ...
    return [
        PrepareComplex(protein_path, poses_paths, os.path.join(out, complex))
        for ...
    ]


def run(name, data, out, complexes=None, force=False):
    out = os.path.join(out, name)
    complexes = _load(data, out, complexes)
    for complex in complexes:
        if force or not complex.prepared():
            complex.prepare()
        protein = EncodeProtein(complex, out)
        if force or not protein.solved():
            protein.solve()
        if force or not protein.encoded():
            protein.encode()


if __name__ == "__main__":
    # Parse args for data dir, out dir, complexes, and force, and nothing else.
    ...
