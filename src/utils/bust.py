import io
import os
import tempfile
from collections import Counter

import gemmi
from posebusters import PoseBusters


VALIDITY = "dock"

# TODO: Evaluate number of excluded poses from this.
# # PoseBusters' `not_too_far_away`, as the result table names it. It fails any pose sitting more
# # than 5 A from the protein, which detects a docking failure rather than an implausible structure.
# FAR = "protein-ligand_maximum_distance"


class BustError(RuntimeError):
    pass


def _write_pdb(model, directory):
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)
    structure.setup_entities()
    path = os.path.join(directory, "protein.pdb")
    with open(path, "w") as file:
        file.write(structure.make_pdb_string())
    return path


def valid(model, poses):
    """
    Indexes of poses PoseBusters finds physically plausible, and the checks the rest of them failed.

    `_clean` has already deleted every heterogen, so the cofactor and water checks are vacuous.
    """
    poses = list(poses)
    with tempfile.TemporaryDirectory(prefix="bust-") as directory:
        table = PoseBusters(VALIDITY, max_workers=0).bust(
            poses, None, _write_pdb(model, directory)
        )
    # See TODO above.
    # table = table.drop(columns=FAR, errors="ignore")

    if len(table) != len(poses):
        raise BustError(
            f"PoseBusters returned {len(table)} rows for {len(poses)} poses"
        )

    failed = Counter({
        check: int(count) for check, count in (~table).sum().items() if count
    })
    return [index for index, ok in enumerate(table.all(axis=1)) if ok], failed
