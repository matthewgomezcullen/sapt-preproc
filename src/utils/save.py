import glob
import os
import zipfile

import numpy as np
from pyscf import lib


UNREADABLE = (OSError, EOFError, ValueError, zipfile.BadZipFile)

# The poses' SCFs are kept together.
POSE_SCF = "pose_scf"


def prepared_path(name, dir):
    return os.path.join(dir, f"{name}_prepared.npz")


def load_prepared(name, dir):
    return _load(prepared_path(name, dir))


def save_prepared(record, name, dir):
    _supersede(prepared_path, name, dir)
    _save(record, prepared_path(name, dir))


def scf_path(name, dir):
    return os.path.join(dir, f"{name}_rhf.chk")


def load_scf(name, dir):
    return _load_chk(scf_path(name, dir))


def save_scf(record, name, dir):
    _supersede(scf_path, name, dir)
    _save_chk(record, scf_path(name, dir))


def pose_scf_path(name, dir, index):
    return os.path.join(dir, POSE_SCF, f"{name}_pose{index}_rhf.chk")


def load_pose_scf(name, dir, index):
    return _load_chk(pose_scf_path(name, dir, index))


def save_pose_scf(record, name, dir, index):
    """
    Only SAPT's scores are built on a pose's SCF, so they are discarded with the file replaced.
    """
    path = pose_scf_path(name, dir, index)
    if os.path.exists(path):
        os.remove(path)
    _supersede(sapt_path, name, dir)
    _save_chk(record, path)


def solved_path(name, dir):
    return os.path.join(dir, f"{name}_solved.npz")


def dice_log_path(name, dir):
    return os.path.join(dir, f"{name}.dice.out")


def load_solved(name, dir):
    return _load(solved_path(name, dir))


def save_solved(record, name, dir):
    _supersede(solved_path, name, dir)
    _save(record, solved_path(name, dir))


def encoded_path(name, dir):
    return os.path.join(dir, f"{name}_encoded.npz")


def load_encoded(name, dir):
    return _load(encoded_path(name, dir))


def save_encoded(record, name, dir):
    _supersede(encoded_path, name, dir)
    _save(record, encoded_path(name, dir))


def casci_path(name, dir):
    return os.path.join(dir, f"{name}_casci.npz")


def load_casci(name, dir):
    return _load(casci_path(name, dir))


def save_casci(record, name, dir):
    _supersede(casci_path, name, dir)
    _save(record, casci_path(name, dir))


def sapt_path(name, dir):
    return os.path.join(dir, f"{name}_sapt.npz")


def load_sapt(name, dir):
    return _load(sapt_path(name, dir))


def save_sapt(record, name, dir):
    _supersede(sapt_path, name, dir)
    _save(record, sapt_path(name, dir))


def _load(path):
    try:
        with np.load(path, allow_pickle=False) as stored:
            record = {key: stored[key] for key in stored.files}
    except UNREADABLE:
        return None
    return {key: value.item() if value.ndim == 0 else value for key, value in record.items()}


def _save(record, path):
    """
    Records are read with pickling off, so one that needs it is refused here.

    Every value is made an array before the file is opened.
    """
    arrays = {key: np.asarray(value) for key, value in record.items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, allow_pickle=False, **arrays)


def _load_chk(path):
    try:
        return lib.chkfile.load(path, "scf")
    except (OSError, KeyError):
        return None


def _save_chk(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lib.chkfile.save(path, "scf", record)


def _supersede(artefact, name, dir):
    """
    Discard `artefact`, and everything written after it.

    The poses' SCFs are built on the preparation alone, so a new preparation discards them. SAPT's 
        scores are built on everything, the poses' SCFs included, so they come last.
    """
    order = [
        prepared_path, scf_path, dice_log_path, solved_path, encoded_path, casci_path, sapt_path
    ]
    for later in order[order.index(artefact):]:
        path = later(name, dir)
        if os.path.exists(path):
            os.remove(path)
    if artefact is prepared_path:
        for path in glob.glob(pose_scf_path(glob.escape(name), glob.escape(dir), "*")):
            os.remove(path)
