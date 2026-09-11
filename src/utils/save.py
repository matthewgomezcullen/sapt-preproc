import os
import zipfile

import numpy as np
from pyscf import lib


UNREADABLE = (OSError, EOFError, ValueError, zipfile.BadZipFile)


def prepared_path(name, dir):
    return os.path.join(dir, f"{name}_prepared.npz")


def load_prepared(name, dir):
    return _load(prepared_path(name, dir))


def save_prepared(record, name, dir):
    _save(record, prepared_path(name, dir))


def scf_path(name, dir):
    return os.path.join(dir, f"{name}_rhf.chk")


def load_scf(name, dir):
    try:
        return lib.chkfile.load(scf_path(name, dir), "scf")
    except (OSError, KeyError):
        return None


def save_scf(record, name, dir):
    path = scf_path(name, dir)
    os.makedirs(dir, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    lib.chkfile.save(path, "scf", record)


def solved_path(name, dir):
    return os.path.join(dir, f"{name}_solved.npz")


def dice_log_path(name, dir):
    return os.path.join(dir, f"{name}.dice.out")


def load_solved(name, dir):
    return _load(solved_path(name, dir))


def save_solved(record, name, dir):
    _save(record, solved_path(name, dir))


def encoded_path(name, dir):
    return os.path.join(dir, f"{name}_encoded.npz")


def load_encoded(name, dir):
    return _load(encoded_path(name, dir))


def save_encoded(record, name, dir):
    _save(record, encoded_path(name, dir))


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
