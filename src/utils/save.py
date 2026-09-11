import os


def prepared_path(name, dir):
    return os.path.join(dir, f"{name}_prepared.npz")


def load_prepared(name, dir):
    ...


def save_prepared(record, name, dir):
    ...


def scf_path(name, dir):
    return os.path.join(dir, f"{name}_rhf.chk")


def load_scf(name, dir):
    ...


def save_scf(record, name, dir):
    ...


def solved_path(name, dir):
    return os.path.join(dir, f"{name}_solved.npz")


def dice_log_path(name, dir):
    return os.path.join(dir, f"{name}.dice.out")


def load_solved(name, dir):
    ...


def save_solved(record, name, dir):
    ...


def encoded_path(name, dir):
    return os.path.join(dir, f"{name}_encoded.npz")


def load_encoded(name, dir):
    ...


def save_encoded(record, name, dir):
    ...
