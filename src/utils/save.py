import os


SCF = "scf"


def preparation_path(name, out):
    return os.path.join(out, f"{name}_preparation.npz")


def load_prepared(name, out):
    ...


def save_prepared(name, out):
    ...


def encoding_path(name, out):
    return os.path.join(out, f"{name}_encoding.npz")


def dice_log_path(name, out):
    return os.path.join(out, f"{name}.dice.out")


def get_scf_dir(name, out):
    return os.path.join(out, SCF) 


def scf_path(name, scf):
    return os.path.join(scf, f"{name}.chk")


def load_encoded(name, out):
    ...


def save_encoded(name, out):
    ...
