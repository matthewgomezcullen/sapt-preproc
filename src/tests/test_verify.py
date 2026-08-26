"""
Scope enforcement for EncodeProtein._verify, per the Eligibility Rules in README.md.

Each rejection test uses a real complex chosen so that it violates exactly one rule; the
    confounder-free choices are recorded in the comment beside each fixture name.

Two README rules have no test because the dataset contains no case that violates them: covalent
    ligands, and ligands that are not closed-shell singlets.
"""

import os
import re

import pytest

from encode import EncodeProtein, OutOfScopeError, OutOfScopeErrorType

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(SRC, "data")
POSEBUSTERS = os.path.join(DATA, "posebusters")
DIFFDOCK = os.path.join(DATA, "diffdock")

POSE_PATTERN = re.compile(r"rank\d+_confidence-[-\d.]+\.sdf")
DROPPED_CONFIDENCE = "confidence-1000.00.sdf"


def _paths(name):
    """
    Resolve a complex to its protein PDB and its candidate poses, dropping the failed-confidence
        poses that README excludes.
    """
    standalone = os.path.join(DATA, name)
    if os.path.isdir(os.path.join(standalone, "poses")):
        protein = os.path.join(standalone, f"{name}_protein.pdb")
        poses_dir = os.path.join(standalone, "poses")
    else:
        protein = os.path.join(POSEBUSTERS, name, f"{name}_protein.pdb")
        poses_dir = os.path.join(DIFFDOCK, name)
    poses = sorted(
        os.path.join(poses_dir, f)
        for f in os.listdir(poses_dir)
        if POSE_PATTERN.fullmatch(f) and not f.endswith(DROPPED_CONFIDENCE)
    )
    return protein, poses


def rejection(name):
    """
    Verify a complex expected to be out of scope, returning the error type it was rejected with.
    """
    encode = EncodeProtein(*_paths(name))
    encode._fetch()

    with pytest.raises(OutOfScopeError) as rejected:
        encode._verify()
    return rejected.value.error_type


@pytest.mark.parametrize(
    "name",
    [
        "6TW5_9M2",     # 29 residues, 261 heavy atoms, no heterogens, neutral ligand
        "5S8I_2LY",     # 12 residues, 108 heavy atoms, apo PDB with no heterogens at all
    ],
)
def test_verify_accepts_in_scope_complex(name):
    """
    A complex violating no eligibility rule passes verification and keeps its structure usable.
    """
    encode = EncodeProtein(*_paths(name))
    encode._fetch()
    encode._verify()

    assert sum(len(chain) for chain in encode.whole) > 0
    assert len(encode.poses) == len(encode.poses_paths)


def test_verify_rejects_metal_in_retained_region():
    """
    README: any metal atom or metal-containing cofactor in the retained region is out of scope.

    6XM9_V55 has a Co ion inside the 4.5 Å shell.
    """
    assert rejection("6XM9_V55") is OutOfScopeErrorType.METAL


@pytest.mark.parametrize(
    "name",
    [
        "6M2B_EZO",     # FMN inside the shell
        "6T88_MWQ",     # FAD inside the shell
    ],
)
def test_verify_rejects_biological_cofactor(name):
    """
    README: a biological cofactor inside the 4.5 Å shell forces an arbitrary A/B partition choice.
    """
    assert rejection(name) is OutOfScopeErrorType.COFACTOR


def test_verify_rejects_other_heterogen_in_shell():
    """
    README: heterogens other than cofactors inside the 4.5 Å shell also reject the complex, even
        though crystallisation additives outside the shell are simply deleted.

    5SAK_ZRY has a DMS molecule inside the shell.
    """
    assert rejection("5SAK_ZRY") is OutOfScopeErrorType.HETEROGEN


def test_verify_rejects_element_without_631g_basis():
    """
    README: elements with no 6-31G basis raise on mol.build(), so reject during encoding.

    7UTW_NAI has a Cd ion, which lies outside 6-31G's H-Zn coverage. Cd is also a metal, and
        metals are checked first, so the rejection is reported as METAL.
    """
    assert rejection("7UTW_NAI") is OutOfScopeErrorType.METAL


def test_verify_rejects_split_metal_coordination_sphere():
    """
    README: reject if any residue in the cutout has a heavy atom within 2.8 Å of any metal, since
        deleting a coordinating residue gives nonsensical geometry and protonation.

    7OSO_0V1 keeps no metal in the cutout, but a metal 2.13 Å from a retained residue.
    """
    assert rejection("7OSO_0V1") is OutOfScopeErrorType.SPLIT_METAL_COORDINATION


def test_verify_rejects_charged_ligand():
    """
    README, v1: reject charged ligands rather than resolve ambiguous protonation.

    7TXK_LW8 has a ligand with formal charge +1.
    """
    assert rejection("7TXK_LW8") is OutOfScopeErrorType.CHARGED_LIGAND


def test_verify_rejects_oversized_cutout():
    """
    README: provisional cap of 400 heavy atoms in the cutout.

    7CIJ_G0C's cutout holds 427 heavy atoms, so it also pins the cap's boundary.
    """
    assert rejection("7CIJ_G0C") is OutOfScopeErrorType.SIZE_CAP


def test_verify_rejects_incomplete_residue_in_cutout():
    """
    README, v1: reject incomplete residues in the cutout rather than repair them with PDBFixer.

    6Z1C_7EY retains ARG A42, which is modelled with 6 of its 11 heavy atoms.
    """
    assert rejection("6Z1C_7EY") is OutOfScopeErrorType.INCOMPLETE_RESIDUE


def test_verify_rejects_disulfide_split_by_cutout():
    """
    README, v1: reject when the cutout catches one Cys and not its S-S partner.

    8FO5_Y4U retains one half of a disulfide bond.
    """
    assert rejection("8FO5_Y4U") is OutOfScopeErrorType.SPLIT_DISULFIDE
