"""
Scope enforcement for EncodeProtein._verify, per the Eligibility Rules in README.md.

Each rejection test uses a real complex chosen so that it violates exactly one rule.

The following aren't tested because the dataset contains no case that violates them: covalent
    ligands, and ligands that are not closed-shell singlets.
"""

import pytest

from conftest import paths
from encode import EncodeProtein, OutOfScopeError, OutOfScopeErrorType


def rejection(name):
    """
    Verify a complex expected to be out of scope, returning the error type it was rejected with.
    """
    encode = EncodeProtein(*paths(name))
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
    A complex violating no eligibility rule passes verification.
    """
    encode = EncodeProtein(*paths(name))
    encode._fetch()
    encode._verify()

    assert encode.whole
    assert sum(len(chain) for chain in encode.whole) > 0
    assert len(encode.poses) == len(encode.poses_paths)


def test_verify_rejects_metal_in_retained_region():
    """
    Any metal atom or metal-containing cofactor in the retained region is out of scope.

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
    A biological cofactor inside the 4.5 Å shell forces an arbitrary A/B partition.
    """
    assert rejection(name) is OutOfScopeErrorType.COFACTOR


def test_verify_rejects_other_heterogen_in_shell():
    """
    Heterogens other than cofactors inside the 4.5 Å shell reject the complex, though 
        crystallisation additives outside the shell are simply deleted.

    5SAK_ZRY has a DMS molecule inside the shell.
    """
    assert rejection("5SAK_ZRY") is OutOfScopeErrorType.HETEROGEN


def test_verify_rejects_element_without_631g_basis():
    """
    Elements with no 6-31G basis raise on mol.build().

    7UTW_NAI has a Cd ion, which lies outside 6-31G's H-Zn coverage. Cd is also a metal, and
        metals are checked first, so the rejection is reported as METAL.
    """
    assert rejection("7UTW_NAI") is OutOfScopeErrorType.METAL


def test_verify_rejects_split_metal_coordination_sphere():
    """
    Reject if any residue in the cutout has a heavy atom within 2.8 Å of any metal.

    7OSO_0V1 keeps no metal in the cutout, but a metal 2.13 Å from a retained residue.
    """
    assert rejection("7OSO_0V1") is OutOfScopeErrorType.SPLIT_METAL_COORDINATION


def test_verify_rejects_charged_ligand():
    """
    v1: reject charged ligands rather than resolve ambiguous protonation.

    7TXK_LW8 has a ligand with formal charge +1.
    """
    assert rejection("7TXK_LW8") is OutOfScopeErrorType.CHARGED_LIGAND


def test_verify_rejects_oversized_cutout():
    """
    Provisional cap of 400 heavy atoms in the cutout.

    7CIJ_G0C's cutout holds 427 heavy atoms.
    """
    assert rejection("7CIJ_G0C") is OutOfScopeErrorType.SIZE_CAP


def test_verify_rejects_incomplete_residue_in_cutout():
    """
    v1: reject incomplete residues in the cutout rather than repair them with PDBFixer.

    6Z1C_7EY retains ARG A42, which is modelled with 6 of its 11 heavy atoms.
    """
    assert rejection("6Z1C_7EY") is OutOfScopeErrorType.INCOMPLETE_RESIDUE


def test_verify_rejects_disulfide_split_by_cutout():
    """
    v1: reject when the cutout catches one Cys and not its S-S partner.

    8FO5_Y4U retains one half of a disulfide bond.
    """
    assert rejection("8FO5_Y4U") is OutOfScopeErrorType.SPLIT_DISULFIDE


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TODO: chain breaks go undetected. PDBFixer.findMissingResidues needs SEQRES records to "
        "know the full sequence, and no PoseBusters PDB carries them, so _verify cannot yet see "
        "that residues are missing from the cutout."
    ),
)
def test_verify_rejects_chain_break_in_cutout():
    """
    A cutout spanning missing residues is out of scope, similarly to incomplete residues.
        
    8A2D_KXY retains a residue flanking a break where five residues (A860-A864) are absent,
        leaving a C-N distance of 16.1 Å against a peptide bond of roughly 1.33 Å.
    """
    assert rejection("8A2D_KXY") is OutOfScopeErrorType.CHAIN_BREAK


def test_verify_rejects_zero_occupancy_heavy_atom_in_cutout():
    """
    An atom deposited at zero occupancy has coordinates but no density supporting them.

    7DUA_HJ0 retains LYS A789, whose CE and NZ are modelled at zero occupancy.
    """
    assert rejection("7DUA_HJ0") is OutOfScopeErrorType.ZERO_OCCUPANCY


def test_verify_accepts_zero_occupancy_hydrogens_in_cutout():
    """
    The rule is about heavy atoms only. _clean strips every deposited hydrogen before protonation, 
        so a zero-occupancy hydrogen never reaches the QM region. 

    7YZU_DO7 retains eight zero-occupancy atoms, all of them hydroxyl or imidazole hydrogens.
    """
    encode = EncodeProtein(*paths("7YZU_DO7"))
    encode._fetch()
    encode._verify()
