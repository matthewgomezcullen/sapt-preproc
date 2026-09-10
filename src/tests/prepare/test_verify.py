"""
Scope enforcement for PrepareComplex._verify, per the Eligibility Rules in README.md.

Each rejection test uses a real complex chosen so that it violates exactly one rule.

The following aren't tested because the dataset contains no case that violates them: covalent
    ligands, and ligands that are not closed-shell singlets.

_reverify closes the gap _verify leaves once minimisation moves the poses. Its rules read the
    shell, and by then _clean has deleted the metals and heterogens the shell is judged against and
    _fix has destroyed the occupancies, so _verify stashes all three on the way past.
"""

import copy
import functools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from conftest import paths
from prepare import PrepareComplex, OutOfScopeError, OutOfScopeErrorType

# 6TW5_9M2 is accepted, and carries both kinds of stash outside its shell: three loose Mg ions and
# MYA, a myristoyl chain that is no crystallisation additive.
STASHED = "6TW5_9M2"
MYA = "MYA"


def rejection(name):
    """
    Verify a complex expected to be out of scope, returning the error type it was rejected with.
    """
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()

    with pytest.raises(OutOfScopeError) as rejected:
        prepared._verify()
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
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()
    prepared._verify()

    assert prepared.whole
    assert sum(len(chain) for chain in prepared.whole) > 0
    assert len(prepared.poses) == len(prepared.poses_paths)


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


@pytest.mark.parametrize(
    "name",
    [
        "7TM6_GPJ",     # S3P, shikimate-3-phosphate, the enzyme's own substrate
        "7ES1_UDP",     # JDF, 34 heavy atoms
        "6TW7_NZB",     # MYA, a myristoyl chain of 63 heavy atoms
        "7QE4_NGA",     # A2G, two N-acetylgalactosamines of a glycan
    ],
)
def test_verify_rejects_other_heterogen_in_shell(name):
    """
    Heterogens other than cofactors inside the 4.5 Å shell reject the complex.
    """
    assert rejection(name) is OutOfScopeErrorType.HETEROGEN


@pytest.mark.parametrize(
    "name",
    [
        "7USH_82V",     # one ethylene glycol, EDO A504, in a 13-residue cutout
        "7OPG_06N",     # two glycerols, GOL A502 and A503
        "7QF4_RBF",     # a chloride ion, CL A203
        "7FB7_8NF",     # MPD, a cryoprotectant of 8 heavy atoms
    ],
)
def test_verify_accepts_crystallisation_additive_in_shell(name):
    """
    A cryoprotectant, precipitant, buffer, or simple ion inside the shell is deleted by _clean
        rather than rejected here.
    """
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()
    prepared._verify()


def test_verify_rejects_a_heterogen_sharing_the_shell_with_an_additive():
    """
    The additive exemption applies per residue, not per complex. 7D5C_GV6 holds GVU, a 47 heavy-atom
        ligand, alongside an ethylene glycol; excusing the EDO must not excuse the GVU with it.
    """
    assert rejection("7D5C_GV6") is OutOfScopeErrorType.HETEROGEN


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


@pytest.mark.parametrize(
    "name",
    [
        "7W06_ITN",     # itaconic acid, two carboxylic acids, a dianion at 7.4
        "7TBU_S3P",     # shikimate-3-phosphate, a carboxylic acid and a phosphate
        "7YZU_DO7",     # a sulfonate
    ],
)
def test_verify_rejects_ligand_ionised_at_physiological_pH(name):
    """
    These three carry a group that is ionised at 7.4 and encoded as the neutral acid it is not.
    """
    assert rejection(name) is OutOfScopeErrorType.ACIDIC_LIGAND


def test_verify_accepts_a_ligand_with_no_ionisable_group():
    prepared = PrepareComplex(*paths("7NFB_GEN"))
    prepared._fetch()
    prepared._verify()


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


@pytest.mark.parametrize(
    "name",
    [
        "7YZU_DO7",     # eight, every one a hydroxyl or imidazole hydrogen
        "7WUX_6OI",     # six
    ],
)
def test_verify_does_not_reject_zero_occupancy_hydrogens(name):
    """
    A zero-occupancy hydrogen never reaches the QM region.
    
    Both of these reach the acidic-ligand rule, which proves the occupancy rule passed.
    """
    assert rejection(name) is OutOfScopeErrorType.ACIDIC_LIGAND


@functools.lru_cache(maxsize=None)
def _reduced(name):
    """
    A complex carried as far as _reverify.
    """
    prepared = PrepareComplex(*paths(name))
    prepared._fetch()
    prepared._verify()
    prepared._fix()
    prepared._clean()
    prepared._protonate()
    prepared._minimise()
    prepared._bust()
    prepared._reduce()
    return prepared


def reduced(name):
    """
    A copy of the shared run.
    """
    return copy.copy(_reduced(name))


def stashed(prepared, metal):
    return [position for _, position, is_metal in prepared.deleted if is_metal is metal]


def onto(poses, target):
    """
    Every pose translated so that its first atom sits on `target`.
    """
    moved = []
    for pose in poses:
        copied = Chem.Mol(pose) # pyright: ignore[reportAttributeAccessIssue]
        conformer = copied.GetConformer()
        shift = np.array(target) - np.array(conformer.GetAtomPosition(0))
        for index in range(copied.GetNumAtoms()):
            position = np.array(conformer.GetAtomPosition(index)) + shift
            conformer.SetAtomPosition(index, Point3D(*position))
        moved.append(copied)
    return moved


@pytest.mark.long_protonate
def test_verify_stashes_the_molecules_clean_deletes():
    prepared = PrepareComplex(*paths(STASHED))
    prepared._fetch()
    prepared._verify()

    assert stashed(prepared, metal=True)
    assert MYA in {name for name, _, _ in prepared.deleted}

    prepared._fix()
    prepared._clean()

    remaining = {residue.name for chain in prepared.whole for residue in chain}
    assert MYA not in remaining
    assert not any(
        atom.element.is_metal
        for chain in prepared.whole
        for residue in chain
        for atom in residue
    )
    assert stashed(prepared, metal=True)


@pytest.mark.long_protonate
def test_reverify_accepts_a_cutout_the_poses_did_not_change():
    prepared = reduced(STASHED)

    prepared._reverify()


@pytest.mark.long_protonate
def test_reverify_rejects_a_metal_the_moved_poses_reached():
    """
    A metal outside the input shell can fall inside it once minimisation has moved the pose.
    """
    prepared = reduced(STASHED)
    prepared.poses = onto(prepared.poses, stashed(prepared, metal=True)[0])

    with pytest.raises(OutOfScopeError) as rejected:
        prepared._reverify()

    assert rejected.value.error_type is OutOfScopeErrorType.METAL


@pytest.mark.long_protonate
def test_reverify_rejects_a_heterogen_the_moved_poses_reached():
    prepared = reduced(STASHED)
    myristoyl = next(position for name, position, _ in prepared.deleted if name == MYA)
    prepared.poses = onto(prepared.poses, myristoyl)

    with pytest.raises(OutOfScopeError) as rejected:
        prepared._reverify()

    assert rejected.value.error_type is OutOfScopeErrorType.HETEROGEN


@pytest.mark.long_protonate
def test_reverify_rejects_a_split_metal_coordination_sphere():
    """
    This rule reads the retained residues rather than the poses, so the stashed metal is put on one
        rather than the poses moved onto the metal.
    """
    prepared = reduced(STASHED)
    retained = next(
        (atom.pos.x, atom.pos.y, atom.pos.z)
        for chain in prepared.reduced
        for residue in chain
        for atom in residue
        if not atom.element.is_hydrogen
    )
    prepared.deleted = [("MG", np.array(retained), True)]

    with pytest.raises(OutOfScopeError) as rejected:
        prepared._reverify()

    assert rejected.value.error_type is OutOfScopeErrorType.SPLIT_METAL_COORDINATION


@pytest.mark.long_protonate
def test_reverify_rejects_a_zero_occupancy_residue_the_cutout_reached():
    """
    _fix destroys the occupancies, so this rule reads a stash of every residue in the structure
        holding a zero-occupancy heavy atom.

    Which residue is retained depends on where minimisation left the poses, so one of the retained
        ones is put into the stash rather than a fixture hunted for that happens to have a
        zero-occupancy atom just outside its shell.
    """
    prepared = reduced(STASHED)
    assert isinstance(prepared.unoccupied, set)
    retained = next(
        (chain.name, residue.seqid.num, residue.seqid.icode)
        for chain in prepared.reduced
        for residue in chain
        if residue.name not in {"ACE", "NME"}
    )
    prepared.unoccupied = {retained}

    with pytest.raises(OutOfScopeError) as rejected:
        prepared._reverify()

    assert rejected.value.error_type is OutOfScopeErrorType.ZERO_OCCUPANCY
