"""
Net charge, electron count, and geometry export for PrepareComplex.

_calculate_charge, _verify_num_electrons and xyz write nothing back into the cutout.

The charge is determined by the set of hydrogens the residue holds.

The AVAS default rule addresses target orbitals by zero-based PySCF atom index, and PySCF indexes 
    in file order, so we traverse back from an index to a residue by walking `self.reduced` in the 
    same order the file was written in.

Three complexes cover the cases, all accepted by _reduce:

    5S8I_2LY    one charged side chain and no free terminus, the smallest cutout of the three
    7WPW_F15    four arginines and a lysine against two aspartates, a cutout that is net positive
    6TW5_9M2    chain C is kept to residue 410, an uncapped C-terminus carrying a fourth charge
"""

import functools

import pytest
from pyscf import gto

from conftest import paths
from prepare import PrepareComplex, PrepareError

SIMPLE = "5S8I_2LY"
CHARGED = "7WPW_F15"
TERMINUS = "6TW5_9M2"

COMPLEXES = [SIMPLE, CHARGED, TERMINUS]

# What each cutout comes to once every residue is counted. 6TW5_9M2 is -1 from its side chains and
# -1 again from the terminus.
NET_CHARGE = {SIMPLE: -1, CHARGED: 3, TERMINUS: -2}

CAPS = frozenset({"ACE", "NME"})

# 6TW5_9M2 keeps LEU C410, where chain C ends. _reduce leaves a chain end uncapped, so protonation's
# OXT and the missing HXT stay, and the carboxylate keeps its charge.
FREE_TERMINUS = ("C", 410)


@functools.lru_cache(maxsize=None)
def prepare(name):
    """
    A complex carried to the end of _reduce.

    Cached because the pipeline is far slower than the three methods under test, and none of them
        changes the structure they read.
    """
    _prepare = PrepareComplex(*paths(name))
    _prepare._fetch()
    _prepare._verify()
    _prepare._fix()
    _prepare._clean()
    _prepare._protonate()
    _prepare._reduce()
    return _prepare


def formal_charge(model, termini=True):
    """
    The charge of a cutout read off the hydrogens each residue holds.

    An independent reading of the same chemistry the implementation is asked for: arginine is
        charged whatever the pH, lysine is charged while it keeps all three hydrogens on NZ,
        aspartate and glutamate are charged while they lack the one on the carboxylate, and
        histidine runs from -1 to +1 on the two it can carry. A cap is neutral. A chain end that
        kept an OXT without an HXT carries the charge protonation left on it.
    """
    total = 0
    for chain in model:
        residues = list(chain)
        for position, residue in enumerate(residues):
            if residue.name in CAPS:
                continue
            names = {atom.name for atom in residue}
            if residue.name == "ARG":
                total += 1
            elif residue.name == "LYS":
                total += {"HZ1", "HZ2", "HZ3"} <= names
            elif residue.name == "ASP":
                total -= "HD2" not in names
            elif residue.name == "GLU":
                total -= "HE2" not in names
            elif residue.name == "HIS":
                total += ("HD1" in names) + ("HE2" in names) - 1
            if not termini:
                continue
            if position == 0 and {"H2", "H3"} <= names:
                total += 1
            if position == len(residues) - 1 and "OXT" in names and "HXT" not in names:
                total -= 1
    return total


def nuclear_charge(model):
    return sum(
        atom.element.atomic_number
        for chain in model
        for residue in chain
        for atom in residue
    )


def atoms(model):
    return [
        (atom.element.name, (atom.pos.x, atom.pos.y, atom.pos.z))
        for chain in model
        for residue in chain
        for atom in residue
    ]


def written(path):
    """
    An .xyz file as (element, coordinates) in the order it was written.
    """
    lines = path.read_text().splitlines()
    return [
        (fields[0], tuple(float(value) for value in fields[1:4]))
        for fields in (line.split() for line in lines[2:])
    ]


@pytest.mark.parametrize("name", COMPLEXES)
def test_calculate_charge_counts_every_charged_residue(name):
    """
    q_A is the sum of the formal charges of what the cutout holds.
    """
    prepared = prepare(name)

    prepared._calculate_charge()

    assert prepared.charge == formal_charge(prepared.reduced)
    assert prepared.charge == NET_CHARGE[name]


def test_calculate_charge_counts_a_free_terminus():
    """
    An uncapped chain end brings its charge into q_A.
    """
    prepared = prepare(TERMINUS)
    chain_name, seqid = FREE_TERMINUS
    terminus = [
        residue
        for chain in prepared.reduced
        for residue in chain
        if (chain.name, residue.seqid.num) == (chain_name, seqid)
    ]
    assert terminus and "OXT" in {atom.name for atom in terminus[0]}

    prepared._calculate_charge()

    assert formal_charge(prepared.reduced, termini=False) == -1
    assert prepared.charge == -2


@pytest.mark.parametrize("name", COMPLEXES)
def test_verify_num_electrons_accepts_an_even_count(name):
    """
    N_e = sum_I Z_I - q_A, over every atom of the cutout, hydrogens included.
    """
    prepared = prepare(name)
    prepared._calculate_charge()

    prepared._verify_num_electrons()

    assert prepared.electrons == nuclear_charge(prepared.reduced) - prepared.charge
    assert not prepared.electrons % 2


def test_verify_num_electrons_rejects_an_odd_count():
    """
    An odd count means the preparation is wrong.
    """
    prepared = prepare(SIMPLE)
    prepared._calculate_charge()
    prepared.charge += 1

    with pytest.raises(PrepareError):
        prepared._verify_num_electrons()


@pytest.mark.parametrize("name", COMPLEXES)
def test_xyz_writes_every_atom_of_the_cutout(name, tmp_path):
    """
    The file holds the whole cutout and nothing else, hydrogens included.
    """
    prepared = prepare(name)
    prepared._calculate_charge()
    path = tmp_path / "cutout.xyz"

    prepared.xyz(str(path))

    lines = path.read_text().splitlines()
    expected = atoms(prepared.reduced)
    assert int(lines[0].split()[0]) == len(expected)
    assert len(lines) == len(expected) + 2


@pytest.mark.parametrize("name", COMPLEXES)
def test_xyz_keeps_the_order_of_the_cutout(name, tmp_path):
    """
    The file is written in the order `self.reduced` iterates, and the coordinates are unchanged.
    """
    prepared = prepare(name)
    prepared._calculate_charge()
    path = tmp_path / "cutout.xyz"

    ordered = prepared.xyz(str(path))

    expected = atoms(prepared.reduced)
    assert len(ordered) == len(expected)
    for (element, coordinates), (symbol, position), (_, _, atom) in zip(
        written(path), expected, ordered
    ):
        assert element == symbol == atom.element.name
        assert coordinates == pytest.approx(position, abs=1e-3)


def test_xyz_records_the_charge_and_multiplicity(tmp_path):
    """
    An .xyz carries no charge of its own. The comment line fills this gap.
    """
    prepared = prepare(TERMINUS)
    prepared._calculate_charge()
    path = tmp_path / "cutout.xyz"

    prepared.xyz(str(path))

    comment = path.read_text().splitlines()[1]
    assert f"charge={prepared.charge}" in comment
    assert f"multiplicity={prepared.multiplicity}" in comment


def test_xyz_is_read_back_by_pyscf(tmp_path):
    """
    What is written is what compression is handed.
    """
    prepared = prepare(SIMPLE)
    prepared._calculate_charge()
    prepared._verify_num_electrons()
    path = tmp_path / "cutout.xyz"

    ordered = prepared.xyz(str(path))

    mol = gto.M(atom=str(path), charge=prepared.charge, spin=prepared.spin, basis=prepared.basis)
    assert mol.natm == len(ordered)
    assert mol.nelectron == prepared.electrons
    assert [mol.atom_symbol(i) for i in range(mol.natm)] == [a.element.name for _, _, a in ordered]


def test_xyz_refuses_a_structure_that_was_never_reduced():
    """
    Writing the whole protein where the cutout was meant would only show in the cost of the SCF.
    """
    prepared = PrepareComplex(*paths(SIMPLE))

    with pytest.raises(PrepareError):
        prepared.xyz("cutout.xyz")
