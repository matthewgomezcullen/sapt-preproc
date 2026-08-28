import io

import gemmi
from openmm.app import Modeller, PDBFile

from utils import protonate


CAPS = frozenset({"ACE", "NME"})

# What a cap keeps of the residue it stands in for, under the atom names OpenMM's hydrogen
# definitions expect. An ACE is that residue's carbonyl and its CA turned into a methyl; an NME is
# its amide and, again, its CA. NME's methyl carbon is "C" there rather than "CH3".
ACE = {"C": "C", "O": "O", "CA": "CH3"}
NME = {"N": "N", "H": "H", "CA": "C"}


def _identifier(chain, residue):
    """
    The OpenMM spelling of `verify.identifier`.
    """
    return (chain.id, int(residue.id), residue.insertionCode)


def _bridge(topology, keep):
    """
    `keep` widened to swallow every residue sitting alone between two kept ones.

    Capping around such a residue would take its backbone into an NME on one side and an ACE on the
        other, placing its CA twice at one point. Widening cannot open a new gap of the same shape:
        a residue left out with both neighbours in the result has both of them in `keep`, and would
        have been swallowed here.
    """
    bridged = set(keep)
    for chain in topology.chains():
        residues = [_identifier(chain, residue) for residue in chain.residues()]
        for before, gap, after in zip(residues, residues[1:], residues[2:]):
            if before in keep and after in keep:
                bridged.add(gap)
    return bridged


def _cap(residue, name):
    """
    Turn a residue into the cap that stands in for it, returning the atoms left over.

    The cap is not built, it is what remains of the residue the truncation removed, so it carries
        that residue's own coordinates and the peptide geometry is untouched.
    """
    wanted = ACE if name == "ACE" else NME
    names = {atom: atom.name for atom in residue.atoms()}
    doomed = [atom for atom in residue.atoms() if names[atom] not in wanted]
    for atom in residue.atoms():
        if names[atom] in wanted:
            atom.name = wanted[names[atom]]
    residue.name = name
    return doomed


def truncate(model, keep, protonation, pH, seed):
    """
    The residues of `keep`, bridged and capped, with everything else deleted.

    A cut is capped and a chain end is not: a cap stands in for a removed residue, and at a chain
        end there is no such residue. Modeller decides that the same way, by position in the chain,
        so a run that starts at one keeps the terminus protonation gave it while a run that starts
        at a cut is left alone behind its ACE.

    The protonation states are passed back in so that Modeller reaches the same tautomer for every
        residue it already decided, and adds nothing but the hydrogens of the caps.
    """
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)
    structure.setup_entities()
    pdb = PDBFile(io.StringIO(structure.make_pdb_string()))
    modeller = Modeller(pdb.topology, pdb.positions)

    kept = _bridge(modeller.topology, keep)
    doomed = []
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        identifiers = [_identifier(chain, residue) for residue in residues]
        for position, residue in enumerate(residues):
            if identifiers[position] in kept:
                continue
            if position + 1 < len(residues) and identifiers[position + 1] in kept:
                doomed.extend(_cap(residue, "ACE"))
            elif position and identifiers[position - 1] in kept:
                doomed.extend(_cap(residue, "NME"))
            else:
                doomed.append(residue)
    modeller.delete(doomed)

    # A cap has no protonation state, and the one belonging to the residue it replaced would be
    # rejected as illegal for an ACE or an NME.
    variants = [
        None if residue.name in CAPS else protonation.get(_identifier(residue.chain, residue))
        for residue in modeller.topology.residues()
    ]
    protonate.add(modeller, pH, seed, variants)

    buffer = io.StringIO()
    PDBFile.writeFile(modeller.topology, modeller.positions, buffer, keepIds=True)
    return gemmi.read_pdb_string(buffer.getvalue())[0] # pyright: ignore[reportAttributeAccessIssue]
