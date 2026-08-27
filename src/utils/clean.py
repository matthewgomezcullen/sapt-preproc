import gemmi


def strip(model):
    """
    Delete everything that is not part of the polypeptide: heterogens, waters, free ions, and the
        hydrogens that came with the deposited structure.

    _verify has already rejected any complex whose cutout holds a heterogen or a metal, so what is
        deleted here lies outside the cutout and cannot change the retained region. Hydrogens go so
        that _protonate owns every hydrogen in the model and assigns it at one consistent pH.

    gemmi decides what is polymer from entity types rather than residue names, so an ACE or NME
        terminating a real chain is kept while a chemically identical free acetate is not.
    """
    structure = gemmi.Structure() # pyright: ignore[reportAttributeAccessIssue]
    structure.add_model(model)

    # Entity types are not carried by a structure read from a PDB, and are what decides polymer
    # membership below.
    structure.setup_entities()
    structure.remove_ligands_and_waters()
    structure.remove_hydrogens()

    # Repair splits heterogens into chains that reuse the polymer chain names, so deleting them
    # leaves duplicate empty chains behind and `verify.identifier` stops being unique.
    structure.remove_empty_chains()
    return structure[0]
