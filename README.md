# Docking

This repository contains experimental code for preprocessing for SAPT0 and active-space SAPT(VQE).

## Pre-processing

Given a protein structure and candidate poses, `encode.py` encodes the protein and reduces the problem to a tractable active space for SAPT with VQE/CASCI. The implementation aims to map as closely as possible to the original SAPT(VQE) paper, while remaining applicable to any protein-ligand complex and across several candidate poses. The encoding and compression process follow. Deviations from the original paper are marked with (*), although an extended list of these deviations are detailed in the next section.

### Encoding

Protein structure is given by $A = \set{element_I, R_I}^{N_{A}}_{I=1}$.

1. Protonate the entire protein, mapping sites, $pK_{a}$, to their protonation state with PROPKA.
    1. (*) The original paper used Protonate3D.
1. Given a protein, $A$, and candidate poses, $\set{B_i}$, truncate $A$ to complete residues which contain at least one atom within 4.5 Å of a pose, and cap the truncated protein with ACE/NME caps, producing $A^{\cup}$. Possibly covered with SparcleQC, or MDAnalysis/RDKit with additional tooling/custom code for capping.
    1. (*) The original paper uses MOE, which is commercial, along with manual pruning, and a native ligand, instead of a union over poses.
1. Calculate the net charge, $q_{A}$, given $A^{\cup}, pK_{a}$.
1. Verify that the number of electrons $$N_{e} = \sum_{I} Z_{I} - q_{A}$$ is even, so that RHF has $N_{\alpha} = N_{\beta} = N_{e}/2$.
    1. Spin and multiplicity are assumed ($S = 0, M = 1$), so the number of electrons must be even ($N_{\alpha} = N_{\beta} = N_{e}/2$).

### Compression

1. Solve the Restricted Hartree-Fock (RHF) equations, given $A^{\cup}$ (`.xyz` file), $q_{A}$, $S$, and a `basis` (default: "6-31G"). PySCF returns the molecular orbital (MO) coefficients, occupations, orbital energies, etc.
1. Run Atomic Valence Active Space (AVAS) over the MOs, returning the number of active orbitals, active electrons, and transformed molecular orbitals. PySCF supports this on the SCF object. AVAS requires targeted atomic orbitals.
    1. (*) The original paper selected a system-specific set of Fe $3d$, O $2p$, and N $2p$ orbitals using chemical knowledge of KDM5A. In this work, we provide a minimal deterministic default for arbitrary nonmetallated protein pockets.
1. Run Semistochastic Heat-Bath Configuration Interaction (SHCI) with Dice on AVAS active space, and restrict orbitals to $\text{lo} \le n_{i} \le \text{hi}$.
    1. (*) Matching the original occupation window does not guarantee $(8e, 8o)$, as in the original paper. For now, complexes yielding hardware-infeasible active spaces will be ignored.
1. Finally, map the active-space fermionic Hamiltonian to a qubit Hamiltonian following the Jordan-Wigner transformation.

#### Chemically relevant atomic valence orbitals

**Note**: this is not the focus of this study. The aim is only to provide a reproducible MVP for generating plausible AVAS target atomic orbitals from an arbitrary prepared protein cutout and its candidate-pose ensemble. The molecular orbitals are not used to decide which atoms are chemically relevant; AVAS uses the chosen atomic orbitals to identify the corresponding molecular-orbital subspace.

The default rule is:

1. For every non-cap protein heavy atom $a$ in the fixed cutout $A^{\cup}$, calculate its minimum distance to any ligand heavy atom in the preregistered candidate-pose ensemble:
   $$
   d_a = \min_{i,b\in B_i^{\mathrm{heavy}}}\lVert\mathbf R_a-\mathbf R_b\rVert.
   $$
1. Keep atoms with $d_a \le 4.5$ Å whose element has one of the following target valence shells:
   ```python
   VALENCE_P_SHELL = {"N": "2p", "O": "2p", "S": "3p"}
   ```
   This includes backbone and side-chain atoms but excludes hydrogens and ACE/NME cap atoms.
1. Generate an atom-specific PySCF AVAS label for each retained atom using its final, fixed, zero-based PySCF atom index, for example `"12 N 2p"`. Sort targets by atom index, remove duplicates, and verify that every label resolves to the expected reference atomic orbitals before running AVAS.
1. Run AVAS with fixed and recorded settings. The initial implementation uses the PySCF defaults `threshold=0.2`, `minao="minao"`, `with_iao=False`, and `canonicalize=True`.

This MVP deliberately does not infer aromatic-carbon $p$ orbitals, conjugated groups, metal oxidation states, or metal valence $d$ shells. If the automatic target set is empty, or if the pocket contains a transition metal, the automatic method must stop and require an explicit `target_aos` override rather than silently inventing targets. The KDM5A reproduction therefore uses the paper's explicit Fe/O/N target set and tests the original procedure; it does not test the nonmetallated automatic-target heuristic.

### Deviations from SAPT(VQE) Original Paper

The following deviations apply relative to the KDM5A workflow in the original [SAPT(VQE) paper](https://doi.org/10.1039/D1SC05691C) and its [supporting information](https://www.rsc.org/suppdata/d1/sc/d1sc05691c/d1sc05691c6.pdf):

- **Binding-site definition:** the original model was cut at 4.5 Å from crystallographic ligand 5 in PDB 6BH4. This work takes the union of complete residues within 4.5 Å of every preregistered candidate pose so that one fixed protein cutout can be used across the pose ensemble.
- **Protein preparation:** the original used MOE to repair missing side chains and a chain break, add ACE/NME caps, run Protonate3D, and perform tethered minimizations. This work aims to use an open-source, deterministic preparation pipeline based on PROPKA and additional capping/preparation tooling.
- **Pruning:** the original manually removed residues and individual side-chain or backbone atoms after the distance cut. This work retains complete selected residues and does not perform subjective manual pruning.
- **Coordinates across candidates:** the original performed ligand-specific relaxation and, for one ligand, reran Protonate3D to optimize the hydrogen-bond network. This work constructs and freezes one $A^{\cup}$ across all poses; input heavy-atom pose coordinates are not changed by the encoding/compression MVP.
- **Waters:** the original retained three manually selected crystallographic waters. The general workflow requires the retained-water set to be declared explicitly. The KDM5A validation below keeps only the two metal-coordinating waters, A/714 and A/726, and therefore does not reproduce the original third-water choice.
- **AVAS targets:** the original selected Fe $3d$ orbitals and particular O $2p$ and N $2p$ orbitals from the metal centre, two waters, glutamate, and histidines. The automatic MVP instead targets atom-specific N $2p$, O $2p$, and S $3p$ shells using the distance rule above and requires an explicit override for metalloproteins.
- **Electronic-structure software:** the original used TeraChem/Lightspeed for classical SCF and integral generation, Gaussian for structural calculations, and in-house quantum code. This implementation substitutes PySCF and Dice where possible.
- **Final active-space size:** the original SHCI natural-orbital occupation window $0.02\le n_i\le1.97$ produced $(8e,8o)$ for KDM5A. The same window is retained here, but its output size is system-dependent; no automatic truncation to eight orbitals is attributed to the original method.

### KDM5A Reproduction

This is a validation of the encoding and compression procedure, not an attempt to reproduce the paper's coordinates, energies, or SAPT(VQE) results. The goal is to determine whether the open-source pipeline reduces a related KDM5A model to the same *scale* of active space as the paper. The paper's $(36e,27o)$ AVAS space and $(8e,8o)$ SHCI-selected space are reference points, not required exact outputs.

1. Use [`src/data/6BH4_KDM5A.pdb`](src/data/6BH4_KDM5A.pdb), the deposited [RCSB 6BH4 structure](https://www.rcsb.org/structure/6BH4), as the immutable source structure. It contains KDM5A, crystallographic ligand DQS in chain A residue 601, Mn in chain A residue 602, and crystallographic waters.
1. Obtain the crystallographic DQS coordinates as an SDF from the [RCSB ligand model endpoint](https://models.rcsb.org/v1/6bh4/ligand?auth_asym_id=A&auth_seq_id=601&encoding=sdf). Use this single crystallographic pose in place of a DiffDock pose ensemble when defining the validation cutout.
1. Work on a prepared copy rather than editing the deposited PDB. Replace Mn A/602 with Fe(III) for the quantum-chemistry model, as in the paper, and pass the coordinating waters as `retained_water_ids=["A/714", "A/726"]`. Discard the remaining crystallographic waters for this MVP and record that choice.
1. Repair and protonate the full receptor at the fixed pH, then select complete protein residues with a heavy atom within 4.5 Å of DQS. Retain the Fe centre and the two declared waters, add ACE/NME caps at peptide cuts, and do not apply the paper's additional manual atom/residue pruning.
1. Remove DQS from the protein monomer before RHF. Export the final atom order, coordinates, charge, spin, multiplicity, retained source residue IDs, cap flags, retained water IDs, and source-to-final atom-index map. The index map is required because AVAS labels refer to the final zero-based PySCF atom indices, not PDB serial numbers.
1. Run singlet RHF/6-31G in PySCF and retain the `Mole` and converged SCF objects, MO coefficients, occupations, and orbital energies.
1. Supply an explicit, atom-specific KDM5A AVAS target list after preparation:
   - Fe A/602: $3d$;
   - O atoms of HOH A/714 and HOH A/726: $2p$;
   - coordinating O of GLU A/485: $2p$;
   - coordinating N atoms of HIS A/483 and HIS A/571: $2p$.

   Resolve each source atom through the final index map and verify every generated PySCF label before AVAS. Do not use the generic nonmetallated target generator for this model.
1. Run AVAS with the fixed recorded settings (`threshold=0.2`, `minao="minao"`, `with_iao=False`, and `canonicalize=True`) and report the resulting active electrons and spatial orbitals. A result different from $(36e,27o)$ is expected because the preparation and coordinates differ.
1. Run Dice SHCI in the complete AVAS space with `eps1=1e-4`, construct the spin-summed one-particle RDM, diagonalize it to obtain natural orbitals and occupations, and select orbitals satisfying $0.02\le n_i\le1.97$. Report the selected electron and spatial-orbital counts without manually truncating them to eight.
1. Treat a final space in the same small-space regime as $(8e,8o)$—roughly 6–12 spatial orbitals—as a successful MVP validation. If it is substantially larger, diagnose the preparation, explicit target resolution, AVAS settings, SHCI convergence, and occupation spectrum; do not change the occupation window after seeing the result merely to force a match.
