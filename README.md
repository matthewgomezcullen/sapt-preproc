# Docking

This repository contains experimental code for proprocessing for SAPT0 and active-space SAPT(VQE).

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
    1. (*) The original paper hardcoded `['Fe 3d', 'O 2p', ...]`. In this work, we provide a rough method for automatically estimating chemically relevant atomic valence orbitals.
1. Run Semistochastic Heat-Bath Configuration Interaction (SHCI) with Dice on AVAS active space, and restrict orbitals to $\text{lo} \le n_{i} \le \text{hi}$.
    1. (*) Matching the original occupation window does not guarantee $(8e, 8o)$, as in the original paper. For now, complexes yielding hardware-infeasible active spaces will be ignored.
1. Finally, map the active-space fermionic Hamiltonian to a qubit Hamiltonian following the Jordan-Wigner transformation.

#### Chemically relevant atomic valence orbitals

**Note**: this is not the focus of this study. We should provide an MVP for generating relevant atomic valence orbitals given

Example proposal, to be replaced:

Target valence $p$ orbitals on chemically relevant N/O/S atoms in the fixed cutout. Use a fixed element rule, such as `N: 2p, O: 2p, S: 3p`. Chemically relevant atoms may be defined as atoms that can directly participate in intermolecular chemistry -- initially all N/O/S side-chain atoms plus aromatic $sp^2$ carbons belonging to residues that contact the candidate-pose union—and then let AVAS + SHCI occupations decide which corresponding MOs survive.

### Deviations from SAPT(VQE) Original Paper
