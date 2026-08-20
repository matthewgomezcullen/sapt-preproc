# Research plan: can a quantum-derived electronic signal improve DiffDock-L pose ranking?

**Scope date:** 11 August 2026  
**Decision this document supports:** whether to implement a small post-hoc reranking study before attempting quantum-guided diffusion.

## Executive decision

The reranking question is scientifically sensible, but only in a narrower form than “quantum binding-affinity prediction.” A fixed-pose electronic interaction energy can contain pose information without being a binding free energy. It can reward electrostatic, exchange, induction, and dispersion complementarity, yet omit ligand deformation, receptor reorganization, desolvation, protonation equilibria, and entropy. It must therefore be tested as one feature in a pose ranker, with steric-validity and ligand-strain controls, rather than interpreted as $\Delta G_{\mathrm{bind}}$.

The best *physical target* in the supplied literature is second-order SAPT(VQE), because it computes an interaction energy directly and contains electrostatics, exchange, induction, dispersion, and the exchange counterparts. It avoids estimating a small interaction as the difference of three large noisy total energies. It is not, however, ready for hundreds of protein-pocket poses: its protein-scale evidence is absent, all results are ideal statevector simulations, and the prototype ERPA/SAPT post-processing was limited to roughly 130 spatial orbitals ([SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §§III–IV; Appendix D, especially the paragraph following Eq. C1).

The recommended first implementation is therefore a **SAPT hierarchy**:

1. compute a conventional second-order SAPT0-like score on every fixed candidate pose;
2. replace its first-order electrostatic-plus-exchange part with an active-space SAPT(VQE) estimate, equivalently

   $$
   E_{\mathrm{hyb}}^{Q}
   = E_{\mathrm{SAPT0}}
   + \left[E_{\mathrm{elst+exch}}^{\mathrm{VQE}}
   - E_{\mathrm{elst+exch}}^{\mathrm{RHF}}\right];
   $$

3. evaluate the identical correction with CASCI, which is the exact active-space classical control; and
4. run full second-order SAPT(VQE) only on a very small size-qualified validation subset if the available implementation can do so.

This mixed-level score is a **proposal**, not a method demonstrated in either SAPT paper. Both terms inside the square-bracket correction must use the same monomer-centred basis, orbitals, active-space convention, and exchange approximation; it is a delta correction to the chosen SAPT0 base, not a direct subtraction of raw components computed in incompatible bases. Its virtue is diagnostic separation: SAPT0 tests whether a direct physical interaction score helps; CASCI-minus-RHF tests whether active-space correlation changes ranking; VQE-minus-CASCI tests only quantum-estimation error. Keeping the protein monomer and its active space fixed across all poses of a complex permits one protein VQE/RDM calculation to be reused across all $N$ poses when a monomer-centred basis is used. The ligand can remain RHF in experiment 1, as in the published KDM5A calculation ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.C, Fig. 4 and Tables VII–VIII).

The smallest convincing pilot is **24 deterministically selected PoseBusters Benchmark complexes, holo receptors, and 20 frozen DiffDock-L poses per complex**. It is a conditional reranking study, not an end-to-end claim about apo docking. The principal endpoint is top-1 success among complexes for which DiffDock-L actually sampled a pose below 2 Å. A classical same-method baseline is mandatory. Real-QPU work is optional and cannot establish advantage.

**Go/no-go summary.** Proceed to implementation only if the project accepts the likely negative outcome: the physical SAPT0 signal may help but the active-space quantum correction may be negligible. Do not make per-step quantum guidance experiment 1; even a central finite-difference implementation costs $2N T(m+6)$ electronic-energy evaluations—9,600 evaluations for $N=20$, $T=20$, and $m=6$—before accounting for VQE iterations or shots.

## 0. Evidence protocol and literature inventory

The supplied Markdown files were treated as the main evidence. Titles, headings, abstracts, conclusions, algorithms, tables, and targeted terms were inspected first; only decision-relevant sections were then read closely. No PDF was consulted.

| Local paper | Primary role in this plan | Most relevant locations |
|---|---|---|
| [DiffDock.md](literature/mds/DiffDock.md) | DiffDock manifold, score model, reverse process, confidence | §§3–4.5; Appendix B, Algorithms 1–4; Appendix C; §§D.3, E.1–E.2, F.1/F.3 |
| [DiffDock-L.md](literature/mds/DiffDock-L.md) | DockGen, scaled model, Confidence Bootstrapping | §§3–5; Appendix B; Appendix C.1–C.4; Appendix E |
| [CompassDock.md](literature/mds/CompassDock.md) | Empirical energy, strain/clash assessment, fine-tuning comparator | §§2.2–2.4, §4, Fig. 4; Appendix B, Table 1 |
| [Q-Score.md](literature/mds/Q-Score.md) | Closest “quantum docking score” prior work | §IV.A–F, Eqs. in §§IV.B–F; §V, Tables I–VI and Figs. 1–5 |
| [FlowDock.md](literature/mds/FlowDock.md) | Flexible co-folding and learned affinity context | §§3.1, 3.4–3.6, 4.1–4.3; Appendix A, Algorithms 1–2 |
| [PoseBench.md](literature/mds/PoseBench.md) | Modern method comparison and benchmark design | §§3.1–3.5, 5.1–5.5; Appendices D–G |
| [PoseBusters.md](literature/mds/PoseBusters.md) | Physical validity and post-2021 benchmark | §§2.2–2.6, 3.2–3.3; Supplement §§S3–S8 |
| [SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md) | First-order SAPT(VQE), KDM5A demonstration | §§I.B–I.E, II.C, III; Appendix B; SI §III and Tables VI–VIII |
| [SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md) | Second-order SAPT(VQE)/ERPA | §§II–IV; Appendices B–D; Table SII |
| [DMET_VQE_PL.md](literature/mds/DMET_VQE_PL.md) | Actual protein–ligand DMET-VQE hardware study | §§2.1–2.7, 3.1–3.5; Appendix 6.1 |
| [DMET_SGD.md](literature/mds/DMET_SGD.md) | DMET plus sample-based quantum diagonalization (the filename says SGD) | §II, especially “Density matrix embedding theory” and “Sample-based quantum diagonalization”; §§III–V, Table I |
| [QCPMD.md](literature/mds/QCPMD.md) | Quantum nuclear forces and classical shadows | §II.A.2–3, §II.B, Eqs. 2–10; §III, Figs. 2–4 |

Four material gaps required primary-source web checks; they were not covered by the supplied Markdown corpus:

- Boltz-2 architecture and affinity scope: [Boltz-2 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/), Abstract, §§1 and 4–5.
- ChemGuide and BADGER as named guidance comparators: [ChemGuide, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1ba3a079c5922a70929316fda4389eb8-Abstract-Conference.html), §§3–4 and Algorithm 1; [BADGER preprint](https://arxiv.org/abs/2406.16821), Abstract and method.
- Fermionic-shadow RDM scaling: [Low, *Classical shadows of fermions with particle number symmetry*](https://arxiv.org/abs/2208.08964), abstract and main theorem.
- Quantum analytic gradients and ext-SQD: [Hohenstein et al., quantum analytic nuclear gradients](https://pubmed.ncbi.nlm.nih.gov/36948843/), abstract; [Barison et al., extended SQD](https://arxiv.org/abs/2411.00468), abstract. A very recent hardware-gradient preprint is treated cautiously later.

### Evidence labels

- **Demonstrated** means reported in a cited paper, within its stated system and conditions.
- **Deduction** means a consequence inferred here; it must not be attributed to the authors.
- **Proposal** means an experimental choice to be pre-registered and tested.

## 1. The quantities must not be conflated

| Quantity | Operational meaning here | Important omissions or caveats |
|---|---|---|
| Native-pose confidence | A learned estimate or ranking proxy for whether a pose has symmetry-corrected RMSD below 2 Å | It is supervised by crystal coordinates; it is neither energy nor affinity. |
| Pose-dependent electronic interaction energy | Electronic interaction of two frozen monomers at one geometry, preferably decomposed by SAPT | Does not include monomer deformation, full solvent, configurational entropy, or receptor motion. |
| Binding energy | An energy difference between a bound complex and separated components under a specified electronic/embedding model | Still normally excludes thermal and entropic terms; definition changes with geometry and environment. |
| Experimental binding affinity / $\Delta G_{\mathrm{bind}}$ | A solution-phase free-energy difference, related to an equilibrium constant under specified conditions | Includes solvent, protonation, conformational ensembles, receptor/ligand reorganization, and entropy. |
| Ligand strain | Internal energy cost of adopting the bound conformation relative to a reference relaxed conformation | Reference state and force field matter; it is not an intermolecular interaction. |
| Steric validity | Whether geometry passes bond, planarity, clash, overlap, and related checks | A necessary filter, not evidence that a pose is native. |

**Demonstrated.** DiffDock confidence is trained on a binary RMSD-below-2-Å label, whereas the SAPT studies compute frozen-geometry interaction terms. The KDM5A SAPT paper explicitly warns that its electronic interaction-energy differences are not the experimental binding-free-energy differences and notes missing thermodynamic-cycle and water effects ([DiffDock.md](literature/mds/DiffDock.md), §4.4; [SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.C around Fig. 4).

**Deduction.** An interaction score can still rank poses: translation, orientation, and torsions alter intermonomer integrals and therefore electrostatics, exchange, induction, and dispersion. But a strained or poorly solvated decoy can have a favorable intermolecular energy. Even exact frozen-geometry electronic energy is not guaranteed to be monotonic in ligand RMSD. The appropriate hypothesis is incremental, conditional information—not recovery of affinity.

## 2. DiffDock and DiffDock-L reconstructed for this project

### 2.1 DiffDock

#### Inputs and seed geometry

**Demonstrated.** DiffDock takes a protein 3D structure $y$, ligand identity/graph, and an isolated seed conformer $c$. In inference that conformer is normally generated with RDKit ETKDG. Bond lengths, bond angles, and small rings are held as represented in $c$; freely rotatable torsions are the ligand’s internal degrees of freedom. During training, conformer matching replaces the crystal ligand with the closest pose reachable from an RDKit conformer so training and inference share local geometry ([DiffDock.md](literature/mds/DiffDock.md), §§4.1–4.2; Appendix B before Algorithm 1).

#### Pose space and actions

For a ligand with $m$ rotatable bonds, the reachable pose manifold $\mathcal M_c\subset\mathbb R^{3n}$ has $m+6$ dimensions and is parameterized by

$$
\mathcal P=T(3)\times SO(3)\times SO(2)^m,
\qquad
T_g\mathcal P\simeq\mathbb R^3\oplus\mathbb R^3\oplus\mathbb R^m.
$$

Translation adds one vector to every ligand atom. Rotation acts around the **unweighted atomic centroid** $\bar x=n^{-1}\sum_i x_i$, called the centre of mass in the paper. A torsion update may initially rotate either side of a rotatable bond, after which the whole ligand is RMSD-aligned to the pre-update coordinates. This defines the minimum-RMSD torsion action and makes its infinitesimal displacement orthogonal to global translation and rotation—zero linear and angular momentum. The authors note that the torsion product is not used as an exact group under composition; the direct-pose algorithm treats it approximately as one for small updates ([DiffDock.md](literature/mds/DiffDock.md), §4.2, Definition and Proposition 1; Appendix A; Appendix B before Algorithms 3–4).

#### Score model and reverse diffusion

**Demonstrated.** Independent forward diffusions are defined on translation, rotation, and torsion using a Gaussian on $T(3)$, IGSO(3) on rotations, and wrapped normals on the torsion torus. The score network receives the current 3D ligand pose, protein, and time. It outputs one equivariant translation vector, one equivariant Euler/rotation vector, and one invariant scalar per rotatable bond. Its protein representation is coarse-grained at Cα/residue level; geometric message passing and separate translation, rotation, and pseudotorque heads yield the three tangent components ([DiffDock.md](literature/mds/DiffDock.md), §§4.3–4.5; Appendix C).

Algorithm 4 is the implementation-relevant reverse sampler. It initializes ligand translation from a broad Gaussian, rotation uniformly on $SO(3)$, and torsions uniformly on their circles. At each time step, it predicts the three scores; multiplies each by that component’s variance decrement; adds component-specific Gaussian noise; and applies translation, rotation, and minimum-RMSD torsion actions directly to the current pose. Multiple independent initializations give $N$ candidate poses. The paper describes 20 nominal inference steps; implementation details report stopping at 18 and omitting noise in the final step ([DiffDock.md](literature/mds/DiffDock.md), Appendix B, Algorithms 2 and 4; Appendix D.3).

#### Confidence model and its exact inference role

**Demonstrated.** A separate, SE(3)-invariant, all-atom receptor–ligand network is trained with cross-entropy on diffusion-generated poses labeled positive iff symmetry-corrected RMSD is below 2 Å. It pools ligand representations to one scalar. During ordinary inference it sees only **completed** sampled poses and ranks the $N$ candidates by predicted probability/confidence of the positive class. It does not alter any reverse-diffusion step. The score network uses the coarse receptor; the confidence network has all receptor atoms ([DiffDock.md](literature/mds/DiffDock.md), §§4.4–4.5; Appendix C.1–C.3). Random single-sample and confidence-ranked success therefore answer different questions; the reported confidence ablations show that ranking is a material part of final accuracy ([DiffDock.md](literature/mds/DiffDock.md), Appendix F.3).

### 2.2 What DiffDock-L changed

**Demonstrated.** DiffDock-L is principally a scaling/data/generalization result, not a new physical inference architecture. The score model grew from roughly 20M to 30M parameters; training added more Binding MOAD examples within known domains and van-der-Mer-inspired synthetic protein–ligand examples. DockGen was constructed from Binding MOAD using ECOD domain clusters absent from PDBBind domains, with filters for cofactors, metals, additives, and large ligands; it contains 141 validation and 189 test complexes. This exposes domain-level leakage that a PDBBind time split misses ([DiffDock-L.md](literature/mds/DiffDock-L.md), §§3, 5.1; Appendices A–B). Table 1 reports DiffDock-L at 43% top-1 on PDBBind with ten samples but only 22.6% on full DockGen (27.6% on the clustered subset), illustrating the generalization gap.

DiffDock-L should not be equated with the small model used in the Confidence Bootstrapping experiments. Those experiments use a faster roughly 4M-parameter DiffDock-S and modify both score and confidence architectures for iterative training. The local confidence model retains a binary task but balances positives and negatives, separates positives below 2 Å from negatives above 4 Å, adds atom-level distance supervision, and crops receptor residues whose Cα is within 20 Å of predicted ligand atoms ([DiffDock-L.md](literature/mds/DiffDock-L.md), Appendix C.2). The paper does not establish that every one of these bootstrapping-specific confidence changes is the production DiffDock-L confidence configuration; implementation work must inspect the selected repository checkpoint rather than infer it from the name.

### 2.3 Confidence Bootstrapping is not classifier guidance

**Demonstrated.** Confidence Bootstrapping repeatedly:

1. rolls the current score model all the way to final poses;
2. scores those final poses with fixed confidence model $c_\phi$;
3. retains poses above threshold $k$ in a buffer, capped per complex;
4. samples buffer entries with probability proportional to $\exp c_\phi$;
5. treats sampled final poses as pseudo-ground truth, forward-noises them to random $t$, and updates the score model by denoising score matching; and
6. mixes genuine MOAD training examples to preserve late denoising and reduce confidence-model exploitation.

The formal target is $p_{\theta,\phi}(x;d)\propto p_\theta(x;d)\exp[c_\phi(x,d)]$. Time weights emphasize high-noise examples, so final-pose selection changes the learned behavior of early reverse steps **across training iterations**. There is no differentiation of confidence through the sampled trajectory and no confidence gradient added to an inference update. The experiments used 60 rollout/update iterations with 200 SGD steps, 32 candidate rollouts per complex in normal iterations, and about eight hours on one A6000 ([DiffDock-L.md](literature/mds/DiffDock-L.md), §§4.2–4.3; Appendix C.1, C.3–C.4).

| Integration point | Can a physical/quantum oracle replace confidence? | Exact qualification |
|---|---|---|
| A. Final-pose reranking | **Yes, directly.** | It only needs a comparable scalar for each completed pose. Lower energy must be sign-converted to a higher-is-better ranking. Calibration as a probability is unnecessary for pure ranking. |
| B. Confidence Bootstrapping | **Architecturally yes, practically difficult.** | A scalar can replace $c_\phi$ in thresholding and $\exp(c)$ weighting after dimensionless scaling. The oracle must be stable over diverse, often bad rollout geometries. Tens of thousands of repeated electronic calculations would eliminate the method’s amortization unless scores are cached or distilled. |
| C. Direct per-step guidance | **Not by substitution.** | This requires $\nabla_{x_t}\log p(\text{desired}\mid x_t)$, an energy gradient, a generalized force, or a black-box estimate at every step. It changes Algorithm 4 itself and is derived in §8. Confidence Bootstrapping supplies no such per-step gradient. |

**Deduction.** Final-pose reranking is the only integration point that preserves a frozen generator, allows paired comparison on exactly the same candidate set, and keeps quantum cost finite. It is therefore the correct first experiment.

## 3. Quantum and hybrid candidates

### 3.1 Why direct SAPT is different from supermolecular subtraction

A supermolecular interaction energy is often written

$$
E_{\mathrm{int}}=E_{PL}-E_P-E_L.
$$

On a quantum device, this requires at least three state/energy calculations. If independent estimators have variances $v_{PL},v_P,v_L$, the difference has variance $v_{PL}+v_P+v_L$ before considering covariance; each large total must therefore be estimated much more accurately than the small final interaction. Variational and basis errors need not cancel, and inconsistent active spaces or embeddings can make subtraction ill-defined.

SAPT instead defines the interaction perturbatively from monomer states. In the supplied formulations, QPU-derived monomer 1- and 2-RDMs are contracted classically with intermonomer integrals to obtain named interaction components. This avoids the ordinary counterpoise/subtraction structure in which basis-set superposition and independent total-energy errors contaminate a small difference, although monomer- versus dimer-centred basis choice still affects SAPT convergence and reuse. It does not make SAPT exact—basis, active-space, response, truncation, and exchange approximations remain—and it is still an interaction energy rather than a binding free energy. In ideal simulations, both SAPT papers found that shallow, coarsely optimized VQE states produced much smaller errors in interaction components than in monomer total energies ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.B–C, Figs. 2–4 and §III; [SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §III, Figs. 2–6).

For pose ranking, SAPT has another important reuse property. With a monomer-centred basis, a rigid translation/rotation of the ligand relative to a fixed receptor changes intermonomer integrals but not either isolated monomer wavefunction. A protein RDM can be reused for every candidate, and a ligand RDM can be reused for candidates with identical internal geometry. Ligand torsion changes generally require a new ligand SCF/correlated state. A dimer-centred ghost basis, pose-dependent electrostatic embedding, or geometry-dependent orbital reoptimization reduces this reuse. The first-order paper explicitly reused a fixed monomer VQE solution along an intermolecular separation coordinate in a monomer-centred basis ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.B, Fig. 2 discussion).

### 3.2 First-order SAPT(VQE)

**Physical quantity.** $E_{\mathrm{elst}}^{(1)}+E_{\mathrm{exch}}^{(1)}$ for two frozen monomers. It is a natural scalar for one pose and is sensitive to relative placement and ligand torsions through density–integral contractions. It omits induction and dispersion, so it is not a credible standalone docking score.

**System and Hamiltonian.** Define ligand and protein-pocket monomers. Classical SCF supplies orbitals and partitions them into frozen core, active, and virtual spaces. The active monomer electronic Hamiltonian is represented in second quantization and mapped to spin-orbital qubits; the other monomer can remain RHF. The KDM5A example used a 163-atom protein cutout within a larger prepared model, a 6-31G treatment, ligand RHF, protein VQE, and a reduced (8e,8o) low-spin active space—16 qubits—from an initial (36e,27o) AVAS/SHCI selection ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.C; SI §III.A). The overall electronic problem was 1,482 electrons in 2,214 spatial orbitals, showing that the classical environment can be much larger than the qubit active space.

**Quantum/classical work.** VQE prepares a $k$-muCJ active-space state and optimizes its energy; the QPU then estimates its 1- and 2-RDM. Classical code constructs integrals and SAPT contractions. Naively the RDM measurement count scales as $O(N_a^4)$; Appendix B shows that the electrostatic contribution can be rotated into one commuting Z-basis group, while exchange is harder and may benefit from factorization ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §I.D–E; Appendix B). No shots, noisy simulation, or hardware experiment were reported: all VQE results were ideal statevectors.

**Evidence and limitation.** The paper evaluated five KDM5A inhibitors on ligand-specific, crystal-derived and relaxed structures; it did not rank multiple poses of the same complex. SAPT(VQE) closely matched SAPT(CASCI), but RHF was also close because the selected state was mainly single-reference. First-order SAPT gave the wrong ligand ordering; missing dispersion and induction were decisive ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.C, Fig. 4; SI Tables VII–VIII). The closest controls are SAPT(CASCI) with identical orbitals/active space and SAPT(RHF) with identical geometry/basis. It has no demonstrated analytic gradient.

**Verdict.** Reject as a standalone score. Retain as a reusable active-space correction to a complete classical SAPT score. This is the most executable way to test a quantum-correlated contribution on many poses without pretending first order is sufficient.

### 3.3 Second-order SAPT(VQE)

**Physical quantity.** A nearly complete low-order interaction energy containing electrostatic, exchange, induction, dispersion, and exchange-induction/dispersion components. This is the best-matched quantum observable for fixed-pose reranking in the corpus.

**Quantum/classical work.** The quantum part is still ground-state VQE and measurement of monomer 1- and 2-RDMs. Second-order response is reconstructed classically using an extended random-phase approximation: a generalized response eigenproblem plus classical response equations avoids preparing excited states on the QPU. Thus there is no per-excitation quantum call, but ERPA tensor construction and solution can dominate classically ([SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §II; Appendix B; Appendix C, especially Eq. C1 and §C.4). Either or both correlated monomers can be quantum; qubits equal twice the active spatial orbitals of each separately treated monomer, not the combined dimer.

**Resources and evidence.** Reported active spaces were (8e,8o), 16 qubits, for stretched water; (6e,6o), 12 qubits, for p-benzyne and the Mn–nitrosyl system. All were ideal statevector simulations; no shot count or hardware run exists. The biological example was a small Mn–nitrosyl complex hydrogen-bonded to HF, water, ammonia, or methane—not a protein–ligand complex. The prototype’s lack of density fitting and core/active/virtual optimization restricted it to about 130 spatial orbitals and forced mixed small basis sets on the metal model ([SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §III.A–B, Figs. 2–6; §IV; Appendix D; Table SII).

**Pose reuse and gradients.** Monomer-centred orbitals would permit the reuse described above, but this paper used dimer-centred ghost functions and held related geometries similar partly to stabilize active orbitals. RDM samples are state- and geometry-specific; one may warm-start VQE and track orbitals by maximum overlap, but cannot reuse measurements after a torsional Hamiltonian change without a quantified approximation. The paper does not derive forces; its conclusion explicitly says gradients and other properties require specialized quantum adaptation ([SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §IV; Appendix D).

**Failure modes.** Active-space selection can change discontinuously across poses; ERPA is approximate; a small basis distorts dispersion; a pocket cutout omits long-range polarization; pairwise/frozen interaction omits strain, solvent, entropy, and receptor relaxation. SAPT can also make an invalid pose appear favorable if the cutout, basis, or active treatment misses the relevant repulsion, although exact exchange itself is repulsive at overlap.

**Verdict.** Rank first as the long-run post-hoc observable and second for experiment-1 executability. Use full SAPT(VQE) only after a size gate and compare to SAPT(CASCI) on exactly the same monomer spaces.

### 3.4 DMET-VQE on a protein–ligand system

**What it actually computed.** The BACE1 study did not calculate a full protein–ligand interaction energy and did not rank poses. For each of 12 oxazine inhibitors it represented the protein by fixed AMBER10:EHT point charges, treated a ligand in that protein field, and compared it with the ligand in an explicit-water/dd-COSMO solvent environment. Protein–protein terms were assumed to cancel. The resulting relative ligand-energy proxy was compared across different ligands and experimental activities ([DMET_VQE_PL.md](literature/mds/DMET_VQE_PL.md), §§2.1–2.2, 3.1 and 3.5; Appendix 6.1).

**Hamiltonian and split.** A global RHF calculation and Löwdin localization define DMET fragments and baths; a chemical potential enforces electron count. Only a [NH$_2$CNH]$^+$ ligand head fragment was sent to VQE, while the rest of the ligand and environment were mean-field. STO-3G and a (2e,2o) active space yielded four spin-orbital qubits with a one-parameter $YXXX$ circuit and Jordan–Wigner mapping ([DMET_VQE_PL.md](literature/mds/DMET_VQE_PL.md), §§2.3–2.6).

**Calls and evidence.** Each ligand required the correlated fragment in protein and solvent environments. IBM Casablanca VQE used 6,000 shots per optimizer iteration and a final 60,000-shot estimate in ten blocks; the trapped-ion run used a statevector-optimized parameter and 8,000 hardware shots. Error cancellation was substantial and partly fortuitous. Ideal VQE, IBM, and trapped-ion rank correlations did not establish a quantum improvement over the very similar HF/STO-3G result; the paper reports $R^2$ around 0.55 ideal, 0.77 IBM, 0.56 trapped-ion, and 0.61 HF ([DMET_VQE_PL.md](literature/mds/DMET_VQE_PL.md), §§3.3–3.5, figures and ranking table).

**Pose relevance.** Moving a ligand changes the external point-charge one-electron Hamiltonian, so the proxy is defined and pose-dependent. Protein exchange, charge transfer, dispersion, and quantum polarization are absent because protein electrons are absent. Penetrative poses are a particular risk. The published fragment was chosen with known binding interactions, which would leak crystal knowledge if copied to unknown poses. Bound calculations must be repeated per pose; a solvent reference can be reused only for identical ligand internal conformations. No force formulation is reported. The correct controls are DMET-FCI/CASCI/CCSD/HF under the identical embedding, basis, and fragment—not an unrelated DFT score.

**Verdict.** Hardware-accessible runner-up for a tiny demonstration, but a weaker pose signal than SAPT and already shown at a scale where HF was competitive. Do not select it merely because it has an actual protein–ligand hardware result.

### 3.5 DMET-SQD

**Physical quantity and pipeline.** DMET constructs fragment-plus-bath Hamiltonians from a global mean-field state. SQD uses a LUCJ circuit, initialized from classical CCSD information, as a determinant sampler. The QPU is measured in the computational basis; S-CORE repairs particle/spin sectors; batches of selected determinants define subspaces; and a classical Slater–Condon Hamiltonian plus Davidson diagonalization yields energy, a classical eigenvector, and RDMs. There is no VQE energy optimization loop in the reported workflow ([DMET_SGD.md](literature/mds/DMET_SGD.md), §II, “Density matrix embedding theory,” “Sample-based quantum diagonalization,” and “Computational details”).

**Resources and evidence.** The paper studied H$_{18}$ and cyclohexane conformers, not proteins or ligands. DMET reduced 41- and 89-qubit full systems to 27- and 32-qubit subsystems on IBM Cleveland. It used three H$_{18}$ and six cyclohexane fragment circuits; roughly 1,000–5,000 and 6,000–10,000 raw configurations per batch were explored. Cyclohexane subspaces reached about 8.7–10.4 million determinants against an 11.8-million active Hilbert space, making distributed classical diagonalization the practical bottleneck. Correct conformer ordering appeared only at sufficiently large configuration counts ([DMET_SGD.md](literature/mds/DMET_SGD.md), §§II–III, Table I and energy/convergence figures).

**Pose use.** SQD naturally returns a total embedded fragment energy, not a direct protein–ligand interaction. A pose score would require a common QM/MM/DMET total-energy definition or supermolecular subtraction. Fragment definitions, atom composition, bath construction, and active orbital identity must be fixed across poses for comparability. Protein-only mean-field and fragment topology can be cached; orbitals can be maximum-overlap tracked; LUCJ/CCSD parameters and prior determinant unions can warm-start neighboring geometries. The bitstrings are samples from a geometry-specific state and cannot be treated as quantitative measurements for a new Hamiltonian without validation.

Once SQD has produced a classical eigenvector, its 1-/2-RDM can be computed classically in the retained determinant space. This is useful for derivative development but means a second randomized-shadow layer is not naturally helpful. The closest baselines are DMET-FCI, DMET-CCSD, and selected-CI/HCI using the same embedding and orbital spaces.

**Verdict.** Promising quantum-centric scaling, but not experiment 1: no protein–ligand demonstration, no direct interaction observable, tens of qubits, and a large classical diagonalization per changed subsystem Hamiltonian.

### 3.6 Proposed QM/MM + SQD or ext-SQD

No supplied paper demonstrates a QM/MM+SQD protein–ligand pose scorer. This row is therefore a **proposal**, not prior evidence. A fixed atomistic protocol could place ligand plus selected pocket residues in a QM region, the rest in MM point charges, localize orbitals, downfold to an active Hamiltonian, sample a LUCJ state with SQD, and combine its embedded electronic energy with MM terms. If atom composition and all boundary definitions remain identical, the total QM/MM energy of different frozen poses is comparable without separately subtracting $E_P$ and $E_L$. It would include ligand strain in the QM region but remain sensitive to boundary charges, double counting, fixed receptor, and absent free-energy terms.

The qubit count is twice the number of active spatial orbitals—plausibly 20–40 for a deliberately small 10–20-orbital active region—but the DMET-SQD data warn that the determinant subspace and classical Davidson cost, not nominal qubits alone, determine feasibility. A QPU call means one or more deep state-preparation circuits with many computational-basis samples for every materially different Hamiltonian. The correct classical controls are SHCI/CASCI, CCSD, and DFT/QM-MM with exactly the same atoms, basis, embedding, and boundaries.

“ext-SQD” should not be used as a prestige suffix. The primary ext-SQD work extends SQD to low-lying excited states and reports N$_2$ and a [2Fe–2S] cluster; a ground-state docking score has no obvious need for that extra step ([Barison et al.](https://arxiv.org/abs/2411.00468), abstract). Neither SQD nor ext-SQD currently supplies a demonstrated analytic protein–ligand force.

**Verdict.** A defensible later energy/gradient platform if SAPT proves impossible, but less clean for interaction-only interpretation and much heavier than the 4-qubit DMET-VQE model.

### 3.7 QCPMD and analytic-gradient VQE are force candidates, not first scorers

QCPMD evolves nuclear coordinates and circuit parameters together using fictitious masses and Langevin updates, avoiding a fully converged VQE at every molecular-dynamics step. Nuclear forces are Hellmann–Feynman expectations of $\partial H/\partial R$; the paper neglects Pulay/state-response terms under a near-ground-state assumption and notes that finite differences could also be used. Its only experiment is simulator H$_2$/STO-3G with a four-qubit circuit, fixed dissipation coefficients, and an effective-temperature caveat ([QCPMD.md](literature/mds/QCPMD.md), §II.A.2, Eqs. 2–10; §III, Figs. 2–4). It neither produces a validated docking score nor maps to DiffDock’s manifold.

By contrast, Lagrangian quantum analytic-gradient work has recovered relaxed 1-/2-RDM information and demonstrated classically simulated active-space VQE/QM-MM examples with up to 327 quantum-region and 18,470 total atoms ([Hohenstein et al.](https://pubmed.ncbi.nlm.nih.gov/36948843/), abstract). This establishes methodological compatibility, not a docking or QPU result. A preprint posted 9 August 2026 reports orbital-optimized VQE gradients/Hessians on hardware only for H$_2$ and water ([Olarte Hernandez et al.](https://arxiv.org/abs/2608.08758), abstract); it is too new and too small to change experiment 1.

**Verdict.** Analytic active-space QM/MM VQE is the best quantum-method direction for later force guidance. QCPMD is a dynamics research program, not a drop-in DiffDock oracle.

## 4. Classical shadows: compatibility does not imply usefulness

### 4.1 Observable-by-observable assessment

| Candidate | Observables actually needed | Can fermionic shadows estimate them? | Better experiment-1 choice |
|---|---|---|---|
| First-order SAPT(VQE) | Monomer 1-RDM and 2-RDM; equivalently contractions for electrostatic/exchange | Yes. Particle-number-preserving fermionic shadows target all 1-/2-RDM elements simultaneously. | Exact statevector RDM on simulator; then deterministic commuting groups/low-rank contractions on a tiny shot ablation. |
| Second-order SAPT(VQE) | Same ground-state 1-/2-RDMs, consumed by ERPA response and SAPT contractions | Yes in principle, and one shadow can support all energy components. | Conventional/grouped or tailored RDM measurement first; validate propagated error in total and individual SAPT terms. |
| DMET-VQE | Energy terms and fragment 1-/2-RDM for DMET assembly | Yes, but the Hamiltonian and DMET contractions are known beforehand. | Hamiltonian grouping/double factorization plus direct RDM measurement. |
| DMET-SQD / QM-MM+SQD | Computational-basis determinant samples for selection; then energy/RDM of the classically diagonalized eigenvector | Not naturally useful. Random orbital/basis rotations change the sample distribution SQD needs. | Use SQD’s native bitstrings and compute RDMs from the classical subspace eigenvector. |
| QCPMD / nuclear forces | Many known Pauli expansions of $\partial H/\partial R_{i\alpha}$ for one state; parameter forces use shifted states | Generic shadows can reuse a state across the $3N$ nuclear-force observables. They cannot remove the two shifted-state preparations per circuit parameter. | Grouped/derandomized measurement unless a multi-observable force benchmark shows lower total variance. |
| Analytic-gradient VQE | Relaxed 1-/2-RDMs, derivative integrals, orbital/circuit response, Pulay terms | Shadows can estimate density observables, not eliminate response equations or changed-state preparations. | Factorized/grouped density measurement and analytic derivative machinery. |

For $\eta$ fermions in $n$ modes, Low proves simultaneous estimation of all $k$-RDMs to **average-element variance** $\epsilon^2$ with at most

$$
M \leq
\binom{\eta}{k}
\left(1-\frac{\eta-k}{n}\right)^k
\frac{1+n}{1+n-k}\frac{1}{\epsilon^2}
$$

random particle-number-preserving single-particle-basis measurements, with $O(k^2\eta)$ estimator work per element ([Low](https://arxiv.org/abs/2208.08964), abstract/main theorem). For SAPT, $k=2$ is the key case. This is genuine compatibility, but it is not a bound on the error of a weighted SAPT energy, an ERPA eigenvalue, the maximum RDM-element error, or a pose-energy difference. Those depend on integral coefficients, covariances, conditioning, and the accuracy required to preserve rankings separated by perhaps fractions of a kcal/mol.

**Demonstrated caution.** QCPMD’s local-Pauli shadows reused 51 snapshots across force observables, while ordinary measurement used 51 shots per Pauli. In a four-qubit test the shadow estimator had variance about 0.072–0.077 versus 0.016 for direct measurement. The authors explicitly state that when the Pauli observables are known, commuting groups, derandomized shadows, or shadow grouping can be more effective because random bases waste samples ([QCPMD.md](literature/mds/QCPMD.md), §II.B and §III, Figs. 2–3). Its advertised $O(N_S)$ independence from the number of nuclei suppresses the dependence hidden in shadow norm and precision; it should not be extrapolated to protein-scale forces without measurement accounting.

**Deduction.** SAPT is the most intellectually compelling shadow use case because one 2-RDM data product supports electrostatic, exchange, induction, dispersion, response, and potentially other observables. But experiment 1 is initially a statevector study; shadows would add estimator variance without testing the primary scientific question. A small hardware study should first exploit known structure: the first-order SAPT paper reduces electrostatics to one commuting Z group and suggests factorization for exchange ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), Appendix B).

**Verdict: classical shadows should be a later ablation.** Add them only after (i) the physical score changes pose rankings, (ii) a shot-noise budget is measured for ordinary grouping, and (iii) there is value in estimating several outputs—SAPT components plus forces, dipoles, or diagnostics—from the same prepared state. They do not currently help SQD and should not be part of the success claim for experiment 1.

## 5. Closest prior work and what novelty remains

### 5.1 CompassDock reconstructed and used as a control

**Demonstrated: inference assessment.** CompassDock wraps DiffDock with a “Compass” module composed of PoseCheck and AA-Score. PoseCheck computes (i) ligand strain as the UFF energy difference between generated and relaxed ligand conformations, (ii) protein–ligand clashes when atoms approach within the sum of van der Waals radii minus 0.5 Å, and (iii) ProLIF interaction fingerprints. AA-Score is an empirical affinity-energy score with amino-acid/main-chain/side-chain-specific hydrogen-bond, van der Waals, and electrostatic terms plus generic hydrophobic, π–π, π–cation, metal–ligand, and rotatable-bond entropy terms ([CompassDock.md](literature/mds/CompassDock.md), §§2.2.2–2.2.3; Appendix B.1, Eq. 8 and Table 1). In inference mode, Compass evaluates a completed DiffDock pose and describes a recursive redocking procedure with threshold stopping, but does not present a controlled demonstration that AA-Score reranks a fixed candidate set to improve top-1 RMSD ([CompassDock.md](literature/mds/CompassDock.md), §2.3, Eq. 1).

**Demonstrated: Compass Score and fine-tuning.** The Compass Score is not simply “AA-Score plus strain and clashes.” It is the equally weighted LAN-MSE discrepancy between a sampled pose’s AA-Score/strain/clash values and those computed on its ground-truth pose. It is added to DiffDock training loss as a stated “non-gradient-tracked penalizer” ([CompassDock.md](literature/mds/CompassDock.md), §§2.4.1–2.4.4, Eqs. 2–7). On 261 selected PDBBind test complexes, Fig. 4 reports 11.38% below 2 Å for DiffDock-L, 11.57% after ordinary fine-tuning, and 11.42% with Compass Score; the favorable-property rate was 4.21%, 3.83%, and 4.98%, respectively ([CompassDock.md](literature/mds/CompassDock.md), §4.2, Fig. 4).

**Deduction/implementation risk.** As written, a detached scalar added to a differentiable loss contributes zero parameter gradient. “Indirectly” influencing gradients would require an unreported selection, weighting, or differentiable path. This ambiguity and the small RMSD change mean the paper is evidence for useful *features and baselines*, not strong evidence for its fine-tuning mechanism.

Answers to the five comparison questions:

1. **Novelty over CompassDock.** A SAPT reranker replaces an empirical atom-type interaction score with an explicitly electronic, component-resolved, active-space interaction calculation on the *same frozen DiffDock-L candidates*. The new scientific test is whether correlated electronic terms add within-complex pose information after controlling for an identical classical approximation. It is not novel merely to compute “more physics.”
2. **Baseline.** Yes: report AA-Score alone, PoseCheck strain/clash alone, and their combination. Do not call the ground-truth-relative LAN-MSE Compass Score an inference score; it cannot be evaluated prospectively without the crystal pose.
3. **Replacement.** Replace only the interaction-energy role of AA-Score for the cleanest comparison. Keep strain and clash handling identical. Compare `AA`, `SAPT`, `AA + validity`, and `SAPT + validity`; optionally test `AA + SAPT + validity` for complementarity.
4. **Invalid poses.** Yes, an interaction-only scorer can favor a strained ligand because strain is a monomer deformation cost. Severe overlap should raise exact exchange, but a truncated pocket, small basis, active-space approximation, or embedding can miss/attenuate repulsion. Run PoseBusters first and retain an explicit hard-overlap rule plus continuous strain/clash features.
5. **Clean ablation.** Use one frozen candidate table and alter only the score columns. The primary four-way ablation is DiffDock confidence; AA-Score; SAPT hierarchy; and each interaction score combined with the same PoseCheck policy. Do not redock, minimize only one arm, or tune thresholds against test RMSDs.

### 5.2 Q-Score reconstructed exactly

**Candidate interactions.** Q-Score starts from an already assembled 3D protein–ligand complex and extracts protein atoms within 5 Å of any ligand atom. SIMG*, a GNN trained on QM9/GEOM-derived data, first predicts lone-pair counts and an orbital graph and then predicts pairwise Natural Bond Orbital donor–acceptor $E^{(2)}$ values from coordinates/connectivity. The $E^{(2)}$ term is a perturbative orbital-stabilization estimate proportional to donor occupancy and squared donor–acceptor Fock coupling divided by their energy separation; it is neither a total interaction energy nor measured on a QPU ([Q-Score.md](literature/mds/Q-Score.md), §IV.A–B, orbital-interaction equations and Fig. 1).

Intermolecular orbital values are aggregated to ligand-atom/protein-atom anchor weights. A greedy diversity rule selects $N\in\{6,10,12\}$ anchors, one per qubit. An edge connects two anchors if the ligand internal distance and protein-anchor distance agree within a target-specific tolerance $\tau$. The highest-weight mutually compatible set is a maximum-weight vertex clique. It is encoded as a QUBO with incompatibility penalty $P=6$, then mapped to an Ising Hamiltonian ([Q-Score.md](literature/mds/Q-Score.md), §IV.C–D, QUBO/Ising equations and Fig. 2).

**Genuinely quantum part.** DC-QAOA prepares a Hadamard state and alternates term-wise cost rotations, $R_X$ mixers, and counterdiabatic $R_Y$ rotations. The paper uses three or six layers, independent angles per cost term, COBYLA with up to 5,000 iterations, and samples bitstrings. Only this combinatorial clique optimization—not NBO electronic structure—is executed on the QPU. The optimal clique-weight sum is Q-Score. In redocking, Kabsch rigid alignment maps ligand anchors onto selected protein anchors to construct a pose; torsions are not optimized ([Q-Score.md](literature/mds/Q-Score.md), §§IV.D–F, Eqs. for the circuit, Kabsch objective, and Q-Score).

**What was validated.** The 11-target redocking graph was built from a co-crystal complex and used target-specific $\tau$. At 10 qubits/three layers, simulation recovered the classical graph optimum for 8/11; three failures were invalid penalty-violating cliques, and penalty tuning was instance-dependent. The reported 8SKH reconstruction remained around 3.34 Å. The separate “scoring” study docked 100 Pocket2Mol molecules for each of ten targets with Vina and compared rankings *between different molecules*, not multiple decoy poses of one fixed protein–ligand pair ([Q-Score.md](literature/mds/Q-Score.md), §V.A–D, Tables I–III, V–VI and Figs. 3–5). Hence Spearman $\rho=0.05$ versus Vina shows difference, not correctness, and $\rho=0.90$ with summed orbital energy is largely expected from how weights are constructed.

**Hardware.** IBM Eagle ran 1,000 circuits with 10,000 shots each. Six-qubit hardware returned the simulator’s most-probable string around 65% of the time, but valid-solution rates were only roughly 26–30%; at ten qubits, match rates were about 12–14% and validity 10–12%. Classical exact enumeration and simulated annealing solved all ten-qubit instances, and greedy selection was about 99.4% relative to exact versus roughly 96.9% for DC-QAOA ([Q-Score.md](literature/mds/Q-Score.md), §V.E–F, Tables III–V). No quantum advantage is demonstrated.

**Can it directly rerank arbitrary DiffDock-L poses?** It can ingest each pose’s coordinates into SIMG*, but the published method does not merely evaluate that pose. Its compatibility graph asks which anchors can be satisfied by a new rigid Kabsch placement, and its final score can select a subset inconsistent with the input pose as a whole. It ignores DiffDock torsions, strain, clashes outside selected anchors, and most electronic interactions. Using its clique weight as if it were a fixed-pose energy would therefore alter the score’s semantics; freezing the supplied coordinates and summing their predicted interactions would be a new, classical GNN score, not published Q-Score.

| Feature | Q-Score | SAPT(VQE) reranker | DMET-VQE/SQD reranker |
|---|---|---|---|
| Electronic input | GNN-predicted NBO-like $E^{(2)}$ anchors | Electronic Hamiltonians and monomer RDMs | Embedded active Hamiltonian(s) |
| QPU task | Combinatorial maximum-weight clique | State preparation/VQE and RDM measurement | VQE energy/RDM or determinant sampling |
| Output | Best compatible anchor-clique weight; optionally a new rigid pose | Frozen-pose direct interaction energy/components | Embedded total or ligand-environment energy proxy |
| Arbitrary fixed-pose semantics | No, not without modification | Yes | Yes if region/embedding is fixed |
| Same-pair decoy validation | No | Not yet—the proposed novelty | No |
| Quantum advantage | No | No | No |

**Defensible novelty today.** “The first controlled test, to our knowledge, of an active-space quantum-derived SAPT correction for *within-complex reranking of a frozen DiffDock-L candidate set*, with a same-Hamiltonian classical control and independent strain/clash handling.” This must be qualified by a search at thesis-submission time. “First quantum scoring for docking” is already false.

## 6. The modern docking landscape and why DiffDock-L remains useful here

### 6.1 Context, not a claim that DiffDock-L is best

PoseBench compares traditional docking, DiffDock-L, flexible DynamicBind, and co-folding systems including NeuralPLexer, RoseTTAFold-All-Atom, Chai-1, Boltz-1, and AlphaFold 3 across Astex Diverse, PoseBusters Benchmark, DockGen-E, and CASP15. It finds that co-folding models often have stronger aggregate pose accuracy, but performance tracks training-set structural similarity and falls on novel targets; pose RMSD and interaction-fingerprint recovery also expose different failure modes ([PoseBench.md](literature/mds/PoseBench.md), §§3.1–3.5; §§5.2–5.5; Appendices D–G). This prevents a claim that DiffDock-L represents 2026 state of the art.

DiffDock-L is nevertheless the right **experimental instrument** because:

- it generates many explicit candidate poses from a frozen, reproducible sampler;
- its confidence model is a separate post-hoc scalar with a clear binary training target;
- its low-dimensional translation/rotation/torsion manifold makes later force projection definable; and
- all rerankers can receive exactly the same receptor and candidate table, isolating scoring from generation.

This modular control is methodologically valuable even if another generator has higher absolute accuracy. A later replication can swap in a stronger generator if the scorer passes the first gate.

### 6.2 FlowDock and Boltz-2 do not make the test redundant

**FlowDock.** It is a flexible protein–ligand co-folding/flow-matching model that can start from sequences/SMILES or an initial protein, samples protein and ligand heavy-atom structures over about 40 steps, and has confidence and affinity heads. The affinity head consumes learned molecular features and a stop-gradient predicted structure; inference ranks structural samples by confidence, not affinity. Its reported affinity evaluations compare different complexes/ligands on PDBBind and CASP16, not controlled decoys of the same pair ([FlowDock.md](literature/mds/FlowDock.md), §§3.1, 3.4–3.6, 4.1–4.3; Appendix A, Algorithms 1–2). Thus its affinity is structurally conditioned, but the paper does not establish that it is a calibrated same-pair pose discriminator or an arbitrary-pose plug-in.

**Boltz-2.** It jointly advances structure prediction and affinity prediction; its affinity module relies on the model’s predicted 3D structure and learned trunk representation, and is evaluated for affinity/virtual-screening tasks across ligands. It adds method conditioning, distance constraints, multi-chain templates, and ensemble training ([Boltz-2 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/), Abstract, §§1 and 4–5). The paper does not validate feeding 20 externally generated DiffDock poses through a standalone affinity head and recovering the lowest-RMSD pose for the same pair. “Uses structure” is not equivalent to “validated same-pair pose ranking.”

In the terminology needed here, FlowDock exposes both structure confidence and an affinity prediction, but uses confidence to rank its own sampled structures; Boltz-2 likewise couples structure/confidence prediction with an affinity module. Neither paper establishes that its affinity output should replace native-pose confidence for arbitrary same-pair decoys. Affinity calibration across ligand–target examples and pose discrimination within one ligand–target pair are different supervised tasks ([FlowDock.md](literature/mds/FlowDock.md), §§3.5–3.6 and 4.1–4.3; [Boltz-2 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/), §§1, 4–5).

**Context-only methods.** AlphaFold 3, Boltz-2, FlowDock, NeuralPLexer, and DynamicBind should appear in the thesis landscape and, if resources permit, contribute one end-to-end top-1 pose per complex. They should not be primary reranking baselines because they do not score the identical frozen DiffDock candidate set under identical inputs. The actual scorer baselines are DiffDock-L confidence, AA-Score/PoseCheck, Vina/GNINA score-only, and same-approximation classical electronic structure.

### 6.3 Benchmark and receptor choice

PoseBusters tests chemical consistency, bond lengths/angles, planarity, internal clashes, UFF energy ratio, protein/cofactor distances, and volume overlap. Its recent 308-complex benchmark is harder and less exposed to older training than the original PDBBind time split; force-field minimization can rescue validity but also move poses and change RMSD ([PoseBusters.md](literature/mds/PoseBusters.md), §§2.2–2.6, 3.2–3.3; Supplement §§S3–S8). PoseBench further offers a post-September-2021 filtered PoseBusters subset for newer co-folding models and DockGen-E for novel ECOD pockets ([PoseBench.md](literature/mds/PoseBench.md), §5.2; Appendix D).

**Proposal.** Use holo receptors deliberately in experiment 1. This removes the largest avoidable confound: DiffDock-L holds receptor coordinates rigid while a quantum score can only judge the supplied geometry. With apo/predicted receptors, a correct ligand may clash because side chains/backbone are wrong, and no frozen-pose scorer can separate ranker error from receptor error. The claim must therefore be “cognate holo, fixed-receptor reranking.” Repeat on apo/flexible inputs only after a signal exists.

## 7. Experiment 1: an executable, falsifiable pilot

### 7.1 Revised hypotheses and estimands

The original H1 mixes two questions. Pre-register them separately:

> **H1-physical.** Conditional on a frozen DiffDock-L set containing at least one near-native pose, a fixed-geometry SAPT0 interaction score adds within-complex pose-ranking information beyond DiffDock-L confidence and AA-Score when all methods receive identical steric/strain handling.

> **H1-correlation.** Replacing SAPT0 first-order electrostatic/exchange terms by an (8e,8o or smaller) active-space correlated estimate changes rankings in a direction that improves near-native selection; ideal VQE reproduces the same-active-space CASCI correction closely enough not to erase that improvement.

H1-physical can be true while H1-correlation is false. Only H1-correlation tests whether the quantum-representable part is scientifically relevant. A simulator cannot establish computational quantum advantage; it can only establish observable usefulness and circuit/resource feasibility.

Define two estimands:

- **End-to-end top-1 success:** success among all included complexes, where missing oracle coverage is a generator failure.
- **Conditional top-1 reranking success:** success only among complexes with at least one sampled pose at symmetry-corrected RMSD <2 Å. This is the primary ranker estimand.

Both must be reported. Conditional performance alone cannot be advertised as docking success.

### 7.2 Dataset and locked cohort

**Benchmark.** Use the 308-complex **PoseBusters Benchmark set** distributed/evaluated by PoseBusters and PoseBench, not the original PDBBind time split. PoseBusters constructed it from recent high-quality complexes and designed explicit validity checks; PoseBench uses the set as its intermediate-difficulty benchmark ([PoseBusters.md](literature/mds/PoseBusters.md), §2.6.2 and §3.2; Supplement §§S3–S5; [PoseBench.md](literature/mds/PoseBench.md), §§3.3, 5.2 and Appendix D.2).

**Pilot size: 24 complexes.** This is enough to reveal gross paired effects and workflow failures while keeping 480 pose evaluations finite; it is not enough for a definitive small effect. The follow-up validation size must be determined from the pilot’s paired discordance/effect estimates, not retroactively by significance.

**Selection, performed before viewing any generated-pose RMSD or electronic score:**

1. Require a complete holo protein model and one noncovalent primary ligand; exclude covalent binders, unresolved ligand atoms, alternate ligand occupancies that cannot be resolved deterministically, and complexes requiring a metal/cofactor for the primary pilot.
2. Require ligand elements in H/C/N/O/F/P/S/Cl, 12–35 heavy atoms, 1–8 RDKit rotatable bonds, formal charge $-1,0,+1$, and a closed-shell singlet under the chosen protonation. These restrictions define the scope; they are not claims about general docking.
3. Generate the 20 DiffDock-L candidates without looking at native coordinates. Form the union of complete receptor residues having any heavy atom within 4.5 Å of any candidate ligand heavy atom. Require at most 100 protein heavy atoms in that fixed union and no disjoint region whose covalent capping is chemically ambiguous. This feasibility filter depends on generator geometry, not native RMSD.
4. Bin eligible complexes by neutral/charged ligand and 1–4/5–8 rotatable bonds. Sort PDB IDs within each of four bins by SHA-256 of the literal string `PDB_ID|q-sapt-pilot-v1`; take six from each bin. Freeze IDs, exclusions, and reasons in a manifest before computing RMSD or scores. If a bin has fewer than six, use the next pre-declared bin in cyclic order and record the deviation.
5. Do not replace a complex because its SCF, VQE, or scorer later fails. Method failure is an outcome. A replacement is allowed only for an objectively corrupted input discovered before any score is inspected, using the next hashed ID and an audit log.

**Leakage audit.** PoseBusters recency reduces direct DiffDock training leakage but does not prove domain novelty. For every retained complex, record PDB release date, maximum receptor sequence identity and ligand similarity to DiffDock/DiffDock-L training data where manifests permit, and whether the ECOD domain occurs in training. Report subgroup results; do not filter after observing accuracy. Holo coordinates and crystallographic waters can reveal pocket geometry, so the claim is explicitly cognate-holo reranking—not blind apo generalization.

**Scientific validity of filtering.** Publish the full flow diagram: 308 initial, count/reason at every gate, 24 selected IDs, and descriptor distributions versus all eligible complexes. Conclusions apply only to small-to-medium, closed-shell, nonmetallated ligands and computationally bounded pockets. A later metal/multireference stratum is necessary if the active-space correction is hypothesized to matter most there.

### 7.3 Pose generation and frozen-candidate contract

1. Use one pinned **DiffDock-L** checkpoint and repository commit. Record whether its confidence architecture is the original or bootstrapping variant.
2. Remove the crystallographic ligand and every feature derived from its coordinates before inference. Supply the holo receptor coordinates and ligand graph/SMILES; do not supply a pocket centre or native conformer.
3. Generate $N=20$ poses per complex in one prescribed run, with seeds `base_seed + 0…19` (or the repository’s equivalent batched seeds). Preserve the RDKit ETKDG seed conformer, raw trajectory metadata, raw pose, and DiffDock confidence.
4. Never insert the native pose or a native-perturbed pose. Oracle coverage must come from the generator.
5. Primary scores operate on **raw** DiffDock coordinates after adding hydrogens in the fixed preparation pipeline. Do not energy-minimize before primary ranking: relaxation changes the candidate, may collapse several candidates to one basin, and can conceal invalid generation.
6. Secondary sensitivity analysis applies the identical restrained protocol to every candidate: protein heavy atoms fixed; ligand heavy-atom harmonic restraint weak enough to resolve hydrogen contacts but strong enough to report displacement; fixed iteration limit; score both before and after; recompute RMSD and PoseBusters. Never minimize only the quantum-scored arm.

Archive a single candidate table keyed by `(complex_id, pose_id, generation_seed)`. Every scorer reads it and may not change coordinates. This makes comparisons paired and prevents silent scorer-specific docking.

### 7.4 Fixed quantum-region construction without crystal-pose leakage

This protocol is a **proposal** and must be frozen before RMSD evaluation.

#### Atoms and boundary

- **Ligand monomer B:** the complete ligand, with the same bond order, protonation, charge, and heavy-atom geometry for every scorer.
- **Protein monomer A:** the union of complete residues within 4.5 Å of *any of the 20 candidate poses*, not the native ligand. Use the same atom set for all 20 poses. Include one adjacent residue on each side of a retained contiguous peptide segment where required for chemically sensible caps.
- Cut peptide bonds only. Cap exposed N/C termini with ACE/NME in a deterministic internal-coordinate construction; optimize cap hydrogens only with all original heavy atoms fixed. Reject, before scoring, regions needing arbitrary cuts through a cofactor or covalent network.
- Exclude crystallographic waters in the primary analysis to avoid inconsistent occupancy. A predeclared secondary analysis can add waters within 3.5 Å of any candidate and fixed by receptor hydrogen-bond criteria, again as one common set across all poses.

The union rule may include atoms irrelevant to a particular pose, but it preserves comparable Hamiltonian composition. A per-pose nearest-residue cutout would make energies incomparable and can reward poses merely for producing a smaller region.

#### Protonation, charge, and spin

- Add missing receptor atoms once. Assign protonation at pH 7.4 with one pinned tool/version; manually adjudicate only a predeclared list of histidine, Asp/Glu, Lys, Cys, and termini rules while blind to RMSD/scores. Store all decisions.
- Standardize the ligand protonation/tautomer from its benchmark chemical component and input SMILES. Do not enumerate states in primary analysis. Flag ambiguous cases before cohort locking; later state enumeration is a separate experiment.
- Require integral total charges for both monomers and a closed-shell global reference. Active-space $N_\alpha,N_\beta$ and multiplicity are fixed per complex and identical across poses.

#### Basis, orbitals, and active space

- Use 6-31G for the primary feasibility run because it matches the large KDM5A precedent and keeps the classical integral problem bounded ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.C and SI §III). This is not a production-quality dispersion basis. On four smallest complexes, repeat SAPT0 with jun-cc-pVDZ or the largest supported SAPT basis to quantify basis-induced rank changes.
- Compute one RHF protein-monomer reference in a **monomer-centred** basis. Localize occupied orbitals (Pipek–Mezey or IAO, chosen once). Treat the ligand at RHF for every unique internal conformer; rigidly related candidates reuse its density exactly up to coordinate transformation.
- Select the protein active space without the native pose. Build an AVAS candidate space from valence orbitals on chemically nontrivial protein atoms that contact at least 25% of the candidate ensemble (polar/charged/aromatic atoms ranked by contact frequency, with deterministic tie-breaking). If larger than 8 spatial orbitals, use a loosely converged SHCI/CASSCF natural-orbital calculation on the *isolated fixed protein monomer* and retain an (8e,8o) space by a frozen occupation rule. If electron count requires it, use (6e,6o); never choose the size after seeing ranking performance.
- Track and report natural occupations. If every occupation is effectively 0 or 2 and CASCI–RHF corrections are negligible, that is evidence against H1-correlation, not permission to switch complexes.
- Map 8 spatial active orbitals to 16 spin-orbital qubits (or 6 to 12) by Jordan–Wigner and use the paper’s shallow number/spin-preserving $1$-muCJ ansatz first. Optimize statevector VQE from RHF; compare total energy and every required RDM contraction to CASCI.

#### Remainder of protein and solvent

- Assign the receptor outside the SAPT cutout fixed AMBER point charges and Lennard–Jones parameters. Do **not** silently put those charges into only one monomer Hamiltonian. Primary reporting separates `SAPT cutout interaction` from a classical `ligand–remainder MM` term computed identically for all SAPT arms.
- Remove cutout atoms and artificial cap atoms from the remainder interaction list. At severed peptide bonds, apply one frozen link-charge rule—zero the boundary-crossing charge group and redistribute its net charge to the first retained remainder shell, or use an established charge-shift scheme—and verify that total receptor charge is conserved. Because the protein is rigid, omitted cutout–remainder self-energy is pose-constant; ligand–remainder terms are not.
- Define `E_phys = E_SAPT(cutout) + E_MM(ligand,remainder)`. Report both components so any gain from the classical environment cannot be attributed to the quantum term.
- Use no continuum solvent in primary ranking. On the four-complex sensitivity subset, add a consistent GB/continuum desolvation estimate to all interaction arms. This tests robustness but still does not yield $\Delta G_{bind}$.

#### Reuse ledger

Per complex, cache one protein geometry, SCF, localized orbitals, active Hamiltonian, optimized VQE parameters, and 1-/2-RDM. Reuse them for all 20 poses. Cache ligand RHF densities by exact internal-coordinate hash; rigid translation/rotation only changes transformed intermonomer integrals. Recompute ligand SCF for a new torsional conformer. Compute pose-specific mixed integrals and classical SAPT contractions for every candidate. Log whether a cache hit was exact, symmetry-equivalent, or merely a warm start.

### 7.5 Scores and baseline matrix

All scores are higher-is-better only after a documented sign conversion; raw values remain archived.

| ID | Score/ranker | Purpose |
|---|---|---|
| R0 | Uniform random candidate, repeated over 10,000 seeded draws | Chance reference conditional on each candidate set |
| R1 | DiffDock-L confidence | Learned native-pose baseline |
| R2 | Vina score-only and, if stable on fixed coordinates, GNINA CNN pose score | Conventional/learned docking-rescore baseline; no search |
| R3 | AA-Score alone | Closest CompassDock interaction baseline |
| R4 | PoseCheck strain, clash count, and PoseBusters validity | Noninteraction physical controls |
| R5 | SAPT0-like second-order score + fixed remainder MM term | Tests H1-physical |
| R6 | $E_{\mathrm{SAPT0}}+[E_1^{CASCI}-E_1^{RHF}]$ | Exact active-space correlation control |
| R7 | $E_{\mathrm{SAPT0}}+[E_1^{VQE(statevector)}-E_1^{RHF}]$ | Ideal quantum-representable estimator; must match R6 within tolerance |
| R8 | Shot-sampled/noise-model R7 on a predeclared subset | Ranking stability under measurement error, not advantage |
| R9 | Full second-order SAPT(CASCI/VQE) on size-qualified four-complex subset | Checks whether the mixed hierarchy changes conclusions |

Two composites are predeclared; no weights are fit on 24 test complexes:

1. **Validity-controlled interaction:** poses failing PoseBusters’ intermolecular distance/volume-overlap checks are placed after passing poses; within each tier rank by the interaction score. Strain is reported but is not a hard exclusion.
2. **Confidence augmentation:** average the within-complex percentile ranks of DiffDock confidence and the chosen physical score (equal 0.5/0.5 weight), after the same validity tiering. Repeat equal-weight fusion for AA-Score so the learned-confidence contribution is controlled.

Report pure scores and composites separately. A win caused solely by the validity tier is a PoseCheck/PoseBusters result, not a quantum result.

**Q-Score decision.** Do not place published Q-Score in the primary matrix because it re-solves an anchor graph and reconstructs a rigid pose rather than scoring the supplied pose. If its code is available, an exploratory appendix may: (i) run its exact classical clique solver, not QAOA; (ii) freeze one global $\tau$ without crystal tuning; and (iii) quantify how much the Kabsch output moves from each input. Label any fixed-coordinate interaction sum as a modified SIMG* baseline, not Q-Score.

### 7.6 Metrics and failure attribution

For every score, report:

- **Oracle coverage:** proportion with $\min_{j\leq20}\mathrm{RMSD}_{ij}<2$ Å; additionally top-5/top-10 coverage by the generator’s original order.
- **Top-1 and top-3 success:** symmetry-corrected ligand RMSD <2 Å, both end-to-end and conditional on oracle coverage.
- **Continuous pose quality:** selected-pose RMSD and centroid distance; do not let a few huge errors dominate—report median and full paired distribution.
- **Within-complex rank relation:** Spearman correlation between higher score and negative RMSD for each complex, summarized by median and a complex bootstrap. Correlations are undefined when RMSD/score is constant; report the count rather than impute zero.
- **Pairwise discrimination:** probability that a near-native pose (<2 Å) outranks a clear decoy (>4 Å), using all eligible pairs but cluster-bootstrap by complex. Report intermediate 2–4 Å poses separately.
- **Physical validity:** PoseBusters pass rate of selected top-1 and joint success `(RMSD <2 Å AND PB-valid)`. Include ligand strain and clash distributions.
- **Interaction recovery:** if feasible, PoseBench PLIF-WM or ProLIF similarity to the native interaction fingerprint as a secondary metric, never as a score input ([PoseBench.md](literature/mds/PoseBench.md), §§5.3–5.4 and Appendix E).
- **Operational feasibility:** preparation/SCF/VQE/SAPT failure rate, wall time, peak RAM/GPU memory, unique QPU states, circuits, shots, and cache reuse.

Classify every complex-score result:

1. **Generator failure:** no pose below 2 Å exists among 20; no ranker could succeed.
2. **Ranker failure:** a below-2-Å pose exists but the selected pose is not below 2 Å.
3. **Validity conflict:** the lowest-RMSD candidate fails a physical check or the most physical candidate has high RMSD.
4. **Method failure:** the scorer did not return a comparable value under the locked protocol.

### 7.7 Minimum ablations

Run these in order; stop expensive later items if an earlier gate fails.

1. **Interaction model:** confidence vs AA-Score vs Vina/GNINA vs SAPT0, with no quantum correction.
2. **Quantum correlation:** SAPT0/RHF first order vs CASCI-corrected vs ideal-VQE-corrected on identical orbitals. This is the decisive quantum ablation.
3. **Validity/strain:** every interaction score alone, plus identical clash tier, plus identical clash tier and equal-rank strain feature. This detects gains caused by PoseCheck rather than interaction physics.
4. **Learned confidence:** physical score alone versus fixed 50/50 rank fusion with confidence. This tests replacement versus augmentation.
5. **Region/active space:** on four locked representatives only, 4.5 versus 6.0 Å region where feasible and (6e,6o) versus (8e,8o). Do not search for the best setting on all 24.
6. **Measurement:** exact statevector, finite-shot grouped measurement, and a pinned device-noise model on two complexes with the smallest and largest correlated rank changes. Repeat sampling 100 times to estimate top-1 stability.
7. **Full second order:** on up to four systems whose **entire locked cutout** genuinely fits the implementation’s orbital/memory limit, compare the mixed hierarchy with full SAPT(CASCI/VQE)-ERPA across their poses. If no whole cutout fits, do not quietly shrink the region per pose. One fixed, predeclared ligand–residue microcomplex may be used to verify equations, but it is not a validation of pocket-pose ranking. If full-method rankings differ materially, the hybrid cannot be generalized without the full method.

### 7.8 Statistical analysis

- Treat the **complex**, not a pose pair, as the independent sampling unit.
- For top-1 success differences, report the paired difference in percentage points, exact McNemar test on discordant complexes, and a 10,000-replicate stratified complex bootstrap confidence interval. Report the discordance table itself.
- For RMSD, per-complex Spearman, pairwise accuracy, and validity, use paired complex bootstrap intervals. A permutation/sign-flip test is suitable for a predeclared continuous paired summary if its exchangeability assumptions hold.
- Use a hierarchical test order: H1-physical first (SAPT0 composite vs confidence control), H1-correlation second (CASCI correction vs SAPT0), VQE fidelity third (VQE vs CASCI). Stop formal testing after a failed gate; label remaining comparisons exploratory. If many baseline contrasts are interpreted, use Holm correction.
- Report effect sizes and intervals even when a p-value is large. With 24 complexes, “not significant” is not equivalence. Define practical equivalence bands before analysis, e.g. ±5 percentage points in conditional top-1 and ±0.03 pairwise accuracy.
- Keep all 24 in operational summaries. The conditional analysis excludes only generator failures by definition, not method failures; method failures should be assigned a prespecified worst rank or reported in a parallel complete-case sensitivity analysis.

### 7.9 Pre-registered success, continuation, and falsification criteria

**Feasibility gate:** at least 20/24 complexes yield comparable SAPT0 and active-space results for all 20 poses; the 16-qubit statevector/CASCI RDM agreement meets a fixed tolerance; and at least 12 complexes have oracle coverage. Otherwise redesign preparation or increase $N$, but do not interpret ranking.

**Progress to a larger physical-score validation** if SAPT0 augmentation has at least four more conditional top-1 wins than losses versus confidence-only, improves pairwise accuracy by at least 0.05, has a positive lower bound on an 80% pilot bootstrap interval, and does not reduce joint PB-valid/top-1 success. These are pilot continuation thresholds, not proof.

**Physical score useful, quantum unnecessary** if SAPT0 passes that gate but CASCI correction changes conditional top-1 by less than the ±5-point equivalence band, changes pairwise accuracy by less than ±0.03, and natural occupations/RHF–CASCI differences show weak correlation. This is a scientifically valuable likely result.

**Quantum component adds measurable information** only if CASCI-corrected SAPT beats SAPT0 by the same predeclared directional criteria, the effect is not driven by one complex or by validity penalties, VQE reproduces CASCI rankings within uncertainty, and the active-space selection is stable. On a simulator, phrase this as “a quantum-representable correlated observable adds information,” not “quantum advantage.”

**Falsify reranking for this scope** if oracle coverage is adequate but SAPT0 and its correlated variants have no practically positive conditional effect, systematically favor high-strain/invalid decoys, or lose to AA-Score/Vina under identical validity handling. Also stop if region/basis/protonation sensitivity reverses rankings more often than the purported gain.

**Hardware go gate:** only after a correlated effect exists, grouped shot estimates preserve at least 90% of the CASCI pairwise decisions at an affordable predeclared shot budget, and the QPU result is compared with a noise-matched classical sample. A decorative one-pose circuit is not evidence for docking utility.

## 8. Future work: how quantum-informed reverse-diffusion guidance could actually be defined

This section is a derivation and cost test, not an experiment-1 commitment.

### 8.1 Scalar-energy finite differences on $m+6$ coordinates

At current pose $x$, let $A_g(\delta,x)$ be DiffDock’s exact action for component $g$: Cartesian translation, exponential-map rotation around the unweighted ligand centroid, or its RMSD-aligned torsion action. For coordinate basis vector $e_a$, a central derivative is

$$
\frac{\partial E}{\partial q_a}
\approx
\frac{E(A(+h_a e_a,x))-E(A(-h_a e_a,x))}{2h_a}.
$$

There are $d=m+6$ coordinates, so a full central gradient costs $2(m+6)$ energy evaluations per reverse step per trajectory; forward differences cost $m+7$ including a shared reference but are biased and fragile under noisy energies. Use separate, convergence-tested $h$ for Å translation and radian rotation/torsion. The displaced geometries must use DiffDock’s actions, not arbitrary Cartesian coordinate nudges.

For $N$ samples and $T$ reverse steps,

$$
N_{E,\mathrm{central}}=2NT(m+6).
$$

With $N=20$, nominal $T=20$, and $m=6$, this is 9,600 electronic-energy evaluations (8,640 if the effective 18-step implementation is used). Every “energy evaluation” may itself require VQE optimization and many measurement groups. Activating guidance only for the last four steps still costs 1,920 energies. Monomer-state reuse helps SAPT under rigid motions, but torsion changes, dimer-centred bases, and pose-dependent embeddings erode it.

A ChemGuide-like simultaneous perturbation can estimate a stochastic direction from two oracle calls,

$$
\widehat{\nabla E}=\frac{E(A(+h u,x))-E(A(-h u,x))}{2h}\,u,
$$

with random tangent direction $u$. This removes the explicit $m+6$ call factor but greatly increases directional variance and mixes coordinates with different units/metrics. ChemGuide uses SPSA with a non-differentiable GFN2-xTB oracle in unconditional small-molecule diffusion and reports that late, cleaner steps are especially effective; it is evidence for the black-box pattern, not for protein docking or a quantum oracle ([ChemGuide](https://arxiv.org/abs/2410.06502), §3.1, Algorithm 1 and §§4.2–4.5).

### 8.2 Projection of Cartesian forces onto DiffDock’s tangent space

Suppose an electronic method returns ligand-atom Cartesian forces

$$
F_i=-\nabla_{x_i}E.
$$

For generalized coordinate $q_a$, let $v_{a i}=\partial x_i/\partial q_a$. The generalized force is the virtual-work contraction

$$
Q_a=\sum_i F_i\cdot v_{a i}
=-\frac{\partial E}{\partial q_a}.
$$

This is a covector. Converting it to a score/gradient vector requires raising its index with the **same product-space metric used by the diffusion**, denoted $M$: the vector is $M^{-1}Q$. In DiffDock’s standard translation/Euler-angle/torsion coordinates this is normally a blockwise identity after the method’s component-specific scale/noise conventions, but it must be confirmed in the implementation. A different, optional choice is the Cartesian RMSD-induced metric

$$
G_{ab}=\sum_i v_{a i}\cdot v_{b i}.
$$

Using $G^{-1}Q$ would give a least-squares Cartesian steepest direction; it is a physical preconditioner, not automatically DiffDock’s native diffusion metric. These two choices must not be silently conflated.

#### Translation

For translation component $a\in\{x,y,z\}$, $v_{ai}=e_a$ for every ligand atom, giving

$$
\boxed{Q_{\mathrm{tr}}=\sum_i F_i.}
$$

This is a force in energy per length. It is not a Cartesian displacement until multiplied by the correct diffusion coefficient/preconditioner.

#### Rotation

Let $c=n^{-1}\sum_i x_i$ be DiffDock’s unweighted centroid and $r_i=x_i-c$. An infinitesimal Euler/axis-angle update $\delta\omega$ gives $\delta x_i=\delta\omega\times r_i$. Hence

$$
\delta E
=-\sum_iF_i\cdot(\delta\omega\times r_i)
=-\delta\omega\cdot\sum_i r_i\times F_i,
$$

so

$$
\boxed{Q_{\mathrm{rot}}=\tau_c=\sum_i(x_i-c)\times F_i.}
$$

Using a mass-weighted centre of mass instead would not match DiffDock’s action. Shifting the centre changes torque by the lever arm crossed with net force, so this detail is implementation-critical.

#### RMSD-aligned torsions

For rotatable bond $k$, choose the same oriented axis and moving atom subset $S_k$ as DiffDock. With unit bond axis $u_k$ through point $a_k$, a raw unit-angle torsion tangent is

$$
w_{ki}=
\begin{cases}
u_k\times(x_i-a_k),&i\in S_k,\\
0,&i\notin S_k,
\end{cases}
$$

with sign verified against the repository’s torsion convention. DiffDock then Kabsch-aligns the torsion-updated ligand to the original pose. The infinitesimal form removes the best-fit rigid translation and rotation from $w_k$. Let

$$
\bar w_k=\frac1n\sum_iw_{ki},\qquad
I=\sum_i\left(\|r_i\|^2\mathbf 1-r_i r_i^\top\right),
$$

and solve, using a pseudoinverse for a degenerate/linear geometry,

$$
I\omega_k=\sum_i r_i\times(w_{ki}-\bar w_k).
$$

The aligned torsion tangent is

$$
v_{ki}=w_{ki}-\bar w_k-\omega_k\times r_i.
$$

It obeys $\sum_i v_{ki}=0$ and $\sum_i r_i\times v_{ki}=0$, matching the zero linear/angular-momentum property of DiffDock’s torsion definition ([DiffDock.md](literature/mds/DiffDock.md), §4.2, Definition and Proposition 1). Therefore

$$
\boxed{Q_{\mathrm{tor},k}=\sum_iF_i\cdot v_{ki}.}
$$

Multiple torsion tangents need not be mutually orthogonal in Cartesian space, particularly for nested rotating subsets. If the optional RMSD-induced preconditioner is used, assemble $G_{kl}=\sum_i v_{ki}\cdot v_{li}$ and solve $Gq=Q$ with regularization. If the native $SO(2)^m$ product metric is used, retain its angular-coordinate convention instead. In either case, raw components are not comparable until angular units and the torsion noise schedule are applied. Recompute axes, subsets, centroid, Kabsch residual, and any induced metric at every pose. Validate every analytic projection by central differences through the repository’s finite torsion action; this catches sign, subset, and alignment errors.

### 8.3 How a physical term enters the reverse update

For component $g\in\{\mathrm{tr,rot,tor}\}$, DiffDock’s approximate reverse step has the schematic form

$$
\Delta q_g
=\Delta\sigma_g^2,s_{\theta,g}(x_t,t)
+\sqrt{\Delta\sigma_g^2}\,z_g.
$$

If the desired time-dependent target is tilted by a physical Boltzmann-like factor,

$$
p_t^{*}(x)\propto p_{\theta,t}(x)\exp[-\beta_t E(x)],
$$

its additional score covector is $-\beta_t\nabla_qE=\beta_t Q$. Under the selected diffusion metric, the corresponding vector is $\beta_tM^{-1}Q$. The guided update would be

$$
\Delta q_g
=\Delta\sigma_g^2
\left[s_{\theta,g}+\lambda_g(t)\,\beta_t(M_g^{-1}Q_g)\right]
+\sqrt{\Delta\sigma_g^2}\,z_g,
$$

followed by the same group/manifold action. $\lambda_g(t)$ is not optional hand-waving: translation scores have units Å$^{-1}$, while rotation/torsion scores have rad$^{-1}$; forces/torques inherit energy units; and each component has a different noise schedule.

A safe calibration protocol would:

1. convert electronic energy to one fixed unit and choose a reference $\beta$ or energy scale before test evaluation;
2. on a validation set only, estimate the RMS norm of the learned score and the selected-metric physical term separately for every component and time bin;
3. set $\lambda_g(t)$ so the median physical drift is a predeclared fraction (for example 0.1) of learned drift, then clip each physical norm to a validation percentile;
4. activate only at low-noise times where atom identities, pocket region, protonation, and SCF remain meaningful; and
5. test the sign by verifying that a noise-free infinitesimal update lowers $E$ without excessive deviation from the learned trajectory.

If energy/forces are evaluated on a denoised $\hat x_0(x_t)$ rather than noisy $x_t$, correct guidance formally requires the Jacobian $\partial\hat x_0/\partial x_t$. Ignoring it is an approximation. Quantum-region membership must not jump with each noisy pose; use a fixed region or a smooth embedding.

### 8.4 The guidance mechanisms are materially different

| Mechanism | Signal and where it acts | Oracle calls | Main issue for this thesis |
|---|---|---:|---|
| Classifier guidance | $\nabla_{x_t}\log p_\phi(y\mid x_t)$ added at each reverse step | One differentiable classifier pass/step | Requires a classifier trained across noise levels; DiffDock’s endpoint confidence is not automatically such a classifier. |
| ChemGuide | SPSA estimate from ± black-box oracle perturbations, applied during diffusion | Two oracle calls/guided step/direction sample | Dimension-light but noisy; demonstrated with GFN2-xTB for isolated generated molecules, not docking. |
| BADGER | Gradient of a learned differentiable energy proxy trained to mimic AutoDock Vina | One neural pass/backprop/step after training | Cheap amortization, but guidance inherits Vina/proxy errors; it is not direct quantum feedback ([BADGER](https://arxiv.org/abs/2406.16821), abstract/method). |
| Confidence Bootstrapping | Completed-pose confidence selects/weights pseudo-ground-truth poses; score model retrained, especially at high noise | Many endpoint rollouts during offline iterations; none at ordinary inference | Final feedback reaches early states through parameter updates, not trajectory gradients. A quantum oracle would be prohibitively expensive unless cached/distilled. |
| Direct generalized-force guidance | Analytic Cartesian forces projected as $J^TF$, metric-preconditioned and added each step | Ideally one electronic state/gradient per guided step | Most physically faithful; still $NT$ state/gradient solves and must include response/Pulay terms. |

The practical route, if reranking succeeds, is likely **oracle distillation**: compute high-quality final-pose SAPT labels, train a differentiable equivariant proxy on coordinates and decomposed components, validate it on unseen targets/decoys, and use its gradients during late reverse steps. This is conceptually closer to BADGER than to real-time QPU guidance and must be described as an amortized surrogate, not quantum computation at inference.

### 8.5 QCPMD, shadows, analytic VQE, and LUCJ+SQD for forces

**QCPMD + classical shadows.** A single QCPMD electronic state can support all $3N$ Hellmann–Feynman force observables, which is why shadows look attractive. But its derivation neglects Pulay/state-response terms, assumes the electronic state stays close enough to ground, couples PQC/nuclear fictitious dynamics, and uses measurement noise as a thermostat. DiffDock needs a stable conditional drift, not finite-temperature electronic/nuclear dynamics. The H$_2$ simulator study used fixed damping precisely to avoid force-variance estimation and admits that the nominal 70 K interpretation may then fail ([QCPMD.md](literature/mds/QCPMD.md), §II.B; §III, discussion after Figs. 3–4). Verdict: not a near-term guidance engine; use it only as conceptual evidence that many force observables share a state.

**Analytic VQE gradients.** These are the most natural direct-force candidate. A correct molecular gradient needs derivative one-/two-electron integrals, relaxed 1-/2-RDMs, orbital/circuit response and Pulay terms—not just Hellmann–Feynman expectations. Double-factorized Lagrangian work shows that this can be organized with modest active spaces in large QM/MM environments, albeit in classical simulation ([Hohenstein et al.](https://pubmed.ncbi.nlm.nih.gov/36948843/), abstract). Warm-starting parameters/orbitals from the prior diffusion step is plausible. It still requires a state and derivative measurement at each guided step and has no same-pair docking demonstration.

**LUCJ + SQD gradients.** The supplied DMET-SQD paper demonstrates energies and RDMs, not analytic nuclear gradients. After classical subspace diagonalization, one can in principle compute a 1-/2-RDM and contract it with derivative integrals; a fully analytic derivative also needs response of orbitals, embedding, bath, and selected eigenvector. More seriously, S-CORE-selected determinant spaces can change discontinuously with geometry, making a smooth trajectory difficult. Bitstring unions and LUCJ parameters can warm-start neighboring steps, but reused samples are not unbiased samples of the new state. Finite-difference SQD energies multiply the already large circuit/diagonalization cost. Verdict: later research, behind analytic VQE for guidance.

**SAPT forces.** Neither SAPT(VQE) paper derives nuclear gradients, and second-order response already has a substantial classical ERPA layer. Differentiating all SAPT terms, orbitals, active-space response, cutout, and caps is a separate method-development project. SAPT should remain an endpoint energy until that machinery exists.

### 8.6 Trajectory resource reality check

For $N=20$ and 20 steps:

- central finite differences at $m=6$: 9,600 electronic energies;
- SPSA: 800 energies for one random direction per step, with high variance;
- direct analytic forces: 400 electronic state/gradient solves;
- late-four-step force guidance: 80 state/gradient solves.

Those are per complex. A VQE “solve” contains many circuit evaluations during optimization plus all measurement groups; an SQD solve contains state-preparation samples plus million-dimensional classical diagonalizations; a QPU queue is not captured by nominal call count. The post-hoc experiment, by contrast, can reuse one static protein VQE/RDM across 20 poses. Real-time QPU guidance is therefore not remotely competitive in experiment 1. Late-step proxy guidance is the only credible intermediate milestone.

## 9. Final ranking and recommendation

### 9.1 Method evidence/resource table

“Qubits” below means logical/problem qubits used in the cited demonstration, not error-corrected physical qubits. An em dash means not reported or not applicable. Proposed resource values are explicitly marked estimates.

The condensed resource entries come from the detailed analyses above: first-order SAPT ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §§I.D–E, II.C and SI Tables VI–VIII); second-order SAPT ([SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), §§II–IV, Appendix D and Table SII); DMET-VQE ([DMET_VQE_PL.md](literature/mds/DMET_VQE_PL.md), §§2.3–2.7 and 3.3–3.5); DMET-SQD ([DMET_SGD.md](literature/mds/DMET_SGD.md), §§II–III and Table I); QCPMD ([QCPMD.md](literature/mds/QCPMD.md), §§II.B–III); and Q-Score ([Q-Score.md](literature/mds/Q-Score.md), §§IV–V and Tables I–V). Proposed QM/MM and hybrid rows have no claimed paper demonstration.

| Method | Physical quantity | Protein treatment | Quantum primitive | Qubits | Measurements/QPU calls | Reranking suitability | Guidance suitability | Classical-shadows compatibility | Major approximation | Closest classical baseline | Demonstrated protein–ligand? | Hardware demonstrated? | Verdict |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|
| **Proposed SAPT0 + first-order SAPT(VQE) correction** | Classical second-order interaction plus correlated $E_{elst}^{(1)}+E_{exch}^{(1)}$ correction | Fixed capped pocket monomer; optional classical remainder | VQE state + 1-/2-RDM; classical SAPT | 12–16 proposed, matching papers | One reusable protein VQE/RDM per complex plus grouped RDM measurements; pose-specific classical contractions | **High for experiment 1**, if mixed-level approximation is validated | Low; no SAPT gradient | High in principle for all RDM-derived components; later ablation | Mixed correlation levels, cutout, small basis, no solvent/strain | Identical SAPT0 and SAPT(CASCI) correction | Components demonstrated on KDM5A, but not this hybrid or pose ranking | No SAPT hardware | **Recommended now**, explicitly experimental |
| **First-order SAPT(VQE) alone** | Electrostatic + exchange interaction | Capped pocket/ligand monomers; one may be RHF | VQE + 1-/2-RDM | 16 on KDM5A (54 before reduction) | Naive $O(N_a^4)$ RDM terms; exact shots unreported | Low: incomplete physics | Low: no gradient | High | Missing induction/dispersion; active-space/exchange/basis errors | SAPT(CASCI), SAPT(RHF) | Yes: five KDM5A ligand complexes, not decoy poses | No; ideal statevector only | **Reject standalone** |
| **Second-order SAPT(VQE)-ERPA** | Electrostatic, exchange, induction, dispersion and exchange counterparts | Two monomers; current prototype small enough for ~130 orbitals | VQE 1-/2-RDM; classical ERPA/response | 12–16 reported | VQE optimization + RDM groups; shots unreported; no excited-state QPU calls | **Highest physical fit**, limited engineering scale | Low until SAPT gradients exist | High, but ERPA error propagation must be tested | ERPA, active space, basis, cutout; classical tensor bottleneck | SAPT(CASCI) same active space; SAPT0/SAPT(DFT) | No; dimers and Mn–nitrosyl model only | No; ideal statevector only | **Runner-up / small-subset validator** |
| **DMET-VQE protein-field model** | Ligand/fragment embedded energy; bound-versus-solvent proxy | Full protein as fixed point charges; ligand DMET fragment | Iterative VQE energy/RDM | 4 reported | IBM 6,000 shots/iteration + 60,000 final; ion 8,000 after classical optimization; bound and solvent runs | Medium-low; pose-dependent but missing protein electrons | Low; no gradient | Possible, not compelling at 4q | STO-3G, one 2e/2o fragment, fixed charges, error cancellation | DMET-FCI/CASCI/CCSD/HF same setup | Yes: 12 BACE1 ligands, affinity ranking not poses | Yes: IBM Casablanca and Honeywell H1-S2 | **Hardware fallback, not primary** |
| **DMET-SQD** | Embedded fragment total energies/RDMs | DMET fragment+bath; no protein example | LUCJ computational-basis sampling; S-CORE; classical Davidson | 27 and 32 reported (full 41/89) | 3/6 fragment circuits; ~1k–10k raw configurations per batch; several batches | Low-medium after a new interaction/total-energy definition | Low; no analytic gradient | Low: native bitstrings already serve SQD; RDM classical after diagonalization | DMET bath, STO-3G, determinant truncation; millions-dimensional diagonalization | DMET-FCI/CCSD; HCI/CCSD(T) | No: H$_{18}$, cyclohexane | Yes: IBM Cleveland | **Later, not experiment 1** |
| **Proposed QM/MM + SQD** | Comparable total QM/MM energy for fixed atom set | Ligand + pocket QM; remainder MM | LUCJ/SQD and classical subspace solve | ~20–40 estimated | At least one sampled state and large diagonalization per changed Hamiltonian | Medium in principle | Medium-low after new derivative theory | Low for native SQD; possible for derived RDMs | Boundary/embedding, total-energy cancellation, subspace discontinuity | SHCI/CASCI/CCSD/DFT QM/MM same region | No supplied demonstration | SQD hardware exists, not this workflow | **Longer-term platform** |
| **ext-SQD** | Ground and low-lying excited states | Active molecular Hamiltonian; no docking workflow | Extended SQD | Up to 77 reported externally, system-dependent | Extra classical step beyond SQD | No clear gain for ground-state pose rank | Low; no demonstrated force | As for SQD | Excited-state machinery irrelevant to primary score | Multistate selected CI/CASCI | No | Yes for molecular excited-state studies, not docking | **Reject for experiment 1** |
| **Analytic active-space VQE/QM-MM gradient** | Total energy and relaxed nuclear gradient | QM active region in large MM environment | VQE + relaxed RDM/Lagrangian response | Active spin-orbital count; case-specific | State optimization plus factorized density/response measurements at each geometry | Medium as an energy model | **Highest quantum-method fit for future direct forces** | Compatible for densities, but shadows do not remove response | Active-space, QM/MM boundary, VQE/response/Pulay error | CASSCF/CCSD/DFT analytic QM/MM gradient | Large QM/MM examples, not docking | Very recent H$_2$/H$_2$O preprint only | **Future guidance rank 1** |
| **QCPMD + local-Pauli shadows** | Hellmann–Feynman nuclear forces along coupled fictitious dynamics | Only H$_2$ tested | PQC dynamics; random local Pauli shadows | 4 in force-variance test | 51 shadows vs 51 shots per Pauli in study; every MD step | Not a natural scalar reranker | Conceptually medium, practically low | Native feature, but higher observed variance and known-observable caveat | Neglected Pulay/response, near-ground assumption, effective thermostat | Ordinary grouped force measurement; classical AIMD | No | No; simulator only | **Do not use now** |
| **Q-Score/DC-QAOA** | Maximum compatible GNN-predicted NBO-like anchor weight | 5 Å extracted site; graph reconstructs rigid pose | DC-QAOA clique optimization | 6/10/12 | Hardware: 1,000 circuits × 10,000 shots; COBYLA up to 5,000 iterations in simulation | Low for arbitrary fixed poses without changing method | None: no coordinate gradient | Not relevant; quantum state is combinatorial | Learned $E^{(2)}$, sparse anchors, target $\tau$, rigid Kabsch, QAOA noise | Exact/greedy MW clique; direct SIMG* sum | 11 co-crystals and 1,000 cross-ligand scores, not fixed decoys | Yes: IBM Eagle; 10q noise-limited | **Closest novelty prior; not primary baseline** |

### 9.2 Separate rankings for the three goals

#### A. Post-hoc pose reranking now

1. **Proposed SAPT hierarchy** (SAPT0 plus first-order SAPT(VQE)/CASCI correction): best compromise of direct interaction physics, protein precedent, monomer reuse, and a clean same-method control.
2. **Full second-order SAPT(VQE)-ERPA:** best observable, but only on a small size-qualified subset until classical scaling and code availability improve.
3. **DMET-VQE protein-field model:** easiest genuine hardware reproduction, but its missing protein electrons and 4-qubit fragment make it a weak pose scorer.
4. **QM/MM+SQD:** plausible but must be invented and validated; too expensive for the first candidate table.
5. **DMET-SQD:** strong hardware-scale demonstration in other molecules, yet no protein, direct interaction score, or efficient per-pose workflow.

First-order SAPT alone, QCPMD, ext-SQD, and published Q-Score are rejected for the primary fixed-pose reranking endpoint for the reasons in the table.

#### B. Quantum-derived diffusion guidance in future work

1. **Analytic active-space VQE/QM-MM gradients**, projected through the exact DiffDock tangent Jacobian.
2. **QM/MM+SQD with newly developed smooth analytic derivatives**, if determinant/bath continuity can be controlled.
3. **QCPMD-inspired multi-force measurement**, only after full relaxed forces and deterministic guidance replace its thermostat assumptions.
4. **SAPT gradients**, last because neither supplied implementation derives them and second-order response differentiation is substantial.

In practice, a distilled differentiable proxy trained on endpoint quantum labels is likely to precede all four, but it is an ML-guidance result, not QPU-in-the-loop guidance.

#### C. Long-term fault-tolerant quantum advantage

1. **Fault-tolerant correlated active-space/QM-MM energies and analytic derivatives**, potentially using phase estimation or another systematically improvable solver rather than VQE. This targets the hardest electronic subspace while classical embedding handles the environment.
2. **Fault-tolerant monomer solvers inside complete SAPT response**, if RDM/response extraction can avoid prohibitive expectation-value sampling.
3. **Quantum-centric embedding/SQD variants**, if the required determinant support and classical diagonalization are shown not to erase scaling benefits.

No item has demonstrated end-to-end advantage for protein–ligand docking. Fault tolerance removes some noise/optimization constraints but does not remove active-region construction, solvent/entropy, geometry sampling, or the need for a better classical comparator.

### 9.3 Recommended method, runner-up, and explicit rejections

**Recommended experiment-1 method:** the locked **SAPT0 + active-space first-order SAPT(VQE) correction** defined in §7.4–7.5. Use one static protein active-space state per complex, ligand RHF, monomer-centred basis for the correction, a fixed union pocket, and a separately reported classical remainder. CASCI is the scientific reference; VQE is the estimator under test.

**Runner-up:** full second-order SAPT(VQE)-ERPA on up to four genuinely size-qualified fixed cutouts. Promote it to the main method only if an available implementation handles the whole fixed pocket/basis and reproduces SAPT(CASCI) efficiently. If no whole cutout fits, omit the pose-ranking comparison rather than present a ligand–residue microcomplex as an equivalent system.

**Explicitly rejected for experiment 1:**

- first-order SAPT(VQE) alone, because its own protein–ligand example shows missing dispersion/induction reverse the useful ordering;
- DMET-SQD/QM-MM+SQD, because the protein–ligand energy definition and per-pose workflow are not demonstrated and classical diagonalization is already large;
- ext-SQD, because excited states do not solve the ground-state pose problem;
- QCPMD or direct force guidance, because they multiply quantum work across the trajectory and do not deliver the required relaxed docking force today;
- classical shadows, because statevector simulation has no measurement problem and targeted grouping should be measured first;
- Q-Score as an arbitrary-pose reranker, because its published graph/QAOA/Kabsch method solves a different discrete reconstruction problem; and
- a hardware-first experiment, because a tiny noisy circuit cannot distinguish useful physics from quantum computation.

### 9.4 Experiment architecture

```text
PoseBusters Benchmark (308 holo complexes)
        |
        +-- blind deterministic chemistry/size filters + hash-stratified selection
        |                           -> locked 24-complex manifest
        |
        v
Frozen DiffDock-L generator -- 20 seeded poses/complex -- native never inserted
        |
        +---------------- candidate table ----------------+
        |                                                  |
        v                                                  v
Learned/empirical baselines                       Fixed region built from
confidence | Vina/GNINA | AA-Score               UNION of all 20 candidates
PoseCheck | PoseBusters                                  |
                                                         v
                                     protein RHF/localization/AVAS -> CASCI/VQE
                                     one reusable protein 1-/2-RDM per complex
                                                         |
                       ligand RHF by conformer + pose-specific intermonomer integrals
                                                         |
                          +------------------------------+-------------------+
                          v                              v                   v
                       SAPT0                 CASCI-corrected SAPT     VQE-corrected SAPT
                          +------------------------------+-------------------+
                                                         |
                                    fixed classical remainder + identical validity policy
                                                         |
                                                         v
                        paired ranking metrics, uncertainty, failure attribution, cost ledger
                                                         |
                            go: larger validation | no-go | physics-only/no-quantum result
```

### 9.5 Staged implementation plan

No code is to be written in the present task. If approved, implementation should proceed through gates:

1. **Protocol freeze (week 1).** Pin repositories/checkpoints, benchmark version, seeds, filters, protonation, region/basis rules, metrics, tolerances, and statistical notebook in a timestamped preregistration. Create the empty manifest schema and provenance fields.
2. **Candidate/benchmark layer (weeks 1–2).** Run DiffDock-L, lock the 24 IDs and 480 poses, compute RMSD only after locking, run PoseBusters/PoseCheck, and establish oracle coverage. Stop if fewer than 12 covered complexes.
3. **Classical scorer gate (weeks 2–4).** Run confidence, Vina/GNINA, AA-Score, fixed region construction, remainder MM, RHF/SAPT0. Inspect only predefined feasibility diagnostics. If SAPT0 cannot score ≥20 complexes or its rank signal is clearly absent, do not build quantum machinery yet.
4. **Active-space validation (weeks 4–6).** For each static protein monomer, build AVAS/local orbitals, run CASCI/SHCI diagnostics, verify region/basis reproducibility, and compute CASCI-corrected pose scores. This decides whether correlated physics is non-negligible before VQE.
5. **Ideal VQE (weeks 6–8).** Reproduce CASCI 1-/2-RDM contractions at 6e/6o then 8e/8o; test one-muCJ and only deepen under a predeclared convergence rule. Verify energy-component and ranking tolerances, not total energy alone.
6. **Conditional full-second-order subset and sensitivity (weeks 8–10).** Run ERPA/full-SAPT only for the zero-to-four whole cutouts that pass the explicit orbital/memory gate; separately run region/basis/solvent sensitivity. Record an empty full-SAPT subset as an engineering limitation, not a reason to change regions after seeing scores.
7. **Statistics and locked report (week 10).** Execute the prewritten paired analysis, publish all failures, and classify the outcome using §7.9.
8. **Optional measurement/QPU gate (after the scientific result).** Benchmark deterministic groups, then shadows as an ablation; run at most two complexes on hardware if ranking precision survives finite shots. Otherwise report a simulator-only thesis result.
9. **Only after validation:** enlarge the cohort and/or train a differentiable proxy. Do not start direct QPU guidance from the pilot.

### 9.6 Expected dependencies and compute

**Docking/structure:** pinned DiffDock-L/PyTorch/e3nn stack; RDKit; PDBFixer or equivalent repair; a pinned protonation tool such as PROPKA/Reduce; OpenMM/AmberTools for caps, remainder terms, and restrained sensitivity; PoseBusters; PoseCheck/AA-Score; ProLIF; Vina and GNINA; a symmetry-aware RMSD package.

**Electronic structure:** PySCF for SCF, integrals, AVAS, localization, CASSCF and QM/MM utilities; PSI4 for classical SAPT0 where compatible; Dice/SHCI if the active-space selection follows the papers; the SAPT(VQE) authors’ implementation if obtainable. The papers used TeraChem/Lightspeed and in-house Quasar/Vulcan, so lack of accessible SAPT-RDM/ERPA code is a major schedule dependency, not a minor package install ([SAPT_VQE_PL.md](literature/mds/SAPT_VQE_PL.md), §II.A; [SAPT_VQE_II.md](literature/mds/SAPT_VQE_II.md), Appendix D).

**Quantum:** Qiskit/Qiskit Nature or an equivalent fermion-to-qubit and circuit stack; exact statevector and shot simulator; optional runtime provider; explicit circuit transpilation and measurement-group logging. Do not mix several quantum SDKs until the CASCI/RDM convention tests pass.

**Resource envelope:**

- DiffDock-L and ordinary scorers: one modern GPU plus CPU, likely hours rather than the project bottleneck.
- 480 fixed-pose classical intermonomer integral/SAPT jobs: parallel CPU/GPU node-days; benchmark 3 representative systems before reserving the cohort. Plan for 128–512 GB RAM on the largest pocket jobs until tensor scaling is measured.
- 24 static 12–16-qubit VQE states: exact simulation is feasible on a GPU/workstation, but optimizer iterations and RDM construction may require GPU-hours to low GPU-days per difficult state. The protein state is reused across 20 poses.
- Full second-order ERPA: one A100-class GPU and high-memory host for at most four ≤130-orbital whole cutouts, consistent with the supplied paper’s prototype scale. A 100-heavy-atom first-order region will generally exceed this gate; zero qualifying systems is plausible unless an optimized density-fitted implementation is available.
- Storage: tens of GB for coordinates, integrals/checkpoints, RDMs, score tables, and logs; save hashes and metadata, not redundant statevectors where regeneration is deterministic.
- Optional QPU: 12–16 logical/problem qubits, shallow number-preserving circuits, and an empirically determined grouped-measurement budget. Expect at least $10^5$–$10^7$ aggregate shots if many 2-RDM contractions need sub-kcal/mol ranking precision; this is a planning range, not a literature-demonstrated SAPT count. Abort when confidence intervals cannot separate relevant pose gaps.

### 9.7 Main scientific and engineering risks

1. **No quantum-relevant correlation.** Drug-like closed-shell pockets may be accurately single-reference; KDM5A already showed CASCI and RHF nearly coincide. This is the most likely negative result.
2. **Interaction energy is the wrong ranking target.** Solvation, strain, protonation, entropy, and receptor response can dominate native-pose selection. Validity controls mitigate but do not repair this mismatch.
3. **Mixed-level SAPT is not systematically balanced.** Replacing only first-order terms may create inconsistent correlation. The full second-order subset is a required diagnostic.
4. **Region instability or noncomparability.** Per-pose cutouts, changing caps, changing active orbitals, or dimer-centred ghost functions can manufacture score differences. The fixed union and cache audit are mandatory.
5. **Basis/fragment artifacts.** 6-31G is chosen for feasibility, not converged dispersion. Rank reversal under the larger-basis subset is a stop signal.
6. **Physical invalidity.** Interaction-only energies can reward pathological contacts or strained conformers. Report raw and validity-controlled results; never hide failures by minimization.
7. **Benchmark leakage/scope.** Holo receptors and feasibility filters simplify the task. The result cannot be generalized to apo, flexible, metallated, covalent, or large-ligand docking.
8. **Classical bottleneck.** Integral transformation, SAPT/ERPA tensors, and SQD diagonalization can dominate, making qubit count a misleading feasibility statistic.
9. **Measurement noise changes ranks.** Chemical-accuracy energy error is not sufficient if pose gaps are smaller; rank-stability intervals are the relevant criterion.
10. **Unavailable research code.** Reimplementing active-space SAPT RDM contractions or ERPA could exceed an MSc schedule. If code access fails, narrow to classical SAPT0 plus a reproduced small-molecule SAPT(VQE) component study rather than silently substituting a different method.

### 9.8 Claims that are defensible—and claims that are not

**Defensible if the protocol is completed:**

- a controlled test of whether a direct SAPT interaction signal improves *within-complex* ranking of a frozen DiffDock-L candidate set on a locked cognate-holo PoseBusters subset;
- a decomposition of any gain into classical physical interaction, active-space correlation, VQE estimation, learned confidence, and strain/clash contributions;
- a resource and reuse analysis showing what one static protein RDM can and cannot amortize across poses; and
- if positive, evidence that a quantum-representable correlated RDM observable contains incremental pose information in this restricted setting.

**Do not claim:**

- “the first quantum docking/scoring method” or novelty from using a QPU—Q-Score and earlier quantum docking work preclude it;
- quantum speedup, advantage, supremacy, or better scaling from a simulator/noisy pilot;
- prediction of binding affinity or $\Delta G_{bind}$ from a frozen electronic interaction score;
- that VQE is more accurate than CASCI under the same active space, or more useful than the classical same-method calculation;
- that first-order SAPT is a complete interaction energy;
- that a detached Compass Score is a validated inference-time ranker;
- that FlowDock/Boltz-2 affinity is irrelevant or already a same-pair pose score without direct decoy evidence;
- that holo rigid-receptor results generalize to apo/flexible docking;
- that classical shadows reduce total shots without an observable-weighted variance comparison;
- that one successful hardware execution establishes chemical accuracy or docking utility; or
- that direct quantum diffusion guidance is practical at current trajectory call counts.

## 10. Immediate implementation decision

The next action should be a **two-complex dry-run specification review**, not a 24-complex calculation: verify that the selected SAPT implementation can consume one fixed capped protein monomer, reuse its active RDM across rigidly varied ligand poses, and produce SAPT0/RHF/CASCI-corrected scores with consistent basis and sign conventions. This is a cheap workflow validation, not part of the statistical result.

If that review confirms code access and region feasibility, implement the 24×20 plan exactly as frozen. If it fails, preserve the thesis question but change the first experiment to a classical SAPT0 reranking study plus a small, separately labeled SAPT(VQE) reproducibility case. That fallback can establish whether the physical observable is worth pursuing without inventing a quantum advantage.
