# SAPT

First-order symmetry-adapted perturbation theory over the encoded protein cutout and each candidate pose, producing an interaction energy per pose. The stage after `encode.py`; nothing in preprocessing changes.

## Glossary

- $A$, $B$ — the monomers: the frozen cutout $A^{\cup}$, and one ligand pose. $N_A$, $N_B$ are their electron counts.
- $\mu$, $\nu$ — atomic orbital (AO) indices, $\mu$ on $A$ and $\nu$ on $B$. Basis functions, not orbitals.
- $i$, $t$ — molecular orbital indices: $i$ core (doubly occupied, frozen), $t$ active.
- $C$ — MO coefficients; row $\leftrightarrow$ AO; column $p$ $\leftrightarrow$ orbital $p$ over the AOs. Kept as `orbital_initial`. Maps $\text{AO} \mapsto \text{MO}$
- $\gamma$, $\Gamma$ — spin-summed one- and two-particle density matrices of a monomer's ground state. Kept as `rdm1`, `rdm2`.
- Superscripts $c$, $a$ — which orbital block the indices fall in: $\Gamma^{cc}$, $\Gamma^{ac}$, $\Gamma^{aa}$.
- $\lambda_{pqrs}$ — the cumulant: $\Gamma$ minus the part factorising into two $\gamma$'s. Zero for a single determinant, so it is the correlation, isolated.
- $D^{A}$ — $A$'s density in the AO basis, core plus active.
- $S$ — overlap matrix; $S_{\mu\nu}$ its $A$--$B$ block. Exchange exists only because that block is nonzero.
- $J[\gamma]$, $K[\gamma]$ — the mean field a charge distribution $\gamma$ exerts in the AO basis, and its antisymmetry counterpart.
- $\tilde{v}$, $\tilde{J}$, $\tilde{K}$ — the same with electron-nucleus and nucleus-nucleus terms folded in (Eq. 10), so a contraction returns a total interaction energy.
- $T_1 \ldots T_5$ — the five terms of first-order exchange (Eq. 14).
- $S^{2}$ — the single-exchange approximation: truncation at two powers of intermonomer overlap.

## Summary of what is provided so far

`prepare.py` freezes one cutout per complex and relaxes each pose inside it with the protein's atoms fixed, so $A$ is the same nuclei for every pose and only $B$ moves.

| Monomer | Artefact | Supplies |
| --- | --- | --- |
| $A$, per complex | `_solved.npz`, `_encoded.npz`, `_casci.npz` | $C$, `active_electrons`, `active_space_size`, $\gamma$, $\Gamma$ |
| $B$, per pose | `pose_scf/<complex>_pose<i>_rhf.chk` | RHF `mo_coeff`, `mo_occ` |

`e_core`, `h1`, `h2` and the qubit operator are not read. They were the means of finding $\gamma$ and $\Gamma$.

## Method

Per pose,

$$E^{(1)}_{\text{int}} = E^{(1)}_{\text{elst}} + E^{(1)}_{\text{exch}}(S^{2}),$$

each term a contraction of the two monomers' density matrices against integrals spanning both. Poses are then ranked by $E^{(1)}_{\text{int}}$ and the top-ranked one held against the deposited ligand.

**First order only.** Dispersion is an $O(o_A v_A o_B v_B)$ contraction, around $9 \times 10^{9}$ amplitudes for the smallest chosen cutout, and Psi4's SAPT works in a dimer-centred basis, so it would re-solve the protein's SCF for each of the 430 poses rather than once per complex. First order is Fock-build scaling. Whether to add a two-body D3 term as an attractive counterweight is open: the paper found first order alone insufficient to rank distinct *ligands*, though pose ranking is a different task.

**Appendix A is not transcribed.** Every block of $\Gamma$ holding a core index is its separable part (Eq. 28), and the ligand's $\Gamma$ is separable whole. Separating $\Gamma^{aa}$ too leaves one AO expression in the two densities and one cumulant term over the active space (step 4), in place of the ten block expressions the paper's route would need.

### 1. Monomer densities

$$D^{A}_{\mu\mu'} = 2\sum_{i}^{n_c} C_{\mu i} C_{\mu' i} + \sum_{tt'} C_{\mu t}\, \gamma_{tt'}\, C_{\mu' t'}, \qquad n_c = \frac{N_A - N_{\text{act}}}{2}$$

and $D^{B} = 2 C^{B}_{\text{occ}} C^{B\,T}_{\text{occ}}$. Narrowing never reorders, so the active block is columns $n_c$ to $n_c + N_{\text{act}}$ of `orbital_initial`. Checked by $\mathrm{tr}(D^{A}S^{A}) = N_A$ and $\mathrm{tr}(D^{B}S^{B}) = N_B$.

### 2. Dimer integrals

All PySCF, none held as a four-index tensor:

- `gto.conc_mol(mol_A, mol_B)` for the concatenated AO basis, $A$'s functions then $B$'s.
- `gto.intor_cross("int1e_ovlp", mol_A, mol_B)` for $S_{\mu\nu}$.
- `int1e_rinv` under `set_rinv_origin`, summed over one monomer's nuclei, for $V_B$ over $A$'s AOs and $V_A$ over $B$'s (`potential`), and through `gto.intor_cross` for either monomer's nuclei between $A$'s functions and $B$'s (`attraction`), which exchange's mixed pairs need. $V_{AB}$ is the cross nuclear repulsion.
- `scf.jk.get_jk` over four molecules, which contracts one block of the concatenated basis without storing its ERIs. `generalized` adds Eq. (10)'s one-electron terms to it in factorised form, so any block of $\tilde{v}$ is contracted as `get_jk` contracts the plain integrals. It screens only when handed a `vhfopt`.

For electrostatics the $1/N_A$, $1/N_B$ weights in $\tilde{v}$ cancel against the densities' traces, leaving $\tilde{J}[\gamma^{B}] = J[\gamma^{B}] + V_B + \frac{S_A}{N_A}(\langle V_A \rangle_{\gamma^{B}} + V_{AB})$. Exchange contracts $\tilde{v}$ with matrices whose trace is no electron count, $R^{B}$ and $W$ in step 4, so `generalized_coulomb` keeps the weights.

### 3. Electrostatics

$E^{(1)}_{\text{elst}} = \sum_{\mu\mu'} D^{A}_{\mu\mu'} \tilde{J}[D^{B}]_{\mu\mu'}$ collapses to the classical Coulomb interaction of two charge distributions: electron-electron, each monomer's electrons against the other's nuclei, and nucleus-nucleus. Four contractions over what step 2 already built.

### 4. Exchange

Split each $\Gamma$ into its separable part and its cumulant, $\Gamma_{pqrs} = \gamma_{pq}\gamma_{rs} - \frac{1}{2}\gamma_{ps}\gamma_{rq} + \lambda_{pqrs}$, in PySCF's `rdm2` order $\Gamma_{pqrs} = \langle p^{\dagger} r^{\dagger} s\, q \rangle$. The order is pinned by $\sum_{r} \lambda_{pqrr} = \frac{1}{2}(\gamma^{2} - 2\gamma)$, which also checks the cutout's saved `rdm2`. Eq. (14) is linear in each monomer's $\Gamma$, and the ligand, a determinant, has no cumulant, so

$$E^{(1)}_{\text{exch}}(S^{2}) = E_{\text{exch}}[D^{A}, D^{B}] + E_{\text{exch}}[\lambda_A].$$

**Separable part** (`sapt.exchange`). Only the two AO densities enter. Written out, $T_5$ cancels the part of $T_4$ that factorises into $\langle P \rangle E^{(1)}_{\text{elst}}$, which leaves six contractions:

$$E_{\text{exch}}[D^{A}, D^{B}] = -\tfrac{1}{2} D^{A}\cdot\tilde{K}[D^{B}] - \tfrac{1}{2} W\cdot\big(\tilde{J} - \tfrac{1}{2}\tilde{K}\big)[D^{B}] - \tfrac{1}{2} W\cdot\big(\tilde{J} - \tfrac{1}{2}\tilde{K}\big)[D^{A}] + \tfrac{1}{4} D^{A}\cdot\tilde{J}[R^{B}] + \tfrac{1}{4} R^{A}\cdot\tilde{J}[D^{B}] - \tfrac{1}{8} W\cdot\tilde{K}[W]$$

with $W = D^{A} S D^{B}$, $R^{A} = W S^{T} D^{A}$, $R^{B} = D^{B} S^{T} W$ and $X \cdot Y = \sum_{\mu\nu} X_{\mu\nu} Y_{\mu\nu}$. Each $\tilde{J}$ and $\tilde{K}$ runs over the block its partner spans: $\tilde{K}[D^{B}]$ over $AA$ through $(AB|BA)$ integrals, the two $(\tilde{J} - \frac{1}{2}\tilde{K})$ over $AB$ through $(AB|BB)$ and $(AA|AB)$, and $\tilde{K}[W]$ over $AB$ through $(AA|BB)$. This is the SAPT(HF) $E^{(10)}_{\text{exch}}(S^{2})$ expression, but derived from Eq. (14) it never uses $DSD = 2D$, so the non-idempotent $D^{A}$ goes through it as $D^{B}$ does. With both densities RHF it is SAPT(RHF).

**Cumulant part** (`sapt.cumulant_exchange`). $\lambda_A$ lives in the active space, and only the terms holding $\Gamma_A$, $T_3$ and $T_4$, can carry it. They are the paper's A5 and A13 with $\lambda$ in place of $\Gamma^{aa}$:

$$E_{\text{exch}}[\lambda_A] = -\frac{1}{2}\sum_{tuvw}\lambda_{tuvw}\Big(\tilde{v}(vw|u\,x_t) + M_{tu}\,\tilde{J}[D^{B}]_{vw} - \frac{1}{2}\,\tilde{v}(vw|x_t\,x_u)\Big),$$

where $x_t = \sum_{\nu}(C_a^{T} S D^{B})_{t\nu}\,\chi_{\nu}$ is a function over $B$'s AOs and $M = C_a^{T} S D^{B} S^{T} C_a$. Both integrals come from J builds of the $N_{\text{act}}(N_{\text{act}}+1)/2$ active pair densities $C_v C_w^{T}$ over the $AB$ and $BB$ blocks, contracted afterwards with $u$ and $x_t$. Only the contraction with $\lambda$ itself is $O(N_{\text{act}}^{4})$.

**In place of the paper's block expressions.** Those insert Eq. (28) wherever a core index appears and keep $\Gamma^{aa}$ whole. Separating $\Gamma^{aa}$ as well is the same algebra, and the core/active distinction then drops out of everything but $\lambda$. Nothing is transcribed from the main text's Eqs. (30)--(39), which the OCR of `SAPT_VQE_PL.md` garbles, nor is A22 or $T_2$'s $cc$ blocks, which the paper never wrote out. Against Eq. (14) written out over the water dimer's occupied orbitals the split holds to $10^{-17}$, and that reference reproduces Tables III and IV: 0.0054820 against 0.005482 Hartree at SAPT(RHF), and 0.0055594 against 0.005559 at SAPT(CASCI).

### 5. Where the correlation lives

In two places: $D^{A}$, through every term of steps 3 and 4, and $\lambda_A$, through $E_{\text{exch}}[\lambda_A]$ alone. A determinant has no cumulant and an idempotent density, so a doubly occupied active space has to give back SAPT(RHF) exactly, whatever the window. That is the consistency test, and it runs through the same code as the correlated path. Over the water dimer the correlated density moves the exchange by $+1.0 \times 10^{-4}$ Hartree and the cumulant moves it by $-2.2 \times 10^{-5}$, both past the tables' last digit. The stage keeps each pose's $E_{\text{exch}}[\lambda_A]$ apart as `cumulants`, so the cumulant's share is read without a second run. The density's share would be the difference from a SAPT(RHF) run against the protein's own RHF density, which is future work.

### 6. Reranking

`sapt.join` gives each row of `confidence.csv` its pose's `elst`, `exch` and `cumulant`, the share of `exch` the cumulant carries, keyed by complex and source as `confidence.join` keys its scores, and `interaction` = `elst` + `exch`. A pose without energies is dropped and reported. `sapt.rank` numbers each complex's poses from the lowest `interaction` as `rank_sapt`, a tie going to DiffDock's rank, and `sapt.summarise` adds `top1_sapt` to `confidence.summarise`'s row, so the three top-1s are over the same poses. They are written as `sapt.csv` and `sapt_summary.csv` beside `confidence.py`'s tables, through its `write_table` and `read_table`. `sapt.run` does all of this over a screen, `python sapt.py --name v1_1_mm_unsize`: it reads `confidence.csv` from the screen's directory and each complex's `<complex>_sapt.npz` from the complex's own, and leaves out, and reports, a complex with none.

## Implementation

- `src/utils/sapt.py` — steps 1--5: `density` and `active`; the integral layer, `dimer`, `overlap`, `potential`, `attraction`, `repulsion`, `coulomb`, `generalized` and `generalized_coulomb`; `electrostatics`, `exchange`, `cumulant` and `cumulant_exchange`.
- `src/sapt.py` — a `SAPT` stage class mirroring `EncodeProtein`: reads $A$'s artefacts and the pose SCFs, and `elst`, `exch` and `interaction` fill a list per pose, `exch` keeping the cumulant's share as `cumulants`. Once every pose has all four, `interaction` keeps them in one record per complex, `<complex>_sapt.npz`, beside each pose's `source`, and a stage handed that directory again reads them back rather than scoring the poses again. Nothing is kept part-way, so a complex interrupted mid-ensemble is scored again from its first pose. Step 6's `join`, `rank`, `summarise`, `FIELDS` and `SUMMARY_FIELDS` sit beside it, and `run`, the script's entry point.
- `src/utils/save.py` — `sapt_path`, `load_sapt` and `save_sapt`. The record comes last in `_supersede`'s order, so a new preparation, SCF, solved space, Hamiltonian or CASCI state discards it, and `save_pose_scf` discards it too, since it is the only artefact built on the poses' SCFs.
- `src/confidence.py` — `read_table` learns the new columns: `rank_sapt` an integer, the four energies decimals, `top1_sapt` a boolean. `rank` numbers through `number`, which takes the order as a key and which `sapt.rank` shares, and `is_near_native` is public for `sapt.summarise`.
- `src/run.py` — a fourth stage after `CASCI`, skipped when the complex's scores are kept. It is built after CASCI, so a new window, which discards the scores, scores the poses again. A complex whose protein fails CASCI stops there and is never scored, so nothing falls back to SAPT(RHF), and `sapt.run` leaves it out.

**Oracle.** `tests/sapt/reference.py` writes Eq. (10) and Eq. (14) out as the four-index tensors they define over the water dimer's 26 functions, Eq. (14) from its definition rather than the paper's blocks. The CASCI vector is carried into the core-plus-active space, so FCI's own density matrices supply every block, Eq. (28)'s included. It reproduces Tables III and IV to their last digit, so Psi4 is not needed: its SAPT0 density-fits the integrals and would not match to this precision anyway.

**Cost.** Per pose, eight J/K builds for the separable part, each over one block of the dimer's functions, and two J builds of $N_{\text{act}}(N_{\text{act}}+1)/2 = 105$ pair densities for the cumulant. The largest class is $(AA|AB)$, about $n_A^{3} n_B / 2 \approx 3 \times 10^{11}$ integrals a pass for the 1695-function cutout, and the separable part's two builds over it share one `get_jk` call. A pair over one monomer's functions keeps its permutational symmetry wherever `get_jk` has a kernel for the contraction, which halves that pass. `get_jk` screens nothing unless handed a `vhfopt`, and most of those integrals vanish away from the interface, so screening is the first thing to try if the cluster run is slow. The cumulant's 105 pair densities are held at once over $A$'s functions, about 2.4 GB for that cutout. Twelve complexes, 430 poses.

**Future work.** Whether a two-body D3 term is in scope, as an attractive counterweight to first order's exchange. It needs only the geometries, so it would be one more column, and the kept scores would not change. And whether to run SAPT(RHF) beside SAPT(CASCI), against the protein's own RHF density, which would give the correlated density's share per pose and a SAPT(RHF) ranking to hold the CASCI one against. It costs a second electrostatics and separable exchange per pose, but no cumulant term, about half again on the test fragment.
