# Complex Log

Every complex that changed how this pipeline works, rather than merely exercising a rule it already had. A few appear twice because they forced separate decisions at different stages.

Only the 36 complexes the tests use are kept under `src/tests/data`; the rest were measured during dataset-wide surveys and would need re-downloading to re-check.

## Fetching

| Complex | What it changed, and why |
| --- | --- |
| `7OPG_06N`, `7R9N_F97`, `7UQ3_O2U`, `6M73_FNR`, `7SCW_GSP` | DiffDock writes a positive confidence without a sign, so a pose pattern assuming a leading minus silently discarded 692 poses across 40 complexes and left three of these with none; the pattern now makes the sign optional. |

## Scope

| Complex | What it changed, and why |
| --- | --- |
| `7USH_82V` | An ethylene glycol (EDO A504) inside a 13-residue cutout was being rejected as a heterogen, which is wrong: nothing is bonded to it and it is no part of the biology, so `ADDITIVES` now lists cryoprotectants, precipitants, buffers and simple non-metal ions for `_clean` to delete instead. This took the accepted set from 83 to 110. |
| `7D5C_GV6` | GVU, a 47 heavy-atom ligand, sits in the shell alongside an ethylene glycol, so the additive exemption had to be applied per residue rather than per complex, or excusing the EDO would have excused the GVU with it. |
| `7DUA_HJ0` | LYS A789 is modelled with CE and NZ at zero occupancy inside the cutout, meaning the crystallographer did not place them; added the `ZERO_OCCUPANCY` rule rather than hand invented coordinates to RHF. |
| `7YZU_DO7` | Its eight zero-occupancy atoms are all hydroxyl or imidazole hydrogens, which `_clean` discards before `_protonate` assigns its own, so the rule was narrowed to heavy atoms. |
| `7LMO_NYO` | Chain A holds both 27A GLY and 27B ASN, two different residues sharing a sequence number, which the old `(chain, seqid)` key conflated; residues are now identified by `(chain, seqid, icode)` throughout. |
| `7UTW_NAI` | Its Cd ion falls outside 6-31G's H-Zn coverage and is also a metal, which fixed the check order: metal is reported first, so the basis check only ever speaks about elements that would really have reached `mol.build()`. |
| `8A2D_KXY` | A retained residue flanks a break where A860-A864 are absent, and PDBFixer cannot see it without SEQRES records the PoseBusters structures do not carry; chain-break detection is deferred to v2 and the test is a strict xfail rather than a silent gap. |

## Preparation

| Complex | What it changed, and why |
| --- | --- |
| `5S8I_2LY` | Missing side-chain atoms (LYS A1323), a missing terminal atom (ARG A1434) and four single-residue chain breaks in one structure; it is the baseline `_fix` case and the cheapest complex to run twice. |
| `7W06_ITN` | Six selenomethionines, whose selenium has no 6-31G basis and which Modeller has no hydrogen definitions for; `_fix` now calls `replaceNonstandardResidues()` so MSE becomes MET before either problem can arise. |
| `7NFB_GEN` | 4317 deposited hydrogens alongside an ACE cap: `_clean` strips every deposited hydrogen so `_protonate` owns them all at one pH, and decides what is polymer from gemmi entity types rather than residue names, or the cap would be deleted as a heterogen. |
| `7WUX_6OI` | Two crystallographic copies of the very ligand being docked sit in a second site; `_clean` deletes them, since leaving them in puts the answer inside the structure. |
| `6TW5_9M2` | Crystallisation additives (MYA, DMS, CL, SO4) and three loose Mg ions outside the cutout, plus 21 histidines and an N-terminal ASP A27, which together pin what `_clean` removes and what `_protonate` titrates. |
| `6YT6_PKE` | CYS A456-A459 at an SG-SG distance of $2.05\ \mathrm{\AA}$ against six free cysteines confirmed Modeller assigns CYX from the distance alone, so a disulfide cysteine keeps no thiol hydrogen and must not be charged as a thiolate. |
| `7P1F_KFN` | SNN is absent from PDBFixer's substitution table and survives `_fix` intact; it lies outside that cutout, but it is the reason `_protonate` is guarded by a test that no retained residue comes out bare. |

## Truncation and capping

| Complex | What it changed, and why |
| --- | --- |
| `7MWN_WI5` | Eleven residues sit alone between two retained runs, the most in the dataset; capping both sides would take one residue's backbone into an ACE and an NME and place its CA twice at one point, so single-residue gaps are bridged instead. Two nuclei at one position give a divergent repulsion and a linearly dependent basis. |
| `7WPW_F15` | Eight such gaps and, unlike `7MWN_WI5`, still inside the size cap once capped, so it replaced it as the bridging fixture. |
| `6TW5_9M2` | Chain C is retained to residue 410, where the chain ends; this reversed the decision to cap unconditionally, because a cap inherits the backbone of the residue truncation removed and a chain end has no such residue to inherit from. The terminus keeps the charge protonation gave it. |
| `7VBU_6I4` | 47 residues holding 422 heavy atoms, which 42 caps take to 527; the size cap is therefore applied again to the capped cutout, since that is the system RHF is handed, and 22 of the 110 complexes that reach capping cross it there. |
| `8FLV_ZB9` | GLN A412 is deposited as backbone plus CB and PDBFixer rebuilds the rest, closing from $4.52\ \mathrm{\AA}$ of a pose to $3.34\ \mathrm{\AA}$; `_reduce` re-checks for repaired residues because `_verify` runs before `_fix` and cannot see them arrive. |

## Determinism

| Complex | What it changed, and why |
| --- | --- |
| `6ZCY_QF8` | PHE A382's rebuilt ring landed up to $6\ \mathrm{\AA}$ apart between two runs of the same input, enough to cross the $4.5\ \mathrm{\AA}$ cutoff on one run and not the next; `addMissingAtoms` is now seeded and both minimisations pinned to OpenMM's Reference platform, which made the whole pipeline bit-identical run to run. |
| `6TW5_9M2` | Its hydrogens moved by $0.6\ \mathrm{\AA}$ between runs, showing the same fault in `Modeller.addHydrogens`, which starts every hydrogen at a random offset and minimises from there. |
| `7MMH_ZJY`, `7XFA_D9J`, `7NP6_UK8` | Each showed 40 to 78 pairs of atoms closer than $0.5\ \mathrm{\AA}$ on some runs and none on others, which is what an unseeded hydrogen minimisation settling badly looks like; all three are clean once seeded. |

## Charge

| Complex | What it changed, and why |
| --- | --- |
| `5S8I_2LY` | Its $q_A = -1$ comes from a single glutamate for which `self.protonation` records `None`, which is true of every arginine, lysine, aspartate and glutamate; the charge is therefore read from the hydrogens the residues hold, via the Amber templates, and not from the protonation map. |
| `7THI_PGA` | $q_A = +8$ from ten arginines, the largest in the set, and an artefact of the cut as much as of the protein, since the counter-charges neutralising that site lie outside $4.5\ \mathrm{\AA}$. |
| `5SAK_ZRY` | $q_A = -7$ from seven aspartates, the other extreme; together these two set the range $q_A$ spans and warn that electrostatics and induction may not be comparable across complexes. |
| `7WQQ_5Z6` | With `6TW5_9M2`, one of only two accepted cutouts that reach a chain end, so the charge an uncapped terminus contributes is rare but real and $q_A$ has to count it. |
