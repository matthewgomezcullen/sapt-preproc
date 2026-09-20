"""
First order SAPT between each complex's protein and its poses, and the ranking it gives them.

SAPT scores every pose of a complex against the correlated protein. run.py runs it after CASCI, and
    each pose's scores are kept in <complex>_sapt.npz beside the protein's artefacts as soon as it has
    them, so a stage stopped part-way resumes after the last pose it kept.

Ran as a script, it reranks a screen run.py has scored: confidence.csv is read from the screen's
    directory, out/filter or out/filter_<name>, and each complex's scores from the complex's own.
    Each complex's poses are ranked on E_elst + E_exch, lowest first, into sapt.csv, a row a pose,
    and sapt_summary.csv, a row a complex, which are written beside confidence.py's tables.

    python sapt.py --name v1_1_mm_unsize
"""

import argparse
import os
from datetime import datetime

import confidence
import filter
from encode import EncodeProtein, EncodingError, SolveLigand
from utils import report, sapt, save

# confidence.csv's columns, and what SAPT adds to each pose
FIELDS = confidence.FIELDS + ["elst", "exch", "cumulant", "interaction", "rank_sapt"]
SUMMARY_FIELDS = confidence.SUMMARY_FIELDS + ["top1_sapt", "discrimination_sapt"]

RANKINGS = [("top1_sapt", "discrimination_sapt", "ranked on E_int")] + confidence.RANKINGS

SCORED = ("electrostatics", "exchanges", "cumulants", "int_energies")

NAME = filter.NAME

TABLE_NAME = "sapt.csv"
SUMMARY_NAME = "sapt_summary.csv"

# TEMPORARY, a last-minute change for this experiment's deadline. The stage segfaulted inside PySCF's
# (AA|AB) pass on these two, whose cutout and ligand are large enough that the cumulant's 105 pair
# densities take that pass's thread buffers to 43 and 47 GB, where 33 GB ran: 2^32 doubles is the
# line, and PySCF's C driver appears to overflow a 32-bit size beyond it. Chunking the pass would
# cost about 150 hours a complex, which the deadline does not have, so these two are scored over
# their correlated densities alone. That keeps the larger share of the correlation, which moved the
# water dimer's exchange by +1.0e-4 Hartree against the cumulant's -2.2e-5, and none of the
# cumulant's; their `cumulants` are kept as zero, so the tables say so. Remove once the pass is
# chunked.
WITHOUT_CUMULANT = ("7XFA_D9J", "7PRM_81I")


class SAPT:
    """
    SAPT takes a correlated EncodeProtein and a solved SolveLigand of the same complex, and scores
        each pose against the protein at first order.

    The protein is monomer A, the same for every pose, so its density is built once. Each pose is
        monomer B, at RHF in its own basis.
    """

    def __init__(
        self,
        protein: EncodeProtein,
        ligand: SolveLigand,
        out=None
    ):
        self.protein = protein
        self.ligand = ligand
        self.density = None # the protein's AO density, core plus active
        self.electrostatics = None # E^(1)_elst per pose
        self.exchanges = None # E^(1)_exch per pose
        self.cumulants = None # the share of E^(1)_exch the protein's cumulant carries, per pose
        self.int_energies = None # E_int per pose

        # Load
        self.out = out
        if out:
            self._load()


    def densities(self):
        """
        The protein's AO density, from the orbitals the window was cut over and CASCI's rdm1 over
            its active block.
        """
        if not self.protein.correlated():
            raise EncodingError("Cannot build the protein's density before CASCI has solved it")
        self.density = sapt.density(
            self.protein.orbital_initial,
            self.protein.rdm1,
            self.protein.active_space_size,
            self.protein.active_electrons,
            self.protein.mol.nelectron,
        )
        return self.density


    def interaction(self):
        """
        E^(1)_int = E^(1)_elst + E^(1)_exch(S^2) of every pose against the protein. The exchange
            carries what the protein's cumulant adds, which is kept apart too.

        Each pose is kept as soon as it is scored, so a stage stopped part-way resumes after the last
            pose it kept.
        """
        if self.scored():
            return self.int_energies
        if not self.ligand.solved():
            raise EncodingError("Cannot score the poses before RHF has solved every one")
        if self.density is None:
            self.densities()
        active = sapt.active(
            self.protein.orbital_initial,
            self.protein.active_space_size,
            self.protein.active_electrons,
            self.protein.mol.nelectron,
        )
        # TEMPORARY: see WITHOUT_CUMULANT.
        without = bool(self.out) and self._name() in WITHOUT_CUMULANT
        cumulant = None if without else sapt.cumulant(self.protein.rdm1, self.protein.rdm2)
        if without:
            print(
                f"[{datetime.now():%H:%M:%S}] Skipping  the cumulant for {self._name()}, whose "
                "exchange is over the correlated densities alone",
                flush=True,
            )
        if self.int_energies is None:
            self.electrostatics, self.exchanges, self.cumulants, self.int_energies = [], [], [], []
        poses = len(self.ligand.mols)
        for index in range(len(self.int_energies), poses):
            mol, density = self.ligand.mols[index], self.ligand.mean_fields[index].make_rdm1()
            electrostatic = sapt.electrostatics(self.protein.mol, self.density, mol, density)
            separable, share = sapt.exchange_parts(
                self.protein.mol, self.density, mol, density, active, cumulant
            )
            self.electrostatics.append(electrostatic)
            self.exchanges.append(separable + share)
            self.cumulants.append(share)
            self.int_energies.append(electrostatic + self.exchanges[-1])
            self.save()
            print(
                f"[{datetime.now():%H:%M:%S}] Scored    pose {index + 1} of {poses}, "
                f"{self.ligand.prepared.source[index]}: E_int {self.int_energies[-1]:.6f} Hartree",
                flush=True,
            )
        return self.int_energies


    def scored(self):
        """
        Whether every pose has all of its scores.
        """
        return self.int_energies is not None and len(self.int_energies) == len(
            self.ligand.prepared.source
        )


    def save(self):
        """
        Write the scores of every pose scored so far, each against the file its pose came from.
        """
        if not self.out or not self.int_energies:
            return
        save.save_sapt(
            {
                "source": self.ligand.prepared.source[:len(self.int_energies)],
                **{key: getattr(self, key) for key in SCORED},
            },
            self._name(),
            self.out,
        )


    def _load(self):
        """
        Read back the scores `out` holds.
        """
        kept = save.load_sapt(self._name(), self.out)
        if kept is None:
            return
        for key in SCORED:
            setattr(self, key, list(kept[key]))


    def _name(self):
        """
        The complex directory.
        """
        return os.path.basename(os.path.normpath(self.out))


def join(rows, energies):
    """
    Each of confidence.csv's rows given its pose's energies, keyed by (complex, source) 
    
    A pose without energies is dropped and reported.
    """
    joined, missing = [], []
    for row in rows:
        pose = (row["name"], row["source"])
        if pose not in energies:
            missing.append(pose)
            continue
        scored = energies[pose]
        joined.append({
            **row,
            "elst": scored["elst"],
            "exch": scored["exch"],
            "cumulant": scored["cumulant"],
            "interaction": scored["elst"] + scored["exch"],
        })
    return joined, missing


def rank(rows):
    return confidence.number(
        rows, lambda row: (row["interaction"], row["rank_docked"]), "rank_sapt"
    )


def summarise(rows, threshold=filter.NEAR_NATIVE):
    summary = confidence.summarise(rows, threshold)
    for entry in summary:
        theirs = [row for row in rows if row["name"] == entry["name"]]
        entry["top1_sapt"] = confidence.is_near_native(
            min(theirs, key=lambda row: row["rank_sapt"]), threshold
        )
        entry["discrimination_sapt"] = report.pairwise_discriminate(
            [-row["interaction"] for row in theirs],
            [confidence.is_near_native(row, threshold) for row in theirs],
        )
    return summary


def energies(name, kept):
    """
    A complex's kept scores, keyed by (complex, source).
    """
    return {
        (name, str(source)): {"elst": elst, "exch": exch, "cumulant": cumulant}
        for source, elst, exch, cumulant in zip(
            kept["source"], kept["electrostatics"], kept["exchanges"], kept["cumulants"]
        )
    }


def run(name=NAME):
    """
    Every complex confidence.py ranked, given its poses' kept scores, reranked on them and summarised
        into the two tables, which are written into the screen's directory beside confidence.py's.

    A complex with no scores kept is left out of both, and reported.
    """
    job = filter.job_dir(name)
    ranked = os.path.join(job, confidence.TABLE_NAME)
    if not os.path.isfile(ranked):
        raise SystemExit(f"{ranked} is missing; run confidence.py over the screen first")
    rows, scores, skipped = report.read_table(ranked), {}, []
    for complex in sorted({row["name"] for row in rows}):
        kept = save.load_sapt(complex, confidence.complex_dir(complex, job))
        if kept is None:
            skipped.append(complex)
        else:
            scores.update(energies(complex, kept))
    if not scores:
        raise SystemExit(f"No complex under {job} has been scored; run run.py first")

    rows, missing = join([row for row in rows if row["name"] not in skipped], scores)
    rows = rank(rows)
    summary = summarise(rows)
    report.write_table(rows, os.path.join(job, TABLE_NAME), FIELDS)
    report.write_table(summary, os.path.join(job, SUMMARY_NAME), SUMMARY_FIELDS)
    _report(summary, skipped, missing)
    return rows, summary


def _report(summary, skipped=(), missing=()):
    """
    Each ranking's top-1 and pairwise discrimination, against a random picker.
    """
    print("Complexes", len(summary))
    if skipped:
        print("  not scored", len(skipped), sorted(skipped))
    if missing:
        print("  poses left unscored", len(missing))
    report.rankings(summary, RANKINGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=NAME,
        help="The screen to rerank, as it was named for filter.py. confidence.csv and each "
             "complex's scores are read from, and the tables written into, out/filter_<name>, or "
             "out/filter without a name.",
    )
    run(parser.parse_args().name)
