"""
First order SAPT between each complex's protein and its poses, and the ranking it gives them.

SAPT scores every pose of a complex against the correlated protein. run.py runs it after CASCI, and
    the scores are kept in <complex>_sapt.npz beside the protein's artefacts once every pose has them.

Ran as a script, it reranks a screen run.py has scored: confidence.csv is read from the screen's
    directory, out/filter or out/filter_<name>, and each complex's scores from the complex's own.
    Each complex's poses are ranked on E_elst + E_exch, lowest first, into sapt.csv, a row a pose,
    and sapt_summary.csv, a row a complex, which are written beside confidence.py's tables.

    python sapt.py --name v1_1_mm_unsize
"""

import argparse
import os
import statistics

import confidence
import filter
from encode import EncodeProtein, EncodingError, SolveLigand
from utils import sapt, save

# confidence.csv's columns, and what SAPT adds to each pose
FIELDS = confidence.FIELDS + ["elst", "exch", "cumulant", "interaction", "rank_sapt"]
SUMMARY_FIELDS = confidence.SUMMARY_FIELDS + ["top1_sapt"]

SCORED = ("electrostatics", "exchanges", "cumulants", "int_energies")

NAME = filter.NAME

TABLE_NAME = "sapt.csv"
SUMMARY_NAME = "sapt_summary.csv"


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


    def elst(self):
        """
        E^(1)_elst of every pose against the protein.
        """
        if not self.ligand.solved():
            raise EncodingError("Cannot score the poses before RHF has solved every one")
        if self.density is None:
            self.densities()
        self.electrostatics = [
            sapt.electrostatics(self.protein.mol, self.density, mol, mean_field.make_rdm1())
            for mol, mean_field in zip(self.ligand.mols, self.ligand.mean_fields)
        ]
        return self.electrostatics


    def exch(self):
        """
        E^(1)_exch(S^2) of every pose against the protein: the exchange of the two densities, and
            what the protein's cumulant adds, which is kept apart too.
        """
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
        cumulant = sapt.cumulant(self.protein.rdm1, self.protein.rdm2)
        self.exchanges, self.cumulants = [], []
        for mol, mean_field in zip(self.ligand.mols, self.ligand.mean_fields):
            density = mean_field.make_rdm1()
            share = sapt.cumulant_exchange(self.protein.mol, active, cumulant, mol, density)
            self.exchanges.append(
                sapt.exchange(self.protein.mol, self.density, mol, density) + share
            )
            self.cumulants.append(share)
        return self.exchanges


    def interaction(self):
        """
        E^(1)_int = E^(1)_elst + E^(1)_exch(S^2) of every pose against the protein. Kept once every
            pose has it.
        """
        if self.electrostatics is None:
            self.elst()
        if self.exchanges is None:
            self.exch()
        self.int_energies = [
            electrostatic + exchange
            for electrostatic, exchange in zip(self.electrostatics, self.exchanges)
        ]
        self.save()
        return self.int_energies


    def scored(self):
        """
        Whether every pose has all of its scores.
        """
        return self.int_energies is not None


    def save(self):
        """
        Write the scores once every pose has all of them.
        """
        if not self.out or not self.scored():
            return
        save.save_sapt(
            {"source": self.ligand.prepared.source, **{key: getattr(self, key) for key in SCORED}},
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
    rows, scores, skipped = confidence.read_table(ranked), {}, []
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
    confidence.write_table(rows, os.path.join(job, TABLE_NAME), FIELDS)
    confidence.write_table(summary, os.path.join(job, SUMMARY_NAME), SUMMARY_FIELDS)
    _report(summary, skipped, missing)
    return rows, summary


def _report(summary, skipped=(), missing=()):
    """
    Each ranking's top-1 over the same complexes, against the rate a random pick off the ensemble
        would manage.
    """
    print("Complexes", len(summary))
    if skipped:
        print("  not scored", len(skipped), sorted(skipped))
    if missing:
        print("  poses left unscored", len(missing))

    answered = [row for row in summary if row["top1_sapt"] is not None]
    if not answered:
        return
    print(f"\nTop-1 over {len(answered)} complexes\n")
    for field, ranking in [
        ("top1_sapt", "ranked on E_int"),
        ("top1_minimised", "re-scored by DiffDock's confidence model"),
        ("top1_docked", "as DiffDock ranked them"),
    ]:
        right = sum(1 for row in answered if row[field])
        print(f"  {right:4d}  {ranking}, {right / len(answered):.1%}")
    chance = statistics.mean(row["fraction"] for row in answered)
    print(f"        a random pick off the ensemble, {chance:.1%}")


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
