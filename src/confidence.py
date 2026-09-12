OUT = ...
DATA = ...


class DiffDockConfidence:

    def __init__(self, complexes):
        self.complexes = complexes
        self.rmsds = None # { complex : { source : rmsd } }
        self.confidences = None # { complex : { source : confidence } }


    def npz_to_sdf(self):
        """
        Expects each complex directory to contain an `_prepared.npz` file.

        Writes one multi-pose SDF per complex from the `_prepared.npz` into the same directory.
        """
        ...


    def rmsd(self):
        """
        Lables each pose with RMSD to native ligand.
        """
        ...


    def confidence(self):
        """
        Invokes DiffDock's confidence model to evaluate each pose.
        """
        ...


    def save(self):
        """
        Record results in OUT/confidence.csv. 

        Columns: complex, source, rmsd, rank (minimised), confidence (minimised), rank (as-docked), 
            confidence (as-docked).
        Sort by complex, confidence (new).
        """
        ...


    def complex_to_difficulty(self):
        """
        Record difficulty (n_near / n_kept) and top-1 success for each complex in OUT/difficulty.csv
        """
        ...


if __name__ == "__main__":
    # --complexes --resuse --force
    ...
