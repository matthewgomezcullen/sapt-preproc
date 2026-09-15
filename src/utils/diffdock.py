"""
Score poses with DiffDock's confidence model.

Run as a script by confidence.py, in the interpreter DiffDock was installed into. Cannot be 
    imported, as `utils` will be shadowed.

```
python utils/diffdock.py --manifest manifest.csv --scores scores.csv \
    --models data/diffdock_models --esm data/diffdock_models/esm2_t33_650M_UR50D.pt
```

The manifest is a row a complex: name, protein (deposited), sdf (poses, titled with source file). 
    The scores table is a row a pose: name, source, confidence. Scoring failures are reported and 
    leaves its poses out of the table.

Nothing is downloaded: every set of weights is read from the path it was given.
"""

import argparse
import csv
import os
import sys



HERE = os.path.dirname(os.path.abspath(__file__))

DIFFDOCK = os.path.join(os.path.dirname(HERE), "diffdock")

CONFIDENCE = "confidence_model"
SCORE = "score_model"
CHECKPOINT = "best_model_epoch75.pt"
PARAMETERS = "model_parameters.yml"

# torus.py precomputes lookup tables into the working directory under these names, and decides they
# are both there on the strength of the first alone.
TABLES = [".p.npy", ".score.npy"]

# v1.1's confidence model is the older architecture, as DiffDock's own inference defaults have it.
OLD = True

# Poses of one complex scored together, which is inference.py's default.
BATCH = 10

FIELDS = ["name", "source", "confidence"]


def _diffdock():
    """
    Inserts DiffDock's root ahead of everything else on the path to resolve to its packages.

    Its torus module writes its lookup tables to the working directory by relative path.
    """
    if DIFFDOCK not in sys.path:
        sys.path.insert(0, DIFFDOCK)
    os.chdir(DIFFDOCK)
    started, finished = [os.path.join(DIFFDOCK, table) for table in TABLES]
    if os.path.exists(started) and not os.path.exists(finished):
        os.remove(started)


def trained(models, which=CONFIDENCE):
    from argparse import Namespace

    import yaml

    with open(os.path.join(models, which, PARAMETERS)) as file:
        return Namespace(**yaml.full_load(file))


def knn_only(models):
    """
    Whether the receptor graph is nearest-neighbours alone, off the score model's settings.
    """
    scoring = trained(models, SCORE)
    return False if not hasattr(scoring, "not_knn_only_graph") else not scoring.not_knn_only_graph


def confidence_model(models, arguments, device):
    """
    The confidence model.
    """
    import torch
    from utils.utils import get_model

    confidence = get_model(
        arguments, device, t_to_sigma=None, no_parallel=True, confidence_mode=True, old=OLD
    )
    confidence.load_state_dict(
        torch.load(os.path.join(models, CONFIDENCE, CHECKPOINT), map_location="cpu"), strict=True
    )
    return confidence.to(device).eval()


def language_model(esm, device):
    """
    ESM-2.
    """
    import torch
    from esm.pretrained import load_model_and_alphabet_core

    if not os.path.isfile(esm):
        raise SystemExit(f"{esm} is not there. The ESM-2 650M weights have to be downloaded first.")
    name = os.path.splitext(os.path.basename(esm))[0]
    language, alphabet = load_model_and_alphabet_core(
        name, torch.load(esm, map_location="cpu"), None
    )
    return language.eval().to(device), alphabet


def embeddings(name, protein, language, alphabet):
    """
    ESM-2's per-residue representations, one tensor a chain, which are the receptor's node features.
    """
    from utils.inference_utils import compute_ESM_embeddings, get_sequences_from_pdbfile

    chains = get_sequences_from_pdbfile(protein).split(":")
    labels = [f"{name}_chain_{index}" for index in range(len(chains))]
    computed = compute_ESM_embeddings(language, alphabet, labels, chains)
    return [computed[label] for label in labels]


def poses(path):
    """
    The poses of one complex, each with the file it came from.
    """
    from rdkit.Chem import SDMolSupplier

    molecules = list(SDMolSupplier(path, removeHs=False, sanitize=True))
    if any(molecule is None for molecule in molecules):
        raise ValueError(f"{path} holds a record RDKit could not read")
    return [molecule.GetProp("_Name") for molecule in molecules], molecules


def graph(name, protein, molecule, arguments, lm, knn):
    """
    One complex as the confidence model's graph, holding the pose's own coordinates.

    Every pose of a complex is a conformer of the same molecule, so the graph is built once.
    """
    from torch_geometric.data import HeteroData

    from datasets.process_mols import get_lig_graph_with_matching, moad_extract_receptor_structure

    complex_graph = HeteroData()
    complex_graph["name"] = name
    get_lig_graph_with_matching(
        molecule, complex_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
        num_conformers=1, remove_hs=arguments.remove_hs,
    )
    moad_extract_receptor_structure(
        path=protein,
        complex_graph=complex_graph,
        neighbor_cutoff=arguments.receptor_radius,
        max_neighbors=arguments.c_alpha_max_neighbors,
        lm_embeddings=lm,
        knn_only_graph=knn,
        all_atoms=arguments.all_atoms,
        atom_cutoff=arguments.atom_radius,
        atom_max_neighbors=arguments.atom_max_neighbors,
    )

    centre = complex_graph["receptor"].pos.mean(dim=0, keepdim=True)
    complex_graph["receptor"].pos -= centre
    if arguments.all_atoms:
        complex_graph["atom"].pos -= centre
    complex_graph.original_center = centre
    return complex_graph


def positions(molecule, remove_hs):
    """
    A pose's coordinates in graph order.
    """
    import torch
    from rdkit.Chem import RemoveHs

    pose = RemoveHs(molecule) if remove_hs else molecule
    return torch.from_numpy(pose.GetConformer().GetPositions()).float()


def confidences(complex_graph, molecules, confidence, arguments, device, batch=BATCH):
    """
    Every pose's confidence, in the order the poses came in.

    `rmsd_classification_cutoff` is a list, so the model has a class per cutoff and the first output
        is the confidence, which is how inference.py reads it. A NaN is floored the same way
        `sampling` floors it, so a pose the model could not score ranks last.
    """
    import copy

    import torch
    from torch_geometric.data import Batch

    from utils.diffusion_utils import set_time

    scored = []
    for start in range(0, len(molecules), batch):
        graphs = []
        for molecule in molecules[start:start + batch]:
            one = copy.deepcopy(complex_graph)
            one["ligand"].pos = positions(molecule, arguments.remove_hs) - one.original_center
            graphs.append(one)

        batched = Batch.from_data_list(graphs).to(device)
        set_time(batched, 0, 0, 0, 0, len(graphs), arguments.all_atoms, device)
        with torch.no_grad():
            out = confidence(batched)
        if isinstance(out, tuple):
            out = out[0]
        out = torch.nan_to_num(out, nan=-1000).cpu()
        scored += (out[:, 0] if out.dim() > 1 and out.shape[1] > 1 else out.reshape(-1)).tolist()
    return scored


def read(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def write(rows, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(manifest, scores, models, esm):
    # Resolved before the working directory moves under them.
    manifest, scores, models, esm = (os.path.abspath(path) for path in (manifest, scores, models, esm))
    _diffdock()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arguments, knn = trained(models), knn_only(models)
    confidence = confidence_model(models, arguments, device)
    language, alphabet = language_model(esm, device)
    print(f"Scoring on {device}", flush=True)

    work = read(manifest)
    rows, failed = [], []
    for one in work:
        name = one["name"]
        try:
            sources, molecules = poses(one["sdf"])
            lm = embeddings(name, one["protein"], language, alphabet)
            complex_graph = graph(name, one["protein"], molecules[0], arguments, lm, knn)
            scored = confidences(complex_graph, molecules, confidence, arguments, device)
        except Exception as error: # noqa: BLE001  one complex must not cost the other forty-nine
            failed.append((name, error))
            print(f"[{name}] failed: {error}", file=sys.stderr, flush=True)
            continue
        rows += [
            dict(zip(FIELDS, (name, source, value))) for source, value in zip(sources, scored)
        ]
        print(f"[{name}] {len(scored)} poses, best {max(scored):+.2f}", flush=True)

    write(rows, scores)
    print(f"Scored {len(rows)} poses of {len(work) - len(failed)}/{len(work)} complexes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="The complexes to score.")
    parser.add_argument("--scores", required=True, help="Where to write a confidence a pose.")
    parser.add_argument("--models", required=True, help="The unpacked DiffDock weights.")
    parser.add_argument("--esm", required=True, help="The ESM-2 650M weights.")
    parsed = parser.parse_args()

    run(parsed.manifest, parsed.scores, parsed.models, parsed.esm)
