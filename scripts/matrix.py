"""Half-bit log odds from all non-gap columns of fitting-only EBA alignments."""

from lightning.pytorch.cli import LightningArgumentParser
import json
import math
from pathlib import Path

from scripts.io import partition_ids, read_fasta, read_json, stable_hash, write_json


def fit_matrix_from_counts(alphabet, counts, epsilon=0.01):
    """Fit half-bit log odds from an upper-triangular count matrix."""
    if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet must contain at least two distinct states")
    if not 0 < epsilon < 1:
        raise ValueError("Smoothing epsilon must be strictly between zero and one")
    size = len(alphabet)
    if len(counts) != size or any(len(row) != size for row in counts):
        raise ValueError("Count matrix dimensions do not match alphabet")
    counts = [[float(value) for value in row] for row in counts]
    if any(not math.isfinite(counts[i][j]) or counts[i][j] < 0
           for i in range(size) for j in range(size)):
        raise ValueError("Counts must be finite and nonnegative")
    if any(counts[i][j] != 0 for i in range(size) for j in range(i)):
        raise ValueError("Counts must be stored in the upper triangle")
    mass = math.fsum(counts[i][j] for i in range(size) for j in range(i, size))
    if mass == 0:
        raise ValueError("Cannot fit a matrix without observations")
    empirical = [[0.0] * size for _ in alphabet]
    for i in range(size):
        for j in range(i, size):
            empirical[i][j] = counts[i][j] / mass / (1 if i == j else 2)
            empirical[j][i] = empirical[i][j]
    marginal = [math.fsum(row) for row in empirical]
    # One residue pseudocount per state gives a positive training-derived prior.
    prior = [(2 * mass * p + 1) / (2 * mass + size) for p in marginal]
    joint = [[(1 - epsilon) * empirical[i][j] + epsilon * prior[i] * prior[j]
              for j in range(size)] for i in range(size)]
    background = [math.fsum(row) for row in joint]
    scores = [[2 * math.log2(joint[i][j] / (background[i] * background[j]))
               for j in range(size)] for i in range(size)]
    result = {"schema_version": 1, "alphabet": alphabet, "units": "half-bit",
        "counts_upper_triangle": counts, "observation_weight": mass,
        "epsilon": epsilon, "prior": prior, "joint": joint,
        "background": background, "scores": scores,
        "unknown_policy": "reject", "weighting": "unique-residue-pairs"}
    result["alignment_scores"] = [[round(score) for score in row] for row in scores]
    result["rounding"] = "nearest-integer-half-bit-ties-to-even"
    return result


def fit_matrix(alphabet, observations, epsilon=0.01):
    """Observations are (state_a, state_b, positive_weight) unordered events.

    The prior supplies positive mass for every state.
    """
    size = len(alphabet)
    index = {state: i for i, state in enumerate(alphabet)}
    counts = [[0.0] * size for _ in alphabet]
    for left, right, weight in observations:
        if left not in index or right not in index:
            raise ValueError("Unknown state in positive observation")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Positive finite observation weight required")
        i, j = sorted((index[left], index[right]))
        counts[i][j] += weight
    return fit_matrix_from_counts(alphabet, counts, epsilon)


def fitting_observations(records, sequences, fitting_ids, *, shared=False):
    """Count all non-gap EBA columns within the fitting split."""
    seen = set()
    for record in records:
        left, right = record["chain_a"], record["chain_b"]
        if left not in fitting_ids or right not in fitting_ids:
            if shared:
                continue
            raise ValueError("EBA column outside the fitting split")
        i, j = record["index_a"], record["index_b"]
        left_states, right_states = sequences[left], sequences[right]
        if (left == right or record.get("kind") != "eba_aligned" or record.get("weight", 1) != 1
                or not isinstance(i, int) or not isinstance(j, int)
                or not 0 <= i < len(left_states) or not 0 <= j < len(right_states)):
            raise ValueError("Expected an unweighted, non-gap EBA column between two chains")
        pair = ((left, i), (right, j)) if left < right else ((right, j), (left, i))
        if pair in seen:
            raise ValueError("Duplicate EBA column")
        seen.add(pair)
        yield left_states[i], right_states[j], 1.0


def matrix_inputs(args):
    """Select the chain IDs used to fit the checkpoint."""
    splits = read_json(args.splits)
    fitting = partition_ids(splits, "fit", args.outer_fold)
    conversion = read_json(str(args.sequences) + ".json")
    training = conversion["training_partition"]
    if (training["split_id"], training["outer_fold"], training["fit_chain_ids_sha256"]) != (
        splits["split_id"], args.outer_fold, stable_hash(fitting)
    ):
        raise ValueError("Checkpoint training partition differs from the fitting split")
    if (conversion["checkpoint_id"], conversion["conversion_context"]) != (
        args.checkpoint_id, "full-chain"
    ):
        raise ValueError("Expected the matching full-chain checkpoint conversion")

    specification = {
        "checkpoint_id": args.checkpoint_id,
        "split_id": splits["split_id"],
        "outer_fold": args.outer_fold,
        "subset": "fit",
        "columns": "all-non-gap",
        "encoder_id": conversion["encoder_id"],
        "training_partition": training,
    }
    return read_fasta(args.sequences), set(fitting), specification


def export_mmseqs(matrix, output):
    """Export half-bit scores in the native matrix's canonical letter order."""
    source_alphabet = matrix["alphabet"]
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    if set(source_alphabet) != set(alphabet) or matrix["units"] != "half-bit":
        raise ValueError("MMseqs export requires the 20-state half-bit TEA alphabet")
    order = [source_alphabet.index(state) for state in alphabet]
    output = Path(output)
    if output.suffix != ".out":
        raise ValueError("MMseqs matrix files require the .out extension")
    background = " ".join(format(p, ".17g") for p in [*(matrix["background"][i] for i in order), 1e-5])
    lines = ["# TEA half-bit log odds; STEAM alignment requires --tea-scale 2",
             f"# Background (precomputed optional): {background}",
             f"# Lambda     (precomputed optional): {math.log(2) / 2:.17g}",
             "  " + " ".join([*alphabet, "X"])]
    lines += [state + " " + " ".join(format(matrix["scores"][i][j], ".17g") for j in order) + " 0"
              for state, i in zip(alphabet, order)]
    lines.append("X " + " ".join(["0"] * 21))
    with output.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = LightningArgumentParser(description=__doc__)
    for key in ("alignments", "splits", "sequences", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(4))
    parser.add_argument("--alphabet", default="ACDEFGHIKLMNPQRSTVWY")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--mmseqs-output", type=Path, help="Optional native .out export for STEAM")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sequences, fitting, specification = matrix_inputs(args)
    with args.alignments.open() as handle:
        records = (json.loads(line) for line in handle)
        observations = fitting_observations(records, sequences, fitting, shared=True)
        result = fit_matrix(args.alphabet, observations, args.epsilon)
    result["specification"] = specification
    result["matrix_id"] = stable_hash(result)
    write_json(args.output, result)
    if args.mmseqs_output is not None:
        export_mmseqs(result, args.mmseqs_output)


if __name__ == "__main__":
    main()
