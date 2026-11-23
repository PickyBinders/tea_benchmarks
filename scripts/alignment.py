"""Affine local alignment with explicit integer scores and backend conventions."""

from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from lightning.pytorch.cli import LightningArgumentParser
import csv
import time
import math
import multiprocessing
from importlib.metadata import version
from pathlib import Path

import numpy as np
from biotite.sequence import Alphabet, GeneralSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal

from scripts.io import partition_ids, read_fasta, read_json, sha256_file, stable_hash, write_json
from scripts.parasail import ParasailAligner

ParasailScorer = ParasailAligner


def _run_parallel(args):
    """Score independent shards in a fresh output directory."""
    if args.shard != 0 or args.shards != 1:
        raise ValueError("Use either --workers or explicit --shard/--shards")
    shards = args.parallel_shards or args.workers
    if shards < args.workers:
        raise ValueError("Need at least one shard per worker")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [output / f"scores_{i:03d}-of-{shards:03d}.tsv" for i in range(shards)]
    jobs = [Namespace(**{**vars(args), 'output': path, 'shard': i, 'shards': shards})
            for i, path in enumerate(paths)]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(score_shard, jobs))
    receipts = [read_json(str(path) + ".json") for path in paths]
    cohort = receipts[0]["cohort"]
    pairs = sum(receipt["pairs"] for receipt in receipts)
    if any(receipt["cohort"] != cohort for receipt in receipts) or pairs != len(cohort) * (len(cohort)-1) // 2:
        raise ValueError("Incomplete all-versus-all shard set")
    write_json(output / "manifest.json", {
        "state": "complete", "pairs": pairs,
        "files": [{"path": path.name, "sha256": sha256_file(path)} for path in paths],
    })


def bonus_scores(matrix, symbols, bonus=0.0):
    """Add a bonus to pairs of distinct known states.

    Multiply half-unit scores and both gap penalties by two for integer
    alignment. Divide the exported scores by the same factor.
    """
    if not math.isfinite(bonus) or bonus < 0 or bonus * 2 != round(bonus * 2):
        raise ValueError("Bonus must be a finite nonnegative multiple of 0.5")
    scores = np.asarray(matrix, dtype=float)
    if scores.shape != (len(symbols), len(symbols)) or not np.isfinite(scores).all() or not np.equal(scores, np.rint(scores)).all():
        raise ValueError("Bonus requires an explicitly rounded square matrix")
    known = np.array([symbol != 'X' for symbol in symbols])
    unlike = ~np.eye(len(symbols), dtype=bool) & known[:, None] & known[None, :]
    scale = 2 if bonus != int(bonus) else 1
    return ((scores + bonus * unlike) * scale).tolist(), scale


def make_aligner(matrix, gap_open, gap_extend):
    """Build the Biotite aligner from an explicitly rounded score matrix."""
    if gap_open <= 0 or gap_extend <= 0 or not math.isfinite(gap_open + gap_extend):
        raise ValueError("Positive finite gap penalties required")
    scores = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(scores).all() or not np.equal(scores, np.rint(scores)).all():
        raise ValueError("Biotite requires an explicitly rounded integer matrix export")
    if np.max(np.abs(scores)) >= 2**30:
        raise ValueError("Integer substitution score exceeds the supported safety bound")
    if int(gap_open) != gap_open or int(gap_extend) != gap_extend:
        raise ValueError("Biotite requires integer gap penalties")
    alphabet = Alphabet(range(len(scores)))
    return alphabet, SubstitutionMatrix(alphabet, alphabet, scores.astype(np.int32))


def local_score(a, b, matrix, gap_open, gap_extend):
    alphabet, substitution = make_aligner(matrix, gap_open, gap_extend)
    if not len(a) or not len(b):
        return 0.0
    if np.max(substitution.score_matrix()[np.ix_(np.unique(a), np.unique(b))]) <= 0:
        return 0.0
    alignments = align_optimal(GeneralSequence(alphabet, a), GeneralSequence(alphabet, b),
        substitution, gap_penalty=(-int(gap_open), -int(gap_extend)), local=True, max_number=1)
    return float(alignments[0].score) if alignments else 0.0


def encode(sequence, alphabet):
    index = {state: i for i, state in enumerate(alphabet)}
    try:
        return np.asarray([index[state] for state in sequence], dtype=np.int32)
    except KeyError as error:
        raise ValueError(f"Unmapped state {error.args[0]!r}; no silent unknown substitution") from error


def main(argv=None):
    parser = LightningArgumentParser(description=__doc__)
    for name in ("sequences", "matrix", "splits", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(4))
    parser.add_argument("--training-splits", type=Path,
                        help="Full-data training declaration, required for main TEA scoring")
    parser.add_argument("--subset", choices=("fit", "validation", "test", "all", "tuning"), default="test")
    parser.add_argument("--gap-open", type=int)
    parser.add_argument("--gap-extend", type=int)
    parser.add_argument("--gap-selection", type=Path,
                        help="Use the frozen selected gap pair and record its checksum")
    parser.add_argument("--off-diagonal-bonus", type=float, default=0.0)
    parser.add_argument("--backend", choices=("biotite", "parasail"), default="biotite",
                        help="Parasail permits adjacent opposing gaps; retune when switching")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="With >1 worker, write scores to a new shard directory")
    parser.add_argument("--parallel-shards", type=int,
                        help="Stable output shard count for direct parallel scoring (default: workers)")
    parser.add_argument("--mask", type=Path, help="Fixed common 3Di residue mask, applied equally to all alphabets")
    args = parser.parse_args(argv)
    if args.gap_selection is not None:
        if args.gap_open is not None or args.gap_extend is not None:
            raise ValueError("Use either a frozen gap selection or explicit gaps")
        selection = read_json(args.gap_selection)
        if (selection.get("selection_objective") != "sensitivity-weighted-pr"
                or selection.get("selection_scope", {}).get("subset") != "tuning"):
            raise ValueError("Held-out scoring requires the frozen sampled-tuning selection")
        args.gap_open = selection["selected"]["gap_open"]
        args.gap_extend = selection["selected"]["gap_extend"]
    elif args.gap_open is None or args.gap_extend is None:
        raise ValueError("Alignment requires explicit gaps or a frozen gap selection")
    if args.workers != 1:
        return _run_parallel(args)
    if args.parallel_shards is not None:
        raise ValueError("--parallel-shards requires --workers greater than one")
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard index/count")

    score_shard(args)


def score_shard(args):
    source = read_fasta(args.sequences)
    matrix = read_json(args.matrix)
    splits = read_json(args.splits)
    if splits.get("purpose") == "conversion-timing-only-not-retrieval":
        raise ValueError("A sampled timing panel cannot replace the retrieval benchmark cohort")
    conversion_context = "reference-domain"
    conversion_batch_policy = "reference-domain"
    conversion_path = Path(str(args.sequences) + ".json")
    checkpoint_id = matrix.get("specification", {}).get("checkpoint_id")
    training_partition = None
    if checkpoint_id is not None:
        specification = matrix["specification"]
        training = splits
        training_outer_fold = args.outer_fold
        if args.subset in ("all", "tuning") and args.outer_fold is None:
            if args.training_splits is None:
                raise ValueError("Main TEA scoring requires --training-splits for the full-data model")
            training = read_json(args.training_splits)
            if training.get("purpose") != "full-dataset-training":
                raise ValueError("Main TEA scoring requires a full-data training declaration")
        elif args.outer_fold is not None and args.training_splits is not None:
            training = read_json(args.training_splits)
            if training.get("purpose") == "full-dataset-training":
                training_outer_fold = None
            elif "runs" not in training or training.get("split_id") != splits.get("split_id"):
                raise ValueError("CV evaluation cohort and training split differ")
        elif args.outer_fold is not None and "identifiers" in splits:
            raise ValueError("CV scoring on a standalone cohort requires --training-splits")
        elif args.training_splits is not None:
            raise ValueError("Training splits apply only to TEA full-data or fold-held-out scoring")
        fitting_ids = partition_ids(training, "fit", training_outer_fold)
        if (specification["split_id"], specification["outer_fold"], specification["subset"]) != (training["split_id"], training_outer_fold, "fit"):
            raise ValueError("TEA matrix belongs to a different fitting split")
        conversion = read_json(conversion_path)
        training_partition = conversion.get("training_partition")
        expected = (training["split_id"], training_outer_fold, stable_hash(fitting_ids))
        observed = (tuple(training_partition.get(key) for key in
                    ("split_id", "outer_fold", "fit_chain_ids_sha256"))
                    if isinstance(training_partition, dict) else None)
        if observed != expected:
            raise ValueError("Checkpoint training split does not match this evaluation")
        if args.training_splits is not None and any(training_partition.get(key) != training.get(key)
                                                   for key in ("dataset_id", "mapping_id")):
            raise ValueError("Full-data checkpoint and training declaration have different data identities")
        if specification.get("training_partition") != conversion["training_partition"]:
            raise ValueError("Matrix and alphabet have different checkpoint training partitions")
        if conversion["checkpoint_id"] != checkpoint_id or conversion["output_sha256"] != sha256_file(args.sequences):
            raise ValueError("Alignment alphabet does not match checkpoint/matrix")
        conversion_context = conversion["conversion_context"]
        conversion_batch_policy = conversion["batch_policy"]
    elif args.training_splits is not None:
        raise ValueError("Fixed baselines have no TEA training partition")
    scores = matrix.get("alignment_scores", matrix["scores"])
    alphabet_symbols = matrix["alphabet"]
    size = len(matrix["alphabet"])
    if len(scores) != size or any(len(row) != size for row in scores):
        raise ValueError("Invalid score matrix dimensions")
    if any(not math.isfinite(scores[i][j]) or abs(scores[i][j] - scores[j][i]) > 1e-10
           for i in range(size) for j in range(size)):
        raise ValueError("Unordered pair scoring requires a finite symmetric matrix")
    identifiers = partition_ids(splits, args.subset, args.outer_fold)
    original_count = len(identifiers)
    if args.mask:
        masks = read_json(args.mask)["known_residues"]
        identifiers = [key for key in identifiers if key in masks]
        if "X" not in alphabet_symbols:
            alphabet_symbols += "X"
            scores = [row + [-4] for row in scores] + [[-4] * len(alphabet_symbols)]
        else:
            unknown = alphabet_symbols.index("X")
            scores = [list(row) for row in scores]
            for i in range(len(scores)):
                scores[i][unknown] = scores[unknown][i] = -4
        for key in identifiers:
            if len(masks[key]) != len(source[key]):
                raise ValueError(f"Mask length does not match full chain: {key}")
            source[key] = "".join(state if known else "X" for state, known in zip(source[key], masks[key]))
    if len(identifiers) < 2:
        raise ValueError("Fewer than two chains in evaluation cohort")
    scores, score_scale = bonus_scores(scores, alphabet_symbols, args.off_diagonal_bonus)
    gap_open, gap_extend = args.gap_open * score_scale, args.gap_extend * score_scale
    encoded = {key: encode(source[key], alphabet_symbols) for key in identifiers}
    simd = ParasailAligner(encoded, scores, gap_open, gap_extend) if args.backend == "parasail" else None
    alphabet, substitution = make_aligner(scores, gap_open, gap_extend)
    encoded = {key: GeneralSequence(alphabet, value) for key, value in encoded.items()}
    # With nonpositive substitution scores and positive gap costs, every
    # local optimum is zero. Emit those scores directly to skip tied tracebacks.
    active_states = np.unique(np.concatenate([sequence.code for sequence in encoded.values()]))
    zero_scores = np.max(substitution.score_matrix()[np.ix_(active_states, active_states)]) <= 0
    output = Path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError("Shard already exists; never overwrite completed scientific results")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    count = 0
    residue_pairs = 0
    started = time.perf_counter()
    with open(temporary, "w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query_id", "target_id", "score"])
        for i, left in enumerate(identifiers):
            if i % args.shards != args.shard:
                continue
            for right in identifiers[i + 1:]:
                if zero_scores:
                    score = 0.0
                elif simd is not None:
                    score = simd.score(left, right)
                else:
                    alignments = align_optimal(encoded[left], encoded[right], substitution,
                        gap_penalty=(-gap_open, -gap_extend), local=True, max_number=1)
                    score = float(alignments[0].score) if alignments else 0.0
                writer.writerow([left, right, format(score / score_scale, ".17g")])
                count += 1
                residue_pairs += len(encoded[left]) * len(encoded[right])
    temporary.replace(output)
    write_json(output.with_suffix(output.suffix + ".json"), {
        "schema_version": 1, "mode": "exact-affine-local", "gap_cost": "open+(k-1)*extend",
        "backend": "parasail.sw_striped_profile_32" if simd is not None else "biotite.align_optimal",
        "backend_version": version(args.backend),
        "adjacent_opposing_gaps": "allowed" if simd is not None else "forbidden",
        "score_export": matrix.get("rounding", "integer-input"),
        "off_diagonal_bonus": args.off_diagonal_bonus, "score_scale": score_scale,
        "checkpoint_id": matrix.get("specification", {}).get("checkpoint_id"),
        "training_partition": training_partition,
        "training_splits_sha256": sha256_file(args.training_splits) if args.training_splits else None,
        "conversion_context": conversion_context,
        "batch_policy": conversion_batch_policy,
        "gap_open": args.gap_open, "gap_extend": args.gap_extend, "pairs": count,
        "gap_selection_sha256": (
            sha256_file(args.gap_selection) if args.gap_selection else None
        ),
        "alignment_and_write_seconds": time.perf_counter() - started,
        "residue_pairs": residue_pairs,
        "shard": args.shard, "shards": args.shards, "subset": args.subset, "outer_fold": args.outer_fold,
        "split_id": splits["split_id"], "matrix_sha256": sha256_file(args.matrix),
        "cohort_purpose": splits.get("purpose", "cross-validation"),
        "sequences_sha256": sha256_file(args.sequences),
        "scores_sha256": sha256_file(output),
        "cohort": identifiers,
        "original_cohort_size": original_count,
        "mask_sha256": sha256_file(args.mask) if args.mask else None,
        "masked_state_score": -4 if args.mask else None,
    })
    print(f"Scored {count} unordered non-self pairs")


if __name__ == "__main__":
    main()
