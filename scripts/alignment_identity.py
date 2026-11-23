"""Measure AA and TEA identity within shared-family SCOP domain spans."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.io import partition_ids, read_fasta, read_json, sha256_file, write_json


def shared_family_column_mask(
    mapping, left, right, left_indices, right_indices, mask_cache=None
):
    """Select aligned columns whose residues occupy the same SCOP family."""
    if mapping.get("coordinates") != "zero-based-half-open":
        raise ValueError("SCOP mapping must use zero-based half-open coordinates")
    chains = mapping["chains"]
    domains = mapping["domains"]
    if left not in chains or right not in chains:
        raise ValueError("Aligned chain is absent from the SCOP mapping")
    left_indices = np.asarray(left_indices, dtype=np.int64)
    right_indices = np.asarray(right_indices, dtype=np.int64)
    if left_indices.shape != right_indices.shape:
        raise ValueError("Aligned index arrays differ in shape")

    def family_masks(chain):
        if mask_cache is not None and chain in mask_cache:
            return mask_cache[chain]
        length = int(chains[chain]["length"])
        masks = {}
        for domain_id in chains[chain]["domains"]:
            domain = domains[domain_id]
            family = domain["family"]
            mask = masks.setdefault(family, np.zeros(length, dtype=bool))
            for region in domain["slices"]:
                if region["chain_id"] != chain:
                    continue
                start, stop = int(region["start"]), int(region["stop"])
                if not 0 <= start < stop <= length:
                    raise ValueError(f"Invalid SCOP domain slice for {chain}")
                mask[start:stop] = True
        if mask_cache is not None:
            mask_cache[chain] = masks
        return masks

    left_masks = family_masks(left)
    right_masks = family_masks(right)
    shared_families = sorted(set(left_masks) & set(right_masks))
    if not shared_families:
        raise ValueError(f"EBA pair {left}/{right} has no shared SCOP family")
    if (
        np.any(left_indices < 0)
        or np.any(left_indices >= chains[left]["length"])
        or np.any(right_indices < 0)
        or np.any(right_indices >= chains[right]["length"])
    ):
        raise ValueError("Aligned residue index lies outside its full chain")
    keep = np.zeros(len(left_indices), dtype=bool)
    for family in shared_families:
        keep |= left_masks[family][left_indices] & right_masks[family][right_indices]
    return keep, shared_families


def main():
    from lightning.pytorch.cli import LightningArgumentParser

    parser = LightningArgumentParser(description=__doc__)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--aa-fasta", type=Path, required=True)
    parser.add_argument("--tea-fasta", type=Path, required=True)
    parser.add_argument("--cv-tea-fasta", type=Path, nargs=4, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--minimum-alignment-mean", type=float, default=2.0)
    parser.add_argument("--minimum-residue-score", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = read_json(args.alignments / "manifest.json")
    pairs_path = args.alignments / "pairs.jsonl"
    if source["files"]["pairs.jsonl"] != sha256_file(pairs_path):
        raise ValueError("EBA pair checksum mismatch")
    aa = read_fasta(args.aa_fasta)
    tea = read_fasta(args.tea_fasta)
    cv_tea = [read_fasta(path) for path in args.cv_tea_fasta]
    splits = read_json(args.splits)
    mapping = read_json(args.mapping)
    held_out = [set(partition_ids(splits, "test", fold)) for fold in range(4)]
    chain_fold = {
        chain: fold for fold, chains in enumerate(held_out) for chain in chains
    }
    if len(chain_fold) != sum(map(len, held_out)):
        raise ValueError("Outer-test chain partitions overlap")

    rows = []
    mask_cache = {}
    accepted_source_alignments = 0
    dropped_without_shared_family_columns = 0
    source_paired_columns = 0
    retained_paired_columns = 0
    with open(pairs_path) as handle:
        for line in handle:
            record = json.loads(line)
            alignment = record["alignment"]
            if not record["accepted"] or alignment is None:
                continue
            if alignment["mean_paired_score"] < args.minimum_alignment_mean:
                continue
            accepted_source_alignments += 1
            left, right = record["chain_a"], record["chain_b"]
            fold = chain_fold.get(left)
            right_fold = chain_fold.get(right)
            if (fold is None) != (right_fold is None):
                fold = None
            elif fold != right_fold:
                raise ValueError("A same-family alignment crosses outer folds")
            indices = np.asarray([
                (i, j) for i, j in zip(alignment["index_a"], alignment["index_b"])
                if i >= 0 and j >= 0
            ], dtype=np.int64)
            if len(indices) != alignment["paired_residues"]:
                raise ValueError("EBA paired-residue count mismatch")
            scores = np.asarray(alignment["paired_substitution_scores"])
            if len(scores) != len(indices):
                raise ValueError("EBA substitution-score count mismatch")
            sequence_checks = [(aa, left, "a"), (aa, right, "b"),
                               (tea, left, "a"), (tea, right, "b")]
            if fold is not None:
                sequence_checks.extend(((cv_tea[fold], left, "a"),
                                        (cv_tea[fold], right, "b")))
            if any(len(strings[name]) != alignment[f"length_{side}"]
                   for strings, name, side in sequence_checks):
                raise ValueError("Alignment and converted chain lengths differ")
            i, j = indices.T
            keep, shared_families = shared_family_column_mask(
                mapping, left, right, i, j, mask_cache
            )
            source_paired_columns += len(indices)
            if not np.any(keep):
                dropped_without_shared_family_columns += 1
                continue
            source_paired_residues = len(indices)
            i, j, scores = i[keep], j[keep], scores[keep]
            retained_paired_columns += len(i)
            identity = lambda strings: float(np.mean(
                np.fromiter((strings[left][x] == strings[right][y]
                             for x, y in zip(i, j)), dtype=bool, count=len(i))
            ))
            rows.append({
                "chain_a": left,
                "chain_b": right,
                "shared_families": ";".join(shared_families),
                "shared_family_count": len(shared_families),
                "outer_fold": -1 if fold is None else fold,
                "paired_residues": len(i),
                "source_paired_residues": source_paired_residues,
                "mean_eba_score": alignment["mean_paired_score"],
                "training_positive_residues": int(np.count_nonzero(
                    scores >= args.minimum_residue_score
                )),
                "aa_identity": identity(aa),
                "tea_identity": identity(tea),
                "cv_tea_identity": None if fold is None else identity(cv_tea[fold]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty or not set(range(4)) <= set(frame.outer_fold):
        raise ValueError("Expected accepted alignments in all four outer folds")
    if (
        (frame.paired_residues <= 0).any()
        or (frame.paired_residues > frame.source_paired_residues).any()
        or not frame[["aa_identity", "tea_identity"]].apply(
            lambda column: column.between(0, 1).all()
        ).all()
        or not frame.cv_tea_identity.dropna().between(0, 1).all()
    ):
        raise ValueError("Invalid shared-family-clipped identity table")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    write_json(args.output.with_suffix(args.output.suffix + ".json"), {
        "schema_version": 2,
        "alignments": len(frame),
        "accepted_source_alignments": accepted_source_alignments,
        "dropped_without_shared_family_columns": dropped_without_shared_family_columns,
        "source_paired_columns": source_paired_columns,
        "retained_paired_columns": retained_paired_columns,
        "fold_counts": {str(key): int(value)
                        for key, value in frame.outer_fold.value_counts().sort_index().items()},
        "minimum_alignment_mean_inclusive": args.minimum_alignment_mean,
        "minimum_residue_score_inclusive": args.minimum_residue_score,
        "identity_columns": (
            "paired non-gap columns for which both residues occupy SCOP domain "
            "slices carrying the same family"
        ),
        "clipping_coordinates": mapping["coordinates"],
        "mapping_sha256": sha256_file(args.mapping),
        "pairs_sha256": source["files"]["pairs.jsonl"],
        "inputs": {str(path): sha256_file(path) for path in (
            args.aa_fasta, args.tea_fasta, *args.cv_tea_fasta, args.splits,
            args.mapping,
        )},
        "output_sha256": sha256_file(args.output),
    })
    print(frame.groupby("outer_fold")[["aa_identity", "tea_identity", "cv_tea_identity"]]
          .agg(["count", "mean", "median"]))


if __name__ == "__main__":
    main()
