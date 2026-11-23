"""Alignment-quality metrics for the AlphaFold multidomain benchmark."""

import numpy as np
import biotite.structure as struc

from scripts.parasail import cigar_trace


def paired_residue_quality(query_ca, target_ca, pairs):
    """Measure native paired C-alpha positions against the full query structure.

    DALI reports equivalent residue ranges. This
    shared core keeps its lDDT definition identical to the other engines.
    """
    for atoms in (query_ca, target_ca):
        if (not len(atoms) or not np.all(atoms.atom_name == "CA")
                or struc.get_residue_count(atoms) != len(atoms)
                or not np.isfinite(atoms.coord).all()):
            raise ValueError("Expected one finite C-alpha atom per residue")
        chains, _ = struc.to_sequence(atoms)
        if len(chains) != 1:
            raise ValueError("Expected one protein chain")
    pairs = np.asarray(pairs, dtype=np.int64)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs)
            or np.any(pairs < 0)
            or np.any(pairs[:, 0] >= len(query_ca))
            or np.any(pairs[:, 1] >= len(target_ca))
            or len(np.unique(pairs[:, 0])) != len(pairs)
            or len(np.unique(pairs[:, 1])) != len(pairs)):
        raise ValueError("Alignment contains invalid or repeated paired residues")

    subject = np.full(query_ca.coord.shape, np.nan, dtype=query_ca.coord.dtype)
    subject[pairs[:, 0]] = target_ca.coord[pairs[:, 1]]
    values = struc.lddt(
        query_ca,
        subject,
        aggregation="atom",
        distance_bins=np.nextafter(np.array([0.5, 1.0, 2.0, 4.0]), -np.inf),
        filter_function=lambda contacts: struc.index_distance(query_ca, contacts) < 15.0,
    )
    aligned = values[pairs[:, 0]]
    return {
        "lddt": values,
        "pairs": pairs,
        "mean_aligned_lddt": float(aligned.mean()) if np.isfinite(aligned).all() else None,
        "matched_columns": len(pairs),
    }


def alignment_quality(query_ca, target_ca, query_start, target_start, cigar):
    """Measure one native CIGAR alignment against the full query structure."""
    trace = cigar_trace(
        len(query_ca), len(target_ca), query_start, target_start, cigar
    )
    pairs = trace[np.all(trace >= 0, axis=1)]
    result = paired_residue_quality(query_ca, target_ca, pairs)
    return {
        **result,
        "alignment_columns": len(trace),
        "query_gap_columns": int(np.count_nonzero(trace[:, 0] < 0)),
        "target_gap_columns": int(np.count_nonzero(trace[:, 1] < 0)),
    }


def alignment_quality_batch(query_ca, target_cas, query_starts, target_starts, cigars):
    """Measure multiple alignments to one query with one Biotite contact calculation."""
    target_cas = list(target_cas)
    query_starts = list(query_starts)
    target_starts = list(target_starts)
    cigars = list(cigars)
    size = len(target_cas)
    if not size or not (len(query_starts) == len(target_starts) == len(cigars) == size):
        raise ValueError("Expected equally sized nonempty alignment batches")
    for atoms in (query_ca, *target_cas):
        if (not len(atoms) or not np.all(atoms.atom_name == "CA")
                or struc.get_residue_count(atoms) != len(atoms)
                or not np.isfinite(atoms.coord).all()):
            raise ValueError("Expected one finite C-alpha atom per residue")
        chains, _ = struc.to_sequence(atoms)
        if len(chains) != 1:
            raise ValueError("Expected one protein chain")

    subjects = np.full(
        (size, *query_ca.coord.shape), np.nan, dtype=query_ca.coord.dtype
    )
    traces = []
    pairs_per_alignment = []
    for index, (target_ca, query_start, target_start, cigar) in enumerate(
        zip(target_cas, query_starts, target_starts, cigars)
    ):
        trace = cigar_trace(
            len(query_ca), len(target_ca), query_start, target_start, cigar
        )
        pairs = trace[np.all(trace >= 0, axis=1)]
        if not len(pairs):
            raise ValueError("Alignment contains no paired residues")
        subjects[index, pairs[:, 0]] = target_ca.coord[pairs[:, 1]]
        traces.append(trace)
        pairs_per_alignment.append(pairs)

    values = struc.lddt(
        query_ca,
        subjects,
        aggregation="atom",
        distance_bins=np.nextafter(np.array([0.5, 1.0, 2.0, 4.0]), -np.inf),
        filter_function=lambda contacts: struc.index_distance(query_ca, contacts) < 15.0,
    )
    output = []
    for index, (trace, pairs) in enumerate(zip(traces, pairs_per_alignment)):
        aligned = values[index, pairs[:, 0]]
        output.append({
            "lddt": values[index],
            "pairs": pairs,
            "mean_aligned_lddt": (
                float(aligned.mean()) if np.isfinite(aligned).all() else None
            ),
            "alignment_columns": len(trace),
            "matched_columns": len(pairs),
            "query_gap_columns": int(np.count_nonzero(trace[:, 0] < 0)),
            "target_gap_columns": int(np.count_nonzero(trace[:, 1] < 0)),
        })
    return output


def multidomain_metrics(
    query_plddt,
    ranked_alignments,
    *,
    quality_hits=5,
    coverage_hits=20,
    tp_threshold=0.6,
    fp_threshold=0.25,
    lddt_cutoff=1.0,
    reference_quirks=False,
):
    """Calculate the multidomain metrics used by the Foldseek benchmark."""
    if (quality_hits < 1 or coverage_hits < 1
            or not 0 <= fp_threshold < tp_threshold <= 1
            or not 0 <= lddt_cutoff <= 1):
        raise ValueError("Invalid multidomain metric settings")
    masks = {query: np.asarray(values) >= 70 for query, values in query_plddt.items()}
    if any(np.asarray(values).ndim != 1 or not np.isfinite(values).all()
           for values in query_plddt.values()):
        raise ValueError("Expected finite per-residue query pLDDT")
    total = sum(mask.sum() for mask in masks.values())
    if not total:
        raise ValueError("No query residues pass the pLDDT cutoff")
    counts = {query: np.zeros(len(mask), dtype=np.int32) for query, mask in masks.items()}
    blocked = {query: np.zeros(len(mask), dtype=bool) for query, mask in masks.items()}

    seen = set()
    previous = None
    first_query = None
    rank = 0
    good = 0
    aligned = 0
    undefined = 0
    selected_hits = {query: 0 for query in masks}

    for record in ranked_alignments:
        query = record["query"]
        if query not in masks:
            raise ValueError(f"Query outside the benchmark cohort: {query}")
        if query != previous:
            if query in seen:
                raise ValueError("Noncontiguous alignments for one query")
            seen.add(query)
            previous = query
            rank = 0
            if first_query is None:
                first_query = query
        rank += 1

        limit = coverage_hits - int(reference_quirks and query == first_query)
        if rank > max(quality_hits, limit):
            continue
        positions = np.asarray(record["query_indices"], dtype=np.int64)
        if (positions.ndim != 1 or not len(positions)
                or np.any(positions < 0) or np.any(positions >= len(masks[query]))
                or len(np.unique(positions)) != len(positions)):
            raise ValueError("Invalid matched query residue indices")
        eligible = positions[masks[query][positions]]
        mean = record["mean_aligned_lddt"]
        if mean is not None and (not np.isfinite(mean) or not 0 <= mean <= 1):
            raise ValueError("Expected mean lDDT in [0, 1] or None")
        undefined += mean is None

        if rank <= quality_hits:
            aligned += len(eligible)
            if mean is not None and float(f"{mean:.3f}") >= tp_threshold:
                good += len(eligible)
        if rank <= limit:
            selected_hits[query] += 1
            if mean is not None and mean <= lddt_cutoff:
                if mean >= tp_threshold:
                    counts[query][eligible[~blocked[query][eligible]]] += 1
                elif mean <= fp_threshold:
                    blocked[query][eligible] = True

    denominator = (
        sum(masks[query].sum() for query in seen) if reference_quirks else total
    )
    cumulative = np.concatenate(list(counts.values()))
    return {
        "protocol": "foldseek-multidomain-alignment-mean",
        "reference_revision": "1737c718bfafd9f6ac300fb6a0ddde6330166202",
        "bookkeeping": "upstream-replay" if reference_quirks else "uniform-hit-limit-full-query-cohort",
        "parameters": {
            "quality_hits": quality_hits,
            "coverage_hits": coverage_hits,
            "tp_threshold": tp_threshold,
            "fp_threshold": fp_threshold,
            "lddt_cutoff": lddt_cutoff,
            "query_plddt_min": 70,
        },
        "queries": len(masks),
        "queries_with_hits": len(seen),
        "eligible_query_residues": int(total),
        "undefined_alignment_means": int(undefined),
        "quality": {
            "sensitivity": good / (quality_hits * denominator),
            "selectivity": good / aligned if aligned else None,
            "correct_aligned_residues": good,
            "aligned_residues": aligned,
            "eligible_query_residues": int(denominator),
        },
        "coverage": {
            "tp_hits": list(range(1, coverage_hits + 1)),
            "query_coverage": [
                float(np.count_nonzero(cumulative >= n) / total)
                for n in range(1, coverage_hits + 1)
            ],
        },
        "evaluated_coverage_hits": selected_hits,
    }


def paper_multidomain_metrics(
    query_plddt,
    ranked_alignments,
    *,
    quality_hits=5,
    coverage_hits=20,
    tp_threshold=0.6,
    fp_threshold=0.25,
):
    """Calculate coverage and alignment quality as described in the Methods.

    Search coverage stops at the first alignment classified as a false positive
    by mean lDDT. Alignment quality uses individual residue-pair lDDT scores.
    Values between the two thresholds are neutral in both analyses. To run
    the released Foldseek script's metric, use :func:`multidomain_metrics`
    with ``reference_quirks=True``.
    """
    if not 0 <= fp_threshold < tp_threshold <= 1:
        raise ValueError("Invalid lDDT thresholds")
    masks = {query: np.asarray(values) >= 70 for query, values in query_plddt.items()}
    if not masks or any(mask.ndim != 1 for mask in masks.values()):
        raise ValueError("Expected one query pLDDT vector per protein")
    total = sum(int(mask.sum()) for mask in masks.values())
    if total <= 0:
        raise ValueError("No eligible query residues")
    counts = {query: np.zeros(len(mask), dtype=np.int32) for query, mask in masks.items()}
    stopped = {query: False for query in masks}
    selected = {query: 0 for query in masks}
    seen = set()
    last_query = None
    rank = 0
    tp_residues = aligned_residues = 0
    tp_matches = fp_matches = 0
    for row in ranked_alignments:
        query = row["query"]
        if query not in masks:
            raise ValueError(f"Query outside cohort: {query}")
        if query != last_query:
            if query in seen:
                raise ValueError("Noncontiguous ranked query hits")
            seen.add(query)
            last_query = query
            rank = 0
        rank += 1
        if rank > max(quality_hits, coverage_hits):
            continue
        positions = np.asarray(row["query_indices"], dtype=np.int64)
        if (positions.ndim != 1 or not len(positions) or
                np.any(positions < 0) or np.any(positions >= len(masks[query])) or
                len(np.unique(positions)) != len(positions)):
            raise ValueError("Invalid paired query positions")
        eligible = masks[query][positions]
        mean = row["mean_aligned_lddt"]
        if mean is not None and not np.isfinite(mean):
            raise ValueError("Nonfinite alignment-mean lDDT")
        if rank <= quality_hits:
            residue_values = np.asarray(row["residue_lddt"], dtype=np.float64)
            if residue_values.shape != positions.shape or not np.isfinite(residue_values).all():
                raise ValueError("Missing or invalid residue-wise lDDT for top-five quality")
            aligned_residues += int(eligible.sum())
            tp_residues += int(np.count_nonzero(eligible & (residue_values >= tp_threshold)))
        if rank <= coverage_hits:
            selected[query] += 1
            if not stopped[query]:
                if mean is not None and mean <= fp_threshold:
                    stopped[query] = True
                    fp_matches += 1
                elif mean is not None and mean >= tp_threshold:
                    counts[query][positions[eligible]] += 1
                    tp_matches += 1
    if len(seen) != len(masks):
        raise ValueError("Some benchmark queries have no ranked hits")
    cumulative = np.concatenate(list(counts.values()))
    return {
        "protocol": "Foldseek Methods-prose diagnostic (not released-script replay)",
        "queries": len(masks),
        "eligible_query_residues": total,
        "parameters": {
            "quality_hits": quality_hits, "coverage_hits": coverage_hits,
            "tp_threshold": tp_threshold, "fp_threshold": fp_threshold,
            "query_plddt_min": 70,
        },
        "coverage": {
            "tp_hits": list(range(1, coverage_hits + 1)),
            "query_coverage": [
                float(np.count_nonzero(cumulative >= n) / total)
                for n in range(1, coverage_hits + 1)
            ],
            "first_fp_rule": "stop all later hits for query after first alignment-mean lDDT <= 0.25",
            "tp_matches_before_first_fp": tp_matches,
            "queries_with_first_fp": fp_matches,
            "evaluated_hits": selected,
        },
        "quality": {
            "sensitivity": tp_residues / (quality_hits * total),
            "selectivity": tp_residues / aligned_residues if aligned_residues else None,
            "correct_aligned_residues": tp_residues,
            "aligned_residues": aligned_residues,
            "classification": "residue-wise lDDT >= 0.6 (TP); <= 0.25 (FP); middle ignored",
        },
    }
