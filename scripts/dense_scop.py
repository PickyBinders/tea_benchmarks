"""Vectorized SCOP metrics for complete unordered all-versus-all scores."""

from pathlib import Path
from collections import Counter

import numpy as np

from scripts.io import read_json
from scripts.scop import fp_policy_name, resolve_negative_boundaries


LEVELS = ("family", "superfamily", "fold")


def read_score_matrix(paths, cohort, score_scale, shard_order="row"):
    """Read upper-triangle shards into one symmetric matrix.

    ``score_scale=None`` retains exact floating-point scores for continuous
    methods such as EBA. Alphabet shards interleave query rows, whereas the
    exhaustive EBA scorer interleaves global unordered pair positions.
    """
    import pyarrow as pa
    import pyarrow.csv as csv

    size = len(cohort)
    if shard_order not in ("row", "pair"):
        raise ValueError("Unknown dense-score shard order")
    continuous = score_scale is None
    matrix = np.full((size, size), np.nan if continuous else -1,
                     dtype=np.float64 if continuous else np.int32)
    np.fill_diagonal(matrix, 0)
    seen = set()
    for path in map(Path, paths):
        metadata = read_json(str(path) + ".json")
        shard, shards = metadata["shard"], metadata["shards"]
        if shard in seen:
            raise ValueError("Duplicate score shard")
        seen.add(shard)
        table = csv.read_csv(
            path,
            read_options=csv.ReadOptions(use_threads=True),
            parse_options=csv.ParseOptions(delimiter="\t"),
            convert_options=csv.ConvertOptions(
                include_columns=["score"],
                column_types={"score": pa.float64()},
            ),
        )
        values = table["score"].to_numpy(zero_copy_only=False)
        if continuous:
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Dense evaluation requires finite nonnegative scores")
        else:
            scaled = values * score_scale
            if not np.isfinite(scaled).all() or not np.equal(scaled, np.rint(scaled)).all():
                raise ValueError("Dense evaluation requires exact integer score units")
            values = np.rint(scaled).astype(np.int32)
        total_pairs = size * (size - 1) // 2
        expected = (len(range(shard, total_pairs, shards)) if shard_order == "pair"
                    else sum(size - left - 1 for left in range(shard, size, shards)))
        if len(values) != expected or metadata["pairs"] != expected:
            raise ValueError("Score shard row count does not match its declared order")
        offset = 0
        left_rows = range(size - 1) if shard_order == "pair" else range(shard, size - 1, shards)
        for left in left_rows:
            if shard_order == "pair":
                row_start = left * (2 * size - left - 1) // 2
                first_right = left + 1 + (shard - row_start) % shards
                if first_right >= size:
                    continue
                right = np.arange(first_right, size, shards)
            else:
                right = np.arange(left + 1, size)
            count = len(right)
            row = values[offset:offset + count]
            matrix[left, right] = row
            matrix[right, left] = row
            offset += count
        if offset != len(values):
            raise ValueError("Score shard ordering does not cover its declared pairs")
    if seen != set(range(shards)) or (
        not np.isfinite(matrix).all() if continuous else np.any(matrix < 0)
    ):
        raise ValueError("Incomplete dense score shards")
    return matrix


def _codes(cohort, annotations):
    result = []
    labels = []
    for key in ("families", "superfamilies", "folds"):
        values = []
        for query in cohort:
            states = annotations[query][key]
            if len(states) != 1:
                raise ValueError("Dense domain evaluation requires one SCOP label per level")
            values.append(states[0])
        unique = {value: index for index, value in enumerate(sorted(set(values)))}
        result.append(np.asarray([unique[value] for value in values], dtype=np.int32))
        labels.append(values)
    return result, labels


def _positive_mask(level, query, family, superfamily, fold):
    if level == 0:
        return family == family[query]
    if level == 1:
        return (superfamily == superfamily[query]) & (family != family[query])
    return (fold == fold[query]) & (superfamily != superfamily[query])


def _histogram_curve(positive, negative, lengths, score_scale):
    present = (positive != 0) | (negative != 0)
    length_index, raw_score = np.nonzero(present)
    thresholds = raw_score.astype(np.float64) / score_scale
    thresholds /= np.sqrt(lengths[length_index])
    order = np.argsort(-thresholds, kind="stable")
    thresholds = thresholds[order]
    positive = positive[length_index, raw_score][order]
    negative = negative[length_index, raw_score][order]
    ends = np.r_[np.flatnonzero(np.diff(thresholds)), len(thresholds) - 1]
    return (
        np.cumsum(positive)[ends],
        np.cumsum(negative)[ends],
        thresholds[ends],
    )


def _precision_recall(positive, negative, lengths, score_scale, level,
                      weighting, positive_weight=None):
    from sklearn.metrics import auc

    tp, fp, thresholds = _histogram_curve(
        positive, negative, lengths, score_scale
    )
    observed = float(tp[-1])
    positive_weight = observed if positive_weight is None else float(positive_weight)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = tp / positive_weight
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    thresholds = thresholds[::-1]
    average_precision = float(-np.diff(recall) @ precision[:-1])
    return {
        "level": level,
        "weighting": weighting,
        "queries": int(positive.shape[0] and 0),  # Set by the caller.
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "ties": "whole-score-blocks",
        "average_precision": average_precision,
        "trapezoidal_auc": float(auc(recall, precision)),
        "positive_weight": positive_weight,
        "reported_positive_weight": observed,
        "coverage": "complete-unordered-pairs",
    }


def _error_curve(positive, negative, lengths, score_scale, level, queries,
                 total_positive):
    tp, fp, thresholds = _histogram_curve(
        positive, negative, lengths, score_scale
    )
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    errors = fp / queries
    sensitivity = tp / total_positive
    return {
        "level": level,
        "kind": "cve",
        "queries": queries,
        "denominator": "all-directed-positive-pairs",
        "coverage": "complete-unordered-pairs",
        "ties": "whole-score-blocks; mixed-top-hit-tie-is-error",
        "errors_per_query": errors.tolist(),
        "sensitivity": sensitivity.tolist(),
        "thresholds": [None, *thresholds.tolist()],
        "sensitivity_at_error_budget": {
            str(budget): float(sensitivity[errors <= budget].max())
            for budget in (0.1, 1, 10)
        },
    }


def _top_hit_curve(values, correct, level, queries):
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    correct = correct[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
    tp = np.r_[0, np.cumsum(correct)[ends]]
    fp = np.r_[0, np.cumsum(1 - correct)[ends]]
    errors = fp / queries
    sensitivity = tp / queries
    return {
        "level": level,
        "kind": "top_hit",
        "queries": queries,
        "denominator": "all-queries",
        "coverage": "complete-unordered-pairs",
        "ties": "whole-score-blocks; mixed-top-hit-tie-is-error",
        "errors_per_query": errors.tolist(),
        "sensitivity": sensitivity.tolist(),
        "thresholds": [None, *ranked[ends].tolist()],
        "sensitivity_at_error_budget": {
            str(budget): float(sensitivity[errors <= budget].max())
            for budget in (0.1, 1, 10)
        },
    }


def evaluate_all(cohort, scores, annotations, score_scale=1, query_lengths=None,
                 levels=LEVELS, negative_boundaries=None,
                 weighting_cohort=None):
    """Compute every dense SCOP curve while aggregating discrete score bins once."""
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels) or not set(levels) <= set(LEVELS):
        raise ValueError("Require distinct known hierarchy levels")
    boundaries = resolve_negative_boundaries(negative_boundaries)
    cohort = list(cohort)
    queries = len(cohort)
    if scores.shape != (queries, queries) or not np.array_equal(scores, scores.T):
        raise ValueError("Expected a symmetric dense score matrix")
    family, superfamily, fold = _codes(cohort, annotations)[0]
    code_counts = [np.bincount(code)[code] for code in (family, superfamily, fold)]
    denominators = np.stack((
        code_counts[0] - 1,
        code_counts[1] - code_counts[0],
        code_counts[2] - code_counts[1],
    ))
    # CV can use full-cohort pair weights for weighted PR. Eligibility and
    # the other metrics use the scored holdout domains.
    weight_denominators = denominators
    if weighting_cohort is not None:
        reference = list(weighting_cohort)
        if len(reference) != len(set(reference)) or not set(cohort) <= set(reference):
            raise ValueError("Weighting cohort must uniquely contain all scored domains")
        _, reference_labels = _codes(reference, annotations)
        _, query_labels = _codes(cohort, annotations)
        reference_counts = [Counter(labels) for labels in reference_labels]
        full_counts = [
            np.asarray([counts[label] for label in labels], dtype=np.int64)
            for counts, labels in zip(reference_counts, query_labels)
        ]
        weight_denominators = np.stack((
            full_counts[0] - 1,
            full_counts[1] - full_counts[0],
            full_counts[2] - full_counts[1],
        ))
    hierarchy_codes = dict(zip(LEVELS, (family, superfamily, fold)))
    hierarchy_counts = dict(zip(LEVELS, code_counts))
    negative_counts = np.stack([
        queries - hierarchy_counts[boundaries[level]] for level in LEVELS
    ])
    for level, name in enumerate(LEVELS):
        if name not in levels:
            continue
        if not int(denominators[level].sum()):
            raise ValueError(f"Dense {name} metrics require eligible positive pairs")
        if not int(negative_counts[level].sum()):
            raise ValueError(f"Dense {name} metrics require eligible negative pairs")
    lengths = np.ones(queries, dtype=np.int64) if query_lengths is None else np.asarray(
        [query_lengths[query] for query in cohort], dtype=np.int64
    )
    if np.any(lengths <= 0):
        raise ValueError("Positive query lengths required")
    unique_lengths, length_codes = np.unique(lengths, return_inverse=True)
    score_bins = int(scores.max()) + 1
    counts = np.zeros((3, 2, len(unique_lengths), score_bins), dtype=np.int64)
    weighted = np.zeros((3, 2, len(unique_lengths), score_bins), dtype=np.float64)
    sensitivity_rows = [[None] * queries for _ in LEVELS]
    top_values = np.empty((3, queries), dtype=np.float64)
    top_correct = np.empty((3, queries), dtype=bool)

    for query in range(queries):
        row = scores[query]
        length_code = length_codes[query]
        negative_weight = 1 / max(int(weight_denominators[2, query]),
                                  int(weight_denominators[0, query]), 1)
        for level, name in enumerate(LEVELS):
            positive = _positive_mask(level, query, family, superfamily, fold)
            positive[query] = False
            boundary_code = hierarchy_codes[boundaries[name]]
            negative = boundary_code != boundary_code[query]
            positive_hist = np.bincount(row[positive], minlength=score_bins)
            negative_hist = np.bincount(row[negative], minlength=score_bins)
            threshold = int(row[negative].max()) if negative.any() else None
            counts[level, 0, length_code] += positive_hist
            counts[level, 1, length_code] += negative_hist
            weighted[level, 1, length_code] += negative_hist * negative_weight
            if denominators[level, query]:
                weighted[level, 0, length_code] += (
                    positive_hist / weight_denominators[level, query]
                )

            if not denominators[level, query]:
                sensitivity_rows[level][query] = {
                    "sensitivity": None,
                    "positives": 0,
                    "negatives": int(negative_counts[level, query]),
                    "reason": "no-positive-targets",
                }
            elif not negative_counts[level, query]:
                sensitivity_rows[level][query] = {
                    "sensitivity": None,
                    "positives": int(denominators[level, query]),
                    "negatives": 0,
                    "reason": "no-negative-targets",
                }
            else:
                recovered = int(np.count_nonzero(row[positive] > threshold))
                sensitivity_rows[level][query] = {
                    "sensitivity": recovered / int(denominators[level, query]),
                    "positives": int(denominators[level, query]),
                    "negatives": int(negative_counts[level, query]),
                    "recovered": recovered,
                    "first_fp_score": threshold / score_scale / np.sqrt(lengths[query]),
                }

            eligible = positive | negative
            best = int(row[eligible].max())
            best_hits = eligible & (row == best)
            top_values[level, query] = best / score_scale / np.sqrt(lengths[query])
            top_correct[level, query] = bool(np.all(positive[best_hits]))

    files = {}
    level_results = {}
    for level, name in enumerate(LEVELS):
        if name not in levels:
            continue
        rows = {}
        for query, identifier in enumerate(cohort):
            row = sensitivity_rows[level][query]
            row["scop_folds"] = sorted(annotations[identifier]["folds"])
            rows[identifier] = row
        eligible = [row["sensitivity"] for row in rows.values()
                    if row["sensitivity"] is not None]
        level_results[name] = {
            "auc": sum(eligible) / len(eligible) if eligible else None,
            "eligible_queries": len(eligible),
            "per_query": rows,
        }
        for weighting, histogram, prefix, positive_weight in (
            ("none", counts, "pr", None),
            ("foldseek-corrected", weighted, "weighted_pr",
             (float(weighted[level, 0].sum()) if weighting_cohort is not None
              else int(np.count_nonzero(denominators[level] > 0)))),
        ):
            curve = _precision_recall(
                histogram[level, 0], histogram[level, 1], unique_lengths,
                score_scale, name, weighting, positive_weight,
            )
            curve["queries"] = queries
            curve["negative_boundary"] = boundaries[name]
            curve["fp_policy"] = fp_policy_name(boundaries)
            if weighting == "foldseek-corrected" and weighting_cohort is not None:
                curve["weighting_reference"] = "full-domain-cohort"
                curve["weighting_reference_domains"] = len(reference)
                curve["recall_denominator"] = "scored-positive-pairs-with-full-cohort-weights"
            files[f"{prefix}_{name}.json"] = curve
        cve = _error_curve(
            counts[level, 0], counts[level, 1], unique_lengths, score_scale,
            name, queries, int(denominators[level].sum()),
        )
        cve.update({"negative_boundary": boundaries[name],
                    "fp_policy": fp_policy_name(boundaries)})
        files[f"cve_{name}.json"] = cve
        top_hit = _top_hit_curve(
            top_values[level], top_correct[level], name, queries
        )
        top_hit.update({"negative_boundary": boundaries[name],
                        "fp_policy": fp_policy_name(boundaries)})
        files[f"top_hit_{name}.json"] = top_hit
    files["metrics.json"] = {
        "schema_version": 1,
        "auc_definition": "mean-sensitivity-before-first-fp",
        "ties": "positive-score-strictly-greater-than-highest-negative",
        "queries": queries,
        "levels": level_results,
        "negative_boundaries": boundaries,
        "fp_policy": fp_policy_name(boundaries),
        "coverage": "complete-unordered-pairs",
    }
    return files


def _lossless_step_indices(x, y):
    """Encode the empirical staircase by storing each segment's endpoints.

    Every score block contributes to the metrics and the raw score shards
    retain every threshold. Segment endpoints reproduce the same step path.
    """
    if len(x) != len(y) or not len(x):
        raise ValueError("Expected nonempty paired curve coordinates")
    keep = np.ones(len(x), dtype=bool)
    if len(x) > 2:
        vertical = (x[:-2] == x[1:-1]) & (x[1:-1] == x[2:])
        horizontal = (y[:-2] == y[1:-1]) & (y[1:-1] == y[2:])
        keep[1:-1] = ~(vertical | horizontal)
    return np.flatnonzero(keep)


def _continuous_event_curves(scores, positive, negative, positive_weights,
                             negative_weights, level, query_count, total_positive,
                             weighted_positive_total, boundaries):
    """Exact threshold metrics for continuous, exhaustive directed scores."""
    from sklearn.metrics import auc

    eligible = positive | negative
    flat = np.flatnonzero(eligible)
    values = scores.ravel()[flat]
    is_positive = positive.ravel()[flat]
    query_indices = (flat // query_count).astype(np.int32)
    del eligible, flat
    order = np.argsort(values)[::-1]
    ranked = values[order]
    is_positive = is_positive[order]
    query_indices = query_indices[order]
    del values, order
    if not len(ranked) or not np.isfinite(ranked).all():
        raise ValueError("Continuous SCOP events must be finite and nonempty")
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), len(ranked) - 1]
    thresholds = ranked[ends]
    del ranked

    tp = np.cumsum(is_positive, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    if int(tp[-1]) != total_positive or int(fp[-1]) != int(negative.sum()):
        raise ValueError("Incomplete continuous SCOP event population")
    weighted_tp = np.cumsum(
        np.where(is_positive, positive_weights[query_indices], 0.0),
        dtype=np.float64,
    )[ends]
    weighted_fp = np.cumsum(
        np.where(~is_positive, negative_weights[query_indices], 0.0),
        dtype=np.float64,
    )[ends]
    del is_positive, query_indices

    def pr_curve(weighting, positive_count, negative_count, denominator):
        recall = positive_count / denominator
        precision = np.divide(
            positive_count, positive_count + negative_count,
            out=np.zeros_like(recall), where=(positive_count + negative_count) != 0,
        )
        increments = np.diff(positive_count, prepend=0)
        ap = float(np.sum(increments * precision) / denominator)
        indices = _lossless_step_indices(recall, precision)
        draw_recall = np.r_[recall[indices][::-1], 0.0]
        draw_precision = np.r_[precision[indices][::-1], 1.0]
        result = {
            "level": level, "weighting": weighting, "queries": query_count,
            "precision": draw_precision.tolist(), "recall": draw_recall.tolist(),
            "thresholds": thresholds[indices][::-1].tolist(),
            "ties": "whole-score-blocks", "average_precision": ap,
            "trapezoidal_auc": float(auc(draw_recall, draw_precision)),
            "positive_weight": float(denominator),
            "reported_positive_weight": float(positive_count[-1]),
            "coverage": "complete-unordered-pairs",
            "negative_boundary": boundaries[level],
            "fp_policy": fp_policy_name(boundaries),
            "empirical_score_blocks": int(len(ends)),
            "curve_encoding": "lossless-collinear-step-endpoints; all score blocks evaluated",
        }
        return result

    pr = pr_curve("none", tp, fp, total_positive)
    weighted_pr = pr_curve(
        "foldseek-corrected", weighted_tp, weighted_fp,
        weighted_positive_total,
    )
    errors = np.r_[0.0, fp / query_count]
    sensitivity = np.r_[0.0, tp / total_positive]
    indices = _lossless_step_indices(errors, sensitivity)
    cve = {
        "level": level, "kind": "cve", "queries": query_count,
        "denominator": "all-directed-positive-pairs",
        "coverage": "complete-unordered-pairs",
        "ties": "whole-score-blocks; mixed-top-hit-tie-is-error",
        "errors_per_query": errors[indices].tolist(),
        "sensitivity": sensitivity[indices].tolist(),
        "thresholds": [None, *thresholds[indices[1:] - 1].tolist()],
        "sensitivity_at_error_budget": {
            str(budget): float(sensitivity[errors <= budget].max())
            for budget in (0.1, 1, 10)
        },
        "negative_boundary": boundaries[level],
        "fp_policy": fp_policy_name(boundaries),
        "empirical_score_blocks": int(len(ends)),
        "curve_encoding": "lossless-collinear-step-endpoints; all score blocks evaluated",
    }
    return {f"pr_{level}.json": pr,
            f"weighted_pr_{level}.json": weighted_pr,
            f"cve_{level}.json": cve}


def evaluate_all_continuous(cohort, scores, annotations, query_lengths,
                            levels=("family", "superfamily"),
                            negative_boundaries="superfamily-fp"):
    """Evaluate exhaustive continuous scores at their original precision.

    Yield one output at a time to bound memory. First-FP and top-hit metrics
    use exact scores. PR and CVE use every native score block and store the
    endpoints of each empirical step.
    """
    levels = tuple(levels)
    if not levels or not set(levels) <= {"family", "superfamily", "fold"}:
        raise ValueError("Continuous evaluation supports family, superfamily and fold")
    boundaries = resolve_negative_boundaries(negative_boundaries)
    if any(boundaries[level] != ("fold" if level == "fold" else "superfamily")
           for level in levels):
        raise ValueError("This evaluation requires different-superfamily FPs for family/superfamily and different-fold FPs for fold")
    cohort = list(cohort)
    size = len(cohort)
    if (scores.shape != (size, size) or scores.dtype != np.float64
            or not np.array_equal(scores, scores.T)
            or not np.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError("Expected complete symmetric nonnegative float64 scores")
    lengths = np.asarray([query_lengths[name] for name in cohort], dtype=np.int64)
    if (lengths <= 0).any():
        raise ValueError("Positive query lengths required")
    family, superfamily, fold = _codes(cohort, annotations)[0]
    family_count = np.bincount(family)[family]
    superfamily_count = np.bincount(superfamily)[superfamily]
    fold_count = np.bincount(fold)[fold]
    denominator = {
        "family": family_count - 1,
        "superfamily": superfamily_count - family_count,
        "fold": fold_count - superfamily_count,
    }
    negative_count = {
        "family": size - superfamily_count,
        "superfamily": size - superfamily_count,
        "fold": size - fold_count,
    }
    negative_weights = 1 / np.maximum(
        fold_count - superfamily_count, family_count - 1,
    ).clip(min=1)
    scores /= np.sqrt(lengths)[:, None]
    same_family = family[:, None] == family[None, :]
    same_superfamily = superfamily[:, None] == superfamily[None, :]
    same_fold = fold[:, None] == fold[None, :]
    results = {}
    for level in levels:
        positive = (same_family.copy() if level == "family" else
                    same_superfamily & ~same_family if level == "superfamily" else
                    same_fold & ~same_superfamily)
        negative = ~(same_fold if level == "fold" else same_superfamily)
        np.fill_diagonal(positive, False)
        count = denominator[level]
        total_positive = int(count.sum())
        if not total_positive or not negative.any():
            raise ValueError("Missing eligible SCOP positives or negatives")
        weights = 1 / np.maximum(count, 1)
        weighted_total = int(np.count_nonzero(count))
        per_query = {}
        top_values = np.empty(size, dtype=np.float64)
        top_correct = np.empty(size, dtype=bool)
        for query, name in enumerate(cohort):
            row = scores[query]
            false_positive = negative[query]
            true_positive = positive[query]
            cutoff = float(row[false_positive].max())
            recovered = int(np.count_nonzero(row[true_positive] > cutoff))
            eligible = bool(count[query] and negative_count[level][query])
            per_query[name] = {
                "sensitivity": recovered / int(count[query]) if eligible else None,
                "positives": int(count[query]),
                "negatives": int(negative_count[level][query]),
                "reason": None if eligible else (
                    "no-positive-targets" if not count[query] else "no-negative-targets"
                ),
                "recovered": recovered if eligible else None,
                "first_fp_score": cutoff if eligible else None,
                "scop_folds": sorted(annotations[name]["folds"]),
            }
            candidates = true_positive | false_positive
            best = float(row[candidates].max())
            top_values[query] = best
            top_correct[query] = not bool(np.any(false_positive & (row == best)))
        eligible_values = [row["sensitivity"] for row in per_query.values()
                           if row["sensitivity"] is not None]
        results[level] = {
            "auc": float(np.mean(eligible_values)),
            "eligible_queries": len(eligible_values),
            "per_query": per_query,
        }
        curves = _continuous_event_curves(
            scores, positive, negative, weights, negative_weights,
            level, size, total_positive, weighted_total, boundaries,
        )
        for filename, payload in curves.items():
            yield filename, payload
        top = _top_hit_curve(top_values, top_correct, level, size)
        top.update({"negative_boundary": boundaries[level],
                    "fp_policy": fp_policy_name(boundaries)})
        yield f"top_hit_{level}.json", top
        del positive, curves
    yield "metrics.json", {
        "schema_version": 1,
        "auc_definition": "mean-sensitivity-before-first-fp",
        "ties": "positive-score-strictly-greater-than-highest-negative",
        "queries": size, "levels": results,
        "negative_boundaries": boundaries,
        "fp_policy": fp_policy_name(boundaries),
        "coverage": "complete-unordered-pairs",
    }
