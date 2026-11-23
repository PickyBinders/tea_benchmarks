"""Single-pass SCOP metrics for query-grouped sparse search results."""

import math

import numpy as np

from scripts.scop import fp_policy_name, resolve_negative_boundaries


LEVELS = ("family", "superfamily", "fold")
ANNOTATION_KEYS = {
    "family": "families",
    "superfamily": "superfamilies",
    "fold": "folds",
}


def _curve_from_histogram(histogram, positive_weight, weighting, level,
                          boundaries, queries):
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve

    score_values = []
    labels = []
    weights = []
    positive_index, negative_index = (
        (2, 3) if weighting == "foldseek-corrected" else (0, 1)
    )
    for score, counts in histogram.items():
        for label, index in ((True, positive_index), (False, negative_index)):
            if counts[index]:
                score_values.append(score)
                labels.append(label)
                weights.append(counts[index])
    observed_weight = math.fsum(
        weight for label, weight in zip(labels, weights) if label
    )
    coverage = observed_weight / positive_weight
    if labels:
        labels_array = np.asarray(labels, dtype=bool)
        scores_array = np.asarray(score_values, dtype=np.float64)
        weights_array = np.asarray(weights, dtype=np.float64)
        precision, recall, thresholds = precision_recall_curve(
            labels_array, scores_array, sample_weight=weights_array
        )
        recall *= coverage
        average_precision = (
            float(
                average_precision_score(
                    labels_array, scores_array, sample_weight=weights_array
                )
            )
            * coverage
            if observed_weight else 0.0
        )
    else:
        precision = np.asarray([1.0])
        recall = np.asarray([0.0])
        thresholds = np.asarray([])
        average_precision = 0.0
    return {
        "level": level,
        "weighting": weighting,
        "queries": queries,
        "negative_boundary": boundaries[level],
        "fp_policy": fp_policy_name(boundaries),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "ties": "whole-score-blocks",
        "average_precision": average_precision,
        "trapezoidal_auc": (
            float(auc(recall, precision)) if len(recall) > 1 else 0.0
        ),
        "positive_weight": positive_weight,
        "reported_positive_weight": observed_weight,
        "coverage": "reported-directed-hits",
    }


def _error_curve_from_histogram(histogram, total_positive, level, boundaries,
                                queries, *, top_hit=False):
    thresholds = sorted(histogram, reverse=True)
    true_positives = [0]
    false_positives = [0]
    for score in thresholds:
        counts = histogram[score]
        true_positives.append(true_positives[-1] + counts[0])
        false_positives.append(false_positives[-1] + counts[1])
    errors = np.asarray(false_positives, dtype=np.float64) / queries
    sensitivity = np.asarray(true_positives, dtype=np.float64) / total_positive
    return {
        "level": level,
        "kind": "top_hit" if top_hit else "cve",
        "queries": queries,
        "negative_boundary": boundaries[level],
        "fp_policy": fp_policy_name(boundaries),
        "denominator": "all-queries" if top_hit else "all-directed-positive-pairs",
        "coverage": "reported-directed-hits",
        "ties": "whole-score-blocks; mixed-top-hit-tie-is-error",
        "errors_per_query": errors.tolist(),
        "sensitivity": sensitivity.tolist(),
        "thresholds": [None, *thresholds],
        "sensitivity_at_error_budget": {
            str(budget): float(sensitivity[errors <= budget].max())
            for budget in (0.1, 1, 10)
        },
    }


def _top_hit_curve(top_hits, level, boundaries, queries):
    histogram = {}
    for score, correct in top_hits:
        block = histogram.setdefault(score, [0, 0])
        block[0 if correct else 1] += 1
    return _error_curve_from_histogram(histogram, queries, level, boundaries, queries,
                                       top_hit=True)


def evaluate_all_sparse(cohort, records, annotations, levels=LEVELS,
                        policies=("primary", "strict"), target_cohort=None):
    """Compute all reported search curves while consuming records once.

    Records must be grouped by query, as native MMseqs/Foldseek result files
    are. Missing hits remain absent while population denominators use the full
    target cohort.
    """
    cohort = list(cohort)
    target_cohort = list(cohort if target_cohort is None else target_cohort)
    levels = tuple(levels)
    policies = tuple(policies)
    if not cohort or len(cohort) != len(set(cohort)):
        raise ValueError("Sparse cohort must be nonempty and unique")
    if not target_cohort or len(target_cohort) != len(set(target_cohort)):
        raise ValueError("Sparse target cohort must be nonempty and unique")
    if not set(cohort) <= set(target_cohort):
        raise ValueError("Sparse query cohort must be a subset of the target cohort")
    if not levels or len(levels) != len(set(levels)) or not set(levels) <= set(LEVELS):
        raise ValueError("Sparse all-curves supports distinct Family/Superfamily/Fold levels")
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("Sparse FP policies must be nonempty and distinct")
    boundaries_by_policy = {
        policy: resolve_negative_boundaries(policy) for policy in policies
    }
    identifier_index = {
        identifier: index for index, identifier in enumerate(target_cohort)
    }
    query_index = {
        identifier: index for index, identifier in enumerate(cohort)
    }
    query_target_indices = np.asarray(
        [identifier_index[identifier] for identifier in cohort], dtype=np.int32
    )
    if set(target_cohort) - set(annotations):
        raise ValueError("Missing sparse target-cohort annotations")

    codes = {}
    counts = {}
    for level, key in ANNOTATION_KEYS.items():
        values = []
        for identifier in target_cohort:
            labels = annotations[identifier][key]
            if len(labels) != 1:
                raise ValueError("Sparse domain evaluation requires one label per level")
            values.append(labels[0])
        vocabulary = {value: index for index, value in enumerate(sorted(set(values)))}
        codes[level] = np.asarray([vocabulary[value] for value in values], dtype=np.int32)
        counts[level] = np.bincount(codes[level])

    query_codes = {
        level: codes[level][query_target_indices] for level in ANNOTATION_KEYS
    }
    positive_counts = {
        "family": counts["family"][query_codes["family"]] - 1,
        "superfamily": (
            counts["superfamily"][query_codes["superfamily"]]
            - counts["family"][query_codes["family"]]
        ),
        "fold": (
            counts["fold"][query_codes["fold"]]
            - counts["superfamily"][query_codes["superfamily"]]
        ),
    }
    fold_positive_counts = (
        counts["fold"][query_codes["fold"]]
        - counts["superfamily"][query_codes["superfamily"]]
    )
    negative_weight_denominators = np.maximum(
        np.maximum(fold_positive_counts, positive_counts["family"]), 1
    )
    negative_counts = {
        policy: {
            level: len(target_cohort)
            - counts[boundaries[level]][query_codes[boundaries[level]]]
            for level in levels
        }
        for policy, boundaries in boundaries_by_policy.items()
    }
    for level in levels:
        if not int(positive_counts[level].sum()):
            raise ValueError(f"Sparse {level} metrics require eligible positive pairs")
        for policy in policies:
            if not int(negative_counts[policy][level].sum()):
                raise ValueError(f"Sparse {policy} {level} requires eligible negatives")

    states = {
        policy: {
            level: {"histogram": {}, "per_query": {}, "top_hits": []}
            for level in levels
        }
        for policy in policies
    }

    def update_histogram(histogram, values, positive, positive_denominator,
                         negative_weight_denominator):
        for label, mask in ((True, positive), (False, ~positive)):
            if not mask.any():
                continue
            unique, observed = np.unique(values[mask], return_counts=True)
            for score, count in zip(unique.tolist(), observed.tolist()):
                block = histogram.setdefault(score, [0, 0, 0.0, 0.0])
                if label:
                    block[0] += count
                    block[2] += count / positive_denominator
                else:
                    block[1] += count
                    block[3] += count / negative_weight_denominator

    completed_queries = set()

    def process_query(query, target_names, score_values):
        current_query_index = query_index[query]
        query_target_index = query_target_indices[current_query_index]
        target_indices = np.asarray(
            [identifier_index[target] for target in target_names], dtype=np.int32
        )
        score_values = np.asarray(score_values, dtype=np.float64)
        family_positive = (
            codes["family"][target_indices] == codes["family"][query_target_index]
        )
        superfamily_positive = (
            (codes["superfamily"][target_indices]
             == codes["superfamily"][query_target_index])
            & ~family_positive
        )
        fold_positive = (
            (codes["fold"][target_indices]
             == codes["fold"][query_target_index])
            & ~(codes["superfamily"][target_indices]
                == codes["superfamily"][query_target_index])
        )
        positive_masks = {
            "family": family_positive,
            "superfamily": superfamily_positive,
            "fold": fold_positive,
        }
        for policy, boundaries in boundaries_by_policy.items():
            for level in levels:
                state = states[policy][level]
                positive = positive_masks[level]
                boundary = boundaries[level]
                negative = (
                    codes[boundary][target_indices]
                    != codes[boundary][query_target_index]
                )
                eligible = positive | negative
                population_positive = int(
                    positive_counts[level][current_query_index]
                )
                population_negative = int(
                    negative_counts[policy][level][current_query_index]
                )
                if not population_positive:
                    row = {
                        "sensitivity": None,
                        "positives": 0,
                        "negatives": population_negative,
                        "reason": "no-positive-targets",
                    }
                elif not population_negative:
                    row = {
                        "sensitivity": None,
                        "positives": population_positive,
                        "negatives": 0,
                        "reason": "no-negative-targets",
                    }
                else:
                    negative_scores = score_values[negative]
                    threshold = (
                        float(negative_scores.max())
                        if len(negative_scores) else -math.inf
                    )
                    recovered = int(np.count_nonzero(score_values[positive] > threshold))
                    row = {
                        "sensitivity": recovered / population_positive,
                        "positives": population_positive,
                        "negatives": population_negative,
                        "recovered": recovered,
                        "first_fp_score": threshold if len(negative_scores) else None,
                    }
                row["scop_folds"] = sorted(annotations[query]["folds"])
                state["per_query"][query] = row
                if eligible.any():
                    eligible_scores = score_values[eligible]
                    eligible_positive = positive[eligible]
                    update_histogram(
                        state["histogram"], eligible_scores, eligible_positive,
                        population_positive,
                        int(negative_weight_denominators[current_query_index]),
                    )
                    best = float(eligible_scores.max())
                    best_positive = eligible_positive[eligible_scores == best]
                    state["top_hits"].append((best, bool(best_positive.all())))

    current_query = None
    current_targets = []
    current_scores = []
    targets_seen = set()
    for query, target, score in records:
        if query not in query_index or target not in identifier_index:
            raise ValueError("Sparse hit outside the declared cohort")
        if not math.isfinite(score):
            raise ValueError("Sparse search score must be finite")
        if query != current_query:
            if query in completed_queries:
                raise ValueError("Sparse search records are not grouped by query")
            if current_query is not None:
                process_query(current_query, current_targets, current_scores)
                completed_queries.add(current_query)
            current_query = query
            current_targets = []
            current_scores = []
            targets_seen = set()
        if target in targets_seen:
            raise ValueError("Duplicate directed sparse query-target pair")
        targets_seen.add(target)
        current_targets.append(target)
        current_scores.append(score)
    if current_query is not None:
        process_query(current_query, current_targets, current_scores)
        completed_queries.add(current_query)

    for identifier in cohort:
        if identifier in completed_queries:
            continue
        process_query(identifier, [], [])

    outputs = {}
    queries = len(cohort)
    for policy, boundaries in boundaries_by_policy.items():
        files = {}
        metric_levels = {}
        for level in levels:
            state = states[policy][level]
            rows = {
                identifier: state["per_query"][identifier]
                for identifier in sorted(cohort)
            }
            eligible = [
                row["sensitivity"] for row in rows.values()
                if row["sensitivity"] is not None
            ]
            metric_levels[level] = {
                "auc": math.fsum(eligible) / len(eligible),
                "eligible_queries": len(eligible),
                "per_query": rows,
            }
            unweighted_positive = int(positive_counts[level].sum())
            weighted_positive = int(np.count_nonzero(positive_counts[level]))
            files[f"pr_{level}.json"] = _curve_from_histogram(
                state["histogram"], unweighted_positive, "none", level,
                boundaries, queries,
            )
            files[f"weighted_pr_{level}.json"] = _curve_from_histogram(
                state["histogram"], weighted_positive, "foldseek-corrected",
                level, boundaries, queries,
            )
            files[f"cve_{level}.json"] = _error_curve_from_histogram(
                state["histogram"], unweighted_positive, level, boundaries,
                queries,
            )
            files[f"top_hit_{level}.json"] = _top_hit_curve(
                state["top_hits"], level, boundaries, queries
            )
        files["metrics.json"] = {
            "schema_version": 1,
            "auc_definition": "mean-sensitivity-before-first-fp",
            "ties": "positive-score-strictly-greater-than-highest-negative",
            "queries": queries,
            "targets": len(target_cohort),
            "levels": metric_levels,
            "negative_boundaries": boundaries,
            "fp_policy": fp_policy_name(boundaries),
            "coverage": (
                "reported-directed-hits; complete-target population denominators"
                if target_cohort != cohort else "reported-directed-hits"
            ),
        }
        outputs[policy] = files
    return outputs
