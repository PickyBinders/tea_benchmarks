"""SCOP set-based remote sensitivity with conservative first-FP ties."""

from bisect import bisect_right
import csv
from collections import Counter
import math
from collections.abc import Mapping
from pathlib import Path
import re

from scripts.io import partition_ids, read_json, sha256_file, write_json


LEVELS = ("family", "superfamily", "fold")
BOUNDARIES = LEVELS
FP_POLICIES = {
    "superfamily-fp": {"family": "superfamily", "superfamily": "superfamily", "fold": "fold"},
    "primary": {"family": "superfamily", "superfamily": "fold", "fold": "fold"},
    "strict": {level: level for level in LEVELS},
    "legacy-cross-fold": {level: "fold" for level in LEVELS},
}

def resolve_negative_boundaries(value=None):
    """Resolve a named policy or explicit boundary for each SCOP hierarchy level."""
    if value is None:
        result = dict(FP_POLICIES["primary"])
    elif isinstance(value, str):
        if value in FP_POLICIES:
            result = dict(FP_POLICIES[value])
        elif value in BOUNDARIES:
            order = {level: index for index, level in enumerate(LEVELS)}
            result = dict(FP_POLICIES["primary"])
            for level in LEVELS:
                if order[value] >= order[level]:
                    result[level] = value
        else:
            raise ValueError("Unknown negative-boundary specification")
    elif isinstance(value, Mapping):
        if set(value) != set(LEVELS):
            raise ValueError("Negative-boundary mapping must cover family, superfamily and fold")
        result = dict(value)
    else:
        raise ValueError("Unknown negative-boundary specification")

    order = {level: index for index, level in enumerate(LEVELS)}
    for level, boundary in result.items():
        if boundary not in BOUNDARIES or order[boundary] < order[level]:
            raise ValueError(f"Invalid {level} false-positive boundary: {boundary}")
    return result


def fp_policy_name(boundaries):
    boundaries = resolve_negative_boundaries(boundaries)
    for name, policy in FP_POLICIES.items():
        if boundaries == policy:
            return name
    return "custom"


def transform_search_value(value, specification):
    """Apply a monotone significance transform to the raw scores."""
    value = float(value)
    transform = specification.get("transform")
    if transform is None:
        result = value
    elif transform.get("kind") == "loglinear-evalue":
        required = ("loglinear_m", "loglinear_c", "target_database_size")
        if any(key not in transform for key in required):
            raise ValueError("Incomplete log-linear E-value transform")
        size = transform["target_database_size"]
        if not isinstance(size, int) or size <= 0:
            raise ValueError("E-value target database size must be a positive integer")
        result = size * 10 ** (
            float(transform["loglinear_m"]) * value
            + float(transform["loglinear_c"])
        )
    elif transform.get("kind") == "piecewise-loglinear-evalue":
        required = (
            "loglinear_m_low", "loglinear_m_high", "loglinear_c",
            "loglinear_breakpoint", "target_database_size",
        )
        if any(key not in transform for key in required):
            raise ValueError("Incomplete piecewise log-linear E-value transform")
        size = transform["target_database_size"]
        if not isinstance(size, int) or size <= 0:
            raise ValueError("E-value target database size must be a positive integer")
        m_low = float(transform["loglinear_m_low"])
        m_high = float(transform["loglinear_m_high"])
        intercept = float(transform["loglinear_c"])
        breakpoint = float(transform["loglinear_breakpoint"])
        if (not all(math.isfinite(parameter) for parameter in (
                m_low, m_high, intercept, breakpoint))
                or m_low >= 0 or m_high >= 0 or breakpoint <= 0):
            raise ValueError("Invalid piecewise log-linear E-value transform")
        exponent = intercept + m_low * value
        if value > breakpoint:
            exponent += (m_high - m_low) * (value - breakpoint)
        result = size * 10 ** exponent
    elif transform.get("kind") == "piecewise-loglinear-diversity-evalue":
        required = (
            "loglinear_m_low", "loglinear_m_high", "loglinear_c",
            "loglinear_breakpoint", "effective_target_count",
            "database_diversity_method", "error_unit",
        )
        if any(key not in transform for key in required):
            raise ValueError("Incomplete diversity-adjusted E-value transform")
        size = float(transform["effective_target_count"])
        m_low = float(transform["loglinear_m_low"])
        m_high = float(transform["loglinear_m_high"])
        intercept = float(transform["loglinear_c"])
        breakpoint = float(transform["loglinear_breakpoint"])
        if (
            not all(math.isfinite(parameter) for parameter in (
                size, m_low, m_high, intercept, breakpoint,
            ))
            or size <= 0 or m_low >= 0 or m_high >= 0 or breakpoint <= 0
            or transform["database_diversity_method"]
                != "aa3-tea5-doph64-kmv1024-v1"
            or transform["error_unit"] != "target-rows"
        ):
            raise ValueError("Invalid diversity-adjusted E-value transform")
        exponent = intercept + m_low * value
        if value > breakpoint:
            exponent += (m_high - m_low) * (value - breakpoint)
        result = size * 10 ** exponent
    else:
        raise ValueError("Unknown search-value transform")
    scale = specification.get("scale", 1.0)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Search score scale must be positive and finite")
    result *= scale
    if not math.isfinite(result):
        raise ValueError("Transformed search value must be finite")
    return result


def group_search_records_by_query(records, query_order):
    """Group interleaved search records by query for the sparse evaluator.

    The metrics evaluate whole blocks of tied scores within each query.
    """
    grouped = {query: [] for query in query_order}
    for query, target, score in records:
        if query not in grouped:
            raise ValueError("Search record query is outside the declared order")
        grouped[query].append((target, score))
    for query in query_order:
        rows = grouped.pop(query)
        for target, score in rows:
            yield query, target, score


def relation(query, target, level, negative_boundary=None,
             neutral_if_shared=None, positive_requires_different=None):
    if level not in LEVELS:
        raise ValueError(f"Unknown hierarchy level: {level}")
    for key in (neutral_if_shared, positive_requires_different):
        if key is not None and (key not in query or key not in target):
            raise ValueError(f"Missing pair-policy annotation: {key}")
    if (neutral_if_shared is not None
            and set(query[neutral_if_shared]) & set(target[neutral_if_shared])):
        return "neutral"
    boundary = resolve_negative_boundaries(negative_boundary)[level]
    shared = {key: bool(set(query[key]) & set(target[key]))
              for key in ("families", "superfamilies", "folds")}
    boundary_key = {
        "family": "families",
        "superfamily": "superfamilies",
        "fold": "folds",
    }[boundary]
    if not query[boundary_key] or not target[boundary_key]:
        return "neutral"
    category = "neutral"
    if level == "family":
        if shared["families"]:
            category = "positive"
    if level == "superfamily":
        if shared["superfamilies"] and not shared["families"]:
            category = "positive"
    if level == "fold":
        if shared["folds"] and not shared["superfamilies"]:
            category = "positive"
    if category == "positive":
        if (positive_requires_different is not None
                and set(query[positive_requires_different])
                & set(target[positive_requires_different])):
            return "neutral"
        return "positive"
    return "neutral" if shared[boundary_key] else "negative"


def _domain_population(identifiers, annotations, negative_boundary=None):
    """Positive/negative denominators from the cohort, independent of returned hits."""
    keys = ("families", "superfamilies", "folds")
    if any(len(annotations[q][key]) != 1 for q in identifiers for key in keys):
        raise ValueError("Domain evaluation requires one SCOP classification per query")
    counts = {key: Counter(annotations[q][key][0] for q in identifiers) for key in keys}
    boundaries = resolve_negative_boundaries(negative_boundary)
    population = {}
    for query in identifiers:
        fam, sf, fold = (counts[key][annotations[query][key][0]] for key in keys)
        hierarchy_counts = {"family": fam, "superfamily": sf, "fold": fold}
        positives = {"family": fam - 1, "superfamily": sf - fam, "fold": fold - sf}
        population[query] = {
            level: (positives[level], len(identifiers) - hierarchy_counts[boundaries[level]])
            for level in LEVELS
        }
    return population


def _count_evalue(histogram, evalue, category):
    counts = histogram.setdefault(evalue, [0, 0, 0])
    counts[0] += category == "positive"
    counts[1] += category == "negative"
    counts[2] += 1


def _evalue_summary(histogram, queries, positives, possible_hits, cutoffs):
    thresholds = sorted(histogram)
    tp = fp = hits = 0
    errors, recall, hit_fraction = [], [], []
    for threshold in thresholds:
        true, false, reported = histogram[threshold]
        tp += true
        fp += false
        hits += reported
        errors.append(fp / queries)
        recall.append(tp / positives)
        hit_fraction.append(hits / possible_hits)

    points = []
    for cutoff in cutoffs:
        count = bisect_right(thresholds, cutoff)
        points.append({
            "evalue_cutoff": cutoff,
            "observed_errors_per_query": errors[count - 1] if count else 0.0,
            "recovered_positive_pair_fraction": recall[count - 1] if count else 0.0,
            "reported_hit_fraction": hit_fraction[count - 1] if count else 0.0,
        })
    return {
        "predicted_errors_per_query": thresholds,
        "observed_errors_per_query": errors,
        "recovered_positive_pair_fraction": recall,
        "reported_hit_fraction": hit_fraction,
        "operating_points": points,
    }


_EMPTY_LABELS = {"family": frozenset(), "superfamily": frozenset()}
_NEUTRAL_FOLDS = re.compile(r"^b\.(67|68|69|70).*")


def _multidomain_cohort(query_families, target_families, result_limit, negative_boundary):
    if result_limit < 1 or set(query_families) & set(target_families):
        raise ValueError("Require a positive result limit and disjoint query/target IDs")
    if any(not families for families in query_families.values()):
        raise ValueError("Every query must have a SCOP family annotation")
    if negative_boundary not in ("fold", "superfamily"):
        raise ValueError("Multidomain negative boundary must be fold or superfamily")

    def labels(cohort):
        return {name: {"family": set(families),
                       "superfamily": {family.rsplit(".", 1)[0] for family in families}}
                for name, families in cohort.items()}

    queries, targets = labels(query_families), labels(target_families)
    levels = ("family",) if negative_boundary == "fold" else ("family", "superfamily")
    positives = {}
    for level in levels:
        sizes = Counter(label for target in targets.values() for label in target[level])
        positives[level] = {
            name: min(result_limit, sum(sizes[label] for label in query[level]))
            for name, query in queries.items()
        }
    return queries, targets, positives


def _multidomain_relation(query, target, level, boundary):
    if query[level] & target[level]:
        return "positive"
    if boundary == "superfamily":
        neutral = bool(query["superfamily"] & target["superfamily"])
    else:
        neutral = any(
            q.rsplit(".", 2)[0] == t.rsplit(".", 2)[0]
            or (_NEUTRAL_FOLDS.fullmatch(q) and _NEUTRAL_FOLDS.fullmatch(t))
            or q.startswith("e.") or t.startswith("e.")
            for q in query["family"] for t in target["family"]
        )
    return "neutral" if neutral else "negative"


def _multidomain_hits(records, queries, limit, rows, seen, discarded, *, calibration=False):
    """Apply the native row cap before deduplicating targets per query."""
    for record in records:
        query, target = record[:2]
        if query not in queries:
            raise ValueError(f"Query outside benchmark cohort: {query}")
        if calibration and (not math.isfinite(record[2]) or record[2] < 0):
            raise ValueError("E-values must be finite and nonnegative")
        rows[query] = rows.get(query, 0) + 1
        if rows[query] > limit:
            discarded["over_limit"] += 1
            continue
        retained = seen.setdefault(query, set())
        if target in retained:
            discarded["duplicate"] += 1
            continue
        retained.add(target)
        yield record


def evalue_retrieval_curve(identifiers, records, annotations, level,
                           negative_boundary=None,
                           cutoffs=(0.001, 0.01, 0.1, 1, 10),
                           error_unit="target-rows"):
    """Calibration and recovery under shared E-value cutoffs.

    The positive denominator counts every eligible pair in the full cohort.
    Equal printed E-values form one threshold block. The search reader supplies
    hits between distinct domains.
    """
    identifiers = list(identifiers)
    identifier_set = set(identifiers)
    boundaries = resolve_negative_boundaries(negative_boundary)
    population = _domain_population(identifier_set, annotations, boundaries)
    total_positive = sum(population[query][level][0] for query in identifiers)
    if not total_positive:
        raise ValueError("E-value retrieval requires eligible positive pairs")
    if error_unit != "target-rows":
        raise ValueError("E-value calibration counts false target hits")
    histogram = {}
    identifier_index = {name: i for i, name in enumerate(identifiers)}
    query_count = len(identifiers)
    seen_pairs = bytearray((query_count ** 2 + 7) // 8)
    rows = 0

    for query, target, evalue in records:
        if query not in identifier_set or target not in identifier_set or query == target:
            raise ValueError("E-value hit outside cohort or self pair")
        if not math.isfinite(evalue) or evalue < 0:
            raise ValueError("E-values must be finite and nonnegative")
        pair_index = (
            identifier_index[query] * len(identifiers) + identifier_index[target]
        )
        byte_index, bit_index = divmod(pair_index, 8)
        bit = 1 << bit_index
        if seen_pairs[byte_index] & bit:
            raise ValueError(f"Duplicate query-target E-value hit: {query}, {target}")
        seen_pairs[byte_index] |= bit
        category = relation(annotations[query], annotations[target], level, boundaries)
        _count_evalue(histogram, evalue, category)
        rows += 1


    return {
        "kind": "evalue",
        "level": level,
        "queries": len(identifiers),
        "negative_boundary": boundaries[level],
        "fp_policy": fp_policy_name(boundaries),
        **_evalue_summary(histogram, query_count, total_positive,
                          query_count * (query_count - 1), cutoffs),
        "reported_nonself_pairs": rows,
        "total_directed_positive_pairs": total_positive,
        "ties": "whole reported E-value blocks",
        "duplicate_check": "exact directed-pair bitset; input order independent",
        "error_unit": error_unit,
        "coverage": "reported-directed-hits; full-cohort positive denominator",
    }


def multidomain_recovery(
    query_families,
    target_families,
    records,
    result_limit=4000,
    negative_boundary="fold",
):
    """Evaluate the MMseqs2 multidomain cohort in native hit order.

    The historical ``fold`` mode exactly preserves the supplied family-level
    reversed-background ROC1 protocol.  ``superfamily`` retains its native-row
    cap and denominators, but reports both Family and Superfamily retrieval
    with a different-superfamily first false positive for both views.
    """
    queries, targets, positives = _multidomain_cohort(
        query_families, target_families, result_limit, negative_boundary
    )
    levels = tuple(positives)
    state = {query: {level: Counter() for level in levels} for query in queries}
    rows, seen, discarded = {}, {}, Counter()
    for query, target in _multidomain_hits(records, queries, result_limit, rows, seen, discarded):
        target_labels = targets.get(target, _EMPTY_LABELS)
        for level in levels:
            counts = state[query][level]
            if not counts["negative"]:
                category = _multidomain_relation(queries[query], target_labels, level, negative_boundary)
                counts[category] += 1

    result_levels = {}
    for level in levels:
        per_query = {}
        for query, labels in queries.items():
            denominator = positives[level][query]
            counts = state[query][level]
            if counts["positive"] > denominator:
                raise ValueError(f"Recovered TP count exceeds {level} denominator: {query}")
            per_query[query] = {
                "sensitivity": counts["positive"] / denominator if denominator else None,
                "positives": denominator,
                "recovered": counts["positive"],
                "first_fp_reached": bool(counts["negative"]),
                "neutral_before_first_fp": counts["neutral"],
                "retained_unique_hits": len(seen.get(query, ())),
                "input_hits": rows.get(query, 0),
                "scop_families": sorted(labels["family"]),
                "scop_superfamilies": sorted(labels["superfamily"]),
            }
        values = [
            item["sensitivity"]
            for item in per_query.values()
            if item["sensitivity"] is not None
        ]
        result_levels[level] = {
            "auc": sum(values) / len(values) if values else None,
            "eligible_queries": len(values),
            "negative_boundary": negative_boundary,
            "per_query": per_query,
        }
    protocol = (
        "MMseqs2 multidomain reversed-background ROC1"
        if negative_boundary == "fold"
        else "MMseqs2 multidomain superfamily-FP first-hit recovery"
    )
    return {
        "schema_version": 2,
        "protocol": protocol,
        "queries": len(state),
        "result_limit": result_limit,
        "fp_policy": (
            "legacy-reversed-background" if negative_boundary == "fold"
            else "superfamily-fp"
        ),
        "negative_boundaries": {level: negative_boundary for level in levels},
        "ties": "native order; truncate before first-occurrence target deduplication",
        "auc_definition": "exact mean per-query recovery; no threshold plot approximation",
        "denominator": (
            "sum of annotated-target label sizes for each query label set at the "
            "reported hierarchy, capped at the native result limit"
        ),
        "no_hit_queries": len(queries) - len(seen),
        "levels": result_levels,
    }


def multidomain_evalue_curve(
    query_families,
    target_families,
    records,
    target_count,
    result_limit=4000,
    cutoffs=(0.001, 0.01, 0.1, 1, 10),
    level="family",
    negative_boundary="fold",
):
    """E-value calibration under the large multidomain family protocol.

    The native first-``result_limit`` rows are selected before target
    deduplication, exactly as in :func:`multidomain_recovery`. Reported
    E-values are then pooled into complete equal-value blocks. Unannotated
    targets are reversed-sequence false positives; same-fold, b.67--b.70 and
    class-e comparisons remain neutral.
    """
    queries, targets, positives = _multidomain_cohort(
        query_families, target_families, result_limit, negative_boundary
    )
    if target_count < len(targets) or level not in positives:
        raise ValueError("Invalid multidomain target count or hierarchy level")
    total_positive = sum(positives[level].values())
    if not total_positive:
        raise ValueError("Multidomain E-values require annotated positives")

    histogram = {}
    rows, seen, discarded = {}, {}, Counter()
    for query, target, evalue in _multidomain_hits(
        records, queries, result_limit, rows, seen, discarded, calibration=True
    ):
        category = _multidomain_relation(
            queries[query], targets.get(target, _EMPTY_LABELS), level, negative_boundary
        )
        _count_evalue(histogram, evalue, category)
    if not histogram:
        raise ValueError("Multidomain E-value curve has no retained hits")


    return {
        "schema_version": 1,
        "kind": "evalue",
        "level": level,
        "negative_boundary": negative_boundary,
        "fp_policy": (
            "legacy-reversed-background" if negative_boundary == "fold"
            else "superfamily-fp"
        ),
        "protocol": (
            "MMseqs2 multidomain reversed-background E-value calibration"
            if negative_boundary == "fold"
            else "MMseqs2 multidomain superfamily-FP E-value calibration"
        ),
        "queries": len(query_families),
        "targets": target_count,
        "result_limit": result_limit,
        **_evalue_summary(histogram, len(queries), total_positive,
                          len(queries) * target_count, cutoffs),
        "total_positive_denominator": total_positive,
        "retained_unique_hits": sum(map(len, seen.values())),
        "queries_with_hits": len(seen),
        "duplicate_targets_before_result_limit": discarded["duplicate"],
        "ignored_rows_after_result_limit": discarded["over_limit"],
        "fp_definition": (
            "unannotated reversed target or annotated different-superfamily target"
            if negative_boundary == "superfamily"
            else "unannotated reversed target or annotated different-fold target; "
                 "same-fold, b.67-b.70 and class-e comparisons are neutral"
        ),
        "denominator": (
            f"sum of annotated-target {level} sizes for each query {level} set, "
            "capped at the native result limit"
        ),
        "coverage": "first native-ranked rows, then first-occurrence target deduplication",
        "ties": "whole reported E-value blocks",
    }


def main(argv=None):
    """Evaluate alphabet score shards or native search hits on one SCOP cohort."""
    from lightning.pytorch.cli import LightningArgumentParser
    from scripts.io import read_fasta

    parser = LightningArgumentParser(description=__doc__)
    for name in ('mapping', 'splits', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--subset', choices=('all', 'test', 'validation', 'tuning'), default='all')
    parser.add_argument('--outer-fold', type=int, choices=range(4))
    parser.add_argument('--levels', nargs='+', choices=LEVELS, default=list(LEVELS))
    parser.add_argument('--fp-policy', choices=tuple(FP_POLICIES), default='superfamily-fp')
    parser.add_argument('--query-length-fasta', type=Path,
                        help='Alphabet ranking: score / sqrt(query length); never applied to search E-values')
    parser.add_argument('--weighting-reference-cohort', type=Path,
                        help='Full cohort for CV alphabet PR weights')
    parser.add_argument('--search-specification', type=Path,
                        help='Native search fields, ranking, cohort and input checksums')
    parser.add_argument('--regroup-search-by-query', action='store_true')
    args = parser.parse_args(argv)
    splits, mapping = read_json(args.splits), read_json(args.mapping)
    domain_partitions = splits.get('runs', {}).get(str(args.outer_fold), {}).get('domain_partitions')
    identifiers = (sorted(domain_partitions[args.subset]) if domain_partitions is not None
                   else partition_ids(splits, args.subset, args.outer_fold))
    raw = mapping.get('annotations', mapping.get('chains'))
    annotations = {key: {field: row[field] if isinstance(row[field], list) else [row[field]]
                         for field in ('families', 'superfamilies', 'folds')}
                   for key, row in raw.items()}
    metadata = {'split_id': splits['split_id'], 'outer_fold': args.outer_fold, 'subset': args.subset}
    inputs = {path: sha256_file(path) for path in [args.mapping, args.splits, *args.scores]}

    if args.search_specification:
        from scripts.sparse_scop import evaluate_all_sparse

        if args.outer_fold is not None or args.query_length_fasta or args.weighting_reference_cohort:
            parser.error('Search uses the native ranking on the full query cohort, without alphabet normalization')
        specification = read_json(args.search_specification)
        signature = specification['scoring_specification']
        if (specification['split_id'] != splits['split_id']
                or set(signature['cohort']) != set(identifiers)
                or specification['annotations_sha256'] != inputs[args.mapping]):
            raise ValueError('Search cohort or annotations differ from the specification')
        base = Path.cwd() if specification.get('result_path_base') == 'repository' else args.search_specification.parent
        expected = {(base / row['path']).resolve(): row['sha256'] for row in specification['results']}
        if len(args.scores) != len(expected) or {p.resolve() for p in args.scores} != set(expected):
            raise ValueError('Search files differ from the specification')
        for path in args.scores:
            if inputs[path] != expected[path.resolve()]:
                raise ValueError(f'Search checksum mismatch: {path}')
        fields, ranking = specification['fields'], specification['ranking']
        if ranking['direction'] not in ('higher', 'lower'):
            raise ValueError('Declare whether higher or lower scores rank first')
        q, t, s = (fields.index(field) for field in ('query', 'target', ranking['field']))
        cohort = set(identifiers)
        self_hits = 0

        def records():
            nonlocal self_hits
            for path in args.scores:
                with path.open() as handle:
                    for row in csv.reader(handle, delimiter='\t'):
                        if len(row) != len(fields) or row[q] not in cohort or row[t] not in cohort:
                            raise ValueError('Malformed search row or identifier outside cohort')
                        if row[q] == row[t]:
                            self_hits += 1
                            continue
                        score = transform_search_value(row[s], ranking)
                        yield row[q], row[t], score if ranking['direction'] == 'higher' else -score

        rows = records()
        if args.regroup_search_by_query:
            rows = group_search_records_by_query(rows, identifiers)
        results = evaluate_all_sparse(identifiers, rows, annotations, args.levels,
                                      [args.fp_policy])[args.fp_policy]
        metadata.update(engine=specification['engine'], ranking=ranking,
                        scoring_specification=signature, excluded_self_hits=self_hits)
    else:
        from scripts.dense_scop import evaluate_all, read_score_matrix

        receipts = [read_json(str(path) + '.json') for path in args.scores]
        reference = receipts[0]
        keys = ('cohort', 'split_id', 'outer_fold', 'subset', 'matrix_sha256',
                'sequences_sha256', 'gap_open', 'gap_extend', 'score_scale',
                'checkpoint_id', 'conversion_context', 'shards', 'off_diagonal_bonus',
                'mask_sha256', 'training_partition', 'training_splits_sha256',
                'batch_policy', 'adjacent_opposing_gaps', 'score_precision',
                'ranking', 'encoder_id', 'embedding_manifest_sha256', 'mapping_sha256',
                'grid_id', 'distance_scale')
        for path, receipt in zip(args.scores, receipts):
            if receipt['scores_sha256'] != inputs[path]:
                raise ValueError(f'Score checksum mismatch: {path}')
            if any(receipt.get(key) != reference.get(key) for key in keys):
                raise ValueError('Score shards have different scoring settings or cohorts')
        if (reference['split_id'], reference['outer_fold'], reference['subset']) != (
                splits['split_id'], args.outer_fold, args.subset):
            raise ValueError('Score shards belong to a different evaluation split')
        if not set(reference['cohort']) <= set(identifiers):
            raise ValueError('Score cohort lies outside the declared split')
        identifiers = reference['cohort']
        scale = reference.get('score_scale', 1)
        scores = read_score_matrix(args.scores, identifiers, scale)
        lengths = None
        if args.query_length_fasta:
            sequences = read_fasta(args.query_length_fasta)
            lengths = {key: len(sequences[key]) for key in identifiers}
        weighting_cohort = (partition_ids(read_json(args.weighting_reference_cohort), 'all')
                            if args.weighting_reference_cohort else None)
        results = evaluate_all(identifiers, scores, annotations, scale, lengths,
                               args.levels, args.fp_policy, weighting_cohort=weighting_cohort)
        signature = {key: reference.get(key) for key in keys}
        signature['query_score_normalization'] = 'sqrt-query-length' if lengths else 'none'
        metadata['scoring_specification'] = signature

    args.output.mkdir(parents=True, exist_ok=False)
    for name, result in results.items():
        write_json(args.output / name, {**result, **metadata})
    write_json(args.output / 'manifest.json', {
        'fp_policy': args.fp_policy, 'levels': args.levels,
        'inputs': {str(path): digest for path, digest in inputs.items()},
        'outputs': {name: sha256_file(args.output / name) for name in results},
    })


if __name__ == "__main__":
    main()
