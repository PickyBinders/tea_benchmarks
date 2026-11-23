"""Fit STEAM's E-value model to the exhaustive SCOP40c scores."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile


def fit(scores_path, labels, effective_targets):
    counts = Counter()
    for batch in ParquetFile(scores_path).iter_batches(
        batch_size=1_000_000, columns=["query", "target", "raw"]
    ):
        query, target, raw = (column.to_numpy() for column in batch.columns)
        scores, frequencies = np.unique(raw[labels[query] != labels[target]], return_counts=True)
        counts.update(dict(zip(scores, frequencies)))

    scores = np.array(sorted(counts, reverse=True), dtype=float)
    false_hits = np.cumsum([counts[score] for score in scores])
    observed = false_hits / len(labels)
    use = (observed >= 0.01) & (observed <= 1000)
    x = scores[use]
    y = np.log10(false_hits[use] / (len(labels) * effective_targets))
    candidates = []
    for breakpoint in np.sort(x)[49:-50]:
        design = np.column_stack((x, np.maximum(0, x - breakpoint), np.ones_like(x)))
        m_low, delta, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        m_high = m_low + delta
        if m_low < 0 and m_high < 0:
            candidates.append((np.mean((y - design @ [m_low, delta, intercept]) ** 2),
                               breakpoint, m_low, m_high, intercept))
    _, breakpoint, m_low, m_high, intercept = min(candidates)
    return dict(loglinear_breakpoint=breakpoint, loglinear_m_low=m_low,
                loglinear_m_high=m_high, loglinear_c=intercept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_data", type=Path)
    args = parser.parse_args()
    data = args.calibration_data
    labels = pd.read_parquet(data / "fit_sequences.parquet").sort_values("index")["superfamily"].to_numpy()
    diversity = dict(line.split("\t") for line in (data / "fit_db.steam-diversity").read_text().splitlines())
    result = fit(data / "fit_exhaustive/scores.parquet", labels, float(diversity["effective_targets"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
