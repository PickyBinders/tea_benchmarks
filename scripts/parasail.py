"""Reusable Parasail local alignment for benchmark scoring and traceback."""

import math
import string

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import read_alignment_from_cigar


def cigar_trace(query_length, target_length, query_start, target_start, cigar):
    """Return query/target residue indices for a native protein CIGAR."""
    if query_start < 1 or target_start < 1:
        raise ValueError("Alignment starts are 1-based")
    prefix = f"{query_start - 1}S" if query_start > 1 else ""
    try:
        alignment = read_alignment_from_cigar(
            prefix + cigar,
            target_start - 1,
            ProteinSequence("A" * target_length),
            ProteinSequence("A" * query_length),
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid CIGAR: {cigar}") from error
    trace = alignment.trace[:, ::-1]
    if np.any(trace[:, 0] >= query_length) or np.any(trace[:, 1] >= target_length):
        raise ValueError("CIGAR extends beyond the sequence")
    return trace


class ParasailAligner:
    """Cached 32-bit Smith-Waterman alignments for an integer matrix."""

    def __init__(self, sequences, scores, gap_open, gap_extend):
        import parasail

        scores = np.asarray(scores, dtype=float)
        if (scores.ndim != 2 or scores.shape[0] != scores.shape[1]
                or not np.isfinite(scores).all()
                or not np.equal(scores, np.rint(scores)).all()):
            raise ValueError("Parasail requires a finite square integer matrix")
        if (not math.isfinite(gap_open + gap_extend)
                or int(gap_open) != gap_open or int(gap_extend) != gap_extend
                or gap_open < gap_extend or gap_extend < 0):
            raise ValueError("Parasail requires integer gap open >= gap extend >= 0")

        symbols = string.ascii_letters + string.digits
        if len(scores) > len(symbols):
            raise ValueError("Too many alphabet states for the internal encoding")

        self._parasail = parasail
        self._gaps = int(gap_open), int(gap_extend)
        self._matrix = parasail.matrix_create(
            symbols[: len(scores)], 0, 0, case_sensitive=True
        )
        for i, row in enumerate(scores):
            for j, value in enumerate(row):
                self._matrix[i, j] = int(value)

        self._sequences = {
            name: "".join(symbols[int(state)] for state in sequence)
            for name, sequence in sequences.items()
        }
        longest = max(map(len, self._sequences.values()), default=0)
        if np.abs(scores).max(initial=0) * max(longest, 1) + sum(self._gaps) >= 2**30:
            raise ValueError("Scoring range exceeds the 32-bit safety bound")
        self._profiles = {
            name: parasail.profile_create_32(sequence, self._matrix)
            for name, sequence in self._sequences.items()
            if sequence
        }

    def score(self, query, target):
        """Return the optimal local-alignment score."""
        if not self._sequences[query] or not self._sequences[target]:
            return 0.0
        result = self._parasail.sw_striped_profile_32(
            self._profiles[query], self._sequences[target], *self._gaps
        )
        if result.saturated:
            raise OverflowError("Parasail score saturated")
        return float(result.score)

    def align(self, query, target):
        """Return score, CIGAR, starts, spans, and matched residue indices."""
        if not self._sequences[query] or not self._sequences[target]:
            return None

        expected = self.score(query, target)
        result = self._parasail.sw_trace_striped_profile_32(
            self._profiles[query], self._sequences[target], *self._gaps
        )
        kernel = "striped"
        if result.saturated or result.score != expected:
            result = self._parasail.sw_trace_scan_32(
                self._sequences[query],
                self._sequences[target],
                *self._gaps,
                self._matrix,
            )
            kernel = "scan"
        if result.saturated or result.score != expected:
            raise RuntimeError("Parasail traceback does not match its optimal score")
        if result.score <= 0:
            return None

        query_start = result.cigar.beg_query + 1
        target_start = result.cigar.beg_ref + 1
        cigar = result.cigar.decode.decode()
        trace = cigar_trace(
            len(self._sequences[query]),
            len(self._sequences[target]),
            query_start,
            target_start,
            cigar,
        )
        pairs = trace[np.all(trace >= 0, axis=1)]
        return {
            "score": float(result.score),
            "query_start": query_start,
            "query_end": int(trace[:, 0].max()) + 1,
            "target_start": target_start,
            "target_end": int(trace[:, 1].max()) + 1,
            "cigar": cigar,
            "trace": trace,
            "pairs": pairs,
            "kernel": kernel,
        }
