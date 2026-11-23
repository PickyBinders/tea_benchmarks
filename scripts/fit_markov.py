"""Fit STEAM's first-order background model from converted TEA chains."""

import argparse
from pathlib import Path

import numpy as np

from scripts.io import read_fasta


def fit_markov(sequences, alphabet, span):
    index = {letter: i for i, letter in enumerate(alphabet)}
    counts = np.ones(len(alphabet))
    pairs = np.zeros((len(alphabet), len(alphabet)))
    windows = []
    for sequence in sequences:
        codes = np.array([index[letter] for letter in sequence.upper()], dtype=int)
        counts += np.bincount(codes, minlength=len(alphabet))
        np.add.at(pairs, (codes[:-1], codes[1:]), 1)
        windows.append(max(0, len(codes) - span + 1))
    background = counts / counts.sum()
    transitions = pairs + 20 * background
    transitions /= transitions.sum(axis=1, keepdims=True)
    return background, transitions, np.mean(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("chains", type=Path, help="Fitting chain IDs, one per line")
    parser.add_argument("output", type=Path)
    parser.add_argument("--alphabet", default="LGVNYMEFKHTQASRCIPWD")
    parser.add_argument("--span", type=int, default=7, help="Spaced seed span")
    args = parser.parse_args()
    sequences = read_fasta(args.fasta)
    chains = args.chains.read_text().split()
    if not chains or len(set(chains)) != len(chains) or args.span < 1:
        parser.error("Provide unique fitting chains and a positive seed span")
    background, transitions, mean_windows = fit_markov(
        (sequences[chain] for chain in chains), args.alphabet, args.span
    )
    with args.output.open("x") as handle:
        print("STEAM_SEED_MARKOV_V1", args.alphabet, f"{mean_windows:.17g}", sep="\n", file=handle)
        for row in (background, *transitions):
            print(" ".join(f"{value:.17g}" for value in row), file=handle)


if __name__ == "__main__":
    main()
