"""Replay a captured native search against explicitly supplied native databases."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--query-db", type=Path, required=True)
    parser.add_argument("--target-db", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    recipes = json.loads((ROOT / "data/settings/native_searches.json").read_text())["recipes"]
    if args.recipe not in recipes:
        parser.error("choose one of: " + ", ".join(sorted(recipes)))
    flags = recipes[args.recipe]["flags"].copy()
    matrix = ROOT / "data/checkpoints/public_export/matrix.out"
    model = ROOT / "data/search_results/steam_final/parameters/seed_markov.txt"
    flags = [value.replace("{MATRIX}", str(matrix)).replace("{SEED_MODEL}", str(model)) for value in flags]
    if "--threads" in flags:
        flags[flags.index("--threads") + 1] = str(args.threads)
    else:
        flags.extend(["--threads", str(args.threads)])
    args.output = args.output.resolve()
    args.workdir = args.workdir.resolve()
    command = [args.binary, "search", str(args.query_db.resolve()), str(args.target_db.resolve()),
               str(args.output / "alignments"), str(args.workdir), *flags]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    args.workdir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("STEAM_")}
    if recipes[args.recipe]["engine"] == "steam":
        environment["STEAM_BUCKET_COVERAGE_PCT"] = "10"
    with (args.output / "search.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
    print(f"Search time: {time.perf_counter() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
