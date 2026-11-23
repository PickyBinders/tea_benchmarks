"""Print or run the frozen simple-matrix STEAM recipe on the current FASTAs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/search_results/steam_final"
PRESETS = {
    "scop": ("scope40_2_08", "", ""),
    "scop_domain": ("scope40_2_08", "domain_only.", "domain_only."),
    "afmd": ("af_multidomain", "query.", "target."),
    "mmmd": ("mmseqs_multidomain", "query.", "target."),
    "afdb": ("afdb_clusters", "query.", "target."),
    "archaea": ("archaeal", "paper_rescue_queries.", "paper_targets."),
}


def commands(args):
    corpus, query, target = PRESETS[args.dataset]
    config = json.loads((FINAL / "current/calibrated_configuration.json").read_text())
    fit = json.loads((FINAL / "current/calibration/fit_exhaustive.json").read_text())
    matrix = ROOT / config["score"]["matrix"]
    model = FINAL / "parameters/seed_markov.txt"
    fasta = ROOT / "data/fasta" / corpus
    query_db, target_db = args.workdir / "query", args.workdir / "target"
    native, source_files = [], []
    if query == target:
        target_db = query_db
    for prefix, database in {query: query_db, target: target_db}.items():
        tea = fasta / f"{prefix}tea.fasta.gz"
        aa_prefix = "" if args.dataset == "scop_domain" else prefix
        aa = fasta / f"{aa_prefix}aa.fasta.gz"
        source_files.extend((tea, aa))
        native.append([args.binary, "createdb", str(tea), str(aa), str(database),
                       "--threads", str(args.threads)])
    recipe = json.loads((ROOT / "data/settings/native_searches.json").read_text())["recipes"][args.dataset]
    flags = [value.replace("{MATRIX}", str(matrix)).replace("{SEED_MODEL}", str(model))
             for value in recipe["flags"]]

    def set_flag(option, value):
        if option in flags:
            flags[flags.index(option) + 1] = str(value)
        else:
            flags.extend([option, str(value)])

    for option, value in (("--threads", args.threads),
                          ("--loglinear-m", fit["loglinear_m_low"]),
                          ("--loglinear-m-high", fit["loglinear_m_high"]),
                          ("--loglinear-breakpoint", fit["loglinear_breakpoint"]),
                          ("--loglinear-c", fit["loglinear_c"]), ("--p-fp", fit["p_fp"])):
        set_flag(option, value)
    if args.evalue is not None:
        set_flag("-e", args.evalue)
    if args.max_accept is not None:
        set_flag("--max-accept", args.max_accept)
        set_flag("--max-rejected", args.max_accept)
    # Record backtraces for CIGAR export.
    if "-a" not in flags:
        flags.append("-a")
    search = [args.binary, "search", str(query_db), str(target_db),
              str(args.workdir / "alignments"), str(args.workdir / "tmp"), *flags]
    fields = "query,target,evalue,raw,qlen,tlen,alnlen,qstart,qend,tstart,tend,qcov,tcov,cigar"
    native.extend([search, [args.binary, "convertalis", str(query_db), str(target_db),
                           str(args.workdir / "alignments"), str(args.output / "hits.tsv"),
                           "--format-output", fields, "--threads", str(args.threads)]])
    return native, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PRESETS, required=True)
    parser.add_argument("--binary", default="steam")
    parser.add_argument("--workdir", type=Path, required=True, help="new directory on working scratch")
    parser.add_argument("--output", type=Path, required=True, help="new durable result directory")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--evalue", help="override the capture cutoff, not calibration")
    parser.add_argument("--max-accept", type=int,
                        help="override the frozen cohort-specific acceptance/rejection cap")
    parser.add_argument("--execute", action="store_true", help="otherwise only print commands")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.workdir, args.output = args.workdir.resolve(), args.output.resolve()
    native, source_files = commands(args)
    for command in native:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    missing = [str(path.relative_to(ROOT)) for path in source_files if not path.is_file()]
    if missing:
        parser.error("Missing search FASTAs (see data/README.md): " + ", ".join(missing))
    args.workdir.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("STEAM_")}
    environment["STEAM_BUCKET_COVERAGE_PCT"] = "10"
    for i, command in enumerate(native):
        started = time.perf_counter()
        with (args.output / f"command_{i}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        print(f"{command[1]}: {time.perf_counter() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
