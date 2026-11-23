# TEA benchmark data

Inputs for the four main paper figures and eight supplementary figures.
Extract `tea_bench_data.tar.gz` into the benchmark repository so this file is
at `data/README.md`, then run `python scripts/render_figures.py --all`.
Zenodo DOI: **to be added after upload**.

| Directory | Contents |
| --- | --- |
| `plot_tables/` | Empirical curves, per-protein/alignment tables, bootstrap summaries, Sankey counts, motifs and selected application pairs. |
| `assets/` | Architecture artwork and structure renders used by the notebooks. |
| `fasta/scope40_2_08/` | Matched SCOPe40 amino-acid and alphabet FASTAs, including full-chain-projected/domain-only TEA and ProstT5, structural 3Di and four TEA CV folds. |
| `metadata/`, `splits/` | SCOP annotations, cohort identities and evaluation partitions. |
| `checkpoints/` | Substitution matrices. |
| `search_results/` | Search settings and timing measurements. |
| `settings/` | Portable search command parameters. |

## Scope

The notebooks render the figures from the supplied search results and tables.
The AFDB application uses **AFDB clusters v3**. AFDB examples use E < 0.01 and
archaeal examples use E < 0.001.

SCOP benchmarks use **SCOPe 2.08**; the separate E-value-fitting cohort is
SCOP40c / SCOPe 1.75. Download the domain PDBs listed in the example tables
to regenerate the supplied structure renders.

The companion search archives contain the full query and target FASTAs and hit
tables for the paper benchmarks. See the benchmark README for download and
extraction instructions.
