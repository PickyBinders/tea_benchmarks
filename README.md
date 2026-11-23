# TEA benchmarks

Code and notebooks for the TEA paper.

## Reproduce the figures

1. Download `tea_bench_data.tar.gz` from [Zenodo](https://doi.org/10.5281/zenodo.17725634) and extract
   it in this repository. This creates `data/`; its README describes the inputs.
2. Use Python 3.12, install the Lato font, and install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Run the notebooks from `notebooks/`, or execute all of them:

   ```bash
   python scripts/render_figures.py --all
   ```

The notebooks display and save figures inline. Each main notebook is
independent. All eight supplements are in one
notebook, including the combined alphabet/search superfamily-and-fold comparison.
The application figures use AFDB clusters v3.

## Search data

Extract the relevant Zenodo archives at the repository root with
`tar --zstd -xf ARCHIVE.tar.zst`. Each adds files under `data/`.

| Archive | Contents |
| --- | --- |
| `scop.tar.zst` | SCOPe 2.08 hit tables for STEAM, MMseqs2, Foldseek, ProstT5–Foldseek, Reseek and DALI. The SCOP FASTAs are in the figure bundle. |
| `af_multidomain.tar.zst` | Query and target FASTAs, AlphaFold DB model IDs, and hit tables for all seven methods. |
| `mmseqs_multidomain.tar.zst` | Query and target FASTAs, target labels and hits from STEAM, MMseqs2 (one pass and three iterations) and ProstT5–Foldseek. |
| `afdb_clusters_v3.tar.zst` | AFDB v3 singleton and representative FASTAs, STEAM hits and annotation matches. |
| `archaea.tar.zst` | Archaeal query and target FASTAs, annotations and STEAM hits. |
| `calibration.tar.zst` | Markov fitting cohort, SCOP40c FASTAs, exhaustive calibration and transfer scores, and MinHash effective database sizes. |

[Search table columns](SEARCH_TABLES.md) describes the hit formats. SCOP
coordinates can be downloaded from **SCOPe 2.08**. The figure bundle includes
the SCOP EBA curves. Recomputing them requires the domain embeddings and a
new EBA run.

### Download AF multidomain structures

`data/metadata/af_multidomain/afdb_ids.tsv` lists the 100 query and
56,573 target models. Download the original v4 mmCIF files from the
versioned AlphaFold DB bucket with an authenticated Google Cloud CLI.
[AlphaFold DB archive instructions](https://github.com/google-deepmind/alphafold/blob/main/afdb/README.md#file-manifests)

```bash
gcloud auth login
mkdir -p data/structures/af_multidomain
tail -n +2 data/metadata/af_multidomain/afdb_ids.tsv | cut -f2 | sort -u |
    sed 's|^|gs://public-datasets-deepmind-alphafold-v4/|' |
    gcloud storage cp --read-paths-from-stdin data/structures/af_multidomain/
```

## Rerun benchmark calculations

```bash
python -m scripts.alignment --help
python -m scripts.scop --help
python -m scripts.matrix --help
```

These provide exhaustive alphabet alignment, SCOP evaluation and substitution
matrix calculations. SCOP evaluation writes sensitivity and precision–recall curves in one pass.
The default FP policy is `superfamily-fp`. Evaluate the four CV folds separately.
Use `--search-specification` for native search rankings. Specify a fresh output
directory for each parallel alignment run.

Search the supplied paired FASTAs with an installed STEAM:

```bash
python scripts/steam_search.py --dataset scop --binary /path/to/steam \
    --threads 64 --workdir /path/to/new-scratch-directory \
    --output /path/to/new-results
```

This prints the commands. Add `--execute` to run them. The available presets
are `scop`, `scop_domain`, `afmd`, `mmmd`, `afdb`, and `archaea`.
They use the published substitution matrix, AA weight 3, TEA scale 2, gaps
24/2, and calibrated E-values.
`scripts/native_search.py` similarly prints captured baseline search recipes.

## Refit the STEAM background and E-values

With `calibration.tar.zst` extracted:

```bash
python -m scripts.fit_markov data/markov/chains.tea.fasta \
    data/markov/fitting_chain_ids.txt seed_markov.txt
python -m scripts.fit_evalues data/search_results/steam_final/simple_matrix/calibration_raw
```

The Markov fitter counts letters and adjacent pairs within each of 9,512
SCOPe chains. It adds one pseudocount per letter to the background frequencies
and 20 pseudocounts distributed by those frequencies to each transition row.
The E-value fitter counts every different-superfamily target hit and fits a
continuous two-slope log-linear curve over 0.01–1,000 observed false hits per
query. It uses all fitting queries and the MinHash effective target count.

The published `data/checkpoints/public_export/matrix.json` also includes
the EBA-aligned residue-pair counts used to compute its half-bit scores.

See [README_FILES.md](README_FILES.md) for the short file guide. On sciCORE,
execute notebooks and benchmarks within a Slurm allocation.
