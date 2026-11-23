# File guide

| File or group | Purpose |
| --- | --- |
| `notebooks/alphabet.ipynb` | Architecture, Sankeys, alphabet benchmarks and conversion timing. |
| `notebooks/search.ipynb` | Search sensitivity, calibration, alignment quality and timing. |
| `notebooks/confidence.ipynb` | Entropy, TEA likelihood and pLDDT. |
| `notebooks/examples.ipynb` | AFDB v3 and archaeal connections and examples. |
| `notebooks/supplementary_figures.ipynb` | Eight supplements in labelled sections. |
| `scripts/render_figures.py` | Execute notebooks with inline figures. |
| `scripts/steam_search.py`, `native_search.py` | Print or run benchmark search commands. |
| `scripts/alignment.py`, `parasail.py` | Exhaustive local alignment and its backend. |
| `scripts/scop.py`, `dense_scop.py`, `sparse_scop.py` | SCOP evaluation for score matrices and search hits. |
| `scripts/multidomain.py`, `alignment_identity.py` | Retrieval and alignment-quality metrics. |
| `scripts/matrix.py` | Substitution matrix construction. |
| `scripts/fit_markov.py`, `fit_evalues.py` | Fit the first-order TEA background and E-value calibration used by STEAM. |
| `scripts/io.py`, `__init__.py` | File-format helpers and package definition. |
| `data/` | Figure inputs, search FASTAs and hits from the Zenodo archives. |
| `SEARCH_TABLES.md` | Columns in the exported hit tables. |
| `requirements.txt`, `LICENSE`, `.gitignore` | Dependencies, licence and generated-data exclusions. |
