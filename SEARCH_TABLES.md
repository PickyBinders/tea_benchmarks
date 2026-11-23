# Search table columns

Paths are relative to `data/search_results/`. TSV and M8 files are
tab-delimited and headerless. Coordinates are one-based.

| Directory and file | Columns in order |
| --- | --- |
| `steam_final/current/scop/*/hits.m8` | query, target, raw |
| `scope40_2_08/{mmseqs2,foldseek,prostt5_foldseek,prostt5_foldseek_full_chain}/hits.tsv` | query, target, evalue, bits |
| `scope40_2_08/reseek/hits.tsv` | query, target, pvalue |
| `scope40_2_08/dali/shards/*.tsv` | query, target, Z-score |
| `steam_final/current/af_multidomain/hits.tsv` | query, target, qstart, tstart, cigar, evalue, raw |
| `af_multidomain/{mmseqs2,foldseek,prostt5_foldseek}/hits.tsv` | query, target, qstart, tstart, cigar, evalue, bits |
| `af_multidomain/dali/hits.tsv` | query, target, Z-score |
| `af_multidomain/eba/hits.tsv` | query, target, qstart, tstart, cigar, score |
| `af_multidomain/reseek/hits.tsv` | query, target, qstart, tstart, cigar, pvalue |
| `steam_final/current/mmseqs_multidomain/search/hits.m8` | query, target, evalue, raw |
| `mmseqs_multidomain/prostt5_foldseek/hits.tsv` | query, target, evalue, bits |
| `figure2_iterations/mmmd/mmseqs_matched/one_pass.tsv`, `mmseqs_3iter/three_pass.tsv` | query, target, qstart, tstart, cigar, evalue, raw |
| `afdb_clusters/steam_final/full_hits.tsv` | query, target, evalue, raw, qlen, tlen, alnlen, qstart, qend, tstart, tend, qcov, tcov, cigar |
| `archaeal/{steam_final,paper_rescue_simple}/hits.tsv` | query, target, evalue, raw, qstart, qend, qlen, tstart, tend, tlen, qcov, tcov, cigar |

In the calibration Parquet tables, `query` and `target` refer to the
zero-based `index` column in the matching sequence tables. Fit indices refer
to `fit_sequences.parquet`. Transfer indices refer to `transfer_queries.parquet`
and `transfer_targets.parquet`.
