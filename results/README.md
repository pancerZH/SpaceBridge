# Results

Compact records of the results in the generated tables of the README, the report and the design doc: every table between `<!-- table:NAME -->` markers. They hold scores, configurations and digests only: no document text, no embeddings and no query text.

## Check them from a public checkout

```bash
uv sync
uv run python scripts/verify_results.py
```

This needs nothing but this folder: no caches, no models, no GPU, about 15 seconds. It checks:

1. Every file matches `SHA256SUMS`.
2. Each run holds exactly its planned cells, scored on one query list whose digest matches its manifest.
3. Every field of every row of `retention.csv`:
   - point estimates, absolute scores, differences, seed spread and the worst pair are recomputed exactly from the cell means;
   - every interval bound, including the difference intervals, is recomputed from the per-query scores, to within 0.001;
   - counts, plans and names match exactly;
   - each verdict must follow from its row's numbers and line.
4. The HNSW rows' points, intervals and verdicts, from their own records and per-query scores.
5. Every generated table in the three documents is regenerated from these files and must match, and none may be empty.

The other tables, of timings, setup figures, CPU precision and the Stage-1 study, come from records here that are checksummed, and are regenerated from them. Nothing rawer than those records is recomputed.

`uv run python scripts/review_tables.py` rewrites the tables from these files.

## The files

| File | What it holds |
| --- | --- |
| `retention.csv` | One row per run, row and model family, at full precision. Row `9h` is the served configuration (BM25 in a pool of 60), and row `9` has BM25 off. Columns: retention and its interval, the strict and direct comparisons, absolute `nDCG@5` of every system, the verdict where a line applies, and any planned cells missing (none are) |
| `cells.csv` | Each cell's mean `nDCG@5` by row and system, at full precision |
| `per_query/<run>.npz` | Per-query `nDCG@5` of every cell and system, in units of 1/10,000, so anyone can recompute any interval or aggregate. `fresh_msmarco_hnsw.npz` holds the HNSW check's |
| `fresh_msmarco_hnsw.jsonl`, `fresh_msmarco_hnsw.config.json` | The fresh HNSW check's records, and the configuration file its run wrote: dataset, corpus size, query count, index model, K, M, efConstruction and the BM25 setting. That run predates the index, code and query digests, which later runs record, so this file has none of them. The queries are the fresh MS MARCO run's, whose digest is in `manifests.jsonl` and `datasets.csv` |
| `precision_cpu.jsonl` | CPU against GPU reranking on the same candidates |
| `stage1.jsonl` | The Stage-1 selector study at 128-token anchor documents |
| `timing.csv` | Query time by stage, p50 and p95, for each timed configuration |
| `setup.csv` | Encoding rates, mount times and index builds, for the break-even estimate, each with its source |
| `datasets.csv` | Per run: documents, queries, judgments, positive and distractor documents and their share, self-matches, and SHA-256 digests of the queries, the corpus (IDs and text, in row order) and the judgments |
| `manifests.jsonl` | Each run's configuration, planned cells, code commit, dependency-lock hash, and model and dataset revisions |
| `SHA256SUMS` | A checksum of every file here but this page |

**Keys in `per_query/<run>.npz`:** `<dataset>__<index model>__<query model>__K0__s<seed>/<row>/<system>`. `K0` means the library's default K. The system is one of:

- `bridge`;
- `native`;
- `oracle` (the re-index);
- `native_bm25` and `oracle_bm25`;
- `no_bridge`.

Queries are in the order of the run's query list, whose digest is in `manifests.jsonl` and `datasets.csv`.

**Run names:**

- `recompute_*` are the published cells again, keeping per-query scores. They match the original records to six decimal places.
- `fresh_*` are the [fresh confirmation](../docs/fresh_confirmation_plan.md).

## How they were made

`scripts/review_export.py` writes this folder from the local records in `outputs/`. They came from `scripts/review_experiments.sh` and `scripts/review_followup.sh`, which take about 11 hours with the local caches, and longer without them.

## What this folder doesn't cover

Some numbers in the documents come from earlier studies, written in prose rather than in a generated table: the HotpotQA cause analysis, the reranker and translation comparisons, the mount and solver timings, the score's calibration. Those records stay on the reference machine and aren't exported. Their scripts saved means rather than per-query scores, and no configuration identity. So those figures are reported measurements, and they can't be checked from this folder.

`data/README.md` describes the local caches.
