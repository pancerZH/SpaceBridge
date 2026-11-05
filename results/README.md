# Results

Compact records of the published results, so that the tables can be checked without the 275 GB of local caches. It holds scores and configurations only: no document text, no embeddings and no query text.

Everything here is written by `scripts/review_tables.py`, from the records in `outputs/review/`. The same script fills the result tables of the README, the report and the design doc, so the documents can't drift from these files.

| File | What it holds |
| --- | --- |
| `retention.csv` | One row per corpus, size and model family: retention, its interval, the strict and direct comparisons, absolute `nDCG@5` of every system, the verdict if it had a line |
| `per_query/<run>.npz` | Per-query `nDCG@5` of each published cell, for each system, in units of 1/10,000, so anyone can recompute any interval or aggregate |
| `timing.csv` | Query time by stage, p50 and p95, for each timed configuration |
| `manifests.jsonl` | Each run's configuration, code commit, dependency-lock hash, model and dataset revisions, and a digest of its exact query list |

**Keys in `per_query/<run>.npz`:** `<dataset>__<index model>__<query model>__K0__s<seed>/<row>/<system>`, where `K0` means the library's default K, row `9h` is the served configuration (BM25 in a pool of 60), row `9` is the same with BM25 off, and the system is `bridge`, `native`, `oracle` (re-index), `native_bm25`, `oracle_bm25`, or `no_bridge`. Queries are in the order of that run's query list: `manifests.jsonl` gives its SHA-256, and `src/spacebridge_bench/datasets.py` says how the list is drawn.

**Run names:** `recompute_*` are the published cells again, keeping per-query scores. `fresh_*` are the [fresh confirmation](../docs/fresh_confirmation_plan.md).

## Repeating them

```bash
uv sync --extra hnsw
bash scripts/review_experiments.sh        # every run; about 10 hours with these caches, longer without them. It stops itself at 12 hours
uv run python scripts/review_recompute.py --compare   # each metric against the record it came from
uv run python scripts/review_tables.py    # results/ and the tables
```

To read an interval by hand:

```python
import numpy as np
z = np.load("results/per_query/fresh_lotte.npz")
print(sorted(z.files)[:6])
print(z["lotte-technology__bge-large-en__bge-base-en__K0__s0/9h/bridge"].mean() / 10_000)
```

**What is not here.** Milestone reports are local by choice. Embeddings and rerank caches are too large, and are described in [`data/README.md`](../data/README.md). Older study records (`outputs/phase1/`) aren't exported, so figures quoted from studies outside these runs, such as the causal analysis on HotpotQA, can't be checked from this folder alone. Each run's manifest says what it ran on, and results made before these runs have no manifest of their own, so the `recompute_*` runs' manifests stand in for them, with the comparison of each metric against the original.
