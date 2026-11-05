# SpaceBridge

Search a vector index with a different embedding model, without re-embedding the documents.

An embedding model's vectors only make sense next to that same model's vectors, so changing models normally means re-embedding everything. SpaceBridge instead fits a small map from the new model's queries into the old model's space, using a few thousand of your documents. A reranker then orders what the search returns, together with the best matches of BM25, a keyword search over the same documents.

**It gives you compatibility right away, not the new model's quality.** To get a better model's gain, you still have to re-index. See [how to choose](#use-it-or-re-index).

## Try it

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or later. Tested on Linux only.

```bash
git clone https://github.com/pancerZH/SpaceBridge.git
cd SpaceBridge
uv sync --extra models --extra banks
uv run python examples/quickstart.py
```

This builds a small index (SciFact, 5,183 science abstracts) with one model, then searches it with another. The first run downloads about 2.5 GB of models, on top of about 7 GB for the Python environment. After that it takes about 30 seconds on a GPU and about 3 minutes on a CPU, and it picks the device by itself. It prints a predicted score, a timing breakdown and the top results.

In your own code, once you have your stored vectors and their text:

```python
import spacebridge

index = spacebridge.Index(vectors, texts, titles)      # L2-normalized vectors, and text in the same row order
bundle = spacebridge.freeze(index, old_model)          # once per index; pass None if the old model is gone
bridge = spacebridge.mount(bundle, index, new_model)   # once per new query model

searcher = spacebridge.Searcher(bridge, new_model, spacebridge.ExactBackend(vectors), index, spacebridge.CrossEncoderReranker())
result = searcher.search("your query", k=5)            # .ids, .scores, .texts, .confidence, .pool, .timing_ms
```

`old_model` and `new_model` are encoders with the right prompts for each model. The [usage guide](docs/usage.md) covers what the inputs must be, how to install less, how to restart without rebuilding, HNSW for large indexes, other languages, and what each reported number means. [`examples/deploy.py`](examples/deploy.py) is a full build-then-serve example you can adapt.

## What to expect

Results use three kinds of search over the same corpus, all followed by the same reranker (`bge-reranker-v2-m3`):

- **Bridge:** the new model's queries, mapped, on the old index. This is SpaceBridge, here with BM25.
- **Native:** the old model's own queries on its own index.
- **Re-index:** every document re-embedded with the new model.

**Retention** is the bridge's `nDCG@5` divided by the weaker of native and re-index. `nDCG@5` scores the top 5 results from 0 to 1. So retention of 0.98 means 2% below the weaker alternative. It's a relative figure, not absolute quality. It can exceed 1, since with BM25 the bridge has a source of candidates plain native search doesn't.

**Same family** means `bge-base-en-v1.5` and `bge-large-en-v1.5`, in either direction. **Across families** means `bge-large-en-v1.5` with `Qwen3-Embedding-0.6B` or `harrier-oss-v1-0.6b`. Our lines were 98% and 94% up to 1M documents, and 95% and 90% above.

### On the corpora that shaped the system

These helped choose the design, so reaching a line here shows the chosen system reaches it on these corpora, not that it would elsewhere. "Low" and "high" bound a 95% confidence interval for query sampling, which doesn't cover other corpora or models. 3 seeds, 60 candidates reranked, BM25 providing a quarter of them, exact search. A blank means not measured.

<!-- table:retention_short -->
| Corpus | Documents | Queries | Same family | Low | High | Across families | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- /table:retention_short -->

### On data that chose nothing

The final system, frozen, with [rules fixed before any result existed](docs/fresh_confirmation_plan.md). **Confirmed:** the score and the interval's low end both reach the line. **Point pass:** only the score does. **Miss:** neither.

<!-- table:fresh -->
| Data | Documents | Queries | Models | Retention | Low | High | Line | Verdict | No bridge, retention | Bridge, absolute `nDCG@5` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS MARCO, unused queries |  |  | not run |  |  |  |  | Not run |  |  |
| HotpotQA, unused queries |  |  | not run |  |  |  |  | Not run |  |  |
| LoTTE technology |  |  | not run |  |  |  |  | Not run |  |  |
| FEVER |  |  | not run |  |  |  |  | Not run |  |  |
| Touché 2020 |  |  | not run |  |  |  |  | Not run |  |  |
| MS MARCO, unused queries, HNSW | 8.84M |  | Same family, one direction |  |  |  | 0.95 | Not run |  |  |
| MS MARCO, unused queries, HNSW | 8.84M |  | Across families, one direction |  |  |  | 0.90 | Not run |  |  |
<!-- /table:fresh -->

<!-- value:fresh_reading -->

### Speed

One query, warm, on an idle machine: this leaves out the first query, model loading and index loading. Simultaneous requests weren't measured. The machine: Intel Core i5-12600K using 10 threads, Nvidia RTX 5070 Ti, 96 GB of DDR4 RAM. 60 candidates, `bge-large-en-v1.5` index, `Qwen3-Embedding-0.6B` queries. Above 1M documents, exact search runs on the CPU even in GPU rows, because the vectors don't fit on the GPU.

**Measured** rows timed that exact configuration, and **estimated** rows adjust a measured run, so don't read their p95 as an observed limit. p95 is the time 95% of queries beat, and on few queries it's close to the slowest seen. Short documents are MS MARCO passages, about 95 tokens per query and document pair, and long are TREC-COVID abstracts or Touché arguments, about 320 tokens and more.

<!-- table:speed -->
| Documents | Corpus | Documents' length | Search | CPU p50, s | CPU p95, s | CPU queries | GPU p50, s | GPU p95, s | GPU queries | Tokens per pair | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100K | MS MARCO | Short | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 171.3K | TREC-COVID | Long | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 382.5K | Touché 2020 | Long | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 1M | MS MARCO | Short | HNSW | not run | not run |  | not run | not run |  |  | Measured |
| 2M | MS MARCO | Short | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 4M | MS MARCO | Short | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 5.23M | HotpotQA | Short | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 8.84M | MS MARCO | Short | Exact | not run | not run |  | not run | not run |  |  | Measured |
| 8.84M | MS MARCO | Short | HNSW | not run | not run |  | not run | not run |  |  | Measured |
| 1M | TREC-COVID's rerank time, this size's search | Long | HNSW |  |  |  |  |  |  |  | **Estimated** |
| 8.84M | TREC-COVID's rerank time, this size's search | Long | HNSW |  |  |  |  |  |  |  | **Estimated** |
<!-- /table:speed -->

On a CPU the reranker takes almost all of this: it's roughly 1.91 ms × candidates × tokens per pair. A CPU query of 11 to 38 seconds is usable for occasional queries, not for traffic. An int8 export of the reranker was about 3 times faster on this CPU, with too few queries checked to claim its quality, so it isn't in the library.

## Use it or re-index

**Use it** when a new query model has to work on an existing index now, or when the model that built the index is gone. Then re-index in the background if the last few points justify it.

**Re-index instead** if you want the new model's quality, or if you can't run a reranker.

**Against a re-indexed system with the same BM25 and reranker**, absolute `nDCG@5` (a point is 0.01 of it):

<!-- table:reindex_short -->
| Corpus | Documents | Models | Bridge | Re-index with BM25 | Difference, absolute points | Low | High | Direct retention | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- /table:reindex_short -->

**Compute** depends on what the re-indexed system would do. If it would also rerank, the bridge costs about 1 ms more per query on a GPU, so re-indexing takes a long time to pay back. If it wouldn't rerank, re-indexing pays back within thousands of queries, though at lower quality. The [design doc](spacebridge_design_and_benchmark.md#95-re-index-or-bridge) has the estimates and their assumptions.

## Limits

- **English only, on the model pairs measured.** BM25's tokenizer reads only `[a-z0-9]+`, so it's off for an index not in English, and a query in another language skips it, unless you give it a translator. Other kinds of model pair aren't measured above 171K documents.
- **You state the language.** Nothing detects it: an unmarked Chinese query is treated as English. See the [usage guide](docs/usage.md#languages).
- **BM25 can cost you where relevance isn't about words.** SciDocs, where relevance is citations, drops from 1.013 and 1.018 to 0.984 and 0.970 with it. Nothing at setup tells these corpora apart. If you have labeled queries, compare with `lexical=None`.
- **On questions that name things, BM25 and the reranker do most of the work.** If your old index could be dropped, try that first: see the table below.
- **The predicted score is a rough signal.** It predicts the first stage, not the final result, and over-predicts by 8 to 27 retention points at 1M documents. If you have labeled queries, measure.
- **Absolute quality falls with size for every system.** MS MARCO reads about 0.72 at 1M and 0.42 at 8.84M.
- **Evidence above 1M documents is thin.** 8.84M rests on MS MARCO alone, 5.23M on HotpotQA, and through HNSW only one index, on MS MARCO.
- **Without BM25,** HotpotQA's multi-hop questions read below the lines on the mean over model pairs at every size from 10K documents up. MS MARCO across families reads below them from 100K to 160K documents at 60 candidates. There, use `freeze(index, old_model, K=4096)`.
- **This measures search, not answers.** No answer generation or downstream task was evaluated.

**What the bridge adds over BM25 and the reranker alone**, which don't use the old index:

<!-- table:nobridge_short -->
| Corpus | Documents | Models | No bridge, retention | Low | High | Bridge, retention | Bridge adds, absolute points | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
<!-- /table:nobridge_short -->

## Where to go next

| You want | Read |
| --- | --- |
| Inputs, deployment, languages, what the numbers mean | [Usage guide](docs/usage.md) |
| The whole story, for a reader new to this | [Final report](spacebridge_final_report.md) |
| How it works, and all the results | [Design doc](spacebridge_design_and_benchmark.md) |
| Which data informed which decision | [Decision ledger](docs/decision_ledger.md) |
| The records behind the tables | [`results/`](results/README.md) |
| The local caches, for continuing development | [`data/README.md`](data/README.md) |

## Layout

- `src/spacebridge/`: the library.
- `src/spacebridge_bench/`: the benchmark harness: model registry, cached encoding, datasets, metrics, runner.
- `tests/`: tests on synthetic data, with no downloads and no GPU. Run with `uv sync && uv run pytest`.
- `scripts/`: studies, and the scripts that make the tables.
- `examples/`: `quickstart.py` and `deploy.py`.
- `results/`: compact records of the published results.

No dataset text or embeddings are in this repository. Benchmark runs download them and cache them under `data/`, and the examples use Hugging Face's own cache.
