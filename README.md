# SpaceBridge

Query a frozen vector index with a different embedding model, without re-embedding the corpus. SpaceBridge fits a small query-side map on a few thousand calibration anchors, and a cross-encoder reranks what the search returns.

It gives you compatibility, not the new model's quality. To get a better model's first-stage gain, you must re-index.

The evidence and design are in [spacebridge_design_and_benchmark.md](spacebridge_design_and_benchmark.md).

## Quickstart

```bash
uv sync --extra models --extra banks
uv run python examples/quickstart.py
```

This downloads a small public corpus (SciFact, 5.2K abstracts) and three models, builds an index with one model and serves it with another. Once they're downloaded, it takes about 30 seconds on a GPU and about 3 minutes on a CPU.

The calls it makes:

```python
import spacebridge

index = spacebridge.Index(vectors, texts, titles)      # your frozen vectors and their text
bundle = spacebridge.freeze(index, old_model)          # once, with the model that built the index
bridge = spacebridge.mount(bundle, index, new_model)   # with each new query model

searcher = spacebridge.Searcher(bridge, new_model, spacebridge.ExactBackend(vectors), index, spacebridge.CrossEncoderReranker())
result = searcher.search("your query", k=5)            # .ids, .scores, .texts, .confidence, .pool, .timing_ms
```

**Above 200K documents,** use HNSW (`uv sync --extra hnsw`, then `spacebridge.build_hnsw` and `spacebridge.HnswBackend`). A reranker is required there: `Searcher` raises `ValueError` without one, unless you pass `allow_no_reranker=True`.

## When to use this

**Use it** when a new query model has to work on an existing index now, or when the model that built the index is gone. **Re-index instead** when you'll serve more than a few thousand queries, or when you want the new model's first-stage quality. See [break-even](#re-index-or-bridge).

### How far it goes

Retention is end-to-end `nDCG@5` after `bge-reranker-v2-m3`, against the weaker of the old model's native search and a full re-index. The targets are 98% for models of one family (`bge-base-en-v1.5` and `bge-large-en-v1.5`) and 94% across families (`bge-large-en-v1.5` with `Qwen3-Embedding-0.6B` or `harrier-oss-v1-0.6b`). 3 seeds, mean over pairs, K = 8,192 anchors, pool 300.

| Corpus | Size | Same family (98%) | Across families (94%) |
| --- | --- | --- | --- |
| MS MARCO | 500K | 0.990 | 0.965 |
| NQ | 500K | 0.993 | 0.980 |
| MS MARCO | 1M | 0.986 | 0.957 |
| NQ | 1M | 0.989 | 0.972 |
| HotpotQA | 1M | **0.956** | **0.898** |
| MS MARCO | 8.84M | 0.963 | not measured |

- **Best range: up to 500K,** on MS MARCO and NQ, at 500 queries. HotpotQA hasn't been measured at that size, and it's the hardest corpus we've tested.
- **Supported range: up to 1M.** MS MARCO and NQ meet the targets. HotpotQA, with multi-hop questions, misses them by 2.4 and 4.2 points. Against the lower stretch line of 95% and 90%, it clears the first and falls 0.2 points short of the second.
- **8.84M is a first look, not a recommendation.** Only the same-family pair has been measured there, on one corpus.

**Limits worth knowing:**

- **Your corpus may be harder than these.** The score below doesn't reliably warn you. If you have labeled queries, measure.
- **Scope:** English, on these model pairs. Apart from the MiniLM result below, other kinds of pair aren't measured above 171K.
- **A small old index is weak at 1M:** `all-MiniLM-L6-v2` queried with `bge-large-en-v1.5` reads 0.935, and with `Qwen3-Embedding-0.6B` 0.912.
- **The default pool can miss at mid sizes.** The library serves pool 60 up to 200K. That misses the targets at 150K on MS MARCO and NQ, and at 50K on HotpotQA. Pass `pool=` for a larger one.
- **The quality score over-predicts as the index grows.** It reads 94 at every size on MS MARCO, while real first-stage `Recall@30` retention falls from 95 at 200K to 79 at 8.84M. Above 200K, treat it as a rough signal and heed its warning.

### How long one query takes

Seconds, p50 (p95), on 10 CPU threads or an RTX 5070 Ti. The library's pool sets the time, so it steps at 200K rather than rising with size. **Short** is MS MARCO passages, 97 tokens per (query, document) pair. **Long** is TREC-COVID abstracts, 311 tokens per pair. No long-document corpus has been measured above 171K, so above that the long column applies TREC-COVID's rerank time to pools from short-passage corpora.

| Index size | Library's pool | CPU, short | CPU, long | GPU, short | GPU, long |
| --- | --- | --- | --- | --- | --- |
| Up to 200K | 60 | 11.0 (12.1) | 36 (42) | 0.16 (0.17) | 0.29 (0.33) |
| 500K | 300 | 50.8 (56.8) | **171 (185)** | 0.40 (0.45) | 1.15 (1.28) |
| 1M | 300 | 50.8 (56.7) | over a minute | 0.40 (0.45) | 1.15 (1.28) |

- **Short passages:** a CPU stays under a minute at every size measured, up to 1M.
- **Long documents:** a CPU crosses a minute just above 200K, where the pool goes from 60 to 300. Use a GPU from there.
- **A grade C or D bridge, or one with no source model, gets pool 100 up to 200K:** 17.5 s short, 60.6 s long.
- **Your own documents, on a CPU like this one:** rerank seconds ≈ 1.91 ms × pool × tokens per pair, within 4% on average and 10% at worst over five corpora from 97 to 412 tokens per pair. One query reaches a minute at pool × tokens per pair ≈ 31,400. Long documents go through a 512-token limit.
- **16 threads** is about 30% faster than 10 (28-30% at the three points measured).

### Re-index or bridge

Above 200K the bridge reranks 300 candidates where a re-indexed system would need about 60. That costs an extra 0.25 s per query on GPU and 40 s on CPU. Queries until re-indexing would have been cheaper in compute, on short passages at 1M:

| Query model | GPU | CPU |
| --- | --- | --- |
| `all-MiniLM-L6-v2` | 475 | 131 |
| `bge-base-en-v1.5` | 1,136 | 901 |
| `bge-large-en-v1.5` | 2,929 | 3,168 |
| `Qwen3-Embedding-0.6B` | 6,921 | 5,070 |

Halve these at 500K. So at this size the bridge is a way to serve right away, not a long-term saving. At or below 200K the bridge also uses pool 60, so there's no extra rerank to pay back.

### Queries in another language

For Chinese queries into an English index, a good translator did better than the bridge on our tests: MADLAD-400 led the best bridge by 2.5-4 points of `nDCG@5` at pool 60. **Use translation while the old query model is available, and the bridge once it's retired.** The evidence is Chinese to English only. For migration within one language, translation doesn't apply.

### When the old model is gone

Pass `None` as the old model. The fit uses only the stored vectors and the new model:

```python
bundle = spacebridge.freeze(index, None)
bridge = spacebridge.mount(bundle, index, new_model)
```

It held up on the sets tested: on MS MARCO 1M at pool 300 it reads 0.985 and 0.952-0.970, level with the fit that has the old model. On FiQA it needs a deeper pool. **There's no quality score, and `result.confidence` is NaN,** because there's no old model to compare with. It serves normally otherwise, at pool 100 or more.

## Setup

```bash
uv sync            # Python 3.12, PyTorch (CUDA 12.8 wheels), benchmark + dev dependencies
uv run pytest      # synthetic-data tests, no downloads or GPU required
```

## Layout

- `src/spacebridge/`: the library (anchor selection, solvers, refinement, `SpaceBridge`, `freeze`, `mount`, `Searcher`).
- `src/spacebridge_bench/`: benchmark harness (model registry, cached encoding, datasets, metrics, runner).
- `tests/`: unit tests on synthetic data.
- `scripts/`: studies, summary tables, and `steps.sh` for restartable experiment drivers.
- `examples/`: the quickstart.

Datasets and embeddings are downloaded at runtime and cached under `data/` (git-ignored); no dataset text is committed to this repository.

## Long runs

Encoding, reranker scores and benchmark results are saved as they are produced, so a crash or power loss costs at most a few minutes of work: rerun the same command and it continues from what is on disk. Run long jobs through `scripts/steps.sh`, which gives drivers restartable steps and a synced progress log, and sets the GPU allocator option that exact search needs at 8.84M.
