# SpaceBridge

Query a frozen vector index with a different embedding model, without re-embedding the corpus. SpaceBridge fits a small query-side map on a few thousand calibration anchors. A cross-encoder then reranks what the search returns, along with BM25's best matches over the document text.

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

**BM25 comes on by itself** at any size, if the index is in English and there's a reranker: `Searcher` builds it from your text, which takes about 20 s per 1M documents on one CPU thread. To build it once and keep it, or to turn it off:

```python
lexical = spacebridge.LexicalIndex.build(index)
lexical.save("lexical/")
searcher = spacebridge.Searcher(bridge, new_model, backend, index, reranker, lexical=spacebridge.LexicalIndex.load("lexical/"))
searcher = spacebridge.Searcher(bridge, new_model, backend, index, reranker, lexical=None)   # off
```

**Above 200K documents,** use HNSW (`uv sync --extra hnsw`, then `spacebridge.build_hnsw` and `spacebridge.HnswBackend`). A rerun of `build_hnsw` continues from its last save. A reranker is required there: `Searcher` raises `ValueError` without one, unless you pass `allow_no_reranker=True`. `ExactBackend` on a GPU raises `ValueError` if the vectors won't fit.

## When to use this

**Use it** when a new query model has to work on an existing index now, or when the model that built the index is gone: serve right away, then re-index in the background if the last few points justify it, or if you'd run without a reranker. See [re-index or bridge](#re-index-or-bridge).

### How far it goes

Retention is end-to-end `nDCG@5` after `bge-reranker-v2-m3`, against the weaker of the old model's native search and a full re-index, at the same rerank depth. The targets are 98% for models of one family (`bge-base-en-v1.5` and `bge-large-en-v1.5`) and 94% across families (`bge-large-en-v1.5` with `Qwen3-Embedding-0.6B`), and a lower stretch line of 95% and 90% above 1M. 3 seeds, mean over pairs, exact search, a budget of 60 candidates with BM25 in a quarter of them, which is what the library serves.

**Retention above 1 isn't a gain over native search.** With BM25, the bridge has a channel plain native doesn't. "With BM25 on both" gives native and re-index the same BM25.

| Corpus | Size | Same family | Across families | With BM25 on both | Without BM25, pool 300 |
| --- | --- | --- | --- | --- | --- |
| MS MARCO | 100K | 1.000 | 0.996 | 0.995, 0.995 | |
| NQ | 100K | 0.996 | 0.999 | 0.995, 0.997 | |
| HotpotQA | 100K | 1.000 | 1.012 | 0.995, 0.997 | |
| MS MARCO | 500K | 0.996 | 0.989 | 0.995, 0.989 | 0.990, 0.965 |
| NQ | 500K | 0.989 | 0.988 | 0.993, 0.985 | 0.993, 0.980 |
| HotpotQA | 500K | 0.997 | 1.016 | 0.994, 0.993 | 0.962, 0.909 |
| MS MARCO | 1M | 0.995 | 0.982 | 0.994, 0.983 | 0.986, 0.957 |
| NQ | 1M | 0.985 | 0.979 | 0.985, 0.975 | 0.989, 0.972 |
| HotpotQA | 1M | 0.995 | 1.020 | 0.989, 0.988 | 0.956, 0.898 |
| MS MARCO | 2M | 0.986 | 0.972 | 0.985, 0.973 | |
| HotpotQA | 2M | 0.994 | 1.020 | 0.982, 0.981 | |
| MS MARCO | 4M | 0.985 | 0.962 | 0.982, 0.958 | |
| HotpotQA | 5.23M | 0.985 | 1.009 | 0.967, 0.961 | 0.886, 0.798 |
| MS MARCO | 8.84M | 0.979 | 0.946 | 0.981, 0.945 | 0.963, 0.895 |

Up to 500K is 500 queries, which moves a reading by about ±1.5 points, and Tier 1 there is one direction. 1M and above is 2,000 queries, both directions.

- **With BM25, best range: up to 1M,** on all three corpora, at budget 60, for models of one family and across families.
- **With BM25, supported range: up to 8.84M,** at budget 60, exact search. The top size rests on MS MARCO alone, HotpotQA is measured to its whole corpus, 5.23M, and NQ isn't measured above 1M. Nothing was held out from this budget above 1M. No size between crosses the line: on MS MARCO, across families, it falls steadily from 0.982 at 1M through 0.972 and 0.962 to 0.946.
- **With BM25 off (`lexical=None`), multi-hop questions miss at every size measured,** from 10K (0.968, 0.937) through 50K (0.948, 0.880) to 5.23M. With it, HotpotQA reads 1.003 to 1.006 and 1.005 to 1.011 from 10K to 50K (500 queries). MS MARCO and NQ pass at those sizes either way.

**Limits worth knowing:**

- **BM25 needs document text and English.** Its tokenizer keeps only `[a-z0-9]+`. The reranker needs the text anyway. It's off for an index in another language, and a query in another language skips it.
- **Where relevance isn't lexical, BM25 can cost you.** SciDocs, where relevance is citation links, goes from 1.013 and 1.018 to 0.984 and 0.970 at budget 60, in every comparison, though it still passes. Nothing at mount tells these corpora apart, so BM25 is on for them too. FiQA and TREC-COVID lost 0.9 to 2.2 points in every comparison and still passed: FiQA 1.018 and 0.981, TREC-COVID 1.001 and 0.987 on 50 queries. SciFact and ArguAna gained 0.9 to 2.0. If you have labeled queries, compare with `lexical=None`.
- **Absolute quality falls with size for everyone.** On MS MARCO at 8.84M, `nDCG@5` reads 0.419 and 0.405, against 0.429 for native and re-index. The same search at 1M reads 0.72.
- **Your corpus may be harder than these.** The score below doesn't reliably warn you. If you have labeled queries, measure.
- **Scope:** English, on these model pairs. A small old index holds up too at 1M: `all-MiniLM-L6-v2` queried with `bge-large-en-v1.5` reads 0.981, and with `Qwen3-Embedding-0.6B` 0.975. Other kinds of pair aren't measured above 171K.
- **Through HNSW, measured on MS MARCO only, one index per size:** with BM25, at budget 60, ef 500, it costs at most 0.08 points against exact search at 1M, reading 0.995 and 0.978, and 0.01 and 0.06 points at 8.84M, reading 0.978 and 0.945. At 8.84M that's one direction per tier, so the supported range itself stays stated as exact search. No HotpotQA or NQ index exists.
- **A bridge with no old model or a Stage-2 health flag gets a budget of 100.** The grade doesn't widen it with BM25 on: over 21 pairs at 1M and above the score doesn't track what 100 gains, and HotpotQA across families, graded C, passes by 7.5 to 12.5 points at 60. Pass `pool=100` if you want those last points: on HotpotQA across families that's 0.9 to 1.9 points of absolute `nDCG@5`.
- **The quality score over-predicts as the index grows.** It doesn't see BM25 either: it still grades HotpotQA B and C. Above 200K, treat it as a rough signal, and don't read `report.rerank_pool`'s 300 as what's served with BM25 on: `result.pool` is.
- **The mount budget is tight where K = 8,192 starts,** on short passages, against a line of 10% of re-index tokens: it reads 8.8-9.8% at 160K. K = 2,048 up to there and K = 8,192 above read level with BM25 on: at 160K, 500 queries, Tier 3, 0.996 and 0.998 on MS MARCO, 0.994 and 1.000 on NQ, 1.015 and 1.018 on HotpotQA.
- **With BM25 off, or an index not in English, K = 2,048 misses from 100K to 160K at pool 60.** Tier 3 on MS MARCO reads 0.933, 0.918 and 0.905 at 100K, 150K and 160K, and on NQ 0.938 and 0.936 at 150K and 160K. `freeze(index, old_model, K=4096)` reads 0.947 and 0.940 on MS MARCO and 0.961 on NQ at 100K and 150K, and it's inside the mount budget from 80K. That case isn't measured on an index not in English.
- **On multi-hop questions, BM25 and the reranker do most of the work.** With no bridge at all, budget 60, 2,000 queries, HotpotQA reads 0.958 and 1.001 at 1M, against 0.995 and 1.020 with the bridge, and 0.945 and 1.000 at 5.23M, against 0.985 and 1.009. On MS MARCO and NQ the bridge adds far more: 0.909 and 0.906 become 0.995 and 0.982 at 1M, and 0.863 and 0.870 become 0.985 and 0.979. So if your old index could be dropped, that baseline is worth trying first on questions that name things.

### How long one query takes

Seconds, p50 (p95), on 10 CPU threads or an RTX 5070 Ti, through `Searcher`, on an idle machine. **Short** is MS MARCO passages, 92-99 tokens per (query, document) pair. **Long** is TREC-COVID abstracts, 320 tokens per pair. No long-document corpus has been measured above 171K, so above that the long column applies TREC-COVID's rerank time to pools from short-passage corpora. Retention is the hardest corpus measured at that size.

| Index size | BM25 | Budget | K | Same family | Across families | CPU, short | CPU, long | GPU, short | GPU, long |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.2K | on | 60 | 2,048 | 1.008 | 1.008 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 10K | on | 60 | 2,048 | 1.000 | 1.001 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 20K | on | 60 | 2,048 | 1.002 | 1.003 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 25.7K | on | 60 | 2,048 | 0.984 | 0.970 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 30K | on | 60 | 2,048 | 1.002 | 1.000 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 50K | on | 60 | 2,048 | 1.000 | 1.000 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 100K | on | 60 | 2,048 | 0.996 | 0.996 | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 150K | on | 60 | 2,048 | 0.994 | 0.996 | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 160K | on | 60 | 2,048 | 0.997 | 0.994 | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 160K + 1 | on | 60 | 8,192 | 0.997 | 0.998 | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 171K | on | 60 | 8,192 | 1.001 | 0.987 | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 200K | on | 60 | 8,192 | 0.995 | 0.995 | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 500K | on | 60 | 8,192 | 0.989 | 0.988 | 11.1 (12.5) | 36.9 (41.4) | 0.16 (0.17) | 0.31 (0.34) |
| 1M | on | 60 | 8,192 | 0.985 | 0.979 | 11.0 (12.5) | 36.8 (41.4) | 0.16 (0.17) | 0.31 (0.34) |
| 2M | on | 60 | 8,192 | 0.986 | 0.972 | 10.7 (13.3) | 37.0 (41.6) | 0.33 (0.43) | 0.51 (0.55) |
| 4M | on | 60 | 8,192 | 0.985 | 0.962 | 10.9 (13.5) | 37.3 (41.8) | 0.55 (0.64) | 0.72 (0.76) |
| 5.23M | on | 60 | 8,192 | 0.985 | 1.009 | 11.1 (13.7) | 37.4 (41.9) | 0.68 (0.77) | 0.86 (0.89) |
| 8.84M | on | 60 | 8,192 | 0.979 | 0.946 | 11.5 (14.0) | 37.8 (42.3) | 1.07 (1.16) | 1.24 (1.27) |
| 8.84M, HNSW | on | 60 | 8,192 | 0.978 | 0.945 | 10.6 (13.2) | 36.9 (41.5) | 0.20 (0.22) | 0.36 (0.40) |

**No row misses.** The lines are 98% and 94% up to 1M, and 95% and 90% above. The lowest below 1M is SciDocs, at 25.7K, which BM25 costs points (see the limits). Rows above 1M are exact search on the CPU, about 0.9 s at 8.84M, since the GPU can't hold those vectors. **The last row is the reference deployment:** one HNSW index, M = 32, ef 500, on MS MARCO at 8.84M, with retention through it on one direction per tier. The 8.84M rows are 20 queries each. 2M and 4M add search and BM25 at their size to those runs, by interpolation, and 5.23M does too. Up to 500K, GPU search is exact on the GPU; at 1M it's HNSW.

- **Short passages:** a CPU stays under a minute at every size measured, up to 8.84M, at 11-12 s, p95 up to 14 s.
- **Long documents:** a CPU stays under a minute at budget 60 at every size, at 36-38 s, and reaches it only at budget 100, at 60.6 s, which a bridge with no old model or a Stage-2 health flag gets.
- **HNSW at 8.84M:** 38.6 GB on disk, built on the CPU in about 1.8 h on 12 threads, loaded in about 30 s from a warm page cache, and 45-47 GB resident while serving, with text and BM25, against 60-62 GB with exact search. Search takes 3-7 ms at ef 500 to 1000, against 0.9 s.
- **BM25 adds** 5 ms at 1M and 30 ms at 8.84M, p50, and 41-58 ms in the HNSW runs on the same index, for a reason not tested. It takes 2.3 GB of RAM at 8.84M. The query model and reranker together peak at 2.4 GB of GPU memory on short passages and 3.0 GB on long documents, on `Qwen3-Embedding-0.6B`.
- **With BM25 off,** above 200K the pool is 300: 51 s short and 171 s long on a CPU, and 0.40 s and 1.15 s on a GPU.
- **Your own documents, on a CPU like this one:** rerank seconds ≈ 1.91 ms × pool × tokens per pair, within 4% on average and 10% at worst over five corpora from 97 to 412 tokens per pair. One query reaches a minute at pool × tokens per pair ≈ 31,400. Long documents go through a 512-token limit.
- **16 threads** is about 30% faster than 10 (28-30% at the three points measured).
- **A faster reranker for CPU, measured but not in the library:** an int8 export of `bge-reranker-v2-m3` (`scripts/export_reranker_onnx.py`, needs `--extra onnx`) took 3.5 s short and 13 s long at pool 60 on this CPU, with no quality loss seen on 50-100 queries per corpus. That's too few to rule out a loss of a point or so, and the targets above aren't claimed for it. To use it, pass `Searcher` an object with a `score(query, docs)` method. `bge-reranker-base` and `ms-marco-MiniLM-L6-v2` score faster but lost up to 44 and 12 points of absolute `nDCG@5`, depending on the corpus, so we don't recommend them as general rerankers.

### Re-index or bridge

Serve right away, then re-index in the background if the last few points justify it, or if you'd run without a reranker. The compute answer depends on whether the re-indexed system would also rerank.

- **If it would,** the bridge adds only BM25 and the projection per query: 6-31 ms on a GPU and 16-41 ms on a CPU, from 1M to 8.84M. Re-indexing then takes tens of thousands of queries or more to pay back in compute (table below), so a low-traffic deployment may never reach it.
- **If it wouldn't,** the bridge also costs a reranker on every query, 0.15 s on a GPU and about 11 s on a CPU on short passages, so re-indexing pays back within thousands of queries. But you'd get less: dense search on a re-indexed system reads `nDCG@5` 0.683, 0.651 and 0.811 at 1M on MS MARCO, NQ and HotpotQA, against 0.722, 0.772 and 0.882 for the bridge with BM25 and the reranker (Tier 1).

Queries until re-indexing would have been cheaper, on short passages. Each cell is GPU, CPU:

| Query model | Re-index 1M | It also reranks | It doesn't | Re-index 8.84M | It also reranks | It doesn't |
| --- | --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 2 min, 1.4 h | 19K, 331K | 803, 476 | 17 min, 12.8 h | 33K, 1.1M | 7.2K, 4.4K |
| `bge-base-en-v1.5` | 5 min, 10.0 h | 45K, 2.3M | 1.9K, 3.3K | 41 min, 88.1 h | 80K, 7.8M | 17K, 30K |
| `bge-large-en-v1.5` | 12 min, 35.1 h | 117K, 8.0M | 5.0K, 12K | 1.8 h, 309.9 h | 206K, 27.4M | 44K, 107K |
| `Qwen3-Embedding-0.6B` | 28 min, 56.1 h | 277K, 12.9M | 12K, 18K | 4.2 h, 496.0 h | 487K, 43.8M | 105K, 171K |

If the re-indexed system runs BM25 and a reranker too, the only difference left is the projection: about 1 ms on a GPU and 11 ms on a CPU.

**If you'd rerank either way, re-index for quality, not compute.** Against a re-indexed system with BM25 too, the bridge gives up 0.6 to 2.5 points at 1M, and 1.9 to 5.5 points at the largest sizes, read from "with BM25 on both" above. Also re-index if you need results without a reranker, or your corpus is one where BM25 costs you. To get a better model's first-stage gain, you must re-index.

### Queries in another language

For Chinese queries into an English index, **use a translator feeding BM25, with the bridge doing the dense search.** BM25 reads the translated query, while the bridge and the reranker keep the original. While the old model is alive it can keep doing the dense search, and it reads about the same. This is Chinese to English only, on two Wikipedia question-answering sets, and it's off by default:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer   # uv sync --extra models --group translate

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda")

def translate(query, lang):   # any function (query, language) -> text will do
    output = translator.generate(**tokenizer([query], return_tensors="pt").to(translator.device), max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

searcher = spacebridge.Searcher(
    bridge, new_model, backend, index, reranker, lexical=spacebridge.LexicalIndex.build(index), translate=translate,
)
result = searcher.search("你的问题", k=5, query_lang="zh")   # the bridge needs query_langs=["zh"] at mount
```

Absolute `nDCG@5` at budget 60 on `xquad-zh` and `mlqa-zh`, English `bge-large-en-v1.5` index, 3 seeds:

| | XQuAD | MLQA |
| --- | --- | --- |
| The bridge alone, `Qwen3-Embedding-0.6B` | 0.809 | 0.669 |
| The bridge, BM25 on opus-mt | 0.859 | 0.725 |
| The bridge, BM25 on MADLAD-400 | 0.861 | 0.733 |
| The old model on MADLAD-400's translation, no bridge | 0.849 | 0.694 |
| The old model on MADLAD-400's translation, with BM25 on it | 0.862 | 0.728 |
| Re-index with `Qwen3-Embedding-0.6B` | 0.851 | 0.713 |

So a translator lets the bridge match the old model plus translation, without the old model. opus-mt costs 38 ms on GPU and 196 ms on CPU per query, and MADLAD-400 215 ms and 3.1 s. **The evidence is Chinese to English only,** on two Wikipedia question-answering sets. Without a translator, BM25 can't help: only numbers and Latin names match. For migration within one language, translation doesn't apply.

### When the old model is gone

Pass `None` as the old model. The fit uses only the stored vectors and the new model:

```python
bundle = spacebridge.freeze(index, None)
bridge = spacebridge.mount(bundle, index, new_model)
```

It holds up with BM25, at the budget of 100 it gets: 0.997 and 1.020 on HotpotQA 1M, and 0.997 and 0.982 on MS MARCO 1M (one pair per tier, 500 queries). **There's no quality score, and `result.confidence` is NaN,** because there's no old model to compare with. It serves normally otherwise.

## Setup

```bash
uv sync            # Python 3.12, PyTorch (CUDA 12.8 wheels), benchmark + dev dependencies
uv run pytest      # synthetic-data tests, no downloads or GPU required
```

## Layout

- `src/spacebridge/`: the library (anchor selection, solvers, refinement, `SpaceBridge`, `freeze`, `mount`, `LexicalIndex`, `Searcher`).
- `src/spacebridge_bench/`: benchmark harness (model registry, cached encoding, datasets, metrics, runner).
- `tests/`: unit tests on synthetic data.
- `scripts/`: studies, summary tables, and `steps.sh` for restartable experiment drivers.
- `examples/`: the quickstart.

Datasets and embeddings are downloaded at runtime and cached under `data/` (git-ignored); no dataset text is committed to this repository.

## Long runs

Encoding, reranker scores and benchmark results are saved as they are produced, so a crash or power loss costs at most a few minutes of work: rerun the same command and it continues from what is on disk. Run long jobs through `scripts/steps.sh`, which gives drivers restartable steps and a synced progress log, and sets the GPU allocator option that exact search needs at 8.84M.
