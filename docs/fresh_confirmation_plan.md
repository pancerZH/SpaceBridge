# Fresh confirmation plan

Written and committed on 2026-11-05, before any result below existed. The system was tuned on reused benchmarks, so its earlier passes are exploratory. This run tests the final system, unchanged, on data that chose nothing. Git history dates this file; if it changes after the results, the changes are visible there.

## What is frozen

The system at the commit that adds this file, with the library's defaults:

- `freeze` and `mount` with no overrides. K is 2,048 up to 160,000 documents and 8,192 above.
- BM25 in a pool of 60 candidates, a quarter of them from BM25.
- `bge-reranker-v2-m3`, in bf16 on the GPU.
- Exact search, except where a line below says HNSW.

Nothing in the defaults changes because of these results. They can only change what we claim.

## The data

No item here was used to choose anything.

| Name | Corpus | Queries | Why it counts as fresh |
| --- | --- | --- | --- |
| `msmarco-full-fresh` | MS MARCO, 8,841,823 passages | 2,000 dev queries drawn at random from those the project never loaded | New queries. The corpus is not new, and it helped choose BM25 and the K rules |
| `hotpotqa-full-fresh` | HotpotQA, 5,233,329 passages | 2,000 test queries drawn at random from those the project never loaded | New queries. The corpus is not new, and it motivated BM25 |
| `lotte-technology` | LoTTE technology, test split, 638,509 forum answers, whole | Up to 2,000 forum questions | New corpus, new domain, not Wikipedia or web search |
| `fever-1m` | FEVER, 1,000,000 passages kept with all judged positives | 2,000 test claims | New corpus and query type. Wikipedia text overlaps HotpotQA's in kind |
| `touche2020` | Touché 2020, 382,545 arguments, whole | 49 questions | New corpus, long documents. Too few queries for much precision, which we say now |

NQ has no fresh row: every one of its 3,452 test queries was already read end to end at 1M.

Query draw: seed 20261105, over the query IDs left after removing every query text the base dataset loads. For `fever-1m` and `lotte-technology`, the loader's usual draw, seed 0.

## The comparison

- **Model pairs (index ← query):** `bge-large-en` ← `bge-base-en`, `bge-base-en` ← `bge-large-en` (same family); `bge-large-en` ← `qwen3-0.6b`, `qwen3-0.6b` ← `bge-large-en` (across families).
- **Seeds:** 0, 1, 2. The seed changes the anchors, the split and the distractors, not the queries.
- **Metric:** `nDCG@5` after the reranker, averaged over queries, then seeds, then pairs within a family.
- **Retention:** the bridge's score divided by the lower of native search and re-index, as the project's lines define it.
- **Also reported, with no pass line:** absolute `nDCG@5` for every system; retention against native and re-index with BM25 too; direct retention against re-index with BM25; BM25 and the reranker with no bridge.
- **Interval:** a 95% percentile bootstrap over queries. Each replicate resamples queries jointly across pairs and seeds, retakes the weaker baseline, and recomputes the whole average. It conditions on these models and this corpus, and says nothing about other corpora.

## The verdict, fixed now

Lines, same family then across families: 98% and 94% up to 1M, 95% and 90% above.

| Verdict | Meaning |
| --- | --- |
| **Confirmed** | The point estimate and the interval's lower bound are both at or above the line |
| **Point pass** | The point estimate is at or above the line, and the lower bound is below it |
| **Miss** | The point estimate is below the line |

A miss stays a miss: no re-tuning and no re-run with other settings. The docs then state the corpus and the number next to the range it contradicts. Whichever way the results go, the best range is stated as what was measured on which corpora, and not as a guarantee for an unseen corpus.

## HNSW

`msmarco-full-fresh` also runs through the one existing 8.84M HNSW index, ef 500, pool 60, `bge-large-en` ← `bge-base-en` and ← `qwen3-0.6b`, with the same lines. The index was built on these documents before this plan.

## Time

The whole run stops within 12 hours of its start. The order is fixed, and a step that doesn't start in time is skipped and reported as not run: `msmarco-full-fresh`, its HNSW check, `hotpotqa-full-fresh`, measured latency, `lotte-technology`, `fever-1m`, `touche2020`. Numbers for the published tables are recomputed first, and that recomputation is not part of this confirmation.
