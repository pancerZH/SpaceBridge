# SpaceBridge: final report

**The question.** You have a search index of millions of documents, built with an embedding model. A new model comes along, or the old one is retired. Can you search the same index with the new model, without re-embedding every document?

**The short answer.** Yes, for English text, if you also use a reranker and keyword search over the same documents. On the three main test corpora, up to 1 million documents, the system keeps at least 98% of the search quality of the weaker alternative between models of one family, and at least 94% across families. A run on data that played no part in building the system gave the verdicts in [§7](#7-what-we-achieved). It doesn't improve quality, it gives you compatibility right away. To get a better model's full gain, you still have to re-index.

This report is for a reader who knows what an embedding is. Terms in **bold** at first use are in [§9](#9-terms). How to use the library is in the [README](README.md), and the method in full is in the [design doc](spacebridge_design_and_benchmark.md).

## 1. The goal

Text search today often works like this: an embedding model turns each document into a vector, and a query becomes a vector too, and the nearest vectors are the results. All those vectors belong to one model. A different model's vectors live in a different space, so they can't be compared.

So changing the model means re-embedding every document. At 8.84 million passages, encoding alone takes from about 20 minutes to 4 hours on a GPU here, by model, and up to 20 days on a CPU. Sometimes it isn't possible at all, if the documents were embedded elsewhere, or if you want a phone's small model to query a large index.

**The goal:** map the new model's queries into the old model's space, so the old index can be searched with no change to it. We set numeric lines for how much quality that must keep, relative to the weaker of the old model's own search and a full re-index:

| Size | Models of one family | Across families |
| --- | --- | --- |
| Up to 1M documents | 98% | 94% |
| 1M to 8.84M documents | 95% | 90% |

8.84M, the whole MS MARCO corpus, is as far as this project goes, because it's the limit of the one machine used. The measure is after a reranker, as is usual for top results, because that's what a user sees.

We also wanted the map to be cheap to fit: at most 10% of the tokens a re-index would encode.

## 2. The theory

**A linear map between models.** Good embedding models may organize meaning with similar relative geometry, differing mainly by offsets, rotations and scaling. If so, a matrix W could carry a query across:

$$\hat q = \text{Normalize}\left(\mu_X + (q - \mu_Y)\, W\right)$$

Here $q$ is the query from the new model, $\mu_Y$ is the new model's average query, and $\mu_X$ is the old model's. So: remove the new model's average, rotate and scale into the old space, add back the old model's average.

**Fitting W needs no labels.** We pick a few thousand documents from the corpus, called **anchors**. From each one we take a **proxy**, usually its title, as a stand-in for a query. Both models encode each proxy, which gives pairs of vectors, one per model, for the same text. The new model also encodes the anchor document, which we can compare with its stored vector.

W is then fitted in two stages. **Stage 1** has closed-form solutions: a rotation and a least-squares fit, picked by how well they reproduce the old model's rankings. **Stage 2** runs 50 steps of gradient descent to make each proxy find its own document.

**What a map can't do.** The stored vectors never change. If the old model put two documents almost in the same place, no query can separate them. And a small error in the query matters more as the index grows, since each query's neighbours get closer together.

**Why a reranker and BM25 help.** A **reranker** reads the query and document text together, and gives a far better score than vectors can, but it's too slow to run on everything. So a first search finds a few dozen **candidates** and the reranker orders them. Then the map only has to get the right document into the candidates, not to the top. **BM25**, a classic keyword search, adds candidates that don't depend on the map at all. A document whose exact name appears in the query is what BM25 finds easily and what a map tends to blur.

## 3. The datasets

Every dataset is a public retrieval benchmark: documents, queries, and human labels saying which documents answer which query.

| Corpus | Documents | Queries used | What it's like | Used for |
| --- | --- | --- | --- | --- |
| SciFact | 5,183 | 300 | Science claims and abstracts | A quick check |
| ArguAna | 8,674 | 1,401 | Arguments | Checking BM25 |
| SciDocs | 25,657 | 1,000 | Paper titles, and the papers they cite | Checking BM25 |
| FiQA | 57,638 | 648 | Finance questions and forum answers | Early decisions |
| TREC-COVID | 171,332 | 50 | Questions and long paper abstracts | Long documents |
| MS MARCO | 8,841,823 | 2,000 | Real web queries and short passages | Scale |
| NQ | 1,000,000 of 2.68M | 2,000 | Real questions and Wikipedia | Scale |
| HotpotQA | 5,233,329 | 2,000 | Questions needing two Wikipedia pages | Scale, hardest case |
| T2Retrieval | 50,000 | 2,000 | Chinese web search | Chinese |
| XQuAD, MLQA | 20,958 and 10,894 | 1,190 and 5,137 | Questions in other languages | Other languages |
| **For the final check only** | | | | |
| MS MARCO, unused queries | 8,841,823 | 2,000 | New queries, same corpus | Fresh confirmation |
| HotpotQA, unused queries | 5,233,329 | 2,000 | New queries, same corpus | Fresh confirmation |
| LoTTE technology | 638,509 | 2,000 | Forum questions and answers | Fresh confirmation |
| FEVER | 1,000,000 of 5.42M | 2,000 | Claims and Wikipedia | Fresh confirmation |
| Touché 2020 | 382,545 | 49 | Long arguments on controversial questions | Fresh confirmation |

**Smaller sizes of a corpus** keep every labeled answer for the test queries and add randomly chosen other documents. This makes sizes comparable, but it isn't the same as a natural corpus of that size, particularly at the small end, where up to 42% of a 10,000-document set are labeled answers.

**Models:**

| Role | Model | Dimensions |
| --- | --- | --- |
| One family, smaller | `bge-base-en-v1.5` | 768 |
| One family, larger | `bge-large-en-v1.5` | 1,024 |
| Another family | `Qwen3-Embedding-0.6B` | 1,024 |
| Another family | `harrier-oss-v1-0.6b` | 1,024 |
| Reranker | `bge-reranker-v2-m3` | |

## 4. The experiments

**The machine.** All on one computer: Intel Core i5-12600K, Nvidia RTX 5070 Ti with 16 GB, 96 GB of DDR4 RAM. So timings are for this machine, one query at a time, with nothing else running.

**What we compared.** For each corpus, size and model pair, three systems, each followed by the same reranker: the **bridge** (new model, mapped, old index), **native** (old model, old index) and **re-index** (new model, new vectors). **Retention** is the bridge's score divided by the weaker of the other two. So 0.98 means 2% below the weaker alternative, and above 1 means better than it.

The score is **nDCG@5**, which rewards putting the right documents in the top 5, higher up being better.

**A ladder of configurations.** One piece at a time, so a gain could be pinned on one change: no centering, then centering, then proxies, then Stage 2, then document pairs, then the reranker, then BM25.

**The stages.**

| Stage | The question | The answer |
| --- | --- | --- |
| First experiments | Does a linear map work at all? | Yes up to 171K documents, with a reranker |
| First test at 1M | Does it hold at scale? | Only with many more anchors and a much deeper reranker pool |
| Library | Can one library call deliver it, and how fast? | Yes on two corpora, but too slow on a CPU |
| A held-out corpus | Does it hold on data we hadn't used? | **No.** HotpotQA missed at 1M |
| Why, and how to fix it | What does the map lose? | BM25 in the pool passes, at a fifth of the reranking cost |
| Largest sizes | How far up? | Up to 8.84M, on the lower line |
| Review | Is the evidence sound? | The defects found changed no published number. Claims and statistics were corrected |
| Fresh confirmation | Does the final system, unchanged, hold on data that chose nothing? | [§7](#7-what-we-achieved) |

**Rules that kept results honest.** Three seeds for anything reported. Reading rules written before the result, for the largest decisions. A result that came out mixed stayed mixed. Records saved as they were produced, so a crash loses little, and a rerun continues.

**How much the evidence supports.** Most of this was exploration: we looked at results, changed the system, and looked again, on reused data. So those passes show that the chosen system reaches the chosen lines on those corpora. They aren't independent proof. The [decision ledger](docs/decision_ledger.md) records which data informed which choice. The fresh confirmation is the one run built to test the final system.

## 5. What went well

**The reranker recovers most of what the map loses.** A first stage reading about 0.8 of `Recall@30` retention becomes about 0.99 at the end, because near misses are still in the candidates.

**Proxies that look like queries.** Replacing a document's lead sentence with its title gained 4.7 to 8.6 retention points of first-stage `Recall@30` on TREC-COVID.

**BM25 fixed the hard corpus and the cost at once.** Before BM25, passing at 1M needed a pool of 300 candidates, 51 s per query on a CPU on short passages. BM25 brought that to 60, 11 s, and it passes on a corpus 300 candidates couldn't. It needs no new model, only the text the reranker already reads.

**It runs at 8.84M on one machine.** The whole path, on a graph index, takes about 0.2 s per query on the GPU. Search itself takes 3 to 7 ms.

**A translator in the right place.** For Chinese queries into an English index, feeding only BM25 with a machine translation gains 5.0 to 7.5 absolute `nDCG@5` points, and matches the old model given the same help, without needing the old model.

**It works with the old model gone.** A fit from stored vectors and the new model alone holds up, at a larger pool of 100, on one pair per family, 500 queries, at 1M.

**The recheck held up.** Recomputing every published cell from the caches, with the corrected code, matched the published records: 19,836 values, all identical to six decimal places.

**The fresh confirmation held up.** On data that chose nothing, no result missed a line, and eight of ten were confirmed outright ([§7](#7-what-we-achieved)).

## 6. What went badly, and what we did

### The results

| Problem | What we saw | What we did |
| --- | --- | --- |
| **The map alone misses on a hard domain** | FiQA first read 0.874 of `Recall@30` retention, under the 95% we'd set | Lowered the first-stage target to 88%, kept the end-to-end one, and added a reranker |
| **Retention falls as the index grows** | Across families, first stage: 0.92 at 50K, 0.73 at 1M, same queries, with no bug. Small tests had flattered us | More anchors above 160K, and judging large indexes end to end |
| **The fix was too expensive** | Pool 300 at 1M took 51 s on short passages and 171 s on long ones, on a CPU | BM25, below |
| **"The 1M goal is met" was wrong** | MS MARCO and NQ passed, then HotpotQA read 0.956 and 0.898 at pool 300 | Withdrew the claim, and looked for why, rather than raising parameters |
| **Why HotpotQA** | Positives named in a query were lost. Repairing the name's part of the query brought back 97% to 99% of those, in a test that used the true titles and the old model. That covers the 42% of positives whose titles appear in a query | BM25, which reads the words directly. Why names transfer worse here than on NQ is still open |
| **BM25 isn't free** | SciDocs, where relevance is citations, drops from 1.013 and 1.018 to 0.984 and 0.970. FiQA and TREC-COVID lose 0.9 to 2.2 retention points | On at every size anyway, a judgment stated as one: a small sure loss against a large one, with nothing to tell them apart in advance |
| **On HotpotQA the bridge adds little** | With no bridge at all, BM25 and the reranker read 0.958 and 1.001 at 1M | State this beside every HotpotQA figure. On MS MARCO and NQ the bridge adds far more |
| **Translation beat the bridge across languages** | A translator into the old model led by 2.5 to 4 absolute points, but needs the old model | Combined them: the translator feeds BM25, and the bridge does the rest |

### The quality score

Predicting quality without labels never became reliable. The score predicts within about 4 points on small corpora, and over-predicts by 8 to 27 at 1M. It can't see a failed Stage 2, another language, or BM25. We wanted a system that's never silently bad, and **we didn't get there**. So the score only advises, and anyone with labeled queries should measure.

### The method

| Problem | What we saw | What we did |
| --- | --- | --- |
| **The first quality signal measured nothing** | It read 1.000 for every configuration, including one with 82% real retention | Replaced it with agreement with the old model's ranking |
| **Agreeing with the old model isn't retrieving well** | Stage 2 raises real quality while lowering agreement, so stopping on agreement threw away the gain, in about 85% of fits | A fixed 50 steps, averaging the late ones, under a floor |
| **A silent numeric failure** | One model's compressed scores cost 20 points, and nothing flagged it | A temperature that follows the score spread, and a health flag |
| **Timing guards and drivers** | Scripts printed "DONE" whatever happened. One unanchored `pgrep` killed its own shell | Rules in the drivers. An old driver with this flaw has now been fixed |

### What an independent review found

Reviewers read the code and documents after we'd finished. They found real defects, and the ones that could have changed a published number were checked against the records.

| Defect | Could it have changed a number? | What the records show | Fix |
| --- | --- | --- | --- |
| **A resume could count an old run as a new one**: the key ignored query count and rows | Yes | No run in any log ever skipped a recorded cell. Each published file has one query count and no repeats | Runs carry a hash of their configuration, and another configuration's records raise an error |
| **Missing results could read as a pass** in the HNSW rule | Yes | All 85 records it needed exist, so the published pass is real | Missing evidence now reads "incomplete" |
| **The "95% intervals" were averaged endpoints**, not intervals for the reported average | The intervals, yes | The point estimates are unaffected, and the intervals were wrong | Real intervals from aligned per-query scores, in §7 |
| **Recall at 150 and 200 fell back to Recall@100** | Yes | Only two study files have those pools, and no published figure used them | Computed at the real depth |
| **Latency tails were modeled but labeled measured** | Yes | Sizes without their own run were modeled | Timed at more sizes, and every cell says which it is |
| **An HNSW file at a reused path could pair the old graph with new text** | Not in the benchmarks | Paths named the vectors | Checked against the vectors, in the library |
| **"Upper bound" and "loss against re-indexing"** were overstated | Claims, yes | The denominator is the weaker baseline, not a bound, and not re-indexing | Renamed, and direct re-index comparison added |

The report's earlier claim that a nonlinear map was never tried was also wrong: one was tried, and it didn't help. That result is narrow: one architecture, trained on cosine.

## 7. What we achieved

**The system:** a Python library, `spacebridge`, with `freeze`, `mount` and `Searcher`. The map fits in seconds to about half a minute on a GPU, on at most about 10% of a re-index's tokens.

### Retention on the main corpora

Retention against the weaker of native search and re-index, after the reranker. "Low" and "high" bound a 95% confidence interval for query sampling, on these models and this corpus, which doesn't cover other corpora. 3 seeds. **Reaching a line here is an exploratory result**, since these corpora helped shape the system.

<!-- table:retention_short -->
| Corpus | Documents | Queries | Same family | Low | High | Across families | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS MARCO | 100K | 500 | 1.000 | 0.990 | 1.009 | 0.996 | 0.988 | 1.004 |
| MS MARCO | 500K | 500 | 0.996 | 0.986 | 1.006 | 0.989 | 0.978 | 0.999 |
| MS MARCO | 1M | 2000 | 0.995 | 0.989 | 1.002 | 0.982 | 0.976 | 0.989 |
| MS MARCO | 2M | 2000 | 0.986 | 0.980 | 0.995 | 0.972 | 0.963 | 0.982 |
| MS MARCO | 4M | 2000 | 0.985 | 0.975 | 0.995 | 0.962 | 0.950 | 0.975 |
| MS MARCO | 8.84M | 2000 | 0.979 | 0.966 | 0.993 | 0.946 | 0.930 | 0.964 |
| NQ | 100K | 500 | 0.996 | 0.987 | 1.006 | 0.999 | 0.992 | 1.008 |
| NQ | 500K | 500 | 0.989 | 0.976 | 1.000 | 0.988 | 0.974 | 1.001 |
| NQ | 1M | 2000 | 0.985 | 0.979 | 0.993 | 0.979 | 0.971 | 0.987 |
| HotpotQA | 100K | 500 | 1.000 | 0.992 | 1.008 | 1.012 | 1.003 | 1.022 |
| HotpotQA | 500K | 500 | 0.997 | 0.988 | 1.007 | 1.016 | 1.003 | 1.031 |
| HotpotQA | 1M | 2000 | 0.995 | 0.990 | 1.000 | 1.020 | 1.012 | 1.029 |
| HotpotQA | 2M | 2000 | 0.994 | 0.988 | 1.000 | 1.020 | 1.011 | 1.029 |
| HotpotQA | 5.23M | 2000 | 0.985 | 0.977 | 0.993 | 1.009 | 0.999 | 1.021 |
<!-- /table:retention_short -->

### Fresh confirmation: data that chose nothing

We froze the system, wrote down the rules before any result existed ([plan](docs/fresh_confirmation_plan.md)), and ran it. **Confirmed** means the point estimate and the interval's low end both reach the line. **Point pass** means only the point estimate does. **Miss** means neither. The same model pairs and seeds.

<!-- table:fresh -->
| Data | Documents | Queries | Models | Retention | Low | High | Line | Verdict | No bridge, retention | Bridge, absolute `nDCG@5` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS MARCO, unused queries | 8.84M | 2000 | Same family | 0.971 | 0.959 | 0.985 | 0.95 | Confirmed | 0.862 | 0.428 |
| MS MARCO, unused queries | 8.84M | 2000 | Across families | 0.956 | 0.940 | 0.974 | 0.90 | Confirmed | 0.870 | 0.418 |
| HotpotQA, unused queries | 5.23M | 2000 | Same family | 0.978 | 0.970 | 0.986 | 0.95 | Confirmed | 0.949 | 0.801 |
| HotpotQA, unused queries | 5.23M | 2000 | Across families | 0.999 | 0.989 | 1.010 | 0.90 | Confirmed | 0.999 | 0.777 |
| LoTTE technology | 638.5K | 2000 | Same family | 0.989 | 0.976 | 1.002 | 0.98 | Point pass | 0.838 | 0.302 |
| LoTTE technology | 638.5K | 2000 | Across families | 0.964 | 0.951 | 0.977 | 0.94 | Confirmed | 0.818 | 0.301 |
| FEVER | 1M | 2000 | Same family | 0.997 | 0.993 | 1.001 | 0.98 | Confirmed | 0.977 | 0.947 |
| FEVER | 1M | 2000 | Across families | 0.994 | 0.991 | 0.999 | 0.94 | Confirmed | 0.977 | 0.944 |
| Touché 2020 | 382.5K | 49 | Same family | 1.021 | 0.976 | 1.087 | 0.98 | Point pass | 1.096 | 0.380 |
| Touché 2020 | 382.5K | 49 | Across families | 1.034 | 0.989 | 1.101 | 0.94 | Confirmed | 1.096 | 0.384 |
| MS MARCO, unused queries, HNSW | 8.84M | 2000 | Same family, one direction | 0.973 |  |  | 0.95 | Point pass |  | 0.429 |
| MS MARCO, unused queries, HNSW | 8.84M | 2000 | Across families, one direction | 0.961 |  |  | 0.90 | Point pass |  | 0.420 |
<!-- /table:fresh -->

**How to read it.** No fresh result missed a line. Eight of the ten verdicts with an interval are confirmed and two are point passes: LoTTE same family reads 0.989 with a low end of 0.976, against 0.98, and Touché same family reads 1.021 with a low end of 0.976, on only 49 queries. Through the HNSW index, MS MARCO reads 0.973 and 0.961 against 0.95 and 0.90. Those are point estimates only, because that script keeps means, not per-query scores. Four things qualify the result:

- **New queries are not a new corpus.** The MS MARCO and HotpotQA rows are new queries on corpora that shaped the system. LoTTE, FEVER and Touché are the new corpora.
- **On some corpora BM25 does the work.** On FEVER, BM25 and the reranker with no bridge read 0.977, close to the bridge's 0.997 and 0.994. On HotpotQA they read 0.949 and 0.999. **On Touché, no bridge reads 1.096, above the bridge's 1.021 and 1.034: there the bridge lowers quality against BM25 alone**, on 49 queries. Where the map earns its place: LoTTE, 0.838 and 0.818 with no bridge against 0.989 and 0.964, and MS MARCO, 0.862 and 0.870 against 0.971 and 0.956.
- **Re-indexing still gains at the largest sizes.** Direct retention against a re-index with the same BM25 and reranker reads 0.968 and 0.945 on MS MARCO at 8.84M, 0.958 and 0.943 on HotpotQA at 5.23M, and 0.980 to 0.996 on the three new corpora.
- **It's one confirmation, on four model pairs and three seeds.** It shows the frozen system reaching its lines on these data. It doesn't make the lines a guarantee for an unseen corpus.

### Against re-indexing directly

This is the tradeoff a user faces: what you'd gain by re-indexing, if the re-indexed system also had BM25 and the reranker. Absolute `nDCG@5`:

<!-- table:reindex_short -->
| Corpus | Documents | Models | Bridge | Re-index with BM25 | Difference, absolute points | Low | High | Direct retention | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS MARCO | 1M | Same family | 0.722 | 0.728 | -0.5 | -0.9 | -0.2 | 0.993 | 0.988 | 0.997 |
| MS MARCO | 1M | Across families | 0.715 | 0.727 | -1.2 | -1.7 | -0.8 | 0.983 | 0.976 | 0.989 |
| MS MARCO | 8.84M | Same family | 0.419 | 0.428 | -0.9 | -1.5 | -0.4 | 0.978 | 0.966 | 0.989 |
| MS MARCO | 8.84M | Across families | 0.405 | 0.429 | -2.4 | -3.1 | -1.7 | 0.944 | 0.929 | 0.959 |
| NQ | 1M | Same family | 0.772 | 0.785 | -1.2 | -1.7 | -0.8 | 0.985 | 0.979 | 0.990 |
| NQ | 1M | Across families | 0.761 | 0.783 | -2.2 | -2.7 | -1.7 | 0.972 | 0.966 | 0.979 |
| HotpotQA | 1M | Same family | 0.882 | 0.895 | -1.2 | -1.6 | -0.9 | 0.986 | 0.982 | 0.990 |
| HotpotQA | 1M | Across families | 0.866 | 0.887 | -2.1 | -2.5 | -1.7 | 0.976 | 0.972 | 0.981 |
| HotpotQA | 5.23M | Same family | 0.808 | 0.838 | -3.0 | -3.4 | -2.5 | 0.965 | 0.959 | 0.970 |
| HotpotQA | 5.23M | Across families | 0.783 | 0.827 | -4.5 | -5.0 | -3.9 | 0.946 | 0.940 | 0.953 |
<!-- /table:reindex_short -->

### What the bridge adds

Against BM25 and the reranker with no bridge and no old index:

<!-- table:nobridge_short -->
| Corpus | Documents | Models | No bridge, retention | Low | High | Bridge, retention | Bridge adds, absolute points | Low | High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MS MARCO | 1M | Same family | 0.905 | 0.888 | 0.922 | 0.995 | +6.5 | +5.4 | +7.7 |
| MS MARCO | 1M | Across families | 0.903 | 0.887 | 0.920 | 0.982 | +5.7 | +4.6 | +6.9 |
| MS MARCO | 8.84M | Same family | 0.874 | 0.849 | 0.900 | 0.979 | +4.5 | +3.4 | +5.6 |
| MS MARCO | 8.84M | Across families | 0.873 | 0.848 | 0.899 | 0.946 | +3.1 | +2.1 | +4.2 |
| NQ | 1M | Same family | 0.863 | 0.847 | 0.881 | 0.985 | +9.6 | +8.3 | +10.9 |
| NQ | 1M | Across families | 0.870 | 0.853 | 0.888 | 0.979 | +8.5 | +7.2 | +9.7 |
| HotpotQA | 1M | Same family | 0.958 | 0.950 | 0.966 | 0.995 | +3.3 | +2.7 | +3.9 |
| HotpotQA | 1M | Across families | 1.001 | 0.991 | 1.011 | 1.020 | +1.6 | +1.1 | +2.1 |
| HotpotQA | 5.23M | Same family | 0.945 | 0.935 | 0.955 | 0.985 | +3.3 | +2.6 | +4.0 |
| HotpotQA | 5.23M | Across families | 1.000 | 0.988 | 1.013 | 1.009 | +0.7 | +0.0 | +1.4 |
<!-- /table:nobridge_short -->

### Speed

One query, warm, on an idle machine, `bge-large-en-v1.5` index, `Qwen3-Embedding-0.6B` queries, a pool of 60 candidates. Throughput under simultaneous requests isn't measured. The README has the full table, including estimated rows.

<!-- table:speed_short -->
| Documents | Corpus | Documents' length | Search | CPU p50, s | CPU p95, s | CPU queries | GPU p50, s | GPU p95, s | GPU queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100K | MS MARCO | Short | Exact | 10.4 | 12.6 | 30 | 0.16 | 0.17 | 200 |
| 171.3K | TREC-COVID | Long | Exact | 36.3 | 43.3 | 30 | 0.29 | 0.34 | 49 |
| 382.5K | Touché 2020 | Long | Exact | 34.1 | 41.2 | 20 | 0.30 | 0.37 | 48 |
| 1M | MS MARCO | Short | HNSW | 10.6 | 12.4 | 30 | 0.16 | 0.18 | 200 |
| 5.23M | HotpotQA | Medium | Exact | 14.8 | 18.9 | 30 | 0.71 | 0.75 | 100 |
| 8.84M | MS MARCO | Short | Exact | 11.3 | 14.0 | 30 | 1.07 | 1.11 | 100 |
| 8.84M | MS MARCO | Short | HNSW | 10.5 | 13.1 | 30 | 0.19 | 0.21 | 200 |
<!-- /table:speed_short -->

Longer documents take longer in proportion. On this CPU, rerank seconds ≈ 1.91 ms × candidates × tokens per (query, document) pair. A larger pool of 100 candidates reaches the one-minute line on long documents on a CPU.

## 8. When to use it

**Suitable:**

- A new query model has to work on an existing English index now, and you'll decide about re-indexing later.
- The model that built the index is gone.
- A self-hosted, low-traffic system where a re-index would cost more than it's worth, and you can run a reranker.
- Chinese queries into an English index, with a translator feeding BM25. Measured on two Wikipedia question-answering sets only.

**Unsuitable:**

- **You want a better model's quality.** A bridge keeps quality. It doesn't add the new model's first-stage gain. Re-index for that.
- **You can't run a reranker,** or can't keep document text available. First-stage results lose far more, and BM25 needs the text too.
- **High traffic on a CPU.** A query takes 10 to 40 seconds. Simultaneous requests weren't measured.
- **An index not in English, above 100K documents.** There's no BM25 there, and the concern from English results at those sizes hasn't been measured there.
- **Questions that name things, or arguments, if you can drop the old index.** On HotpotQA, BM25 and a reranker alone do most of the work, and on Touché 2020 they did better than the bridge. Try that first.
- **Corpora where relevance isn't lexical,** like citation links, unless you've compared with `lexical=None` on labeled queries.
- **Other languages, other model kinds, or a corpus unlike these,** with no measurements of your own. The quality score won't reliably warn you.

**What is still open:** why names transfer worse on HotpotQA than on NQ; ranking-trained nonlinear maps; other BM25 settings and tokenizers; a second HNSW index on another corpus; a third corpus above 1M; and a label-free signal that works at scale. A real deployment's corpus and query log were never used. Answer quality in a full retrieval-and-generation system was never measured: this project stops at the ranking.

**Repeating the work.** Compact records of the published tables are in [`results/`](results/README.md), with the commands that made them. Embeddings and raw logs stay on the reference machine, and [`data/README.md`](data/README.md) describes them. Milestone reports stay local, by choice.

## 9. Terms

| Term | Meaning |
| --- | --- |
| **Anchor** | One of the few thousand documents the map is fitted on |
| **BM25** | A classic keyword search. It scores documents by the rare query words they contain |
| **Bridge** | The new model's queries, mapped, searching the old model's index |
| **Candidates, pool** | The documents a first search hands the reranker. A pool of 60 means 60 |
| **Confidence interval** | A range for what a number would be with other queries of the same kind. Here, 95%, by a bootstrap over queries |
| **Cross-encoder, reranker** | A model that reads a query and a document together and scores the pair. Accurate and slow |
| **Embedding** | A vector representing a piece of text |
| **Embedding model** | A model that makes embeddings. Vectors from different models can't be compared directly |
| **Exact search** | Comparing the query against every stored vector |
| **Family** | Models trained by the same recipe: `bge-base-en-v1.5` and `bge-large-en-v1.5`. "Across families" is one of them with `Qwen3-Embedding-0.6B` or `harrier-oss-v1-0.6b` |
| **First stage** | The vector search, before reranking |
| **HNSW** | A graph index for fast, approximate nearest-neighbour search. `ef` sets how far it looks |
| **K** | The number of anchors |
| **Mount** | Fitting the map for a new query model |
| **Native** | The old model's own queries on its own index |
| **nDCG@5** | A score from 0 to 1 for the top 5 results. It rewards right documents, more so higher up |
| **p50, p95** | The time half of queries beat, and the time 95% of queries beat. p95 of few queries is near the slowest one seen |
| **Point (absolute)** | 0.01 of `nDCG@5` |
| **Point (retention)** | 0.01 of retention, so 0.97 to 0.98 is one point |
| **Proxy** | Text from a document standing in for a query, usually its title |
| **Re-index** | Re-embedding every document with the new model |
| **Recall@k** | The share of a query's right documents in the top k |
| **Retention** | The bridge's score divided by the weaker of native and re-index. It's relative, and above 1 is possible with BM25 |
| **Seed** | A number setting the random choices: here, which anchors and how they split. It doesn't change the test queries |
| **Stage 1, Stage 2** | The two steps of fitting the map: closed-form, then 50 steps of gradient descent |
