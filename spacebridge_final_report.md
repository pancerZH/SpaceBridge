# SpaceBridge: final report

**2026-11-04. Phases 0 and 1 are finished: 2026-10-30 to 2026-11-04, on one machine.** How to use the library is in the [README](README.md). The method, every finding in its section and the full tables are in the [design doc](spacebridge_design_and_benchmark.md). This report is the story: what we set out to do, how we tested it, what got in the way, and what we ended up with.

**How to read the numbers.** Retention is the bridged result divided by the weaker of the old model's native search and a full re-index with the new model. Unless a line says otherwise, it's end-to-end `nDCG@5` after `bge-reranker-v2-m3`, the mean over model pairs, 3 seeds, given as Tier 1 then Tier 3. Tier 1 is models of one family (`bge-base-en-v1.5` and `bge-large-en-v1.5`), and Tier 3 crosses families (`bge-large-en-v1.5` with `Qwen3-Embedding-0.6B` or `harrier-oss-v1-0.6b`). **Retention above 1 isn't a gain over native:** with BM25 the bridge has a channel plain native search doesn't. "Strict" gives native and re-index BM25 too.

## 1. The result

**The question:** can a vector index built with one embedding model be searched with a different query model, without re-embedding the corpus?

**The answer: yes, in English, with a cross-encoder reranker and BM25 in its candidate pool.** A map fitted on a few thousand anchors isn't enough on its own at scale. The map, BM25 and the reranker together are.

| Range | Line | Result |
| --- | --- | --- |
| **Best: up to 1M documents,** on MS MARCO, NQ and HotpotQA | 98% and 94% | MS MARCO 0.995, 0.982. NQ 0.985, 0.979. HotpotQA 0.995, 1.020 |
| **Supported: up to 8.84M, exact search** | 95% and 90% | MS MARCO 8.84M 0.979, 0.946. HotpotQA at its whole corpus, 5.23M, 0.985, 1.009 |
| The reference deployment on one HNSW index, MS MARCO 8.84M, ef 500 | 95% and 90% | 0.978, 0.945, one direction per tier |

**Speed:** one query at 8.84M on that HNSW index takes 0.20 s on a GPU and 10.6 s on a CPU on short passages. A CPU stays under a minute at every size at the default budget, on long documents too (36-38 s).

**What it's for: instant compatibility and migration.** Serve a frozen index right away with a new query model, then re-index in the background if the last few points justify it, or if you'd run without a reranker. It isn't a quality upgrade: to get a better model's first-stage gain, you must re-index.

## 2. The initial goals

### Where we started, 2026-10-30

Upgrading an embedding model, querying a large index with a tiny on-device model, or searching an English index in another language normally means re-embedding every document. The first design motivated this with datasets of 10M+ chunks. It rested on a hypothesis: good embedding models organize meaning with largely the same relative geometry, differing mainly by offsets, rotations and scaling, so a small map fitted on calibration anchors could carry queries across without touching the index.

| Goal | Original target |
| --- | --- |
| Tier 1, same family | `Recall@30` ≥ 95%, end-to-end `nDCG@5` ≥ 98% |
| Tier 2, Chinese across families | ≥ 88%, ≥ 96% |
| Tier 3, encoder and decoder-LLM | ≥ 82%, ≥ 94% |
| Tier 4, 384-d against 2560-d | ≥ 80% up, ≥ 85% down, ≥ 92% |
| Tier 5, Chinese queries into an English index | ≥ 78%, ≥ 90% |
| Solver | Under 250 ms |
| Mount | Under 2 s, 4 s or 20 s, by model size |

### How the goals changed, and why

| Date | What changed | Why |
| --- | --- | --- |
| 10-30 | First-stage Tier 1 went from 95% to 88% at K = 2,048 and 90% at K = 4,096, and Tier 3 gained 87% at K = 4,096. **The end-to-end targets didn't change** | A linear map reads 89-92% on FiQA, the hard domain, so 95% was out of reach. The reranker recovers the rest |
| 11-01 | Mount went from seconds to **at most 10% of the tokens a re-index would encode** | Fixed seconds can't hold across corpora and machines. A ratio doesn't depend on the hardware |
| 11-01 | Scale went from 10M+ to **1M as the goal and 8.84M as the stretch and the ceiling**, with a stretch line of 95% and 90% above 1M | 8.84M is the limit of this machine. If a goal didn't fit, we'd optimize the algorithm, not assume more hardware |
| 11-01 | Positioning became **instant compatibility and migration** | At 1M, the compute and quality cases for the bridge were weak |
| 11-01 | No fixed latency budget, but **a CPU and GPU table by size, up to a one-minute line** | It's the honest way to show what "reasonable time" means on your hardware |
| 11-01 | A reranker is required above 200K, and large indexes are judged end to end | First-stage-only use loses far more at 1M |

### What Phase 1 had to deliver

| # | Deliverable | Result |
| --- | --- | --- |
| 1 | A library call that mounts and serves | `freeze`, `mount`, `Searcher`. The real `Searcher` matches the benchmark within 0.02 points at 1M, 0.000 at 5.23M and 8.84M |
| 2 | The 1M goal confirmed | **Met on three corpora,** with BM25. NQ Tier 1, at 0.985, is the narrow one: its CIs straddle 98% |
| 3 | Retention by size up to 8.84M, with two ranges | Done, §1 |
| 4 | CPU and GPU table up to the minute line | Done, §5.3. At the default budget, a CPU doesn't reach the line |
| 5 | A serving path on HNSW with a latency breakdown | Done at 1M and 8.84M, on MS MARCO |
| 6 | Evidence for positioning | Translation, source-free, upgrade, break-even and no-bridge baselines measured |
| 7 | A design doc and README that match | Yes. The quickstart runs from a clean checkout |

## 3. How we experimented

**One machine.** i5-12600K, 94 GB of RAM, RTX 5070 Ti with 16 GB, 284 GB of free NVMe. Timings are on this machine only.

**A ladder of configurations.** Every test pair ran through the same rows, so a gain could be pinned on one change: `0a` re-index and `0b` native as the ceilings, `1` the identity check (a model bridged to itself has to match native), `2` to `8b` adding one piece of the map at a time, `9` the reranker, and `9h` the map with BM25 and the reranker, which is what ships. The benchmark and the library call the same functions, so benchmarks test what ships.

**Corpora chosen to break it.**

| Corpus | Size | Role |
| --- | --- | --- |
| SciFact | 5.2K | Smoke test; flatters results |
| ArguAna, SciDocs | 8.7K, 25.7K | Held out from BM25's choice |
| FiQA | 57.6K | The hard domain, and the authoritative set up to 171K |
| TREC-COVID | 171K | Long documents, 50 queries |
| T2Retrieval, XQuAD, MLQA | | Chinese, and cross-language |
| MS MARCO | 50K, 1M, 8.84M | Short real queries, no titles; the scale test |
| NQ | 50K, 1M | The second scale corpus |
| HotpotQA | 50K, 1M, 5.23M | Multi-hop questions. Held out through Phase 0 and M1, and it's what broke the 1M claim |

**Rules that kept the results honest.**

- **Three seeds and a dataset the decision wasn't made on.** Three early "wins" failed this: a quality-score feature, title proxies for shared titles, and most of the random-anchor gain, +3.0 on 3 MS MARCO pairs and +0.4 on NQ.
- **Reading rules fixed before the results.** The 8.84M go/no-go, HNSW's pass line, and the tests of why HotpotQA loses more were all written down first, and a result that came out "mixed" stayed mixed.
- **Report the limits with the number.** Every range says how many corpora it rests on, and whether exact search or an index.
- **The real path against the benchmark.** The real `Searcher`, with real models, was compared with the benchmark's rows at 1M, 5.23M and 8.84M, and identity is rerun after each engine change.
- **Timings on an idle machine.** A shared GPU moved a top-300 reading from 330 ms to 549 ms.
- **Crash safety.** Encoding, reranker scores and results are saved as they're produced, and rerunning a command continues it. One kill had cost a long run, and waits kept growing. The rules were then tested with a real `kill -9` during the 8.84M HNSW build.

**The sequence.**

| Stage | Question | Answer |
| --- | --- | --- |
| Phase 0, M1-M4 | Does a linear query map work at all, across tiers? | Yes up to 171K, with a reranker |
| Phase 0, closeout | Does it hold at 1M? | K = 2,048 doesn't. K = 8,192 with top-300 reranking does, on one corpus and one seed |
| Phase 1, M1 | Can a library call deliver this, and how fast? | Yes on MS MARCO and NQ, at pool 300, which a CPU can't serve on long documents |
| Phase 1, M2 | Does it hold on a held-out corpus, and how far up? | **No.** HotpotQA misses at 1M, and Tier 1 at 8.84M reads 0.963 |
| HotpotQA study | Why, and can it be fixed? | BM25 in the rerank pool passes, at a fifth of the rerank cost |
| Phase 1, M3 | Does that hold in the library, and at the largest sizes? | Yes: 1M on three corpora, and the stretch line at 5.23M and 8.84M |
| Follow-up, closeout | What's left? | The cause on HotpotQA in part, the sizes between, and HNSW at 8.84M |
| Final checks | Do the last decisions hold where they move a line? | §5.5 |

## 4. The difficulties, and what we did about them

### The map

**A linear map alone misses on a hard domain.** The first full run on FiQA read 0.874 `Recall@30` retention at K = 2,048, under the original Tier 1 target. Most of the loss turned out to be near misses, so the fix wasn't a better map but a second stage. **A reranker over the map's top 60-100 turns a first stage of about 0.8 into about 0.99 end to end.** Bridge quality matters most for recall at 60-100, not for the top-5 order.

**The first quality signal measured nothing.** Own-anchor-document recall read 1.000 for every configuration, including one with 82% real retention, because proxies trivially find their own document. Solver selection and early stopping were being decided by ties. We redefined it as top-10 agreement with the native ranking.

**Agreeing with the teacher isn't retrieving well.** Stage 2 (InfoNCE and KL) improves real retrieval while lowering agreement with the source model, so stopping on agreement threw the gain away. Early stopping quit at step 0 in about 85% of fits. A single 90% agreement floor then broke two ways: a cliff on hard pairs, and selecting on noise. **What ships:** a fixed 50 steps, averaging steps 20-50, under a floor scaled to Stage 1's disagreement.

**Calibration queries matter as much as the solver.** Lead sentences taught Stage 2 the wrong relation on short-question corpora. Unique document titles fixed that (+4.7 to +8.6 points on TREC-COVID). Document-to-document pairs added document-shaped evidence (+1 to +7 points), truncated to 128 tokens to fit the mount budget.

**Score scale failed silently.** `multilingual-e5` compresses its scores, so a fixed Stage-2 temperature cost 20 points on T2Retrieval, and nothing flagged it. The temperature now follows the score spread, and the quality report carries a Stage-2 health flag.

Smaller bugs of the same kind: a K ≤ N/10 cap that cost SciFact 5 points, a τ² factor that made the KL term about 400× too weak, and least-squares shrinkage that needed residual norm matching (+2.9 points).

### Scale

**Retention falls as the index grows, and small subsets hide it.** Everything up to the Phase 0 closeout was at 171K or below. The same queries that retained 0.97 and 0.92 at 50K retained 0.86 and 0.73 at 1M. There's no bug: identity stays at 0.998-1.000, and native holds. A denser index turns the same small query error into more lost ranks. Even 32,768 real query pairs reach only 0.83-0.91, which points to a limit of the linear map; a nonlinear one was never tried. **What helped:** more anchors (K = 2,048 to 8,192 gains +3.8 and +8.7 points at 1M), random instead of k-means at that size (no worse, and the selection step disappears), and a deeper rerank pool, 300 at 1M.

**That fix was too expensive.** The reranker dominates query time on a CPU. Pool 300 at 1M meant 51 s on short passages and 171 s on long documents, over the one-minute line. An int8 export of the reranker is 2.4-3.7× faster with no loss seen on small samples, and it still left long documents at 71 s.

**The goal fell over on the held-out corpus.** MS MARCO and NQ passed at 1M at pool 300, and pool 200 passed on both. Then HotpotQA, which hadn't chosen anything, read 0.956 and 0.898 at pool 300. **The best range fell to 500K, and then, without BM25, HotpotQA missed there too.** Also, 8.84M Tier 1 read 0.963, so the rule stopped that work. Pool 1,000 passed narrowly on one seed (0.981, 0.943, worst pair 0.934), but it's in practice GPU-only.

**The root cause, and the fix that wasn't a parameter.** We asked why rather than raising numbers. The diagnostic showed HotpotQA loses twice MS MARCO's share of native's top 300 (7.7% and 14.0%, against 3.2% and 6.3%), and most of what it loses was easy for native: 57% and 69% had been in native's top 10. Depth, two positives and query length didn't explain it. So we tested what the map loses directly, and **BM25 holds it, because BM25 doesn't use the map.** A quarter of a budget of 60 for BM25 reads 0.994 and 1.018 on HotpotQA at 1M, against 0.879 and 0.794 plain. MS MARCO and NQ, which didn't choose it, confirmed it. At 8.84M Tier 1 went from 0.963 at pool 300 to 0.979 at budget 60. It needs no new model and no change to the index, only the text the reranker already reads.

**That also fixed the cost problem.** With BM25 the pool no longer has to widen with size, so a query at 1M went from 51 s to 11 s on short passages and from 171 s to 37 s on long documents, on a CPU. BM25 itself adds 5 ms at 1M and 30 ms at 8.84M.

**Why HotpotQA, exactly?** Only in part answered, and we stopped when the rules said to. On positives the query names, the map's weakened version of the name is the direct cause: repairing the titles recovers 96-100%, and repairing everything else does little. Repairing one title pushes out the other positive, so a fix has to cover both. No query-side fix works, even given the true titles. The low-variance "tail" theory was rejected. Rare words, name mentions and query length each fail to explain why names transfer worse there than on NQ, and that question stays open. We also learned that cosine misleads as a measure of a map: fits on real queries raise it and lower retrieval.

**HotpotQA's pass says little about the map.** With no bridge at all, BM25 and the reranker read 0.958 and 1.001 at 1M, and 0.945 and 1.000 at 5.23M. On MS MARCO and NQ the bridge adds far more: at 1M, 0.909, 0.906 and 0.863, 0.870 become 0.995, 0.982 and 0.985, 0.979, and at 8.84M no bridge misses both lines at 0.876 and 0.874. So the reports always state this next to the HotpotQA figure.

**BM25 isn't free.** SciDocs, where relevance is citation links, goes from 1.013 and 1.018 to 0.984 and 0.970, and FiQA and TREC-COVID lose 0.9 to 2.2 points. All still pass. SciFact and ArguAna gain 0.9 to 2.0. Nothing at mount tells these corpora apart, and without BM25 HotpotQA misses at every size from 10K up. That's a small sure loss against a large one, so **BM25 is on at every size,** as a judgment, stated as one.

**The rules around K kept moving.** K = 8,192 on short passages costs more than the mount budget below about 150K, so K = 4,096 filled the gap and cleared misses at 100K and 150K. Real mounts then read 9.9-11.0% at 70K, against an estimate of 9.2%, so we moved it to 80K. Once BM25 was in, a fixed-in-advance test showed K = 4,096 no longer earned its budget (at most +0.37 points at Tier 3, and no corpus in all 6 cells). **It's gone:** K = 2,048 up to 160K, K = 8,192 above. The cost of that simplification falls on a bridge without BM25 (§6).

**Serving 8.84M on one machine.**

- **Encoding took 10 h.** Padding was 31-37% of what the GPU processed, and raising the batch size made it slower. Sorting by exact token count under a per-model token budget gave ×1.24-1.63, about 6.8 h for three models, and results didn't move beyond the noise of re-encoding.
- **16 GB of VRAM.** Exact search at 8.84M runs out without `expandable_segments`, which `scripts/steps.sh` now sets. Vectors at that size don't fit, so exact search there stays on the CPU at about 0.9 s.
- **HNSW.** A GPU graph index (cuVS) couldn't be installed under the project's dependency rule, so it has no result. `faiss-cpu` built the 38.6 GB index in about 1.8 h on 12 threads, far under the 3-4 h expected. Search takes 3-7 ms instead of 0.9 s, and through it Tier 3 loses only 0.06 points against exact search.
- **The kill test found a real bug.** A crash during a save would have left a temporary copy as large as the index, up to 39 GB. A rerun now removes it.

### Positioning

**The bridge isn't a quality upgrade, and whether it saves compute depends.** Against a re-indexed system with BM25 too, the bridge gives up 0.6 to 2.5 points at 1M, and 1.9 to 5.5 at the largest sizes. Our first sentence, "not a compute saving", turned out only half right when checked against the break-even:

- **If the re-indexed system would rerank too,** the bridge adds only BM25 and the projection, so re-indexing pays back after 19K to 277K queries at 1M on a GPU, and 331K to 12.9M on a CPU. A low-traffic deployment saves the re-index.
- **If it wouldn't rerank,** re-indexing pays back within 476 to 18K queries, but you get less: 0.683, 0.651 and 0.811 at 1M on MS MARCO, NQ and HotpotQA, against 0.722, 0.772 and 0.882 for the bridge with BM25 and the reranker (absolute `nDCG@5`, Tier 1).

**Translation beat the bridge across languages.** The 7% baseline without a bridge was never the real alternative. MADLAD-400 translating into the old model led the best bridge by 2.5-4 points at pool 60, but that needs the old model alive. **The fix combined the two:** a translator feeding BM25, with the bridge doing the dense search. It gains +5.0 to +7.5 points over the bridge alone, and matches the old model given BM25 too (0.861, 0.733 against 0.862, 0.728). It's `Searcher(translate=)`, off by default, measured Chinese to English only.

**The old model may be gone.** The engine originally couldn't run without it. A source-free fit from stored vectors and the new model alone now holds: 0.997 and 1.020 on HotpotQA at 1M, 0.997 and 0.982 on MS MARCO, at budget 100, on one pair per tier and 500 queries.

### The quality score

**Label-free quality prediction never became reliable.** The score predicts within about ±4-5 points on small corpora, and over-predicts by 8-27 at 1M. It can't see a failed Stage 2, cross-language shift or BM25. Top-30 overlap with native on real queries correlates far better (+0.90) but needs the old model and a query log. **At the end the grade was even spending in the wrong place:** across 21 pairs at 1M and above, its rank correlation with what a wider pool gains is -0.01. So routing stayed advisory, the size and health checks do the warning, and with BM25 on the grade no longer widens the pool.

### Process

**Claims failed when checked, including ours.** "The 1M goal is met" was withdrawn in M2 when HotpotQA came in, and "pool 300 is required" became 200 once a finer grid existed, then was overtaken by BM25. "Best range 500K" didn't survive its own measurement. A first HotpotQA comparison mixed K values and reversed a result. The "range problem" theory and the amplifying fix were each written down as predictions and then rejected. Reading tables against the raw records, and running what the README says, caught most of these. For this report, the tables were regenerated from the records and matched byte for byte, and the headline numbers were recomputed from the raw records independently of the scripts that made them.

**Long agent runs need guards.** Drivers printed "DONE" whatever their steps did. Timing guards could never pass, or passed during a pause. An unanchored `pgrep -f` killed its own shell. A running bash script was edited in place. Each is now a rule: a driver prints "DONE" only on success, timings wait for an empty GPU, and drivers aren't edited while running.

## 5. What we ended up with

### 5.1 The engine

| Part | What ships |
| --- | --- |
| Anchors | K = 2,048 k-means medoids up to 160K documents; K = 8,192 uniform random above |
| Calibration queries | The document title if it has at least 4 tokens and no other anchor shares it, otherwise title and lead sentence |
| Stage 1 | Centered affine map, mean restore, residual norm matching; Stiefel or ridge, picked by agreement with the teacher |
| Document pairs | On for query models up to about 0.6B, anchor docs at 128 tokens; off for the 4B class |
| Stage 2 | InfoNCE and KL, 50 steps, average of steps 20-50 under a scaled agreement floor; temperature follows the score spread |
| Retired model | `freeze(index, None)`: a source-free fit from stored vectors |
| **BM25** | On at any size for an English index with a reranker: k1 = 0.9, b = 0.4, a quarter of the pool, no score fusion |
| **Reranker** | `bge-reranker-v2-m3`. Required above 200K documents |
| **Budget** | 60. 100 for a source-free fit or a Stage-2 health flag. With BM25 off: 60, or 100 for grade C or D, a health flag or a source-free fit, and 300 above 200K |
| Search | Exact up to 200K. HNSW above, M = 32, efConstruction = 200, ef 500 |
| Other languages | A language shift per registered query language (Δμ); a translator feeding BM25 if you pass one |

### 5.2 Retention by size

Budget 60, 3 seeds. Up to 500K is 500 queries, which moves a reading by about ±1.5 points; 1M and above is 2,000.

| Corpus | Size | With BM25 | Strict | No bridge | Without BM25, pool 300 | Absolute `nDCG@5`, with BM25 |
| --- | --- | --- | --- | --- | --- | --- |
| MS MARCO | 1M | 0.995, 0.982 | 0.994, 0.983 | 0.909, 0.906 | 0.986, 0.957 | 0.722, 0.715 |
| NQ | 1M | 0.985, 0.979 | 0.985, 0.975 | 0.863, 0.870 | 0.989, 0.972 | 0.772, 0.761 |
| HotpotQA | 1M | 0.995, 1.020 | 0.989, 0.988 | 0.958, 1.001 | 0.956, 0.898 | 0.882, 0.866 |
| MS MARCO | 2M | 0.986, 0.972 | 0.985, 0.973 | 0.895, 0.897 | | 0.632, 0.622 |
| HotpotQA | 2M | 0.994, 1.020 | 0.982, 0.981 | 0.951, 0.999 | | 0.857, 0.837 |
| MS MARCO | 4M | 0.985, 0.962 | 0.982, 0.958 | 0.881, 0.883 | | 0.530, 0.517 |
| HotpotQA | 5.23M | 0.985, 1.009 | 0.967, 0.961 | 0.945, 1.000 | 0.886, 0.798 | 0.808, 0.783 |
| MS MARCO | 8.84M | 0.979, 0.946 | 0.981, 0.945 | 0.876, 0.874 | 0.963, 0.895 | 0.419, 0.405 |

**Absolute quality falls with size for everyone:** at 8.84M, native and re-index read 0.429 too. Below 1M no corpus misses with BM25 on. The lowest is SciDocs at 0.984 and 0.970.

### 5.3 Speed

Seconds for one query, p50 (p95), through `Searcher`, on 10 CPU threads or an RTX 5070 Ti, budget 60. Retention is the hardest corpus measured at that size. Short is MS MARCO, about 95 tokens per pair; long is TREC-COVID, about 320. The full table is in the README.

| Size | K | Tier 1 | Tier 3 | CPU, short | CPU, long | GPU, short | GPU, long |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.2K | 2,048 | 1.008 (SciFact) | 1.008 (SciFact) | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 25.7K | 2,048 | 0.984 (SciDocs) | 0.970 (SciDocs) | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 100K | 2,048 | 0.996 (NQ) | 0.996 (MS MARCO) | 11.0 (12.5) | 36.8 (41.4) | 0.15 (0.16) | 0.30 (0.34) |
| 160K | 2,048 | 0.997 (HotpotQA) | 0.994 (NQ) | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 160K + 1 | 8,192 | 0.997 (NQ) | 0.998 (MS MARCO) | 11.1 (12.5) | 36.9 (41.4) | 0.15 (0.17) | 0.30 (0.34) |
| 1M | 8,192 | 0.985 (NQ) | 0.979 (NQ) | 11.0 (12.5) | 36.8 (41.4) | 0.16 (0.17) | 0.31 (0.34) |
| 4M | 8,192 | 0.985 (MS MARCO) | 0.962 (MS MARCO) | 10.9 (13.5) | 37.3 (41.8) | 0.55 (0.64) | 0.72 (0.76) |
| 5.23M | 8,192 | 0.985 (HotpotQA) | 1.009 (HotpotQA) | 11.1 (13.7) | 37.4 (41.9) | 0.68 (0.77) | 0.86 (0.89) |
| 8.84M | 8,192 | 0.979 (MS MARCO) | 0.946 (MS MARCO) | 11.5 (14.0) | 37.8 (42.3) | 1.07 (1.16) | 1.24 (1.27) |
| 8.84M, HNSW | 8,192 | 0.978 (MS MARCO) | 0.945 (MS MARCO) | 10.6 (13.2) | 36.9 (41.5) | 0.20 (0.22) | 0.36 (0.40) |

The 8.84M rows are 20 queries each. 2M, 4M and 5.23M add search and BM25 at their size to those runs, by interpolation. Above 171K the long column combines TREC-COVID's rerank time with pools from short-passage corpora. **Budget 100** takes 17.5 s short and 60.6 s long on a CPU. **The int8 reranker** isn't in the library; it takes 3.5 s short and 13 s long, with quality checked only on 50-100 queries per corpus.

### 5.4 Should you use it?

- **Use it** when a new query model has to work on an existing index now, or when the model that built it is gone.
- **Re-index** if you need the last few points, if you'd run without a reranker, or if you want a better model's first-stage gain.
- **On questions that name things,** try BM25 and a reranker with no bridge first, if your old index could be dropped.
- **Compare with `lexical=None`** if you have labeled queries and relevance isn't lexical.

### 5.5 The last decisions

The closeout left five decisions, each with a recommendation. All five were accepted on 2026-11-04 and are in the library and the docs.

| Decision | What it means |
| --- | --- |
| **BM25's size line is gone** | On at any size, for an English index with a reranker. By the rule, the lowest passing size is the smallest corpus measured, 5,183. HotpotQA goes from missing below 50K (0.968, 0.937 at 10K) to passing, +3.6 to +13.0 points. SciDocs loses 2.8 and 4.7 and still passes. Held out, SciDocs loses and SciFact and ArguAna gain, so this rests more on HotpotQA's misses than on held-out evidence |
| **K = 8,192 starts at 160K** | So every pair measured stays inside the mount budget |
| **K = 4,096 is gone** | It doesn't earn its budget with BM25 on. The mount halves from 80K to 150K, and falls to about a quarter from 150K to 160K. A bridge with BM25 off pays for that (§6), and can pass `K=4096` to `freeze` |
| **The grade doesn't widen the budget with BM25 on** | 60 for all, and 100 for a source-free fit or a Stage-2 health flag. HotpotQA Tier 3 gives up 0.3 to 0.7 points of retention, 0.9 to 1.9 of absolute `nDCG@5`, and still passes by 7.5 to 12.5 points |
| **The supported range stays "exact search"** | With one sentence added: one HNSW index on MS MARCO reads 0.978 and 0.945 at ef 500. A second index on another corpus would be needed to drop those words |

**Checked before merging, since nothing had been measured between 150K and 171K.** All pass (`scripts/final_checks.sh`):

- **Mount budget, real mounts at 160,001 MS MARCO documents:** 8.8% to 9.8% of re-index tokens, across 4 pairs, against 9.4% to 10.5% at 150,001. That replaces the estimate behind the decision.
- **Both sides of the line,** with BM25, 500 queries, 3 seeds: K = 2,048 at 160,000 reads 1.002, 0.996 on MS MARCO, 0.999, 0.994 on NQ and 0.997, 1.015 on HotpotQA. K = 8,192 at 160,001 reads 1.003, 0.998, then 0.997, 1.000, then 1.003, 1.018. No step at the switch beyond +0.6 points on the mean. With BM25 off the step is real: MS MARCO Tier 3 0.905 and 0.952.
- **The real `Searcher` below BM25's old line,** on HotpotQA's 50K set, reads +0.000 points against row `9h`, 200 queries, at K = 2,048, grade B, budget 60, with a saved `LexicalIndex`. `lexical="auto"` at small sizes has a unit test, and the quickstart runs it on SciFact, where BM25 builds in 0.3 s at 5,183 documents and one query takes 0.39 s on a GPU.

## 6. Limits

- **English, on these model pairs.** Tiers 1 and 3 only above 171K. Tiers 2, 4 and 5 aren't measured at 1M. `all-MiniLM-L6-v2` as the old index reads 0.981 and 0.975 at 1M; other kinds of pair are untested above 171K.
- **Above 1M the evidence is thin.** One corpus per size, nothing held out from the budget, NQ not measured, and one HNSW index on one corpus, one direction per tier.
- **NQ Tier 1 at 1M is a narrow pass.** Its CIs straddle 98%. Budget 100 reads 0.991.
- **With BM25 off, K = 2,048 now misses at 100K-160K at pool 60:** on MS MARCO 0.933, 0.918 and 0.905 at Tier 3 at 100K, 150K and 160K, against 0.947 and 0.940 at K = 4,096 at the first two, and on NQ 0.938 and 0.936 at 150K and 160K, against 0.961 at 150K. **An index not in English has no BM25,** so the same applies to it, and that case isn't measured at these sizes. Pass `K=4096` to `freeze` there.
- **HotpotQA misses with BM25 off** at every size from 10K up.
- **Your corpus may be harder than these,** and the quality score won't reliably warn you. If you have labeled queries, measure.
- **Evidence is benchmarks on one machine.** No real deployment's corpus or query log has been used.

## 7. What's open

None of these block anything.

- Why names transfer worse on HotpotQA than on NQ.
- `nq-full`, a third corpus above 1M, and a second HNSW index.
- An int8 reranker in the library: first a paired check on 500 queries per corpus.
- A nonlinear map, never tried: the linear one stalls at cosine 0.76-0.79, even on 32K real pairs.
- BM25's settings and other tokenizers; cross-language beyond Chinese to English.
- A better quality signal: real-query overlap is the lead.
- Designed and not built: Operator D, query language detection, `.slice` packaging and storage tiering.

## 8. Where everything is

| What | Where |
| --- | --- |
| Library and benchmark harness | `src/spacebridge/`, `src/spacebridge_bench/`; `uv run pytest` |
| Using it | [README](README.md), `examples/quickstart.py` |
| Method and every finding | [design doc](spacebridge_design_and_benchmark.md) |
| Studies and table scripts | `scripts/`; `scripts/final_checks.sh` for the last checks |
| Milestone reports and raw records | `docs/reports/` and `outputs/` on the reference machine, git-ignored, so not in this repository |
