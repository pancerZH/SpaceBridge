# Decision ledger

Which data informed which choice, and what confirmed it afterwards. "Held out" isn't a permanent property of a dataset: a corpus held out of one decision often informed a later one, so this table states it decision by decision.

The evidence is why the results before the [fresh confirmation](fresh_confirmation_plan.md) are exploratory: decisions were made after looking at those benchmarks, and on the same queries they were later reported on.

**Original targets and the criteria that replaced them** are the first table. **"Rule fixed in advance"** means a reading rule was written down before that result existed, in the milestone reports on the reference machine (`docs/reports/`, kept local). Their dates can be checked against those files, not from this repository alone.

## Targets

| Date | Original | Revised to | Why | Data that showed it |
| --- | --- | --- | --- | --- |
| 10-30 | First-stage same family at least 95% | 88% at K = 2,048, 90% at K = 4,096. End-to-end targets unchanged | A linear map reads 89% to 92% on the hard corpus | FiQA test queries |
| 10-30 | First-stage across families at least 82% | The same at K = 2,048, 87% at K = 4,096 | As above | FiQA test queries |
| 11-01 | Mount in 2, 4 or 20 s by model size, then 5, 10 or 30 s | At most 10% of re-index tokens | Seconds can't hold across corpora and machines | Mount timings on four corpora |
| 11-01 | 10M+ documents | 1M as the goal, 8.84M as the ceiling. A lower stretch line of 95% and 90% above 1M | What this machine can do | MS MARCO at 1M |
| 11-01 | No line for first stage at K = 8,192 | Judged end to end, recall reported | First-stage retention falls with size | MS MARCO at 1M |
| 11-03 | Positioned as re-index "if traffic justifies it" | "If the last few points justify it, or if you'd run without a reranker" | Break-even estimates | Timing and encoding rates |

## Decisions

| Date | Decision | Data used to choose | Rule fixed in advance? | Later confirmation, and on what | That confirmation's limits |
| --- | --- | --- | --- | --- | --- |
| 10-30 | Reranker over the map's top 60 to 100 | FiQA, SciFact | No | HotpotQA 50K (11-01), TREC-COVID | Same queries as later reports |
| 10-31 | Document pairs on | Six BEIR sets, XQuAD | No | HotpotQA 50K: +0.3 to +0.6 points, 28 of 36 cells. Same-family alone was within noise | Then HotpotQA was reused |
| 10-31 | Language shift registered at mount | XQuAD, Chinese | No | MLQA, XQuAD's other languages | One corpus family. XQuAD's languages share questions and contexts |
| 10-31 | Title proxies | TREC-COVID | No | SciDocs, NQ, HotpotQA, ArguAna | Same benchmarks later |
| 10-31 | Unique titles only | XQuAD, ArguAna | No | ArguAna neutral. XQuAD -2.1 ± 1.5 on one model | Decision sets |
| 10-31 | Score is agreement alone | Six BEIR sets, leave-one-dataset-out | No | Quora and HotpotQA: other features failed there | The same labels chose the feature and the regularization |
| 10-31 | Advisory routing, pool at least 60 | FiQA end to end | No | | The grade's later fate is below |
| 10-31 | Second-stage selection: late average under a scaled floor | T2Retrieval, TREC-COVID test queries, scored checkpoint by checkpoint | No | HotpotQA +0.0, SciDocs -0.0, FiQA with an mE5 index +0.4 | The rule was chosen by looking at test queries |
| 10-31 | Temperature rule | T2Retrieval | No | FiQA with an mE5 index: +2.1 to +5.0, 12 of 12 cells | One other corpus |
| 11-01 | Random anchors above about 200K | MS MARCO 1M | No | NQ 1M: +0.4, a tie at K = 8,192 | "No worse, and no selection step", rather than better |
| 11-01 | Anchor documents at 128 tokens | FiQA, TREC-COVID | No | HotpotQA +0.0 | Long documents do lose up to 0.8 points |
| 11-02 | Pool 300 above 200K with BM25 off | MS MARCO and NQ 1M, then HotpotQA 1M | Yes: the size and corpus rule | HotpotQA missed. The default stayed 300 with a stated miss | |
| 11-02 | Stop 8.84M work if retention is under 0.97 | First look, MS MARCO 8.84M | Yes | It read 0.963, and work stopped until BM25 | |
| 11-02 | BM25 in the pool, budget 60 | **HotpotQA 1M**, then MS MARCO and NQ 1M | Reading rules, yes. Choosing the fix came after the miss | MS MARCO and NQ pass at 1M, but they then also helped set the default. FiQA and TREC-COVID lost 0.9 to 2.2 points | All three corpora since reused |
| 11-03 | K = 4,096 from 80K | MS MARCO, NQ, HotpotQA at 75K to 150K; real mounts | Mount line, yes | Removed the next day | |
| 11-03 | HNSW at 8.84M | MS MARCO, one index | Yes: pass line written first | Fresh queries in this round, §9.7 of the design doc | One index, one corpus |
| 11-03 | Translator feeding BM25 | XQuAD and MLQA, Chinese | Yes: at least 2 points on both, then 3 seeds and a second model | The same two sets | Chinese to English only |
| 11-04 | BM25 at every size | HotpotQA below 50K, and the five corpora that hadn't chosen it, which split | Yes: reading rule first | | A judgment, stated as one. SciDocs, FiQA and TREC-COVID lose |
| 11-04 | K = 2,048 up to 160,000, K = 8,192 above | Real mounts at 150,001 and 160,001, MS MARCO. Both sides on three corpora, 500 queries | Yes | | The prefix sets reuse MS MARCO, NQ, HotpotQA queries |
| 11-04 | K = 4,096 removed | MS MARCO, NQ, HotpotQA | Yes | | A bridge with BM25 off pays for it |
| 11-04 | Grade doesn't widen the pool with BM25 on | 21 pairs at 1M and above | No | | Three corpora |
| 11-05 | No default changes. Final system frozen | | **The whole plan was fixed in advance** | Design doc §9.7 | Reports what's in it |

## Corpus by corpus

| Corpus | Used for decisions | So it is not evidence for |
| --- | --- | --- |
| FiQA | Early defaults, stage selection, routing | The early defaults. It's unused for BM25's choice |
| TREC-COVID | Title proxies, stage selection | Those choices. It's unused for BM25's choice |
| XQuAD Chinese | Language defaults, translator rule | Those choices |
| T2Retrieval | Temperature rule, stage selection | Those choices |
| HotpotQA | Held out until 11-02, then BM25's choice and the K rules | BM25 and K. Its 2,000 queries at 1M and below are reused everywhere |
| MS MARCO | Held out until 11-01, then K, random anchors, pool and BM25 confirmation | K, anchors and pool. Its first 500 queries are in every smaller run |
| NQ | Used in 11-01's scale findings, then K rules | K. Its 3,452 test queries were all read at 1M. It has no fresh queries |
| SciDocs, ArguAna, SciFact | BM25's line reading | Whether BM25 belongs at every size. SciFact is too small to be evidence for much |
| Fresh queries (MS MARCO, HotpotQA), LoTTE, FEVER, Touché | Nothing | They're what §9.7 reads |
