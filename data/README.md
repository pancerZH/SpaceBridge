# Local data index

This folder and `outputs/` hold about 275 GB of embeddings, indexes and results that are not on GitHub. Use this page to find out what a file is and whether you can reuse it, so that work isn't repeated.

Only this page and [`catalog.csv`](catalog.csv) are committed. Figures are from 2026-11-04.

All of it was produced on one machine: Intel Core i5-12600K, Nvidia RTX 5070 Ti, 96 GB of DDR4 RAM.

## Three things to know first

1. **Never rename a cache file.** A name like `0a1d517f2ed648e46e0edec7.npy` is a hash of what went into the file, and the code finds the file by recomputing that hash. Rename it and the code encodes everything again without any error.
2. **You rarely need to look files up to reuse them.** Run a command from the repository root with the same inputs, and it finds what is already on disk.
3. **To learn what a file is, look it up in the catalog:**

   ```bash
   grep 0a1d517f2ed648e46e0edec7 data/catalog.csv
   ```

   That file turns out to be the first 500 MS MARCO test queries, encoded with `bge-large-en`.

## The catalog

`catalog.csv` has 6,885 rows: one per file, except that each BM25 index and the ONNX export get one row per folder. It leaves out the logs, driver scripts and finished-step markers in `outputs/`. It holds no document text and no measured results.

| Column | Meaning |
| --- | --- |
| `area` | Which kind of data: the rows of the next table, plus `results` for `outputs/` |
| `path` | Path from the repository root |
| `bytes` | File size in bytes |
| `modified` | Date the file last changed |
| `rows` | Rows in the array, or records in a results file |
| `dim` | Vector length |
| `dtype` | Number format of the array |
| `model` | The embedding model or reranker |
| `dataset` | The dataset. Several, separated by `;`, when they share the same inputs |
| `role` | What the file holds, explained under [Roles](#roles-of-embedding-files) |
| `detail` | Specifics, such as the number of anchors and the seed |
| `identified_by` | How the row was worked out, explained below |

`identified_by` tells you how far to trust a row:

| Value | Meaning |
| --- | --- |
| `computed` | The hash was recomputed from the inputs and matched the filename, so the identity is certain |
| `name` | Read from a filename that spells out its contents, or from where the file sits |
| empty | Not identified, and `role` says `unidentified` |

Some useful queries:

```bash
# Every file encoded from one corpus's documents
grep -E 'corpus-documents' data/catalog.csv | grep msmarco-full

# Everything cached for one model and dataset
grep 'embeddings/qwen3-0.6b/' data/catalog.csv | grep ',nq-1m,'

# What a results file ran over
grep 'outputs/phase1/m3_full/hp.jsonl' data/catalog.csv
```

To refresh it after new runs (about 20 minutes, no GPU or network, and it changes nothing except the catalog):

```bash
uv run python -u scripts/index_data.py
```

## What is under `data/`

| Area | Location | Files | Size | What it holds | How the name is made |
| --- | --- | --- | --- | --- | --- |
| `embeddings` | `cache/embeddings/<model>/` | 5,025 | 184 GB | Embedding vectors, as float16 | Hash of the model ID, the maximum token length and every input text |
| `hnsw` | `cache/hnsw/` | 4 | 47 GB | Approximate search indexes built with FAISS | `<dataset>_<embedding file>_<dim>d_<settings>.faiss` |
| `datasets` | `cache/datasets/` | 27 | 8 GB | Parsed datasets, including all their text | `<dataset>_<hash of its settings>.pkl` |
| `lexical` | `cache/lexical/` | 49 | 8 GB | BM25 keyword indexes, one folder each | `<dataset>_<hash of its document IDs>/` |
| `onnx` | `cache/onnx/` | 1 folder | 3 GB | `bge-reranker-v2-m3` exported for CPU use, including a smaller int8 version | Model name |
| `rankings` | `cache/rankings/` | 315 | 1 GB | Top 1,000 row IDs per query, from exact search | Hash of the embedding file's name, dimension and length, plus the query vectors |
| `rerank` | `cache/rerank/` | 164 | 0.6 GB | Reranker scores | `<reranker>_L<max length>.npz`, or `.<process ID>.scores` for a log |
| `anchors` | `cache/anchors/` | 694 | 21 MB | Row IDs chosen as calibration anchors by k-means | `<dataset>_<model>_N<rows>_n<allowed rows>_K<anchors>_s<seed>[_tkm].npy` |
| `flores` | `flores/` | 14 | 5 MB | Human-translated sentence pairs, for language calibration | `flores_<languages>_<filter>_v1.json` |
| `translations` | `cache/translations/` | 4 | 0.4 MB | Machine-translated test queries | `<translator>_<hash of the queries>.jsonl` |

In a dataset name, `@` marks a cut: `msmarco-1m-nested@200000` means the first 200,000 rows of `msmarco-1m-nested`, using the same embedding file.

### Roles of embedding files

| Role | Files | What it holds |
| --- | --- | --- |
| `anchor-query-proxies` | 2,579 | Short text from each anchor document (usually its title), encoded as a query |
| `anchor-documents` | 1,312 | Anchor documents encoded as documents, often cut to 128 tokens |
| `language-bank-queries` | 118 | FLORES or Tatoeba sentences encoded as queries |
| `test-queries` | 84 | All the test queries of a dataset |
| `corpus-documents` | 85 | A whole corpus. Seven of these are also the first rows of a larger corpus |
| `test-queries-first-rows` | 51 | The first test queries only, usually a 500-query screening run |
| `log-queries` | 13 | Non-test queries used by `scripts/hotpot_study.py` |
| `language-bank-documents` | 6 | FLORES sentences encoded as documents |
| `corpus-documents-after-first-1m` | 3 | The rows after the first million, from `scripts/encode_suffix.py` |
| `translated-queries` | 2 | Machine-translated test queries |
| `unidentified` | 772 | See [Files not identified](#files-not-identified) |

Terms used above:

- **Anchors:** the few thousand documents used to fit the map from one model's queries to another's.
- **Proxy:** a short piece of text taken from a document to stand in for a real query.
- **Nested corpus:** a corpus ordered so that any first *n* rows are a fair sample, which lets one file serve every size.

### The files that took longest to build

These whole-corpus files are the most expensive to rebuild, so keep them above all others. The hashes below are the start of each filename, under `cache/embeddings/<model>/`.

| Dataset | Rows | `bge-base-en` | `bge-large-en` | `qwen3-0.6b` |
| --- | --- | --- | --- | --- |
| `msmarco-full` | 8,841,823 | `30e8c1fcb1` | `7ccea59fbb` | `41f7a074c2` |
| `hotpotqa-full` | 5,233,329 | `2fc30e1cd9` | `a93d6efd1d` | `acd488f6fb` |
| `hotpotqa-full`, after the first 1M rows | 4,233,329 | `295c6ba6f6` | `b6b605265d` | `a17e7da182` |
| `msmarco-1m` | 1,000,000 | `9fab5f9dcc` | `de4f162f7e` | `dbb6e75c9d` |
| `msmarco-1m-nested` | 1,000,000 | `db0fd56320` | `d2528f1213` | `45c5f4e128` |
| `hotpotqa-1m` | 1,000,000 | `0a5b0f3d47` | `135e4c9e93` | `ef9e329b20` |
| `hotpotqa-1m-nested` | 1,000,000 | `fddf42658a` | `5ab6810c2f` | `f69e03d571` |
| `nq-1m` | 1,000,000 | `2cbbd9ecde` | `7388b0e7fb` | `a2bfb37e25` |
| `nq-1m-nested` | 1,000,000 | `a8ea6df313` | `6bb90ac220` | `f32716ab92` |

`msmarco-1m` also has `harrier-0.6b` (`1f8d46c29a`) and `minilm` (`b7c96f6d53`), and `msmarco-1m-nested` has `harrier-0.6b` (`04423b6d7a`).

Four search indexes sit on top of these:

| File in `cache/hnsw/`, by dataset and settings | Model | Dimensions |
| --- | --- | --- |
| `msmarco-1m`, HNSW (M = 32, efConstruction = 200) | `bge-base-en` | 768 |
| `msmarco-1m`, HNSW (M = 32, efConstruction = 200) | `bge-large-en` | 1,024 |
| `msmarco-1m`, IVF16384 with PQ128 | `bge-large-en` | 1,024 |
| `msmarco-full`, HNSW (M = 32, efConstruction = 200) | `bge-large-en` | 1,024 |

### Files not identified

772 embedding files (5.8 GB, 3% of the embedding bytes) and 12 rankings could not be matched by recomputing their hash. They are still valid: the code uses them whenever the same inputs come up again. Most came from study scripts that build their own inputs.

This table is an inference from each file's date and row count, not a verified fact:

| Modified | Main row counts | Files | Probable source |
| --- | --- | --- | --- |
| 2026-10-30 | 512, 1,024, 2,048, 4,096, 5,183 | 40 | Early Phase 0 runs, before later changes to how anchors and proxies are made |
| 2026-11-01 | 410, 1,024, 2,048, 8,192, 16,384 | 416 | Size-curve and anchor studies (`scale_curve.py`, `scale_anchors_depth.py`). 410 is the 20% validation share of 2,048 anchors |
| 2026-11-02 | 1,024, 2,000, 2,048, 8,192 | 209 | HotpotQA studies (`hotpot_study.py`) |
| 2026-11-03 | 510, 696, 1,024, 1,405, 1,885, 2,000, 8,192 | 107 | Query variants from `hotpot_identity.py` |

If you need disk space, these are the first candidates to delete. Anything deleted is encoded again if a later run asks for it.

## Before you rely on a cached file

The hashes do not cover everything, so a matching filename does not guarantee a matching result:

- **Model and dataset versions are not in the hash.** If a model is updated upstream, the old vectors are reused without warning, mixed with new ones for any new inputs. The versions on this machine are listed below.
- **The device and precision are not in the hash for embeddings or most reranker scores.** A CPU run can reuse scores written by a GPU run. The `-fp32` reranker files are a deliberate exception, written by `scripts/check_reranker_precision.py`.
- **An HNSW index is not checked against its corpus when it is reopened.** Before reusing one, check that its filename names the embedding file you are searching.
- **Paths are relative to the repository root.** Run commands from there, or the caches won't be found.

### Versions on this machine

These are what each download's `refs/main` pointed to on 2026-11-04. That is the latest version seen, not proof of which version produced a given file. The downloads themselves are in the default Hugging Face cache (`~/.cache/huggingface`, about 53 GB), outside this repository.

| Model | Revision |
| --- | --- |
| `BAAI/bge-base-en-v1.5` | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` |
| `BAAI/bge-large-en-v1.5` | `d4aa6901d3a41ba39fb536a557fa166f842b0e09` |
| `BAAI/bge-large-zh-v1.5` | `79e7739b6ab944e86d6171e44d24c997fc1e0116` |
| `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| `google/embeddinggemma-2` | `f9972ac22e696a105c482202646979e33e4b5a1e` |
| `google/madlad400-3b-mt` | `fa184c675da0b5c9e1c8694fccd4e12e2d422094` |
| `Helsinki-NLP/opus-mt-zh-en` | `cf109095479db38d6df799875e34039d4938aaa6` |
| `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| `microsoft/harrier-oss-v1-0.6b` | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` |
| `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `Qwen/Qwen3-Embedding-4B` | `5cf2132abc99cad020ac570b19d031efec650f2b` |
| `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

| Dataset | Revision |
| --- | --- |
| `facebook/belebele` | `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` |
| `google/xquad` | `51adfef1c1287aab1d2d91b5bead9bcfb9c68583` |
| `mteb/arguana` | `6c1bcf74b13dfd823aff056b79d4d93e702f19c7` |
| `mteb/fiqa` | `5e59eeb3a7df6b85882112b747008547c21587ea` |
| `mteb/hotpotqa` | `60b39fafe9b0b81ce6be80c295de49ace5b7a14e` |
| `mteb/msmarco` | `49b09e5f3736c430f448d07150a5c2d425c24ce3` |
| `mteb/nfcorpus` | `52ac3f19d3449632d9f00aab0ad34a110fc03816` |
| `mteb/nq` | `b84726e65fd226125cf7c0cbeeb5c214d49e8187` |
| `mteb/quora` | `7f060c1bdc9b3d5161eecc05aa59921bca4dd185` |
| `mteb/scidocs` | `490848228d0a9ca7a7244f5e77d8fe33e6df6974` |
| `mteb/scifact` | `cf10ab6856b15b0e670ef8ae5dae4e266c12d035` |
| `mteb/T2Retrieval` | `921dd3af6e78d1ae7ee0368aa8d7eaee02c8f08e` |
| `mteb/tatoeba-bitext-mining` | `4f65d2990c55534b6aa84eeb7da78b1f13c97426` |
| `mteb/trec-covid` | `0724773f091de506941c2108e1647af956de07f7` |
| `openlanguagedata/flores_plus` | `1147e2dc59f37fbb5c22fd8478770de02cd399f9` |

`facebook/mlqa` has no `refs/main` recorded here. The pickles in `cache/datasets/` hold the parsed text, so anything already parsed does not need these downloads.

## Other files that aren't on GitHub

| Location | Size | What it holds |
| --- | --- | --- |
| `outputs/` | 25 GB | Every experiment's raw records, logs and fitted maps |
| `docs/reports/` | 0.7 MB | Plans, progress notes and milestone reports. Local by choice; never commit them |
| `quickstart/` | 13 MB | What `examples/quickstart.py` saves |
| `review/` | 0.1 MB | Review comments on the project |

### Finding results in `outputs/`

Each `.jsonl` file holds one record per finished experiment cell. Beside it, `.log` is the run's output and `.done` marks a finished step. These records are the raw evidence behind the published tables; unlike the caches, they can only be replaced by running the experiment again, so **keep them**. `scripts/summarize_*.py` turn them into tables.

Earlier work, at the top level of `outputs/`:

| Location | What it holds |
| --- | --- |
| 111 loose `.jsonl` files (`m2_*`, `m3_*`, `m4_*` and others) | Phase 0 milestones, named after the milestone |
| `closeout/`, `flores/`, `p0end/`, `gate/`, `scale/` | Phase 0 closeout experiments, language studies, final end-to-end checks, and early gate and scale checks |
| `phase1_audit/` | The speed audit and its verification, at the start of Phase 1 |

Phase 1, under `outputs/phase1/`. Each folder's report is in `docs/reports/`. Most folders have their driver script beside them as `<folder>.sh`:

| Stage | Folders | What they hold | Report |
| --- | --- | --- | --- |
| Milestone 1 | `w2_gate`, `w3_encode`, `w3_1m`, `m1_nested`, `m1_grid`, `m1_switch`, `m1_timing`, `w5_grid`, `w5_ann`, `w6_free`, `w6_translate`, `w6_upgrade` | Encoding, the 1M confirmation, curves by size, pool grids, timing, approximate search, translation, source-free fits and model upgrades | `phase1-m1.md` |
| Milestone 2 | `m2_pools`, `m2_lib`, `m2_hotpot`, `m2_ann`, `m2_timing`, `m2_first_look`, `m2_fixes`, `m2_e`, `m2_e_int8`, `m2_e_more`, `m2_e_timing`, `m2_tables` | Candidate pools, library defaults, HotpotQA at 1M, HNSW, timing, the first look at 8.84M, lighter rerankers and tables | `phase1-m2.md` |
| HotpotQA study | `hp_study`, `hp_fixes`, `hp_fixes_arguana`, `hp_identity`, `hp_scale` | Why multi-hop questions miss, and the fixes tried, including BM25. `hp_study/fits/` (9 GB) holds fitted maps | `hotpotqa-followup.md` |
| Milestone 3 | `m3_lib`, `m3_a6`, `m3_b`, `m3_1m`, `m3_encode`, `m3_curves`, `m3_full`, `m3_timing`, `m3_quickstart`, `m3_f`, `m3_tables` | The library with BM25, cross-language checks, end-to-end runs up to 8.84M, timing and the fresh-clone quickstart | `phase1-m3.md` |
| Closeout | `followup`, `closeout`, `final` | Follow-up checks, the closeout decisions, and the final checks at the 160K line | `phase1-closeout.md`, `phase1-final.md` |

`outputs/phase1/m3_quickstart/clone/` (8 GB) is a full fresh checkout with its own Python environment, used to test the quickstart. The catalog skips it. It holds no unique results and can be deleted to save space.
