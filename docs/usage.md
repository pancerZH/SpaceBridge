# Using SpaceBridge

This page covers installing it, what it expects from you, deploying it, and how to read what it reports. The [README](../README.md) says what to expect from it, and the [design doc](../spacebridge_design_and_benchmark.md) says how it works.

## Install

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or later. All results were produced on Linux with Python 3.12.3. Other operating systems and Python versions are untested.

```bash
git clone https://github.com/pancerZH/SpaceBridge.git
cd SpaceBridge
```

Pick the installation you need. Each of these was run from a clean clone on Linux.

| You want | Command | Adds |
| --- | --- | --- |
| The library, with your own encoder and reranker objects | `uv sync --no-default-groups` | NumPy, SciPy, scikit-learn, PyTorch |
| Real models from Hugging Face, as the examples use them | `uv sync --no-default-groups --extra models --extra banks` | sentence-transformers, Transformers, Hugging Face datasets |
| An HNSW index too | add `--extra hnsw` | FAISS (CPU) |
| The benchmark harness and tests | `uv sync` | Everything the benchmark needs, and pytest |

- **Disk:** the Python environment is about 7 GB, because `pyproject.toml` pins PyTorch's CUDA 12.8 wheels. There is no CPU-only install path, but these wheels also run on a CPU.
- **Downloads:** the first run of `examples/quickstart.py` downloads about 2.5 GB of models and a few MB of data.
- **Network:** only needed for downloads. The English examples use the `banks` extra only for its dataset loader.
- **Dependency dates:** `exclude-newer` in `pyproject.toml` is a fixed date, so dependencies resolve as of then. It doesn't move on its own.

**Device:** models and solvers take `device="cuda"` by default and fall back to the CPU if no GPU is visible. To force the CPU, pass `device="cpu"` or run with `CUDA_VISIBLE_DEVICES=` set to nothing.

## What it expects from your index

| Input | Required form | If it's wrong |
| --- | --- | --- |
| `vectors` | Shape `(documents, dimensions)`. A NumPy array, a memory map, or anything with `__len__` and `[row_ids]` | Wrong shape raises `ValueError` |
| Vector scale | **L2-normalized.** All search is by inner product, so this is cosine similarity | Non-finite values raise `ValueError`. Other lengths give a warning, since results then are unmeasured |
| `texts` | One string per vector, in the same row order | Another length raises `ValueError`. A list reordered but equally long is caught when the artifacts are served, on a sample of rows |
| `titles` | The same, or `None`. Empty strings are fine | Another length raises `ValueError` |
| Size | At least 50 documents, and at least 20 anchors if you set `K` | Raises `ValueError`. Nothing below 5,183 documents was measured |
| Language | English for BM25. Set `src_lang` in `freeze` if it isn't, and BM25 turns itself off | See [Languages](#languages) |

**External document IDs:** SpaceBridge works in row numbers. `result.ids` are rows, so keep your own map from row to document ID.

**Your own storage:** instead of `Index`, pass anything with `__len__`, `vectors(ids)`, `texts(ids)` and `titles(ids)`.

**Encoders.** A query model must be given its own prompts, because the wrong prompt silently lowers quality. Plain `SentenceTransformerEncoder("name")` sends text with no prompt:

```python
spacebridge.SentenceTransformerEncoder(
    "BAAI/bge-large-en-v1.5", query_template="Represent this sentence for searching relevant passages: {text}",
)
spacebridge.SentenceTransformerEncoder(
    "Qwen/Qwen3-Embedding-0.6B", query_template="Instruct: Given a question, retrieve passages that answer it\nQuery:{text}",
    processor_kwargs={"padding_side": "left"}, max_seq_length=512,
)
spacebridge.SentenceTransformerEncoder(
    "intfloat/multilingual-e5-large", query_template="query: {text}", doc_template="passage: {body}",
)
```

To use a model not on Hugging Face, write your own object with `size_class`, `encode_queries(texts)` and `encode_docs(titles, texts, max_tokens=None)`, returning L2-normalized float32 rows. Give it an `identity` property, a dict of the model and its settings, and a bridge will refuse to serve with a different one.

## Deploying: build once, serve from a new process

[`examples/deploy.py`](../examples/deploy.py) is the runnable version of this section. Replace its `load_corpus` with your data.

| Step | How often | Needs | Saves |
| --- | --- | --- | --- |
| `freeze(index, old_model)` | Once per index | The old model if you still have it, otherwise `None` | Calibration data, `bundle.save(path)` |
| `mount(bundle, index, new_model)` | Once per new query model | The new model | The fitted map, `bridge.save(path)` |
| `LexicalIndex.build(index)` | Once per corpus | Document text | BM25 index, `.save(path)` |
| `build_hnsw(vectors, path)` | Once per set of vectors | The stored vectors | The graph file |

**What serving needs afterwards:** the bridge, the new query model, the reranker, the BM25 index, the search backend, and the document text. The HNSW graph replaces the vectors for search, but not the text. **The old model is not needed** once `freeze` has run.

```python
index = spacebridge.Index(vectors, texts, titles)
searcher = spacebridge.Searcher(
    spacebridge.SpaceBridge.load("bridge/"),
    new_model,
    spacebridge.HnswBackend("index.faiss", ef_search=500),   # or spacebridge.ExactBackend(vectors, device="cuda")
    index,
    spacebridge.CrossEncoderReranker(),
    lexical=spacebridge.LexicalIndex.load("lexical/"),
)
result = searcher.search("your query", k=5)    # result.ids, .scores, .texts, .confidence, .pool, .timing_ms
```

**HNSW:**

- `ef_search` sets how far the graph search looks: higher is slower and finds slightly more. 500 is what was measured.
- Above 200,000 documents, use HNSW or `ExactBackend(vectors, device="cpu")`. The GPU backend raises `ValueError` if the vectors won't fit.
- `build_hnsw` saves as it grows and continues after a crash, if you rerun it **with the same vectors and settings**. Other vectors or settings at the same path raise `ValueError`. Only one process may build at a path.
- Sizes and times at 8.84M documents are in the README.

**Old model gone?** `spacebridge.freeze(index, None)` fits from the stored vectors and the new model alone. An index whose old model is alive gets a bundle holding that model's embeddings, so `mount` still doesn't need the old model. A `freeze(index, None)` bundle has no quality score, `result.confidence` is NaN, and it gets a candidate pool of 100.

**What protects you from mismatches.** Artifacts remember what they were made for, and `mount` and `Searcher` check them, on a sample of 32 rows. A replaced or reordered corpus raises `ValueError`; a single edited row outside the sample doesn't. Artifacts saved before these checks existed are checked by the size and vector width they recorded, with a warning that their content and row order are unverified. For a custom backend, add a `check_index(index)` method, or the backend goes unchecked.

| Protection | What it covers | What it doesn't |
| --- | --- | --- |
| Checksums in `manifest.json` | A file changed or cut short | Whether it belongs with this index |
| Sampled fingerprints | Another corpus, row order or re-embedding | An edit outside the sample |
| Recorded size and width, for artifacts saved before fingerprints | Another size or vector width | Another corpus of the same size and width, or a reordering |
| Encoder identity | Another model, prompt or length, if the encoder says | Encoders with no `identity` |
| Atomic saves | A crash during a save | Two writers to one bundle, or swapping files under a running server |

**Threads.** `SentenceTransformerEncoder` runs one encode at a time per object, so sharing one is safe but serialized. Simultaneous requests were not measured at all: every time on this page and the README is one query on an idle machine.

## How to read what it reports

These three numbers measure different things, so don't mix them up.

| Number | Where | Means | Scale | Doesn't mean |
| --- | --- | --- | --- | --- |
| **Retention** in the README's tables | Benchmarks only: it needs labeled queries | Final `nDCG@5` with the bridge, divided by the weaker of native search and a re-index | About 0 to above 1 | That you'd get this on your corpus |
| **`report.score`** | `bridge.quality_report()` | A prediction of **first-stage** `Recall@30` retention, in percent, from calibration data alone. It comes with a band and a grade (A: 95 or more, B: 85 to under 95, C: 70 to under 85, D: under 70) | Not clamped: it can read above 100, since retention can. `None` with no old model | The final quality. It over-predicts above about 171K documents, and can't see BM25 or a failed second stage |
| **`result.confidence`** | Every query | Predicted **top-10 agreement with the old model's ranking** for this query, from how close it is to the calibration data | 0 to 1. NaN with no old model | That the answer is relevant: 0.84 is not an 84% chance of being right |

**Trust the score only as a rough signal.** It predicted within about ±4 retention points on the six small corpora it was fitted on, and over-predicted by 8 to 27 at 1M documents. If you have labeled queries, measure instead.

## How many candidates get reranked

`pool` is the number of candidates sent to the reranker, and it drives query time. If you don't set it:

| Your setup | BM25 | Pool |
| --- | --- | --- |
| English index, a reranker, old model known, no warnings | On | 60 |
| The same, with no old model, or a Stage-2 health flag | On | 100 |
| A `LexicalIndex` passed in, whatever the language | On | As the two rows above |
| No reranker, or an index not in English, or `lexical=None`, up to 200K documents | Off | 60 for grade A or B; 100 for C, D, no old model, or a health flag |
| The same, above 200K documents | Off | 300, with a warning |
| A query in another language, with no `translate=` | Skipped for that query | Its language's own advice if it has one, otherwise the row above |
| A query in another language, with `translate=` | On | As the first two rows |
| `pool=` passed to `Searcher` or `search` | No change | That number |

**A quarter of the pool comes from BM25.** At pool 60: BM25's top 15, then the top of the first-stage list until there are 60 different documents. BM25 may find fewer than 15 matches, and the two lists may overlap, so the first-stage list fills what's left. There's no score fusion: the reranker compares everything.

**`result.pool`** counts the candidates retrieved. If there's a reranker, that's what it scored. It can be smaller than asked if the backend returns fewer.

**`k` and `pool`:** `search(q, k=100)` raises the pool to 100 if it was lower, so you get up to 100 results. Invalid values of `k` or `pool` raise `ValueError` before any work is done.

**A Stage-2 health flag** means the second fitting stage was blocked by its safety floor, or the index's scores are so compressed that it ran outside what was calibrated. It appears in `report.notes` and `report.details["stage2"]`.

## Languages

**Nothing detects the language.** You state it:

- **`freeze(index, old_model, src_lang="en")`** says what language the index is in. It's your statement, not a check, and it defaults to English. It decides whether BM25 can turn on and which language queries are centered for.
- **`searcher.search(q, query_lang="zh")`** says what language this query is in. It defaults to the index's language. A Chinese query sent without it is centered as English and reaches BM25 as is.

| Query language | Effect |
| --- | --- |
| The index's, or `None` | Normal centering, BM25 on if it applies |
| Registered at `mount(query_langs=[...])` | Centered for that language. BM25 is skipped unless you pass `translate=` |
| Not registered | English centering with a warning. BM25 is skipped unless you pass `translate=` |

**Registering a language** needs parallel sentences, which `mount` gets for you for an English index:

| Language | Source | Access |
| --- | --- | --- |
| Chinese | FLORES, else Tatoeba | FLORES is gated: accept its terms on Hugging Face and log in. Tatoeba isn't gated |
| German, Spanish, Russian, Arabic, Vietnamese, Hindi, Turkish, Greek, Thai, Romanian | FLORES only | The same. Without access, `mount` raises `ValueError` saying so |
| Any other | None | Pass `shift_banks={"xx": (index-language sentences, xx sentences)}` to `mount` |

An index not in English always needs `shift_banks`, because the default banks all have English on one side.

### Chinese queries into an English index, end to end

This is the one cross-language setup with measured results, which the README's [Queries in another language](../README.md#queries-in-another-language) table gives: Chinese to English only, on two Wikipedia sets up to about 21K documents. The query model has to read both languages. It uses a translator feeding BM25, with the bridge doing the dense search. Only BM25 reads the translation.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer   # uv sync --extra models --extra banks --group translate

import spacebridge

device = "cuda" if torch.cuda.is_available() else "cpu"
new_model = spacebridge.SentenceTransformerEncoder(
    "Qwen/Qwen3-Embedding-0.6B", query_template="Instruct: Given a question, retrieve passages that answer it\nQuery:{text}",
    processor_kwargs={"padding_side": "left"}, max_seq_length=512, device=device,
)
bridge = spacebridge.mount(bundle, index, new_model, query_langs=["zh"], device=device)   # downloads the language bank

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
translator = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(device)

def translate(query, lang):   # any function (query, language) -> text will do
    output = translator.generate(**tokenizer([query], return_tensors="pt").to(device), max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

searcher = spacebridge.Searcher(
    bridge, new_model, spacebridge.ExactBackend(vectors, device=device), index, spacebridge.CrossEncoderReranker(device=device),
    lexical=spacebridge.LexicalIndex.build(index), translate=translate, device=device,
)
result = searcher.search("你的问题", k=5, query_lang="zh")
```

## Measuring on your own corpus

**Query time:** `result.timing_ms` gives `encode`, `project`, `search`, `lexical`, `fetch`, `rerank`, `total` and `translate` if it ran, for each query. Ignore the first query, which warms the models. Time on your own hardware:

```python
times = [searcher.search(q).timing_ms["total"] for q in your_queries[1:]]
```

Model loading, `freeze`, `mount`, BM25 building and HNSW building all come before this and aren't in it. `bridge.metrics["mount"]` holds the mount's parts.

**CPU rerank time:** seconds ≈ 1.91 ms × pool × tokens per (query, document) pair, on the benchmark machine with 10 threads. This fit was within 4% on average and 10% at worst over five corpora from 97 to 412 tokens per pair. Long documents are cut at 512 tokens.

**Quality:** if you have labeled queries, compare with `lexical=None` too. BM25 lowered every measured result on SciDocs, FiQA and TREC-COVID. Nothing at `mount` can tell you which corpora those are.

## What is cached and what resumes

| Workflow | Where its files go | If it's interrupted |
| --- | --- | --- |
| `examples/quickstart.py` | `quickstart/`, plus Hugging Face's own cache for downloads. It caches no embeddings | Rerun: it repeats everything |
| `examples/deploy.py` | The folder you give it | Rerun `build`: everything but HNSW repeats |
| `build_hnsw` | The path you give it | Continues from its last save, at most about 10 minutes back plus the batch in progress |
| Benchmark runs (`spacebridge_bench.run`) | `data/cache/` and `--out` | Rerun the same command. Encoding saves at most every 2 minutes, reranker scores every 30 s, results per finished cell |

- Benchmark runs refuse to continue an output file that holds another configuration's results, so use a new `--out` when you change a setting. The configuration covers every option, the device, the model and dataset revisions, the exact corpus, queries and judgments, and a hash of the library's and the harness's code, so a code change starts a new file too.
- Benchmark caches written on a CPU live apart from the GPU's, under `<name>@cpu-fp32`, and every cache file written from now on has a `.meta.json` sidecar naming its model, revision and device class. A file whose sidecar disagrees with the run raises instead of being reused. Older files have no sidecar and are counted as legacy in each run's manifest.
- These recovery claims were tested by killing processes (`kill -9`), including during the 8.84M HNSW build. A real power loss wasn't tested and rests on `fsync`.
- `scripts/steps.sh` isn't a command. A driver script sets `OUT` and sources it, as its comments show.
- [`data/README.md`](../data/README.md) explains the benchmark's local files.

## Contributing

`uv run pytest` runs the synthetic tests, with no downloads and no GPU. `.github/workflows/tests.yml` runs them on GitHub. Code under `spacebridge_bench` and `scripts` is research code, so its interfaces can change. There's no compatibility promise across versions yet, beyond artifact files stating their format version and refusing newer ones.
