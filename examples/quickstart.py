"""Search an index built with one embedding model using a different query model, without re-embedding the corpus for the new one.

    uv sync --extra models --extra banks
    uv run python examples/quickstart.py

A small demo, so it makes its own "old" index first: it encodes SciFact (5.2K abstracts) with the old model, which a real migration wouldn't do,
and encodes a few thousand anchors with the new model, which it would. The first run downloads about 2.5 GB of models and a few MB of data. After that, it takes
about 30 seconds on a GPU and about 3 minutes on a CPU. It writes to quickstart/ and repeats all its work on every run: examples/deploy.py shows building
once and serving from a new process. It prints something like this (numbers vary by machine):

    predicted first-stage retention 94% (grade B), advisory rerank pool 60; notes: []
    agreement confidence 0.84, one query 0.39 s: encode 8 ms, project 1 ms, search 1 ms, lexical 1 ms, fetch 0 ms, rerank 355 ms
      3.215  row 1234: ...

"predicted retention" is the score of the first stage's Recall@30, not of the final answer, and confidence predicts top-10 agreement with the old model, not
that an answer is right: docs/usage.md explains both.
"""
from datasets import load_dataset

import spacebridge

print("downloading the corpus and models if needed, then encoding the demo index with the old model", flush=True)
corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
titles, texts = [t or "" for t in corpus["title"]], [t or "" for t in corpus["text"]]

# The frozen index: vectors built once with the old model. Nothing below changes them.
old_model = spacebridge.SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
vectors = old_model.encode_docs(titles, texts)
index = spacebridge.Index(vectors, texts, titles)

# Once, with the old model: choose anchors, make query proxies, record what the old model does with them.
print("freeze: anchors and proxies, with the old model", flush=True)
bundle = spacebridge.freeze(index, old_model)
bundle.save("quickstart/bundle")

# With each new query model: fit the query map. Encodes a few thousand anchors, far fewer than re-indexing would.
new_model = spacebridge.SentenceTransformerEncoder(
    "BAAI/bge-small-en-v1.5", query_template="Represent this sentence for searching relevant passages: {text}",
)
print("mount: fitting the query map for the new model", flush=True)
bridge = spacebridge.mount(bundle, index, new_model)
bridge.save("quickstart/bridge")
report = bridge.quality_report()
print(f"predicted first-stage retention {report.score:.0f}% (grade {report.grade}), advisory rerank pool {report.rerank_pool}; notes: {report.notes}")

# Serve: encode, project, search, fetch text, rerank. An English index also gets BM25 in the rerank pool, built here from the
# same text; pass lexical=None to turn it off.
searcher = spacebridge.Searcher(
    spacebridge.SpaceBridge.load("quickstart/bridge"), new_model, spacebridge.ExactBackend(vectors), index,
    spacebridge.CrossEncoderReranker(),
)
result = searcher.search("Do statins reduce the risk of heart disease?", k=3)
print(f"agreement confidence {result.confidence:.2f}, one query {result.timing_ms['total'] / 1e3:.2f} s: " + ", ".join(f"{k} {v:.0f} ms" for k, v in result.timing_ms.items() if k != "total"))
for row, score, text in zip(result.ids, result.scores, result.texts):
    print(f"{score:7.3f}  row {row}: {text[:90]}")
