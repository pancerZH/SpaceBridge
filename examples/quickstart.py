"""Search a frozen index built with one embedding model, using a different query model, without re-embedding anything.

    uv sync --extra models --extra banks
    uv run python examples/quickstart.py

A few minutes on a CPU, seconds on a GPU. Downloads a small public corpus (SciFact, 5.2K abstracts) and three models.
"""
from datasets import load_dataset

import spacebridge

corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
titles, texts = [t or "" for t in corpus["title"]], [t or "" for t in corpus["text"]]

# The frozen index: vectors built once with the old model. Nothing below changes them.
old_model = spacebridge.SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
vectors = old_model.encode_docs(titles, texts)
index = spacebridge.Index(vectors, texts, titles)

# Once, with the old model: choose anchors, make query proxies, record what the old model does with them.
bundle = spacebridge.freeze(index, old_model)
bundle.save("quickstart/bundle")

# With each new query model: fit the query map. Encodes a few thousand anchors, far fewer than re-indexing would.
new_model = spacebridge.SentenceTransformerEncoder(
    "BAAI/bge-small-en-v1.5", query_template="Represent this sentence for searching relevant passages: {text}",
)
bridge = spacebridge.mount(bundle, index, new_model)
bridge.save("quickstart/bridge")
report = bridge.quality_report()
print(f"predicted retention {report.score:.0f}% (grade {report.grade}), rerank pool {report.rerank_pool}; notes: {report.notes}")

# Serve: encode, project, search, fetch text, rerank. Above 50K documents, an English index also gets BM25 in the rerank pool,
# built here from the same text; pass lexical=None to turn it off. This corpus is small enough to go without.
searcher = spacebridge.Searcher(
    spacebridge.SpaceBridge.load("quickstart/bridge"), new_model, spacebridge.ExactBackend(vectors), index,
    spacebridge.CrossEncoderReranker(),
)
result = searcher.search("Do statins reduce the risk of heart disease?", k=3)
print(f"confidence {result.confidence:.2f}, {result.timing_ms['total'] / 1e3:.2f} s: " + ", ".join(f"{k} {v:.0f} ms" for k, v in result.timing_ms.items() if k != "total"))
for row, score, text in zip(result.ids, result.scores, result.texts):
    print(f"{score:7.3f}  row {row}: {text[:90]}")
