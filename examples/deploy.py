"""Deploy on your own index: build once, then serve from a new process that loads everything and never needs the old model.

    uv sync --extra models --extra banks --extra hnsw
    uv run python examples/deploy.py build deployment/
    uv run python examples/deploy.py serve deployment/ "Do statins reduce the risk of heart disease?"

`build` uses SciFact as a stand-in for your data. To use your own, replace load_corpus. What it must return is in docs/usage.md:
L2-normalized float32 vectors, one row per document, with text and titles in the same row order.
"""
import json
import sys
from pathlib import Path

import numpy as np

import spacebridge

OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEW_MODEL = "BAAI/bge-small-en-v1.5"
NEW_QUERY_TEMPLATE = "Represent this sentence for searching relevant passages: {text}"


def new_encoder(device: str = "cuda") -> spacebridge.SentenceTransformerEncoder:
    return spacebridge.SentenceTransformerEncoder(NEW_MODEL, query_template=NEW_QUERY_TEMPLATE, device=device)


def load_corpus(old_model) -> tuple[np.ndarray, list[str], list[str]]:
    """Your frozen vectors, as stored, and their text. Here, a small public corpus, encoded on the spot."""
    from datasets import load_dataset

    corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    titles, texts = [t or "" for t in corpus["title"]], [t or "" for t in corpus["text"]]
    return old_model.encode_docs(titles, texts), texts, titles


def build(out: Path) -> None:
    print("loading the old model and the corpus")
    old_model = spacebridge.SentenceTransformerEncoder(OLD_MODEL)
    vectors, texts, titles = load_corpus(old_model)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    (out / "documents.jsonl").write_text("".join(json.dumps({"title": a, "text": b}) + "\n" for a, b in zip(titles, texts)))
    index = spacebridge.Index(vectors, texts, titles)

    print("freeze: once per index, with the old model if you still have it (pass None if not)")
    spacebridge.freeze(index, old_model).save(out / "bundle")

    print("mount: once per new query model")
    bundle = spacebridge.CalibrationBundle.load(out / "bundle")
    bridge = spacebridge.mount(bundle, index, new_encoder())
    bridge.save(out / "bridge")
    print(f"  mounted in {sum(bridge.metrics['mount'].values()) / 1e3:.1f} s, predicted first-stage retention {bridge.quality_report().score:.0f}%")

    print("BM25: once per corpus and row order")
    spacebridge.LexicalIndex.build(index).save(out / "lexical")

    print("HNSW: once per set of vectors. A rerun over the same vectors continues; other vectors or settings raise ValueError")
    spacebridge.build_hnsw(vectors, out / "index.faiss")
    print(f"done: {out}. The old model isn't needed from here on.")


def serve(out: Path, query: str) -> None:
    # The stored vectors, the text and the new query model must all stay available. The HNSW graph replaces the vectors for search, not the text.
    vectors = np.load(out / "vectors.npy", mmap_mode="r")
    documents = [json.loads(line) for line in (out / "documents.jsonl").open()]
    index = spacebridge.Index(vectors, [d["text"] for d in documents], [d["title"] for d in documents])
    searcher = spacebridge.Searcher(
        spacebridge.SpaceBridge.load(out / "bridge"),      # checked against the index below: a mismatch raises ValueError
        new_encoder(),
        spacebridge.HnswBackend(out / "index.faiss"),      # ef_search = 500 by default: higher is slower and finds slightly more
        index,
        spacebridge.CrossEncoderReranker(),
        lexical=spacebridge.LexicalIndex.load(out / "lexical"),
    )
    result = searcher.search(query, k=3)   # add query_lang="zh" and so on for a language registered at mount: nothing detects it
    stages = ", ".join(f"{k} {v:.0f} ms" for k, v in result.timing_ms.items() if k != "total")
    print(f"{result.timing_ms['total'] / 1e3:.2f} s, {result.pool} candidates ({stages}). The first query also warms the models.")
    for row, score, text in zip(result.ids, result.scores, result.texts):
        print(f"{score:7.3f}  row {row}: {text[:90]}")


if __name__ == "__main__":
    command, target = sys.argv[1], Path(sys.argv[2])
    if command == "build":
        build(target)
    elif command == "serve":
        serve(target, sys.argv[3])
    else:
        raise SystemExit("usage: deploy.py build DIR | deploy.py serve DIR QUERY")
