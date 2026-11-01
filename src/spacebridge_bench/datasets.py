"""BEIR-format retrieval datasets from the Hugging Face hub (design §7.1)."""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    lang: str                         # corpus (index) language
    task: str                         # retrieval instruction for instruction-tuned query encoders
    max_docs: Optional[int] = None    # positive-preserving subsample size for large corpora
    query_lang: Optional[str] = None  # cross-lingual sets: language of the test queries
    max_queries: Optional[int] = None  # random query subsample (fixed seed) before the corpus subsample
    split: str = "test"               # qrels split
    corpus_split: str = "corpus"      # split names of the corpus / queries configs
    queries_split: str = "queries"


DATASETS: dict[str, DatasetSpec] = {
    "scifact": DatasetSpec("mteb/scifact", "en", "Given a scientific claim, retrieve documents that support or refute the claim"),
    "fiqa": DatasetSpec("mteb/fiqa", "en", "Given a financial question, retrieve user replies that best answer the question"),
    "nfcorpus": DatasetSpec("mteb/nfcorpus", "en", "Given a question, retrieve relevant documents that best answer the question"),
    "arguana": DatasetSpec("mteb/arguana", "en", "Given a claim, find documents that refute the claim"),
    "scidocs": DatasetSpec(
        "mteb/scidocs", "en", "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper"
    ),
    # Full 171K corpus: its 17.5K judged-relevant docs would make up 35% of a 50K subsample.
    "trec-covid": DatasetSpec("mteb/trec-covid", "en", "Given a query on COVID-19, retrieve documents that answer the query"),
    "nq": DatasetSpec("mteb/nq", "en", "Given a question, retrieve Wikipedia passages that answer the question", max_docs=50_000),
    # M3 held-out sets (never used to choose a default). 2,000 of 7,405 hotpotqa test queries keep their
    # 4K positives at 8% of a 50K subsample; all 14.8K would be 30%.
    "hotpotqa": DatasetSpec(
        "mteb/hotpotqa", "en", "Given a multi-hop question, retrieve documents that can help answer the question",
        max_docs=50_000, max_queries=2_000,
    ),
    "quora": DatasetSpec(
        "mteb/quora", "en", "Given a question, retrieve questions that are semantically equivalent to the given question",
        max_docs=50_000, max_queries=2_000,
    ),
    # Tier 2 (native Chinese): T2Ranking web passages and real search queries; only a dev split exists.
    # 2,000 of 22.8K queries (~10K positives) in a 50K-passage subsample.
    "t2retrieval": DatasetSpec(
        "mteb/T2Retrieval", "zh", "Given a Chinese search query, retrieve web passages that answer the question",
        max_docs=50_000, max_queries=2_000, split="dev", corpus_split="dev", queries_split="dev",
    ),
    # Close-out validation (never used for a decision): real Bing queries, untitled passages (lead proxies, the
    # case where Stage 2 can hurt unnoticed). 2,000 of the 6,980 dev queries (~1.1 positives each); 50K-passage
    # subsample comparable to the other sets, and a 1M-passage subsample of the 8.8M corpus for the scale test.
    "msmarco": DatasetSpec(
        "mteb/msmarco", "en", "Given a web search query, retrieve relevant passages that answer the query",
        max_docs=50_000, max_queries=2_000, split="dev",
    ),
    "msmarco-1m": DatasetSpec(
        "mteb/msmarco", "en", "Given a web search query, retrieve relevant passages that answer the query",
        max_docs=1_000_000, max_queries=2_000, split="dev",
    ),
    # Tier 5a: XQuAD questions (human-translated) against all ~21K English SQuAD v1.1 paragraphs.
    "xquad-zh": DatasetSpec(
        "google/xquad", "en", "Given a question, retrieve Wikipedia passages that answer the question", query_lang="zh"
    ),
}


@dataclass
class RetrievalDataset:
    name: str
    lang: str
    task: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    queries: list[str]
    qrels: list[dict[int, int]]   # per query: doc row -> graded relevance (> 0 only)
    self_rows: np.ndarray         # per query: corpus row holding the query itself (ArguAna), else -1
    query_lang: Optional[str] = None        # set when queries are not in the corpus language
    src_queries: Optional[list[str]] = None  # parallel queries in the corpus language (native ceiling)

    @property
    def index_lang_queries(self) -> list[str]:
        return self.src_queries if self.src_queries is not None else self.queries

    @property
    def N(self) -> int:
        return len(self.doc_ids)


DATASET_CACHE = Path("data/cache/datasets")


def load_beir(name: str, split: Optional[str] = None, seed: int = 0) -> RetrievalDataset:
    """The parsed dataset, pickled under data/cache/datasets/ (parsing trec-covid's 171K rows takes ~18 s)."""
    spec = DATASETS[name]
    key = hashlib.sha256(repr((spec, split, seed, _LOADER_VERSION)).encode()).hexdigest()[:12]
    path = DATASET_CACHE / f"{name}_{key}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    ds = _load_beir(name, split, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(ds, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ds


_LOADER_VERSION = 1  # bump when parsing changes, so stale pickles are not reused


def _load_beir(name: str, split: Optional[str], seed: int) -> RetrievalDataset:
    from datasets import load_dataset

    spec = DATASETS[name]
    if spec.repo == "google/xquad":
        return _load_xquad(name, spec)
    corpus = load_dataset(spec.repo, "corpus", split=spec.corpus_split)
    queries = load_dataset(spec.repo, "queries", split=spec.queries_split)
    qrels = load_dataset(spec.repo, "default", split=split or spec.split)
    if spec.max_queries is not None:
        qrels = qrels.select(_query_subsample(qrels["query-id"], qrels["score"], spec.max_queries, seed))

    all_ids = [str(i) for i in corpus["_id"]]
    if spec.max_docs is not None and len(all_ids) > spec.max_docs:
        corpus = corpus.select(_positive_preserving_subsample(all_ids, qrels, spec.max_docs, seed))
    doc_ids = [str(i) for i in corpus["_id"]]

    row_of = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    rels: dict[str, dict[int, int]] = {}
    for qid, did, score in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
        if score > 0 and str(did) in row_of:
            rels.setdefault(str(qid), {})[row_of[str(did)]] = int(score)

    query_text = dict(zip(map(str, queries["_id"]), queries["text"]))
    qids = [q for q in rels if q in query_text]
    texts = [t or "" for t in corpus["text"]]
    self_rows = self_match_rows([query_text[q] for q in qids], texts)
    return RetrievalDataset(
        name=name,
        lang=spec.lang,
        task=spec.task,
        doc_ids=doc_ids,
        titles=[t or "" for t in corpus["title"]],
        texts=texts,
        queries=[query_text[q] for q in qids],
        qrels=[rels[q] for q in qids],
        self_rows=self_rows,
    )


def self_match_rows(queries: list[str], texts: list[str]) -> np.ndarray:
    """Per query, the corpus row with identical text (BEIR drops it from results: ArguAna queries are
    corpus arguments), else -1. Matched on text: query and corpus ids live in separate namespaces."""
    row_of_text = {t: i for i, t in enumerate(texts) if t}
    return np.array([row_of_text.get(q, -1) for q in queries], dtype=np.int64)


def _query_subsample(query_ids: list, scores: list, n: int, seed: int) -> np.ndarray:
    """Qrels rows of ``n`` random queries that have at least one positive (all rows if fewer)."""
    qids = sorted({str(q) for q, s in zip(query_ids, scores) if s > 0})
    if len(qids) <= n:
        return np.arange(len(query_ids))
    keep = set(np.random.default_rng(seed).choice(qids, n, replace=False).tolist())
    return np.array([i for i, q in enumerate(query_ids) if str(q) in keep])


def _positive_preserving_subsample(doc_ids: list[str], qrels, n: int, seed: int) -> np.ndarray:
    """Corpus rows: every positive of the split's qrels, then uniform random distractors up to ``n``
    (design §7.1 rule 1). Rows are returned shuffled so positives are not clustered."""
    positives = {str(d) for d, s in zip(qrels["corpus-id"], qrels["score"]) if s > 0}
    is_pos = np.array([d in positives for d in doc_ids])
    rng = np.random.default_rng(seed)
    pos_rows = np.flatnonzero(is_pos)
    if len(pos_rows) >= n:
        raise ValueError(f"{len(pos_rows)} positives do not fit in a {n}-document subsample")
    rest = rng.choice(np.flatnonzero(~is_pos), n - len(pos_rows), replace=False)
    return rng.permutation(np.concatenate([pos_rows, rest]))


def _load_xquad(name: str, spec: DatasetSpec) -> RetrievalDataset:
    """Cross-lingual retrieval: XQuAD questions in ``spec.query_lang`` against the English paragraphs of
    SQuAD v1.1 (train + dev; XQuAD's 240 paragraphs come from dev). Each question's positive is its
    own paragraph; the parallel English question gives the native ceiling (design §2.1, Row 0b)."""
    from datasets import load_dataset

    en = load_dataset(spec.repo, "xquad.en", split="validation")
    xq = load_dataset(spec.repo, f"xquad.{spec.query_lang}", split="validation")
    if en["id"] != xq["id"]:
        raise ValueError("XQuAD language splits are not row-aligned")
    squad = load_dataset("rajpurkar/squad")
    norm = lambda t: " ".join(t.split())
    paragraphs: dict[str, tuple[str, str]] = {}  # normalized context -> (title, context), first wins
    for split in ("train", "validation"):
        for title, ctx in zip(squad[split]["title"], squad[split]["context"]):
            paragraphs.setdefault(norm(ctx), (title.replace("_", " "), ctx))
    # XQuAD's copy of a paragraph replaces SQuAD's when they differ only in whitespace.
    for ctx in en["context"]:
        paragraphs[norm(ctx)] = (paragraphs.get(norm(ctx), ("", ""))[0], ctx)
    keys = list(paragraphs)
    row_of = {k: i for i, k in enumerate(keys)}
    return RetrievalDataset(
        name=name,
        lang=spec.lang,
        task=spec.task,
        doc_ids=[str(i) for i in range(len(keys))],
        titles=[paragraphs[k][0] for k in keys],
        texts=[paragraphs[k][1] for k in keys],
        queries=list(xq["question"]),
        qrels=[{row_of[norm(c)]: 1} for c in en["context"]],
        self_rows=np.full(len(en), -1, dtype=np.int64),
        query_lang=spec.query_lang,
        src_queries=list(en["question"]),
    )


def load_shift_bank(query_lang: str, n: int = 256) -> tuple[list[str], list[str]]:
    """Parallel (English, query-language) sentences for the language-cone shift (design §4.4).

    FLORES (gated) needs an HF token; without one this uses human-translated Tatoeba pairs from
    ``mteb/tatoeba-bitext-mining`` as a stand-in. Returns the first ``n`` pairs.
    """
    from datasets import load_dataset

    config = {"zh": "cmn-eng"}[query_lang]
    ds = load_dataset("mteb/tatoeba-bitext-mining", config, split="test")
    pairs = [(e.strip(), q.strip()) for q, e in zip(ds["sentence1"], ds["sentence2"])][:n]
    return [e for e, _ in pairs], [q for _, q in pairs]
