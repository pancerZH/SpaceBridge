"""BEIR-format retrieval datasets from the Hugging Face hub (design §7.1)."""

from dataclasses import dataclass

DATASETS: dict[str, tuple[str, str, str]] = {
    # name: (hub repo, language, retrieval task instruction for instruction-tuned query encoders)
    "scifact": ("mteb/scifact", "en", "Given a scientific claim, retrieve documents that support or refute the claim"),
    "fiqa": ("mteb/fiqa", "en", "Given a financial question, retrieve user replies that best answer the question"),
    "nfcorpus": ("mteb/nfcorpus", "en", "Given a question, retrieve relevant documents that best answer the question"),
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

    @property
    def N(self) -> int:
        return len(self.doc_ids)


def load_beir(name: str, split: str = "test") -> RetrievalDataset:
    from datasets import load_dataset

    repo, lang, task = DATASETS[name]
    corpus = load_dataset(repo, "corpus", split="corpus")
    queries = load_dataset(repo, "queries", split="queries")
    qrels = load_dataset(repo, "default", split=split)

    row_of = {str(doc_id): i for i, doc_id in enumerate(corpus["_id"])}
    rels: dict[str, dict[int, int]] = {}
    for qid, did, score in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"]):
        if score > 0 and str(did) in row_of:
            rels.setdefault(str(qid), {})[row_of[str(did)]] = int(score)

    query_text = dict(zip(map(str, queries["_id"]), queries["text"]))
    qids = [q for q in rels if q in query_text]
    return RetrievalDataset(
        name=name,
        lang=lang,
        task=task,
        doc_ids=list(corpus["_id"]),
        titles=[t or "" for t in corpus["title"]],
        texts=list(corpus["text"]),
        queries=[query_text[q] for q in qids],
        qrels=[rels[q] for q in qids],
    )
