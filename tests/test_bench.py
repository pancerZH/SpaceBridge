import json

import numpy as np
import pytest

from spacebridge_bench.datasets import _nested_order, _positive_preserving_subsample, self_match_rows
from spacebridge_bench.rerank import rerank, topk_jaccard
from spacebridge_bench.search import search


def test_search_excludes_self_rows():
    E = np.eye(4, dtype=np.float32)
    Q = np.array([[1, 0.5, 0, 0], [0, 0, 1, 0.2]], dtype=np.float32)
    assert search(Q, E, 2, "cpu").tolist() == [[0, 1], [2, 3]]
    assert search(Q, E, 2, "cpu", exclude=np.array([0, -1])).tolist()[0][0] == 1
    assert search(Q, E, 2, "cpu", exclude=np.array([0, -1])).tolist()[1] == [2, 3]


def test_positive_preserving_subsample_keeps_all_positives():
    ids = [str(i) for i in range(1000)]
    qrels = {"corpus-id": ["3", "500", "999", "7"], "score": [1, 2, 1, 0]}
    rows = _positive_preserving_subsample(ids, qrels, 100, seed=0)
    assert len(rows) == len(set(rows.tolist())) == 100
    assert {3, 500, 999} <= set(rows.tolist())


class _FakeReranker:
    """Scores a pair by the doc's trailing integer, so higher doc ids rank first."""

    def score(self, queries, docs, task):
        return np.array([float(d.split()[-1]) for d in docs])


def test_nested_order_prefixes_keep_positives_and_match_the_subsample():
    doc_ids = [f"d{i}" for i in range(1000)]
    qrels = {"corpus-id": ["d5", "d900", "d7"], "score": [1, 1, 0]}
    order = _nested_order(doc_ids, qrels, 100, seed=0)
    assert sorted(order.tolist()) == list(range(1000))
    assert set(order[:2].tolist()) == {5, 900}
    assert set(order[:100].tolist()) == set(_positive_preserving_subsample(doc_ids, qrels, 100, seed=0).tolist())


def test_rerank_reorders_within_depth_only():
    docs = [f"doc {i}" for i in range(10)]
    ranked = np.array([[1, 5, 3, 9, 0]])
    out = rerank(_FakeReranker(), ranked, 3, ["q"], docs, "")
    assert out.tolist() == [[5, 3, 1]]


def test_topk_jaccard():
    a, b = np.array([[1, 2, 3, 4, 5]]), np.array([[1, 2, 3, 6, 7]])
    assert topk_jaccard(a, b, k=5)[0] == 3 / 7


def test_self_match_rows_uses_text_not_ids():
    texts = ["an argument", "another doc", ""]
    assert self_match_rows(["another doc", "unrelated query", ""], texts).tolist() == [1, -1, -1]


def test_mrl_slices_share_the_full_cache_and_renormalize(tmp_path):
    from spacebridge_bench.encode import Encoder
    from spacebridge_bench.models import MODELS

    full, sliced = MODELS["qwen3-0.6b"], MODELS["qwen3-0.6b-256"]
    assert sliced.dim == sliced.truncate_dim == 256 and sliced.hf_id == full.hf_id
    emb = np.random.default_rng(0).standard_normal((3, full.dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    enc = Encoder(sliced, "cpu", cache_dir=tmp_path)
    assert enc.cache_dir == Encoder(full, "cpu", cache_dir=tmp_path).cache_dir
    out = enc._slice(emb)
    assert out.shape == (3, 256)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(out[0] / out[0, 0], emb[0, :256] / emb[0, 0], rtol=1e-4)


def test_query_subsample_keeps_whole_queries():
    from spacebridge_bench.datasets import _query_subsample

    qids = ["a", "a", "b", "c", "c", "d"]
    rows = _query_subsample(qids, [1, 1, 1, 1, 0, 0], 2, seed=0)
    kept = {qids[i] for i in rows}
    assert len(kept) == 2 and "d" not in kept
    assert all(qids[i] in kept for i in rows) and sum(q in kept for q in qids) == len(rows)


def test_flores_filter_drops_belebele_and_cross_split_sentences(tmp_path, monkeypatch):
    import huggingface_hub

    from spacebridge import language as datasets

    files = {
        "dev/eng_Latn.jsonl": ["A cat sat.", "Shared  sentence.", "Kept dev one."],
        "dev/cmn_Hans.jsonl": ["猫坐着。", "共享的句子。", "保留的。"],
        "devtest/eng_Latn.jsonl": ["Shared sentence.", "Inside a passage.", "Kept devtest."],
        "devtest/cmn_Hans.jsonl": ["共享的句子。", "在段落里。", "保留测试。"],
    }
    belebele = [{"flores_passage": "Intro. A cat  sat. More."}, {"flores_passage": "Inside a passage. End."}]

    def fake_download(repo, filename, repo_type):
        path = tmp_path / "hub" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = belebele if repo == datasets.BELEBELE_REPO else [{"id": i, "text": t} for i, t in enumerate(files[filename])]
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(datasets, "FLORES_CACHE", tmp_path / "flores")
    out = datasets.load_flores("zh")
    assert out["dev"]["en"] == ["Kept dev one."] and out["dev"]["zh"] == ["保留的。"] and out["dev"]["ids"] == [2]
    assert out["devtest"]["en"] == ["Kept devtest."] and out["counts"]["devtest"] == {"raw": 3, "kept": 1}
    unfiltered = datasets.load_flores("zh", belebele_filter=False)
    assert unfiltered["dev"]["en"] == ["A cat sat.", "Kept dev one."]  # cross-split duplicates still go
    assert datasets.load_flores("zh") == out  # cached


class _FakeModel:
    """Stands in for a SentenceTransformer: a text's tokens are its words, its embedding depends only on the text."""

    max_seq_length = 512

    def __init__(self, dim=8, fail_after=None, oom_above=None):
        self.dim, self.fail_after, self.oom_above = dim, fail_after, oom_above
        self.encoded, self.batch_tokens = [], []

    def tokenizer(self, texts, truncation=True, max_length=512):
        return {"input_ids": [t.split()[:max_length] for t in texts]}

    def preprocess(self, texts, prompt):
        assert prompt == ""
        return {"texts": texts}

    def __call__(self, features):
        import torch

        texts = features["texts"]
        tokens = max(len(t.split()) for t in texts) * len(texts)
        if self.oom_above is not None and tokens > self.oom_above:
            raise torch.cuda.OutOfMemoryError("fake")
        if self.fail_after is not None and len(self.encoded) >= self.fail_after:
            raise KeyboardInterrupt
        self.encoded.extend(texts)
        self.batch_tokens.append(tokens)
        return {"sentence_embedding": torch.stack([_fake_vector(t, self.dim) for t in texts])}


def _fake_vector(text, dim=8):
    import zlib

    import torch

    gen = torch.Generator().manual_seed(zlib.crc32(text.encode()))
    return torch.randn(dim, generator=gen)


def _fake_encoder(tmp_path, model):
    from spacebridge_bench.encode import Encoder
    from spacebridge_bench.models import MODELS

    enc = Encoder(MODELS["bge-base-en"], "cpu", cache_dir=tmp_path)
    enc._model = model
    return enc


def _fake_texts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(f"w{i}" for _ in range(rng.integers(1, 40))) + f" doc{i}" for i in range(n)]


def _expected(texts):
    import torch

    v = torch.stack([_fake_vector(t) for t in texts])
    return torch.nn.functional.normalize(v, dim=-1).half().float().numpy()


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_token_sorted_encoding_keeps_input_order_and_budget(tmp_path):
    from spacebridge_bench import encode

    texts, model = _fake_texts(), _FakeModel()
    out = _fake_encoder(tmp_path, model).encode(texts)
    np.testing.assert_allclose(out, _unit(_expected(texts)), atol=1e-6)
    assert max(model.batch_tokens) <= encode.CPU_BATCH_TOKENS
    assert sorted(model.encoded) == sorted(texts)  # each text once


def test_encoding_continues_from_saved_parts_after_a_crash(tmp_path, monkeypatch):
    from spacebridge_bench import encode

    monkeypatch.setattr(encode, "PART_ROWS", 16)
    monkeypatch.setattr(encode, "CPU_BATCH_TOKENS", 200)
    texts = _fake_texts()
    crashed = _FakeModel(fail_after=100)
    with pytest.raises(KeyboardInterrupt):
        _fake_encoder(tmp_path, crashed).encode(texts)
    enc = _fake_encoder(tmp_path, _FakeModel())
    path = enc.cache_dir / f"{enc._key(texts)}.npy"
    parts = path.with_suffix(".parts")
    assert not path.exists() and encode._finished_parts(parts)  # no complete-looking file, but saved work
    saved = encode._finished_parts(parts)[-1][2]
    out = enc.encode(texts)
    assert len(enc._model.encoded) == len(texts) - saved  # only the rest was encoded
    assert path.exists() and not parts.exists()
    np.testing.assert_allclose(out, _unit(_expected(texts)), atol=1e-6)


def test_out_of_memory_halves_the_token_budget(tmp_path, monkeypatch):
    from spacebridge_bench import encode

    monkeypatch.setattr(encode, "CPU_BATCH_TOKENS", 1_000)
    texts, model = _fake_texts(), _FakeModel(oom_above=300)
    out = _fake_encoder(tmp_path, model).encode(texts)
    assert max(model.batch_tokens) <= 300
    np.testing.assert_allclose(out, _unit(_expected(texts)), atol=1e-6)


def test_store_reads_what_encode_returns(tmp_path):
    from spacebridge_bench.store import EmbeddingStore

    titles, texts = [""] * 200, _fake_texts()
    enc = _fake_encoder(tmp_path, _FakeModel())
    E = enc.docs(titles, texts)
    store = enc.docs_store(titles, texts)
    assert store.shape == E.shape and len(store) == len(E)
    rows = np.array([5, 1, 150])
    np.testing.assert_allclose(store[rows], E[rows], atol=1e-6)
    np.testing.assert_allclose(store.mean("cpu"), E.mean(0), atol=1e-5)
    short = EmbeddingStore(store.path, truncate_dim=4)
    assert short.shape == (200, 4)
    np.testing.assert_allclose(short[rows], _unit(E[rows][:, :4]), atol=1e-5)


def _store_of(tmp_path, E):
    from spacebridge_bench.store import EmbeddingStore

    np.save(tmp_path / "index.npy", E.astype(np.float16))
    return EmbeddingStore(tmp_path / "index.npy")


def test_streamed_search_matches_search_in_one_piece(tmp_path, monkeypatch):
    from spacebridge_bench import store

    rng = np.random.default_rng(0)
    E, Q = _unit(rng.standard_normal((1_000, 16))), _unit(rng.standard_normal((20, 16))).astype(np.float32)
    s = _store_of(tmp_path, E)
    whole = search(Q, s[np.arange(1_000)], 50, "cpu", exclude=np.arange(20))
    monkeypatch.setattr(store, "CPU_SHARD_ROWS", 170)
    assert len(list(s.shards("cpu"))) == 6
    assert search(Q, s, 50, "cpu", exclude=np.arange(20)).tolist() == whole.tolist()


def test_nested_search_gives_each_prefix_its_own_top_k(tmp_path, monkeypatch):
    from spacebridge_bench import store
    from spacebridge_bench.search import search_nested

    rng = np.random.default_rng(1)
    E, Q = _unit(rng.standard_normal((1_000, 16))), _unit(rng.standard_normal((20, 16))).astype(np.float32)
    s = _store_of(tmp_path, E)
    monkeypatch.setattr(store, "CPU_SHARD_ROWS", 170)
    out = search_nested(Q, s, [100, 350, 1_000], k=30, device="cpu")
    assert set(out) == {100, 350, 1_000}
    for size, ranked in out.items():
        assert ranked.tolist() == search(Q, s[np.arange(size)], 30, "cpu").tolist()


def test_cached_search_reuses_the_ranking_at_any_depth(tmp_path, monkeypatch):
    from spacebridge_bench import search as search_module
    from spacebridge_bench.search import cached_search

    rng = np.random.default_rng(2)
    E, Q = _unit(rng.standard_normal((300, 16))), _unit(rng.standard_normal((5, 16))).astype(np.float32)
    s = _store_of(tmp_path, E)
    first = cached_search(Q, s, 30, "cpu", cache_dir=tmp_path / "r")
    assert first.tolist() == search(Q, s, 30, "cpu").tolist()
    monkeypatch.setattr(search_module, "search", lambda *a, **k: pytest.fail("should come from the cache"))
    assert cached_search(Q, s, 10, "cpu", cache_dir=tmp_path / "r").tolist() == first[:, :10].tolist()


class _FakeCrossEncoder:
    """Scores a pair by the doc's trailing integer."""

    def __init__(self):
        self.seen = []

    def tokenizer(self, queries, docs, truncation=True, max_length=512):
        return {"input_ids": [(q + " " + d).split() for q, d in zip(queries, docs)]}

    def predict(self, pairs, batch_size, convert_to_numpy, show_progress_bar):
        self.seen.extend(pairs)
        return np.array([float(d.split()[-1]) for _, d in pairs])


def _fake_reranker(tmp_path, model):
    from spacebridge_bench.rerank import RERANKERS, Reranker

    rr = Reranker(RERANKERS["bge-reranker-v2-m3"], "cpu", cache_dir=tmp_path)
    rr._model = model
    return rr


def test_rerank_scores_survive_a_crash_and_ignore_a_torn_record(tmp_path):
    from spacebridge_bench import rerank as rerank_module
    from spacebridge_bench.rerank import _pair_key

    docs = [f"doc {i}" for i in range(50)]
    first = _fake_reranker(tmp_path, _FakeCrossEncoder())
    # An older .npz cache stays readable.
    np.savez(first.cache_path, keys=np.array([_pair_key("", "q", "doc 0")], dtype=np.uint64), scores=np.array([123.0], dtype=np.float32))
    scores = first.score(["q"] * 50, docs, "")
    assert scores[0] == 123.0 and scores[7] == 7.0 and len(first.model.seen) == 49
    log = next(tmp_path.glob("*.scores"))
    with log.open("ab") as f:
        f.write(b"torn")  # a power loss in the middle of a record
    rerank_module._CACHES.clear()  # a new process
    after = _fake_reranker(tmp_path, _FakeCrossEncoder())
    assert after.score(["q"] * 50, docs, "").tolist() == scores.tolist()
    assert after.model.seen == []  # nothing rescored


def test_finished_cells_drops_a_torn_line_and_an_incomplete_cell(tmp_path):
    from spacebridge_bench.run import finished_cells

    path = tmp_path / "results.jsonl"
    lines = [
        {"cell": "a", "cell_records": 2, "row": "0a", "config_id": "x"}, {"cell": "a", "cell_records": 2, "row": "8b", "config_id": "x"},
        {"cell": "b", "cell_records": 2, "row": "0a", "config_id": "x"},
        {"row": "old format, kept"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"cell": "c", "cell_rec')
    assert finished_cells(path, "x") == {"a"}
    kept = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r.get("cell") for r in kept] == ["a", "a", None]
    assert finished_cells(tmp_path / "missing.jsonl", "x") == set()


def test_atomic_file_keeps_the_old_file_when_a_write_fails(tmp_path):
    from spacebridge_bench import durable

    path = tmp_path / "a.npy"
    durable.save_npy(path, np.arange(3))
    with pytest.raises(RuntimeError):
        with durable.atomic_file(path) as f:
            f.write(b"partial")
            raise RuntimeError
    assert np.load(path).tolist() == [0, 1, 2] and [p.name for p in tmp_path.iterdir()] == ["a.npy"]


def test_recall_at_rerank_depth_only_when_the_ranking_reaches_it():
    from spacebridge_bench.metrics import per_query

    deep = per_query(np.arange(1_000)[None, :], [{500: 1}])
    assert deep["recall@300"][0] == 0.0 and deep["recall@1000"][0] == 1.0
    shallow = per_query(np.arange(100)[None, :], [{50: 1}])
    assert "recall@300" not in shallow and shallow["recall@100"][0] == 1.0


def test_progress_lines_reach_the_file(tmp_path, monkeypatch):
    from spacebridge_bench import durable

    monkeypatch.setenv(durable.PROGRESS_ENV, str(tmp_path / "progress.log"))
    durable.progress("cell 1/3: start")
    assert (tmp_path / "progress.log").read_text().endswith("cell 1/3: start\n")


def test_store_prefix_is_a_view_with_its_own_rankings(tmp_path):
    from spacebridge_bench.search import cached_search
    from spacebridge_bench.store import EmbeddingStore

    E = np.random.default_rng(0).standard_normal((50, 8))
    np.save(tmp_path / "index.npy", E.astype(np.float16))
    store = EmbeddingStore(tmp_path / "index.npy")
    head = store.prefix(20)
    assert store.prefix(50) is store and len(head) == 20 and head.shape == (20, 8)
    np.testing.assert_array_equal(head[np.arange(5)], store[np.arange(5)])
    Q = E[:3].astype(np.float32)
    ranked_head = cached_search(Q, head, 5, "cpu", cache_dir=tmp_path / "r")
    ranked_full = cached_search(Q, store, 5, "cpu", cache_dir=tmp_path / "r")
    assert ranked_head.max() < 20 and ranked_full.max() >= 20 or (ranked_head == ranked_full).all()
    assert len(list((tmp_path / "r").iterdir())) == 2


def test_first_docs_refuses_a_prefix_that_loses_positives():
    from spacebridge_bench.datasets import RetrievalDataset
    from spacebridge_bench.run import _first_docs

    ds = RetrievalDataset("x", "en", "t", [str(i) for i in range(10)], [""] * 10, ["d"] * 10, ["q"], [{7: 1}], np.array([-1]))
    assert _first_docs(ds, 8).N == 8 and _first_docs(ds, 8).name == "x@8" and len(_first_docs(ds, 8).texts) == 10
    with pytest.raises(ValueError, match="positives"):
        _first_docs(ds, 7)


def test_onnx_reranker_goes_through_the_cache_and_the_cpu_budget(tmp_path):
    from spacebridge_bench.rerank import Reranker, RerankerSpec

    rr = Reranker(RerankerSpec("fake-onnx", "x", instructed=False, onnx="x.onnx"), "cpu", cache_dir=tmp_path)
    rr._model = _FakeCrossEncoder()
    scores = rr.score(["q"] * 3, ["doc 1", "doc 2", "doc 3"], "")
    assert scores.tolist() == [1.0, 2.0, 3.0]
    assert len(rr._model.seen) == 3
    rr.score(["q"] * 3, ["doc 1", "doc 2", "doc 3"], "")
    assert len(rr._model.seen) == 3


def test_rerank_pools_handles_lists_of_any_length_and_keeps_order_on_ties():
    from spacebridge_bench.rerank import rerank_pools

    docs = [f"doc {i}" for i in range(10)]
    out = rerank_pools(_FakeReranker(), [np.array([1, 5, 3]), np.array([2])], ["a", "b"], docs, "")
    assert out.tolist() == [[5, 3, 1], [2, -1, -1]]

    class Flat:
        def score(self, queries, docs, task):
            return np.zeros(len(docs))

    assert rerank_pools(Flat(), [np.array([7, 2, 9])], ["a"], docs, "").tolist() == [[7, 2, 9]]


def test_strict_retention_uses_the_lower_of_the_two_fixed_baselines():
    from spacebridge_bench.run import _strict_retention

    pq = lambda v: {m: np.array(v, dtype=float) for m in ("recall@30", "ndcg@5", "ndcg@10")}
    out = _strict_retention({"bridge": pq([0.4, 0.6]), "native": pq([0.8, 0.8]), "oracle": pq([0.5, 0.7])})
    assert out["native"]["ndcg@5"] == 0.8 and out["oracle"]["ndcg@5"] == 0.6
    assert out["rho_ceil"]["ndcg@5"] == pytest.approx(0.5 / 0.6)
    lo, hi = out["ci_ceil"]["ndcg@5"]
    assert lo <= out["rho_ceil"]["ndcg@5"] <= hi


def test_lexical_rows_are_rerank_rows_with_a_quarter_for_bm25():
    from spacebridge.lexical import LEXICAL_SHARE
    from spacebridge_bench.run import LEXICAL_ROWS, RERANK_DEPTH, ROWS

    assert LEXICAL_ROWS == {"9h": 60, "9hw": 100, "9hx": 300} and LEXICAL_SHARE == 4
    assert all(RERANK_DEPTH[r] == b and r in ROWS for r, b in LEXICAL_ROWS.items())


def test_the_lexical_cache_is_keyed_by_the_corpus_rows(tmp_path, monkeypatch):
    from dataclasses import replace

    from spacebridge import Index
    from spacebridge_bench import run
    from spacebridge_bench.datasets import RetrievalDataset

    monkeypatch.setattr(run, "LEXICAL_CACHE", tmp_path)
    monkeypatch.setattr(run, "_LEXICAL", {})
    texts, titles = [f"word{i} common" for i in range(6)], [""] * 6
    ds = RetrievalDataset("toy", "en", "task", [str(i) for i in range(6)], titles, texts, [], [], np.empty(0, dtype=np.int64))
    index = Index(np.zeros((6, 2)), texts, titles)
    built = run.cached_lexical(ds, index)
    assert built.top("word3", 2).tolist() == [3]
    assert len(list(tmp_path.iterdir())) == 1
    assert run.cached_lexical(ds, index) is built  # the same process keeps it

    monkeypatch.setattr(run, "_LEXICAL", {})
    monkeypatch.setattr(run.LexicalIndex, "build", lambda *_: pytest.fail("a rerun must not rebuild"))
    assert run.cached_lexical(ds, index).top("word3", 2).tolist() == [3]  # a new process loads it

    monkeypatch.undo()
    monkeypatch.setattr(run, "LEXICAL_CACHE", tmp_path)
    monkeypatch.setattr(run, "_LEXICAL", {})
    prefix = replace(ds, doc_ids=ds.doc_ids[:4])
    assert len(run.cached_lexical(prefix, Index(np.zeros((4, 2)), texts[:4], titles[:4]))) == 4  # other rows, another index
    assert len(list(tmp_path.iterdir())) == 2


def test_concat_embeddings_writes_a_file_np_load_reads_and_nothing_on_failure(tmp_path):
    from spacebridge_bench.encode import concat_embeddings

    a, b = np.arange(12, dtype=np.float16).reshape(4, 3), -np.arange(9, dtype=np.float16).reshape(3, 3)
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", b)
    concat_embeddings([tmp_path / "a.npy", tmp_path / "b.npy"], tmp_path / "ab.npy", chunk_rows=2)
    back = np.load(tmp_path / "ab.npy", mmap_mode="r")
    assert back.dtype == np.float16 and back.shape == (7, 3)
    np.testing.assert_array_equal(back, np.concatenate([a, b]))
    np.save(tmp_path / "c.npy", np.zeros((2, 5), dtype=np.float16))
    with pytest.raises(ValueError, match="line up"):
        concat_embeddings([tmp_path / "a.npy", tmp_path / "c.npy"], tmp_path / "bad.npy")
    assert not (tmp_path / "bad.npy").exists()
