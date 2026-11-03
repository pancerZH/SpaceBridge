"""A5: does the library do what the benchmark says? One pair through the real ``Searcher``, for the same queries as a row.

Freezes and mounts through the ``sentence-transformers`` adapters, with ``ExactBackend`` on the GPU, a saved ``LexicalIndex`` and
the real ``CrossEncoderReranker``, then scores each query's pool. Writes one JSON line per budget to --out; a rerun skips those.

    uv run python -u scripts/searcher_check.py hotpotqa-1m --index-model bge-large-en --query-model bge-base-en --n 200 --budgets 60 --out out.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from spacebridge import CrossEncoderReranker, ExactBackend, Index, Searcher, SentenceTransformerEncoder, freeze, mount
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir, load_shift_bank
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import cached_lexical


def adapter(key: str, task: str, device: str) -> SentenceTransformerEncoder:
    spec = MODELS[key]
    return SentenceTransformerEncoder(
        spec.hf_id, spec.query_template.replace("{task}", task), spec.doc_template, device,
        batch_tokens=spec.batch_tokens, max_seq_length=spec.max_seq_length,
        processor_kwargs=spec.tokenizer_kwargs, config_kwargs=spec.config_kwargs or None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--query-model", default="bge-base-en")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--budgets", type=int, nargs="+", default=[60])
    p.add_argument("--K", type=int, default=None, help="anchors (default: the index size's)")
    p.add_argument("--translate", default=None, help="a cached translator of scripts/translation_baseline.py: BM25 reads its output, through Searcher(translate=)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    done = {r["budget"] for r in durable.read_jsonl(args.out)[0]} if args.out.exists() else set()
    budgets = [b for b in args.budgets if b not in done]
    if not budgets:
        return
    ds = load_beir(args.dataset)
    enc = Encoder(MODELS[args.index_model], args.device)
    store = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    enc.unload()
    index = Index(store, ds.texts, ds.titles)
    t0 = time.perf_counter()
    src = adapter(args.index_model, ds.task, args.device)
    bundle = freeze(index, src, ds.lang, args.seed, args.K, device=args.device)
    del src
    tgt = adapter(args.query_model, ds.task, args.device)
    langs = [ds.query_lang] if ds.query_lang else []
    banks = {ds.query_lang: load_shift_bank(ds.query_lang)[:2]} if ds.query_lang else None
    bridge = mount(bundle, index, tgt, langs, device=args.device, shift_banks=banks)
    durable.progress(f"mounted in {time.perf_counter() - t0:.0f} s: K {bundle.K}, grade {bridge.quality.grade}, stored pool {bridge.quality.rerank_pool}")
    vectors = np.empty(store.shape, dtype=np.float32)
    for s in range(0, len(store), 200_000):
        vectors[s : s + 200_000] = store[np.arange(s, min(s + 200_000, len(store)))]
    backend = ExactBackend(vectors, device=args.device)
    del vectors
    translate = None
    if args.translate:
        from translation_baseline import translate_cached

        translations = dict(zip(ds.queries, translate_cached(args.translate, ds.queries, args.device)))
        translate = lambda query, lang: translations[query]
    searcher = Searcher(bridge, tgt, backend, index, CrossEncoderReranker(device=args.device), device=args.device, lexical=cached_lexical(ds, index), translate=translate)
    durable.progress(f"the library's own budget here is {searcher.pool}")
    qs, qrels = ds.queries[: args.n], ds.qrels[: args.n]
    for budget in budgets:
        rows = np.full((len(qs), budget), -1, dtype=np.int64)
        sent = []
        for i, q in enumerate(qs):
            out = searcher.search(q, k=budget, pool=budget, query_lang=ds.query_lang)
            rows[i, : len(out.ids)] = out.ids
            sent.append(out.pool)
            if (i + 1) % 50 == 0:
                durable.progress(f"budget {budget}: {i + 1} / {len(qs)} queries")
        metrics = {m: float(v.mean()) for m, v in per_query(rows, qrels).items()}
        rec = {"dataset": ds.name, "src": args.index_model, "tgt": args.query_model, "seed": args.seed, "K": bundle.K, "n": len(qs), "budget": budget,
               "default_budget": searcher.pool, "grade": bridge.quality.grade, "translate": args.translate, "pool_mean": float(np.mean(sent)), "metrics": metrics}
        durable.append_lines(args.out, [json.dumps(rec)])
        durable.progress(f"budget {budget}: nDCG@5 {metrics['ndcg@5']:.5f}")


if __name__ == "__main__":
    main()
