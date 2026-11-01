"""W6: the real alternative for cross-lingual search. Translate each query with an open MT model, then search natively.

A comparator only: machine translation stays out of calibration (design §2.2). Translations are cached line by line, so a
rerun continues. Reports first-stage metrics and nDCG@5 after reranking the top pools, with the reranker reading either the
translated query or the original one, next to the human-translated parallel query (Row 0b).

    uv run --group translate python -u scripts/translation_baseline.py xquad-zh --translator opus --out x.jsonl
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.rerank import RERANKERS, Reranker, rerank
from spacebridge_bench.run import _DocTexts, _first_queries
from spacebridge_bench.search import cached_search

CACHE = Path("data/cache/translations")
TRANSLATORS = {
    # (hub id, prefix for the source text, generation kwargs). opus-mt: CC-BY-4.0, 78M params. MADLAD-400 3B: Apache-2.0.
    "opus": ("Helsinki-NLP/opus-mt-zh-en", "", {}),
    "madlad": ("google/madlad400-3b-mt", "<2en> ", {}),
}
POOLS = (30, 60, 100, 300)


class Translator:
    def __init__(self, name: str, device: str = "cuda"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name, (self.hf_id, self.prefix, self.kwargs) = name, TRANSLATORS[name]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id, dtype=dtype).to(self.device).eval()

    @torch.inference_mode()
    def translate(self, texts: list[str]) -> list[str]:
        batch = self.tokenizer([self.prefix + t for t in texts], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        out = self.model.generate(**batch, max_new_tokens=128, **self.kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def unload(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


def translate_cached(translator_name: str, queries: list[str], device: str, batch: int = 32) -> list[str]:
    key = hashlib.sha256("\n".join(queries).encode()).hexdigest()[:16]
    path = CACHE / f"{translator_name}_{key}.jsonl"
    done = {}
    if path.exists():
        done = {r["i"]: r["text"] for r in durable.read_jsonl(path)[0]}
    todo = sorted((i for i in range(len(queries)) if i not in done), key=lambda i: -len(queries[i]))
    if todo:
        tr = Translator(translator_name, device)
        for s in range(0, len(todo), batch):
            part = todo[s : s + batch]
            out = tr.translate([queries[i] for i in part])
            durable.append_lines(path, [json.dumps({"i": i, "text": t}, ensure_ascii=False) for i, t in zip(part, out)])
            done.update(zip(part, out))
            durable.progress(f"translate {translator_name}: {len(done):,} / {len(queries):,}")
        tr.unload()
    return [done[i] for i in range(len(queries))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--translator", required=True, choices=TRANSLATORS)
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir(args.dataset)
    if args.max_queries is not None:
        ds = _first_queries(ds, args.max_queries)
    translated = translate_cached(args.translator, ds.queries, args.device)
    enc = Encoder(MODELS[args.index_model], args.device)
    store = enc.docs_store(ds.titles, ds.texts)
    Q_mt, Q_native = enc.queries(translated, ds.task), enc.queries(ds.index_lang_queries, ds.task)
    enc.unload()
    ranked_mt = cached_search(Q_mt, store, 1000, args.device, exclude=ds.self_rows)
    ranked_native = cached_search(Q_native, store, 1000, args.device, exclude=ds.self_rows)
    pq_mt, pq_native = per_query(ranked_mt, ds.qrels), per_query(ranked_native, ds.qrels)
    rec = {"dataset": ds.name, "translator": args.translator, "index_model": args.index_model, "n_queries": len(ds.queries),
           "first_stage": {m: float(v.mean()) for m, v in pq_mt.items()}, "native": {m: float(v.mean()) for m, v in pq_native.items()},
           "exact_match_human": float(np.mean([a.strip().lower() == b.strip().lower() for a, b in zip(translated, ds.index_lang_queries)]))}
    rr = Reranker(RERANKERS["bge-reranker-v2-m3"], args.device)
    docs = _DocTexts(ds.titles, ds.texts)
    rec["rerank"] = {}
    for pool in POOLS:
        native = per_query(rerank(rr, ranked_native, pool, ds.index_lang_queries, docs, ds.task), ds.qrels)
        en = per_query(rerank(rr, ranked_mt, pool, translated, docs, ds.task), ds.qrels)
        orig = per_query(rerank(rr, ranked_mt, pool, ds.queries, docs, ds.task), ds.qrels)
        rec["rerank"][str(pool)] = {"native": float(native["ndcg@5"].mean()), "mt_translated": float(en["ndcg@5"].mean()), "mt_original": float(orig["ndcg@5"].mean())}
        durable.progress(f"{ds.name} {args.translator} pool {pool}: {rec['rerank'][str(pool)]}")
    durable.append_lines(args.out, [json.dumps(rec)])
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
