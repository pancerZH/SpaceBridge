"""Does precision or device change what a reranker ranks? fp32 on the GPU, fp32 on the CPU (``cpu``: what CPU serving runs), bf16 (what
every benchmark scored in, on the GPU) and, for
bge-reranker-v2-m3, int8 (ONNX, on the CPU), on the same candidates: the index model's own first stage, with no bridge, so only the
reranker differs. Each record also has the tokens per pair of the reranked top 5 and of the pool, which shows a pull towards length.

Each variant's nDCG@5 by pool, and its paired per-query difference from fp32 with a 95% bootstrap CI. bf16 scores go through a sigmoid
and have few distinct values, so ties are common, and a tie keeps the first-stage order. The count of distinct scores is in the record.

One JSON line per (dataset, queries, pool, variant) goes to --out, and a rerun skips the dataset if all are there. Scores go through
the crash-safe rerank cache, so a rerun only scores what's missing. fp32 has a cache of its own and leaves the others untouched.

    uv run --extra onnx python -u scripts/check_reranker_precision.py msmarco-1m --queries 100 --out precision.jsonl
    uv run python -u scripts/check_reranker_precision.py nq-1m --queries 100 --reranker bge-reranker-base --variants fp32 bf16 --out length.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from spacebridge.encoders import token_lengths
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query
from spacebridge_bench.models import MODELS
from spacebridge_bench.rerank import RERANKERS, Reranker, RerankerSpec, topk_jaccard
from spacebridge_bench.run import _DocTexts, _first_queries
from spacebridge_bench.search import cached_search

BOOTSTRAPS = 2_000


class Fp32Reranker(Reranker):
    """A reranker in fp32 on the GPU. The same dtype as a CPU run, but other kernels: the ``cpu`` variant measures the CPU itself, and on
    four corpora its top 5 matched this one exactly (outputs/review/followup/precision_cpu.jsonl)."""

    precision = "fp32"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.spec.hf_id, device=self.device, model_kwargs={"torch_dtype": torch.float32}, max_length=self.spec.max_length)
        return self._model


def paired_ci(diff: np.ndarray) -> list[float]:
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), (BOOTSTRAPS, len(diff)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--queries", type=int, required=True)
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--reranker", default="bge-reranker-v2-m3", choices=["bge-reranker-v2-m3", "bge-reranker-base"])
    p.add_argument("--variants", nargs="+", default=["fp32", "bf16", "int8"], choices=["fp32", "bf16", "int8", "cpu"], help="int8 needs --extra onnx and bge-reranker-v2-m3; cpu is fp32 on the CPU")
    p.add_argument("--threads", type=int, default=10, help="CPU threads for the cpu variant")
    p.add_argument("--pools", type=int, nargs="+", default=[60, 100, 300])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    spec = RERANKERS[args.reranker]
    specs = {"fp32": RerankerSpec(f"{spec.key}-fp32", spec.hf_id, instructed=False), "bf16": spec, "int8": RERANKERS["bge-reranker-v2-m3-int8"],
             "cpu": spec}  # the reranker itself, on the CPU: its cache is the CPU namespace, so CPU scores never mix with GPU ones
    variants = {name: specs[name] for name in args.variants}
    assert "fp32" in variants and ("int8" not in variants or args.reranker == "bge-reranker-v2-m3")
    cell = lambda r: (r.get("reranker", "bge-reranker-v2-m3"), r["dataset"], r["n_queries"], r["pool"], r["variant"])
    wanted = {(args.reranker, args.dataset, args.queries, pool, name) for pool in args.pools for name in variants}
    done = {cell(r) for r in durable.read_jsonl(args.out)[0]} if args.out.exists() else set()
    if wanted <= done:
        print("all on disk")
        return

    ds = _first_queries(load_beir(args.dataset), args.queries)
    n = len(ds.queries)
    enc = Encoder(MODELS[args.index_model], "cuda")
    store = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    Q = enc.queries(ds.index_lang_queries, ds.task)
    depth = max(args.pools)
    ranked = cached_search(Q, store, depth, "cuda", exclude=ds.self_rows)[:, :depth]
    enc.unload()
    docs = _DocTexts(ds.titles, ds.texts)
    queries = [ds.queries[i] for i in range(n) for _ in range(depth)]
    texts = [docs[j] for j in ranked.ravel()]
    tokens = token_lengths(AutoTokenizer.from_pretrained(spec.hf_id), queries, texts, max_length=spec.max_length).reshape(n, depth)

    scores = {}
    for name, variant in variants.items():
        if name == "cpu":
            torch.set_num_threads(args.threads)
        rr = (Fp32Reranker if name == "fp32" else Reranker)(variant, "cpu" if name == "cpu" else "cuda")
        scores[name] = rr.score(queries, texts, ds.task).reshape(n, depth)
        rr.unload()

    records = []
    for pool in args.pools:
        rankings, top5_tokens = {}, {}
        for name, s in scores.items():
            order = np.argsort(-s[:, :pool], axis=1, kind="stable")
            rankings[name] = np.take_along_axis(ranked[:, :pool], order, axis=1)
            top5_tokens[name] = float(np.take_along_axis(tokens[:, :pool], order[:, :5], axis=1).mean())
        base = per_query(rankings["fp32"], ds.qrels)["ndcg@5"]
        for name, ranking in rankings.items():
            mine = per_query(ranking, ds.qrels)["ndcg@5"]
            records.append({
                "reranker": args.reranker, "dataset": args.dataset, "n_queries": n, "index_model": args.index_model, "pool": pool, "variant": name, "ndcg@5": float(mine.mean()),
                "difference_from_fp32": float((mine - base).mean()), "ci_difference": paired_ci(mine - base),
                "top5_jaccard_with_fp32": float(topk_jaccard(ranking, rankings["fp32"]).mean()),
                "distinct_scores": len(np.unique(scores[name][:, :pool])), "scores": scores[name][:, :pool].size,
                "top5_tokens": top5_tokens[name], "pool_tokens": float(tokens[:, :pool].mean()),
            })
            print(f"{args.dataset} pool {pool} {name}: nDCG@5 {mine.mean():.4f}, from fp32 {100 * (mine - base).mean():+.2f} points "
                  f"[{100 * records[-1]['ci_difference'][0]:+.2f}, {100 * records[-1]['ci_difference'][1]:+.2f}], top-5 overlap {records[-1]['top5_jaccard_with_fp32']:.3f}", flush=True)
    durable.append_lines(args.out, [json.dumps(r) for r in records if cell(r) not in done])
    durable.progress(f"{args.dataset}: precision check done, {n} queries")


if __name__ == "__main__":
    main()
