"""Is a fit through the sentence-transformers adapter biased against one through the benchmark's cached encoders, or only noisy?
Same anchors and the same test queries through both maps, first stage only, each seed. The adapter encodes fresh in fp32; the cached
encoder reads float16 files written in bigger batches.

    uv run python -u scripts/fit_compare.py xquad-zh --index-model bge-large-en --query-model qwen3-0.6b --K 2048 --seed 0 1 2 --out out.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from searcher_check import adapter  # noqa: E402

from spacebridge import Index, default_anchor_count, freeze, mount  # noqa: E402
from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.datasets import load_beir, load_shift_bank  # noqa: E402
from spacebridge_bench.encode import Encoder, MountEncoder  # noqa: E402
from spacebridge_bench.metrics import per_query  # noqa: E402
from spacebridge_bench.models import MODELS  # noqa: E402
from spacebridge_bench.run import _first_queries, cached_anchors  # noqa: E402
from spacebridge_bench.search import search  # noqa: E402

FIELDS = ("rho_val", "stage2_gain", "stage2_best_step", "stage2_averaged", "solver", "lambda_rel")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--query-model", default="qwen3-0.6b")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir(args.dataset)
    if args.max_queries:
        ds = _first_queries(ds, args.max_queries)
    enc = Encoder(MODELS[args.index_model], args.device)
    store = enc.docs_store(ds.titles, ds.texts).prefix(ds.N)
    enc.unload()
    enc = Encoder(MODELS[args.query_model], args.device)
    Q = np.asarray(enc.queries(ds.queries, ds.task), dtype=np.float32)
    enc.unload()
    index = Index(store, ds.texts, ds.titles)
    lang = ds.query_lang
    banks = {lang: load_shift_bank(lang)[:2]} if lang else None
    K = args.K or default_anchor_count(ds.N)
    for seed in args.seed:
        anchors = cached_anchors(ds, args.index_model, store, np.setdiff1d(np.arange(ds.N), ds.self_rows), K, seed)
        src = MountEncoder(MODELS[args.index_model], ds.task, args.device)
        bundle = freeze(index, src, ds.lang, seed, K, anchor_ids=anchors, device=args.device)
        src.unload()
        rec = {"dataset": ds.name, "src": args.index_model, "tgt": args.query_model, "seed": seed, "K": K}
        maps = {}
        for name in ("cached", "adapter"):
            tgt = MountEncoder(MODELS[args.query_model], ds.task, args.device) if name == "cached" else adapter(args.query_model, ds.task, args.device)
            bridge = mount(bundle, index, tgt, [lang] if lang else [], device=args.device, shift_banks=banks)
            if name == "cached":
                tgt.unload()
            del tgt
            q_hat = bridge.transform_query(Q, query_lang=lang)
            maps[name] = q_hat
            ranked = search(q_hat, store, 300, args.device, exclude=ds.self_rows)
            pq = per_query(ranked, ds.qrels)
            rec[name] = {"recall@30": float(pq["recall@30"].mean()), "recall@300": float(pq["recall@300"].mean()), "ndcg@10": float(pq["ndcg@10"].mean()),
                         **{f: bridge.metrics.get(f) for f in FIELDS}}
        rec["q_hat_cosine"] = float((maps["cached"] * maps["adapter"]).sum(1).mean())
        durable.append_lines(args.out, [json.dumps(rec)])
        durable.progress(f"seed {seed}: recall@30 cached {rec['cached']['recall@30']:.4f}, adapter {rec['adapter']['recall@30']:.4f}; cosine of the mapped queries {rec['q_hat_cosine']:.5f}; steps {rec['cached']['stage2_best_step']}, {rec['adapter']['stage2_best_step']}")


if __name__ == "__main__":
    main()
