"""W5: wall-clock of one mount by query model, and the query model's re-indexing speed. Run it on an idle machine.

``mount`` times one mount in a fresh process, from loading the model to the fitted bridge, with nothing cached for the
query model. ``rate`` encodes a sample of the corpus for re-indexing time at any size (tokens per document / tokens per
second). Use ``CUDA_VISIBLE_DEVICES=`` for CPU numbers. One JSON line goes to --out per configuration; a rerun skips it.

    uv run python -u scripts/mount_timing.py mount msmarco-1m-nested --model qwen3-0.6b --index-model bge-large-en --K 8192 --device cuda --out t.jsonl
    CUDA_VISIBLE_DEVICES= uv run python -u scripts/mount_timing.py rate msmarco-1m-nested --model qwen3-0.6b --device cpu --threads 10 --out t.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spacebridge import Index, SentenceTransformerEncoder, freeze, mount
from spacebridge.encoders import token_lengths
from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import _first_docs


def live_encoder(key: str, task: str, device: str) -> SentenceTransformerEncoder:
    spec = MODELS[key]
    return SentenceTransformerEncoder(
        spec.hf_id, spec.query_template.replace("{task}", task), spec.doc_template, device,
        batch_tokens=spec.batch_tokens if device.startswith("cuda") else None, max_seq_length=spec.max_seq_length,
        processor_kwargs=spec.tokenizer_kwargs, config_kwargs=spec.config_kwargs or None,
    )


def time_mount(args, ds) -> dict:
    store = Encoder(MODELS[args.index_model], args.device).docs_store(ds.titles, ds.texts).prefix(ds.N)
    index = ds.index_over(store)
    exclude = np.union1d(sorted({d for rel in ds.qrels for d in rel}), ds.self_rows[ds.self_rows >= 0])
    src = MountEncoder(MODELS[args.index_model], ds.task, args.device)
    bundle = freeze(index, src, ds.lang, args.seed, args.K, anchor_select="random", exclude=exclude, device=args.device)
    src.unload()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    enc = live_encoder(args.model, ds.task, args.device)
    load_s = time.perf_counter() - t0
    bridge = mount(bundle, index, enc, device=args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    rec = {"mount_s": time.perf_counter() - t0, "load_s": load_s, "stages_ms": bridge.metrics["mount"], "K": bundle.K, "index_model": args.index_model}
    if args.device.startswith("cuda"):
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return rec


def time_rate(args, ds) -> dict:
    spec = MODELS[args.model]
    enc = live_encoder(args.model, ds.task, args.device)
    rows = np.random.default_rng(args.seed).choice(ds.N, min(args.n, ds.N), replace=False)
    titles, texts = [ds.titles[i] for i in rows], [ds.texts[i] for i in rows]
    enc.encode_docs(titles[:32], texts[:32])  # warmup
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    enc.encode_docs(titles, texts)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    inputs = [spec.format_doc(a, b) for a, b in zip(titles, texts)]
    tokens = token_lengths(enc.model.tokenizer, inputs, max_length=spec.max_seq_length)
    return {"seconds": seconds, "n_docs": len(rows), "tokens_per_doc": float(tokens.mean()), "tokens_per_s": float(tokens.sum() / seconds), "docs_per_s": len(rows) / seconds}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("mount", "rate"))
    p.add_argument("dataset")
    p.add_argument("--model", required=True, help="the query model")
    p.add_argument("--index-model", default="bge-large-en")
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=2_000, help="rate: documents to encode")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    key = f"{args.mode}|{args.dataset}|{args.model}|{args.device}|t{args.threads}" + (f"|K{args.K}" if args.mode == "mount" else "") + (f"|N{args.max_docs}" if args.max_docs else "")
    if args.out.exists() and key in {r["key"] for r in durable.read_jsonl(args.out)[0]}:
        durable.progress(f"{key}: already recorded")
        return
    torch.set_num_threads(args.threads)
    ds = load_beir(args.dataset)
    if args.max_docs is not None:
        ds = _first_docs(ds, args.max_docs)
    rec = time_mount(args, ds) if args.mode == "mount" else time_rate(args, ds)
    rec = {"key": key, "mode": args.mode, "dataset": ds.name, "N": ds.N, "model": args.model, "device": args.device, "threads": args.threads, **rec}
    durable.append_lines(args.out, [json.dumps(rec)])
    durable.progress(f"{key}: {json.dumps({k: v for k, v in rec.items() if isinstance(v, float)})}")


if __name__ == "__main__":
    main()
