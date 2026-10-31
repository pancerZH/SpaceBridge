"""Corpus-difficulty features computable at mount from the index alone (M3 L6).

For every fit in the given spacebridge-bench JSONL files, recomputes what the engine sees at mount (anchor
docs, their proxies, the distractor pool of the record's seed) and measures label-free corpus properties.
Writes one line per (dataset, src, K, seed, anchor version); ``calibrate_quality.py --features`` joins them.

    uv run python scripts/difficulty_features.py outputs/m2_*.jsonl --out outputs/m3_difficulty.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from spacebridge import sample_distractors
from spacebridge.anchors import count_tokens
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import ANCHOR_CACHE, make_proxy

N_SAMPLE = 2_000


def anchor_version(path: Path) -> str:
    """M2 files before the torch k-means change used sklearn anchors (cache files without ``_tkm``)."""
    return "sklearn" if path.name.startswith(("m2_", "m2d_")) else "tkm"


def feature_key(r: dict, path: Path) -> tuple:
    """(dataset, src, K, seed, anchor version[, proxy style when not lead]); lead keys predate the style."""
    key = (r["dataset"], r["src"], r["K"], r.get("seed", 0), anchor_version(path))
    return key + ((r["proxy"],) if r.get("proxy", "lead") != "lead" else ())


def features(ds, E: np.ndarray, Xq: np.ndarray, ids: np.ndarray, proxies: list[str], seed: int) -> dict:
    dis = sample_distractors(ds.N, ids, seed=seed)
    pool = np.concatenate([E[ids], E[dis]])
    rng = np.random.default_rng(0)
    s = rng.choice(len(pool), min(N_SAMPLE, len(pool)), replace=False)
    S = pool[s] @ pool.T
    S[np.arange(len(s)), s] = -np.inf
    top = -np.sort(-S, axis=1)[:, :30]
    finite = np.where(np.isfinite(S), S, np.nan)
    gap = top[:, 0] - top[:, -1]
    dsample = E[dis[rng.choice(len(dis), min(N_SAMPLE, len(dis)), replace=False)]]
    own = np.einsum("ij,ij->i", Xq, E[ids])
    rank_own = ((Xq @ pool.T) > own[:, None]).sum(1)
    return {
        "log10_n": float(np.log10(ds.N)),
        "pool_gap": float(gap.mean()),
        "pool_gap_sd": float((gap / np.nanstd(finite, axis=1)).mean()),
        "pool_top1": float(top[:, 0].mean()),
        "anisotropy": float(np.linalg.norm(pool.mean(0))),
        "coverage": float((dsample @ E[ids].T).max(1).mean()),
        "proxy_own_cos": float(own.mean()),
        "proxy_own_top1": float((rank_own == 0).mean()),
        "proxy_doc_len_ratio": float(np.mean([count_tokens(p) for p in proxies]) / np.mean([count_tokens(ds.texts[i]) for i in ids])),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("outputs/m3_difficulty.jsonl"))
    args = p.parse_args()
    done = set()
    if args.out.exists():
        done = {tuple(json.loads(l)["key"]) for l in args.out.open()}
    keys = set()
    for path in args.results:
        for line in path.open():
            r = json.loads(line)
            if "quality" in r.get("fit", {}) and "reranker" not in r and r["row"] != "1":
                keys.add(feature_key(r, path))
    todo = sorted(keys - done)
    print(f"{len(keys)} keys, {len(todo)} to compute")
    cache = {}
    for key in todo:
        dataset, src, K, seed, version = key[:5]
        style = key[5] if len(key) > 5 else "lead"
        if dataset not in cache:
            cache.clear()
            cache[dataset] = load_beir(dataset)
        ds = cache[dataset]
        if (dataset, src) not in cache:
            cache[dataset, src] = Encoder(MODELS[src], "cuda").docs(ds.titles, ds.texts)
        E = cache[dataset, src]
        allowed = np.setdiff1d(np.arange(ds.N), ds.self_rows)
        suffix = "_tkm" if version == "tkm" else ""
        path = ANCHOR_CACHE / f"{ds.name}_{src}_N{ds.N}_n{len(allowed)}_K{K}_s{seed}{suffix}.npy"
        if not path.exists():
            print(f"missing anchors {path}")
            continue
        ids = np.load(path)
        proxies = [make_proxy(ds.titles[i], ds.texts[i], style) for i in ids]
        enc = Encoder(MODELS[src], "cuda")
        Xq = enc.queries(proxies, ds.task)
        enc.unload()
        f = features(ds, E, Xq, ids, proxies, seed)
        with args.out.open("a") as fh:
            fh.write(json.dumps({"key": list(key), **f}) + "\n")
        print(dataset, src, K, seed, version, {k: round(v, 3) for k, v in f.items()})


if __name__ == "__main__":
    main()
