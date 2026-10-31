"""Benchmark runner: tier/pair x ablation rows -> JSONL (design §7.2–§7.4).

Example:
    uv run spacebridge-bench --dataset scifact --tier 1 3
    uv run spacebridge-bench --dataset scifact --pair bge-large-en:qwen3-0.6b --rows 0a 0b 6 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spacebridge import CalibrationBundle, SpaceBridge, extract_query_proxy, make_split, sample_distractors, select_anchors

from .datasets import DATASETS, RetrievalDataset, load_beir
from .encode import Encoder
from .metrics import evaluate, retention
from .models import MODELS
from .search import search

ROWS = {
    "0a": "Oracle full re-index (M_tgt docs + M_tgt queries)",
    "0b": "M_src native (ceiling)",
    "1": "Identity sanity (M_tgt := M_src, full pipeline)",
    "2": "Uncentered full-SVD Procrustes, doc anchors",
    "3s": "+ centering + mean restore + scaled truncated Stiefel, doc anchors",
    "3r": "+ centering + mean restore + ridge (lambda sweep), doc anchors",
    "3b": "Row 3 (best Stage-1 op) without mean restore",
    "6": "+ query-proxy pairs (best Stage-1 op)",
    "6b": "Row 6 without mean restore",
    "8": "+ Stage-2 InfoNCE + KL refinement",
}

# row -> (anchor embeddings used as Y, fit mode, pair type, restore source mean)
BRIDGE_ROWS = {
    "1": ("src_query", "auto", "query", True),
    "2": ("tgt_doc", "naive", "doc", True),
    "3s": ("tgt_doc", "stiefel", "doc", True),
    "3r": ("tgt_doc", "ridge", "doc", True),
    "3b": ("tgt_doc", "stage1", "doc", False),
    "6": ("tgt_query", "stage1", "query", True),
    "6b": ("tgt_query", "stage1", "query", False),
    "8": ("tgt_query", "auto", "query", True),
}

TIERS = {
    "1": [("bge-base-en", "bge-large-en"), ("bge-large-en", "bge-base-en")],
    "3": [
        ("bge-large-en", "qwen3-0.6b"), ("qwen3-0.6b", "bge-large-en"),
        ("bge-large-en", "harrier-0.6b"), ("harrier-0.6b", "bge-large-en"),
    ],
}

SEARCH_K = 100


def run_pair(
    ds: RetrievalDataset, src_key: str, tgt_key: str, K: int, seed: int, rows: list[str], device: str
) -> list[dict]:
    src, tgt = MODELS[src_key], MODELS[tgt_key]
    K = min(K, ds.N // 10)  # §4.1: K <= N/10

    enc = Encoder(src, device)
    E_src = enc.docs(ds.titles, ds.texts)
    Q_src = enc.queries(ds.queries, ds.task)
    anchor_ids = select_anchors(E_src, K, seed)
    proxies = [extract_query_proxy(ds.titles[i], ds.texts[i]) for i in anchor_ids]
    train_idx, val_idx = make_split(K, seed=seed)
    bundle = CalibrationBundle(
        src_lang=ds.lang,
        anchor_ids=anchor_ids,
        anchor_doc_texts=[ds.texts[i] for i in anchor_ids],
        anchor_query_texts=proxies,
        X_doc_raw=E_src[anchor_ids],
        X_query_raw=enc.queries(proxies, ds.task),
        train_idx=train_idx,
        val_idx=val_idx,
        distractor_ids=sample_distractors(ds.N, anchor_ids, seed=seed),
        mu_x_doc=E_src.mean(0),
    )
    enc.unload()

    enc = Encoder(tgt, device)
    E_tgt = enc.docs(ds.titles, ds.texts)
    Q_tgt = enc.queries(ds.queries, ds.task)
    Y_query, mount = enc.timed_encode([tgt.format_query(p, ds.task) for p in proxies])
    enc.unload()
    mount["projected_encode_ms_K2048"] = mount["encode_ms"] * 2048 / K

    native = evaluate(search(Q_src, E_src, SEARCH_K, device), ds.qrels)
    oracle = evaluate(search(Q_tgt, E_tgt, SEARCH_K, device), ds.qrels)
    anchor_Y = {"src_query": bundle.X_query_raw, "tgt_query": Y_query, "tgt_doc": E_tgt[anchor_ids]}
    X_distractor = E_src[bundle.distractor_ids]

    base = {"dataset": ds.name, "src": src_key, "tgt": tgt_key, "K": K, "seed": seed, "n_queries": len(ds.queries)}
    records = []
    for row in rows:
        rec = {**base, "row": row, "description": ROWS[row]}
        if row in ("0a", "0b"):
            rec["metrics"] = oracle if row == "0a" else native
        else:
            y_kind, mode, pairs, restore = BRIDGE_ROWS[row]
            bridge = SpaceBridge(device=device, restore_source_mean=restore)
            fit = bridge.fit(bundle, anchor_Y[y_kind], mode=mode, pairs=pairs, X_distractor=X_distractor)
            Q = Q_src if row == "1" else Q_tgt
            Q_hat = bridge.transform_query(Q)
            rec["metrics"] = evaluate(search(Q_hat, E_src, SEARCH_K, device), ds.qrels)
            rec["fit"] = fit
            rec["projection_us"] = _projection_latency_us(bridge, Q[0])
            if row != "1":
                rec["mount"] = mount
        rec["rho_oracle"], rec["rho_ceil"] = retention(rec["metrics"], native, oracle)
        rec["rho_native"] = {m: v / native[m] if native[m] > 0 else float("nan") for m, v in rec["metrics"].items()}
        records.append(rec)
    return records


def _projection_latency_us(bridge: SpaceBridge, q: np.ndarray, n: int = 1000) -> float:
    bridge.transform_query(q)
    t0 = time.perf_counter()
    for _ in range(n):
        bridge.transform_query(q)
    return 1e6 * (time.perf_counter() - t0) / n


def print_table(records: list[dict]) -> None:
    r0 = records[0]
    print(f"\n{r0['dataset']}  {r0['src']} (index) <- {r0['tgt']} (queries)  K={r0['K']}  n_q={r0['n_queries']}")
    header = f"{'row':>4}  {'R@10':>6} {'R@30':>6} {'nDCG@10':>7}  {'ρceil@30':>8} {'ρorac@30':>8}  {'ρ_val':>6}  {'solver':<16} status"
    print(header)
    print("-" * len(header))
    for r in records:
        m, fit = r["metrics"], r.get("fit", {})
        solver = fit.get("solver", "")
        if "lambda_rel" in fit:
            solver += f" λ={fit['lambda_rel']}"
        if "stage2_best_step" in fit:
            solver += f" +S2@{fit['stage2_best_step']}"
        rho_val = f"{fit['rho_val']:.3f}" if "rho_val" in fit else ""
        print(
            f"{r['row']:>4}  {m['recall@10']:6.3f} {m['recall@30']:6.3f} {m['ndcg@10']:7.3f}  "
            f"{r['rho_ceil']['recall@30']:8.3f} {r['rho_oracle']['recall@30']:8.3f}  {rho_val:>6}  {solver:<16} {fit.get('status', '')}"
        )
    mount = next((r["mount"] for r in records if "mount" in r), None)
    if mount:
        print(
            f"mount: encode {mount['encode_ms']:.0f} ms for {mount['n_inputs']} proxies "
            f"({mount['tokens_per_s']:.0f} tok/s; ~{mount['projected_encode_ms_K2048']:.0f} ms at K=2048)"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact", choices=sorted(DATASETS))
    p.add_argument("--tier", nargs="*", default=[], choices=sorted(TIERS))
    p.add_argument("--pair", nargs="*", default=[], help="src:tgt model keys, e.g. bge-large-en:qwen3-0.6b")
    p.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    p.add_argument("--K", type=int, default=2048, help="anchor budget (capped at N/10)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("outputs/results.jsonl"))
    args = p.parse_args()

    pairs = [tuple(s.split(":")) for s in args.pair] + [pr for t in args.tier for pr in TIERS[t]]
    if not pairs:
        p.error("give --tier and/or --pair")
    for src, tgt in pairs:
        for key in (src, tgt):
            if key not in MODELS:
                p.error(f"unknown model {key!r}; choose from {sorted(MODELS)}")

    ds = load_beir(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for src, tgt in pairs:
        records = run_pair(ds, src, tgt, args.K, args.seed, args.rows, args.device)
        with args.out.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        print_table(records)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
