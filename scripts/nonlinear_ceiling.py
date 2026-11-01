"""W4 candidate 3: is the linear map class the limit? A generous test no mount could afford.

Fits a linear map and a residual MLP on up to 200K real MS MARCO train query pairs (target query -> native source query) and
measures, on the held-out dev queries of msmarco-1m, cosine to the native query and Recall@30 retention. A gain under 0.02
in cosine closes the question: the limit is information, not the map class. Writes one JSON line per fit as it finishes.

    uv run python -u scripts/nonlinear_ceiling.py --pair bge-large-en:qwen3-0.6b --out x.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from spacebridge_bench import durable
from spacebridge_bench.datasets import load_beir
from spacebridge_bench.encode import Encoder
from spacebridge_bench.metrics import per_query, retention
from spacebridge_bench.models import MODELS
from spacebridge_bench.search import cached_search, search

DEV = "cuda"


class ResidualMLP(torch.nn.Module):
    """y -> mean + linear(y) + mlp(y): starts at the linear map, so it can only add to it."""

    def __init__(self, W: torch.Tensor, mu_y: torch.Tensor, mu_x: torch.Tensor, width: int, depth: int, dropout: float):
        super().__init__()
        self.register_buffer("mu_y", mu_y)
        self.register_buffer("mu_x", mu_x)
        self.W = torch.nn.Parameter(W.clone())
        layers: list[torch.nn.Module] = []
        d = W.shape[0]
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            d = width
        last = torch.nn.Linear(d, W.shape[1])
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        self.net = torch.nn.Sequential(*layers, last)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        yc = y - self.mu_y
        return F.normalize(self.mu_x + yc @ self.W + self.net(yc * 10), dim=-1)


def ridge(Y: torch.Tensor, X: torch.Tensor, lam_rel: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_y, mu_x = Y.mean(0), X.mean(0)
    Yc, Xc = (Y - mu_y).double(), (X - mu_x).double()
    A = Yc.T @ Yc
    A.diagonal().add_(lam_rel * A.diagonal().mean())
    return torch.linalg.solve(A, Yc.T @ Xc).float(), mu_y, mu_x


def cos_to_native(Q_hat: torch.Tensor, X: torch.Tensor) -> float:
    return F.cosine_similarity(Q_hat, F.normalize(X, dim=-1), dim=-1).mean().item()


def train_mlp(Y_tr, X_tr, Y_val, X_val, W, mu_y, mu_x, width, depth, dropout, epochs=40, batch=2048, lr=1e-3, wd=1e-2):
    model = ResidualMLP(W, mu_y, mu_x, width, depth, dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(Y_tr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    best, best_state, gen = -1.0, None, torch.Generator(device=DEV).manual_seed(0)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Y_tr), generator=gen, device=DEV)
        for s in range(0, len(perm) - batch + 1, batch):
            sel = perm[s : s + batch]
            loss = 1 - F.cosine_similarity(model(Y_tr[sel]), X_tr[sel], dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.inference_mode():
            c = cos_to_native(model(Y_val), X_val)
        if c > best:
            best, best_state = c, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs="+", required=True)
    p.add_argument("--n-train", type=int, nargs="+", default=[32_768, 100_000, 200_000])
    p.add_argument("--mlp", nargs="+", default=["2048x2", "4096x3"], help="widthxdepth")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ds = load_beir("msmarco-1m")
    qs = load_dataset("mteb/msmarco", "queries", split="queries")
    qtext = dict(zip(map(str, qs["_id"]), qs["text"]))
    tr = load_dataset("mteb/msmarco", "default", split="train")
    tr_ids = sorted({str(q) for q, s in zip(tr["query-id"], tr["score"]) if s > 0})
    dev = set(ds.queries)
    train_texts = [qtext[q] for q in tr_ids if q in qtext and qtext[q] not in dev]
    order = np.random.default_rng(0).permutation(len(train_texts))
    train_texts = [train_texts[i] for i in order[: max(args.n_train) + 4_000]]
    durable.progress(f"{len(train_texts):,} train queries")
    done = {r["key"] for r in durable.read_jsonl(args.out)[0]} if args.out.exists() else set()

    def emit(key: str, **rec) -> None:
        durable.append_lines(args.out, [json.dumps({"key": key, **rec})])
        durable.progress(f"{key}: {rec}")

    for pair in args.pair:
        src_key, tgt_key = pair.split(":")
        src, tgt = Encoder(MODELS[src_key], DEV), Encoder(MODELS[tgt_key], DEV)
        E_src, E_tgt = src.docs_store(ds.titles, ds.texts), tgt.docs_store(ds.titles, ds.texts)
        X_all, Q_src = src.queries(train_texts, ds.task), src.queries(ds.queries, ds.task)
        src.unload()
        Y_all, Q_tgt = tgt.queries(train_texts, ds.task), tgt.queries(ds.queries, ds.task)
        tgt.unload()
        native = per_query(cached_search(Q_src, E_src, 1000, DEV), ds.qrels)
        oracle = per_query(cached_search(Q_tgt, E_tgt, 1000, DEV), ds.qrels)
        native_m, oracle_m = {m: float(v.mean()) for m, v in native.items()}, {m: float(v.mean()) for m, v in oracle.items()}
        Y_val, X_val = (torch.as_tensor(a[-4_000:], device=DEV) for a in (Y_all, X_all))
        Y_dev, X_dev = torch.as_tensor(Q_tgt, device=DEV), torch.as_tensor(Q_src, device=DEV)

        def evaluate(project) -> dict:
            with torch.inference_mode():
                Q_hat = project(Y_dev)
            m = {k: float(v.mean()) for k, v in per_query(search(Q_hat.cpu().numpy(), E_src, 1000, DEV), ds.qrels).items()}
            _, rho = retention(m, native_m, oracle_m)
            return {"cos_dev": cos_to_native(Q_hat, X_dev), "rho_recall30": rho["recall@30"], "rho_recall300": rho["recall@300"], "recall30": m["recall@30"]}

        for n in args.n_train:
            Y_tr, X_tr = torch.as_tensor(Y_all[:n], device=DEV), torch.as_tensor(X_all[:n], device=DEV)
            key = f"{pair}|n{n}|linear"
            best = None
            for lam in (1e-3, 1e-2, 1e-1):
                W, mu_y, mu_x = ridge(Y_tr, X_tr, lam)
                project = lambda y: F.normalize(mu_x + (y - mu_y) @ W, dim=-1)
                c = cos_to_native(project(Y_val), X_val)
                if best is None or c > best[0]:
                    best = (c, lam, W, mu_y, mu_x)
            c, lam, W, mu_y, mu_x = best
            if key not in done:
                emit(key, pair=pair, n=n, map="linear", lam=lam, cos_val=c, **evaluate(lambda y: F.normalize(mu_x + (y - mu_y) @ W, dim=-1)))
            for spec in args.mlp:
                key = f"{pair}|n{n}|mlp{spec}"
                if key in done:
                    continue
                width, depth = map(int, spec.split("x"))
                t0 = time.perf_counter()
                model, cos_val = train_mlp(Y_tr, X_tr, Y_val, X_val, W, mu_y, mu_x, width, depth, 0.1)
                emit(key, pair=pair, n=n, map=f"mlp{spec}", cos_val=cos_val, train_s=time.perf_counter() - t0, **evaluate(model))
                del model
                torch.cuda.empty_cache()
            del Y_tr, X_tr
        del Y_val, X_val, Y_dev, X_dev
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
