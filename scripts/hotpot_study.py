"""Why does the map lose more on HotpotQA than on MS MARCO and NQ at 1M? First-stage experiments, no reranking.

Each experiment appends one synced JSON line per unit to --out/<exp>.jsonl and skips units already there, so a rerun
continues. Fits are kept under --out/fits/, so later experiments start in seconds. Train-split queries stand in for a
query log and never overlap the 2,000 test queries; NQ has no train split, so it uses its unused test queries.

    dose     same map, scaled error: does HotpotQA punish the same error more, or is its error larger?
    offset   how much of the error is one shared vector, and does recentering on real queries remove it?
    fits     real-query pairs, alone and added to the proxies, shrunk and chosen by cosine: shows cosine misleads
    margin   loss by a positive's native margin in units of the map's noise: one curve for every corpus?
    signal   how much of native's score differences survive the map, and how positives fare against that trend
    fits2    the same, norm-matched and chosen by agreement with native, with a nonlinear map: the map class's ceiling
    scale    the balance between the query-specific part and the restored source mean
    tta      averaging the map's output over variants of a query
    local    the residuals of the calibration pairs nearest the query
    cross    maps of one corpus on the queries and proxies of another: is it the queries or the fit?
    form     cosine after changing the shape of the queries: multi-clause, entity-only, concatenated
    features per-positive loss by whether its title is in the query, entity rarity, re-index rank
    cover    how much of a query lies in the span of the calibration pairs, and the spectra of each corpus

    uv run python -u scripts/hotpot_study.py dose --out outputs/phase1/hp_study
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from spacebridge import Index, freeze, mount
from spacebridge.solvers import ridge_path
from spacebridge_bench import durable
from spacebridge_bench.datasets import DATASETS, load_beir
from spacebridge_bench.encode import Encoder, MountEncoder
from spacebridge_bench.metrics import per_query, retention
from spacebridge_bench.models import MODELS
from spacebridge_bench.run import _first_docs, _first_queries
from spacebridge_bench.search import cached_search, search

DEV = "cuda"
CORPORA = ("hotpotqa-1m", "msmarco-1m", "nq-1m")
PAIRS = {"T1": ("bge-large-en", "bge-base-en"), "T1b": ("bge-base-en", "bge-large-en"), "T3": ("bge-large-en", "qwen3-0.6b"),
         "T3b": ("qwen3-0.6b", "bge-large-en"), "T3c": ("harrier-0.6b", "bge-large-en")}
N_TEST, N_LOG, N_DEV = 2_000, 85_000, 500
_DS: dict = {}
_REAL: dict = {}


def dataset(name: str):
    if name not in _DS:
        base, _, size = name.partition("@")
        ds = load_beir(base)
        if size:
            ds = _first_docs(ds, int(size))
        _DS[name] = (ds, _first_queries(ds, N_TEST) if len(ds.queries) > N_TEST else ds)
    return _DS[name]


def real_queries(name: str) -> list[str]:
    """Log queries, none of them a test query; the last ``N_DEV`` are for choosing hyperparameters."""
    if name not in _REAL:
        full, test = dataset(name)
        if name != "nq-1m":
            spec = DATASETS[name]
            ids = sorted({str(q) for q in load_dataset(spec.repo, "default", split="train")["query-id"]})
            np.random.default_rng(0).shuffle(ids)
            rows = load_dataset(spec.repo, "queries", split=spec.queries_split)
            text = dict(zip(map(str, rows["_id"]), rows["text"]))
            seen = set(test.queries)
            _REAL[name] = [text[i] for i in ids[: N_LOG + N_DEV + 1000] if i in text and text[i] not in seen][: N_LOG + N_DEV]
        else:
            _REAL[name] = list(full.queries[N_TEST:])
    return _REAL[name]


def encode_queries(key: str, texts: list[str], task: str) -> np.ndarray:
    enc = Encoder(MODELS[key], DEV)
    out = np.asarray(enc.queries(texts, task), dtype=np.float32)
    enc.unload()
    return out


class Ctx:
    """One map with positives kept out of the anchors, and its test queries. K = 8,192 random anchors as ``scale_curve.py``
    fits them, or K = None for the library's default at this size."""

    def __init__(self, out: Path, corpus: str, tier: str, seed: int = 0, K: int | None = 8192):
        self.corpus, self.tier, self.seed, self.out, self.K = corpus, tier, seed, out, K
        self.src, self.tgt = PAIRS[tier]
        self.ds = ds = dataset(corpus)[1]
        e_src, e_tgt = Encoder(MODELS[self.src], DEV), Encoder(MODELS[self.tgt], DEV)
        self.E_src, self.E_tgt = e_src.docs_store(ds.titles, ds.texts).prefix(ds.N), e_tgt.docs_store(ds.titles, ds.texts).prefix(ds.N)
        self.Q_src = np.asarray(e_src.queries(ds.index_lang_queries, ds.task), dtype=np.float32)
        self.Q_tgt = np.asarray(e_tgt.queries(ds.queries, ds.task), dtype=np.float32)
        e_src.unload()
        e_tgt.unload()
        self.native = cached_search(self.Q_src, self.E_src, 1_000, DEV, exclude=ds.self_rows)
        self.oracle = cached_search(self.Q_tgt, self.E_tgt, 1_000, DEV, exclude=ds.self_rows)
        self.native_m = {m: float(v.mean()) for m, v in per_query(self.native, ds.qrels).items()}
        self.oracle_m = {m: float(v.mean()) for m, v in per_query(self.oracle, ds.qrels).items()}
        self._fit()
        self.Q_hat = self.project(self.Q_tgt)

    def _fit(self) -> None:
        tag = "" if self.K == 8192 else f"_K{self.K or 'auto'}"
        path = self.out / "fits" / f"{self.corpus}_{self.src}_{self.tgt}{tag}_s{self.seed}.npz"
        if not path.exists():
            ds = self.ds
            positives = np.array(sorted({d for rel in ds.qrels for d in rel}))
            pool = Index(self.E_src, ds.texts, ds.titles)
            src, tgt = MountEncoder(MODELS[self.src], ds.task, DEV), MountEncoder(MODELS[self.tgt], ds.task, DEV)
            select = "random" if self.K == 8192 else "auto"
            bundle = freeze(pool, src, ds.lang, self.seed, self.K, anchor_select=select, exclude=positives, device=DEV)
            src.unload()
            bridge = mount(bundle, pool, tgt, device=DEV, doc_pairs=True)
            Y_q = tgt.encode_queries(bundle.anchor_query_texts)
            Y_d = tgt.encode_docs(bundle.anchor_doc_titles, bundle.anchor_doc_texts, max_tokens=128)
            tgt.unload()
            path.parent.mkdir(parents=True, exist_ok=True)
            with durable.atomic_file(path) as f:
                np.savez(f, W=bridge.W, mu_x=bridge.mu_x_query, mu_y=bridge.mu_y_by_lang[ds.lang], Y_q=Y_q, X_q=bundle.X_query_raw,
                         Y_d=Y_d, X_d=bundle.X_doc_raw, tr=bundle.train_idx, va=bundle.val_idx,
                         proxy_words=np.array([len(p.split()) for p in bundle.anchor_query_texts]),
                         rho_val=bridge.metrics["rho_val"], solver=str(bridge.metrics.get("solver")))
        z = np.load(path)
        self.W, self.mu_x, self.mu_y = z["W"], z["mu_x"], z["mu_y"]
        self.Y_q, self.X_q, self.Y_d, self.X_d = z["Y_q"], z["X_q"], z["Y_d"], z["X_d"]
        self.tr, self.va, self.proxy_words = z["tr"], z["va"], z["proxy_words"]

    def project(self, Y: np.ndarray, W=None, mu_y=None, mu_x=None) -> np.ndarray:
        W = self.W if W is None else W
        mu_y = self.mu_y if mu_y is None else mu_y
        mu_x = self.mu_x if mu_x is None else mu_x
        q = (Y - mu_y) @ W + mu_x
        return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)

    def evaluate(self, Q_hat: np.ndarray) -> dict:
        """Retention of a first stage, cosine to the native query, and positives native had in its top 300 that left it."""
        ds = self.ds
        ranked = search(Q_hat, self.E_src, 1_000, DEV, exclude=ds.self_rows)
        m = {k: float(v.mean()) for k, v in per_query(ranked, ds.qrels).items()}
        _, ceil = retention(m, self.native_m, self.oracle_m)
        nr, br = positive_ranks(self.native, ds.qrels), positive_ranks(ranked, ds.qrels)
        lost = float(np.mean((nr <= 300) & (br > 300)))
        return {"cos": float(cos(Q_hat, self.Q_src).mean()), "lost300": lost,
                **{f"ret@{k}": ceil[f"recall@{k}"] for k in (30, 100, 300, 1000)},
                **{f"abs@{k}": m[f"recall@{k}"] for k in (100, 300, 1000)}}


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def positive_ranks(ranked: np.ndarray, qrels) -> np.ndarray:
    """1-based rank of every positive in query order, 10**6 when outside the ranking."""
    out = []
    for row, rel in zip(ranked, qrels):
        pos = {d: i + 1 for i, d in enumerate(row.tolist())}
        out.extend(pos.get(d, 10**6) for d in rel)
    return np.array(out)


class Log:
    def __init__(self, out: Path, exp: str, total: int):
        self.path, self.exp, self.total = out / f"{exp}.jsonl", exp, total
        self.done = {json.loads(l)["key"] for l in self.path.read_text().splitlines() if l.strip()} if self.path.exists() else set()
        self.n = 0

    def has(self, key: str) -> bool:
        self.n += 1
        return key in self.done

    def write(self, key: str, t0: float, **rec) -> None:
        rec = {"key": key, "seconds": round(time.time() - t0, 1), **rec}
        durable.append_lines(self.path, [json.dumps(rec)])
        durable.progress(f"exp {self.n}/{self.total} {self.exp} {key}: done")


def exp_dose(out: Path, tiers, seeds) -> None:
    alphas = (0.25, 0.5, 0.75, 1.0, 1.25)
    log = Log(out, "dose", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                r = c.Q_hat - c.Q_src
                curve = []
                for a in alphas:
                    q = c.Q_src + a * r
                    curve.append({"alpha": a, **c.evaluate(q / np.linalg.norm(q, axis=1, keepdims=True))})
                # The score noise the map adds among the documents native ranks highest, against their spread.
                rng = np.random.default_rng(0)
                stats = []
                for i in rng.choice(len(c.Q_src), 300, replace=False):
                    ids = c.native[i]
                    D = np.asarray(c.E_src[np.sort(ids)], dtype=np.float32)[np.argsort(np.argsort(ids))]
                    s, sh = D @ c.Q_src[i], D @ c.Q_hat[i]
                    noise = (sh - s).std()
                    rank_s, rank_h = np.argsort(np.argsort(-s)), np.argsort(np.argsort(-sh))
                    stats.append((s[0], s[9], s[99], s[299], s[999], s.std(), noise, noise / s.std(), np.corrcoef(rank_s, rank_h)[0, 1], s[0] - s[299]))
                st = np.mean(stats, axis=0)
                names = ("s@1", "s@10", "s@100", "s@300", "s@1000", "spread", "noise", "noise_over_spread", "spearman", "gap_1_300")
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, curve=curve, top1000=dict(zip(names, map(float, st))))


def exp_offset(out: Path, tiers, seeds) -> None:
    log = Log(out, "offset", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                real = real_queries(corpus)
                Y_real = encode_queries(c.tgt, real, c.ds.task)
                X_real = encode_queries(c.src, real, c.ds.task)
                r = c.Q_hat - c.Q_src
                rbar = r.mean(0)
                sv = torch.linalg.svdvals(torch.as_tensor(r - rbar)).pow(2)
                rec = {"baseline": c.evaluate(c.Q_hat), "shared_share": float((rbar @ rbar) / (r * r).sum(1).mean()),
                       "resid_norm": float(np.linalg.norm(r, axis=1).mean()), "top10_pc_share": float(sv[:10].sum() / sv.sum()),
                       "mu_gap_y": float(np.linalg.norm(Y_real.mean(0) - c.mu_y)), "mu_gap_x": float(np.linalg.norm(X_real.mean(0) - c.mu_x)),
                       "proxy_words": float(c.proxy_words.mean()), "proxy_words_p10_p90": [float(np.percentile(c.proxy_words, p)) for p in (10, 90)],
                       "query_words": float(np.mean([len(q.split()) for q in c.ds.queries]))}
                rec["by_n"] = {}
                for n in (10, 100, 1000, 10000):
                    if n > len(real):
                        continue
                    my, mx = Y_real[:n].mean(0), X_real[:n].mean(0)
                    rec["by_n"][n] = {"both": c.evaluate(c.project(c.Q_tgt, mu_y=my, mu_x=mx)),
                                      "tgt_only": c.evaluate(c.project(c.Q_tgt, mu_y=my)),
                                      "src_only": c.evaluate(c.project(c.Q_tgt, mu_x=mx))}
                rec["other_corpus"] = {}
                for other in CORPORA:  # offsets from another corpus's queries: the shape of a question, not this corpus's topics
                    if other != corpus:
                        ot = dataset(other)[1]
                        rec["other_corpus"][other] = c.evaluate(c.project(
                            c.Q_tgt, mu_y=encode_queries(c.tgt, ot.queries, c.ds.task).mean(0), mu_x=encode_queries(c.src, ot.queries, c.ds.task).mean(0)))
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **rec)


def best_ridge(Y_fit, X_fit, Y_dev, X_dev, mu_y, mu_x, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    Yt, Xt = torch.as_tensor(Y_fit, device=DEV).double(), torch.as_tensor(X_fit, device=DEV).double()
    best = None
    for lam, W in ridge_path(Yt, Xt, grid):
        W = W.float().cpu().numpy()
        q = (Y_dev - mu_y) @ W + mu_x
        score = float(cos(q, X_dev).mean())
        if best is None or score > best[0]:
            best = (score, lam, W)
    return best[2], best[1]


def fit_mlp(Y, X, Y_dev, X_dev, W0, mu_y, mu_x, width=2048, epochs=60, batch=1024):
    """A residual MLP on the ridge map, trained on cosine, with the best epoch on the dev queries: the map class's ceiling."""
    d_t, d_s = Y.shape[1], X.shape[1]
    W0t, muy, mux = (torch.as_tensor(a, device=DEV) for a in (W0, mu_y, mu_x))
    net = torch.nn.Sequential(torch.nn.Linear(d_t, width), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(width, width),
                              torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(width, d_s)).to(DEV)
    torch.nn.init.zeros_(net[-1].weight)
    torch.nn.init.zeros_(net[-1].bias)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Yt, Xt = torch.as_tensor(Y, device=DEV), torch.as_tensor(X, device=DEV)
    Yd, Xd = torch.as_tensor(Y_dev, device=DEV), torch.as_tensor(X_dev, device=DEV)
    forward = lambda y: F.normalize((y - muy) @ W0t + net(y - muy) + mux, dim=-1)
    best = (-1.0, None)
    for _ in range(epochs):
        net.train()
        for ids in torch.randperm(len(Yt), device=DEV).split(batch):
            loss = 1 - (forward(Yt[ids]) * Xt[ids]).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            score = float((forward(Yd) * Xd).sum(1).mean())
        if score > best[0]:
            best = (score, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    net.eval()

    def apply(y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return forward(torch.as_tensor(y, device=DEV)).cpu().numpy()
    return apply, best[0]


def exp_fits(out: Path, tiers, seeds) -> None:
    log = Log(out, "fits", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                real = real_queries(corpus)
                dev_at = len(real) - min(N_DEV, len(real) // 4)
                Y_real = encode_queries(c.tgt, real, c.ds.task)
                X_real = encode_queries(c.src, real, c.ds.task)
                Y_dev, X_dev = Y_real[dev_at:], X_real[dev_at:]
                tr = c.tr
                yq, xq, yd, xd = c.Y_q[tr], c.X_q[tr], c.Y_d[tr], c.X_d[tr]
                rec = {"baseline": c.evaluate(c.Q_hat), "n_log": dev_at, "pairs_proxy": len(tr)}
                # Stage 1 alone on the shipped pairs, on the shipped centering, with a lower lambda than the shipped grid.
                Y0 = np.vstack([yq - yq.mean(0), yd - yd.mean(0)])
                X0 = np.vstack([xq - xq.mean(0), xd - xd.mean(0)])
                W, lam = best_ridge(Y0, X0, Y_dev, X_dev, yq.mean(0), xq.mean(0))
                rec["stage1_proxy_doc"] = {"lam": lam, **c.evaluate(c.project(c.Q_tgt, W=W, mu_y=yq.mean(0), mu_x=xq.mean(0)))}
                for n in (1000, 10000, 85000):
                    if n > dev_at:
                        continue
                    y, x = Y_real[:n], X_real[:n]
                    my, mx = y.mean(0), x.mean(0)
                    W_real, lam = best_ridge(y - my, x - mx, Y_dev, X_dev, my, mx)
                    rec[f"real_{n}"] = {"lam": lam, **c.evaluate(c.project(c.Q_tgt, W=W_real, mu_y=my, mu_x=mx))}
                    W_mix, lam = best_ridge(np.vstack([Y0, y - my]), np.vstack([X0, x - mx]), Y_dev, X_dev, my, mx)
                    rec[f"mix_{n}"] = {"lam": lam, **c.evaluate(c.project(c.Q_tgt, W=W_mix, mu_y=my, mu_x=mx))}
                if dev_at >= 85000:
                    y, x = Y_real[:85000], X_real[:85000]
                    my, mx = y.mean(0), x.mean(0)
                    W_real, _ = best_ridge(y - my, x - mx, Y_dev, X_dev, my, mx)
                    apply, dev_cos = fit_mlp(y, x, Y_dev, X_dev, W_real, my, mx)
                    rec["mlp_85000"] = {"dev_cos": dev_cos, **c.evaluate(apply(c.Q_tgt))}
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **rec)


def exp_cross(out: Path, tiers, seeds) -> None:
    log = Log(out, "cross", len(tiers) * len(seeds))
    for tier in tiers:
        for seed in seeds:
            key = f"{tier}|s{seed}"
            if log.has(key):
                continue
            t0 = time.time()
            ctxs = {name: Ctx(out, name, tier, seed) for name in CORPORA}
            table = {}
            for a, ca in ctxs.items():  # map of a
                for b, cb in ctxs.items():  # inputs of b
                    table[f"{a}->{b}"] = {"queries": float(cos(ca.project(cb.Q_tgt), cb.Q_src).mean()),
                                          "proxies": float(cos(ca.project(cb.Y_q[cb.va]), cb.X_q[cb.va]).mean()),
                                          "docs": float(cos((cb.Y_d[cb.va] - cb.Y_d[cb.tr].mean(0)) @ ca.W + cb.X_d[cb.tr].mean(0), cb.X_d[cb.va]).mean())}
            log.write(key, t0, tier=tier, seed=seed, table=table)


_CAPS = re.compile(r"\b[A-Z][\w'’.-]*(?:\s+(?:of|the|de|la|and|in|for|van|von|on|to)?\s*[A-Z0-9][\w'’.-]*)*")
_QWORD = {"what", "who", "which", "when", "where", "why", "how", "is", "are", "was", "were", "do", "does", "did", "in", "the", "a", "an", "on", "at", "of"}


def entities(q: str) -> list[str]:
    spans = []
    for m in _CAPS.finditer(q):
        words = m.group(0).split()
        while words and words[0].lower() in _QWORD:
            words = words[1:]
        if words:
            spans.append(" ".join(words))
    return spans


def exp_form(out: Path, tiers, seeds) -> None:
    log = Log(out, "form", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                qs = c.ds.queries
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(qs))
                variants = {
                    "original": qs,
                    "concat_two": [f"{qs[i]} {qs[j]}" for i, j in enumerate(perm)],
                    "concat_same": [f"{q} {q}" for q in qs],
                    "first_half": [" ".join(q.split()[: max(2, len(q.split()) // 2)]) for q in qs],
                    "second_half": [" ".join(q.split()[max(1, len(q.split()) // 2):]) or q for q in qs],
                    "first_8_words": [" ".join(q.split()[:8]) for q in qs],
                    "entities_only": [", ".join(entities(q)) or q for q in qs],
                    "one_entity": [(entities(q) or [q])[0] for q in qs],
                    "no_qmark_lower": [q.rstrip("?").lower() for q in qs],
                }
                res = {}
                for name, texts in variants.items():
                    Y = encode_queries(c.tgt, texts, c.ds.task)
                    X = encode_queries(c.src, texts, c.ds.task)
                    res[name] = {"cos": float(cos(c.project(Y), X).mean()), "words": float(np.mean([len(t.split()) for t in texts]))}
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, variants=res,
                          val_proxy_cos=float(cos(c.project(c.Y_q[c.va]), c.X_q[c.va]).mean()),
                          n_entities=float(np.mean([len(entities(q)) for q in qs])))


def _tokens(t: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split()


def exp_features(out: Path, tiers, seeds) -> None:
    log = Log(out, "features", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        ds = dataset(corpus)[1]
        sample = np.random.default_rng(0).choice(ds.N, 100_000, replace=False)
        df = Counter()
        for i in sample.tolist():
            df.update(set(_tokens(ds.titles[i] + " " + ds.texts[i])))
        idf = lambda w: np.log(100_000 / (1 + df.get(w, 0)))
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                ranked = search(c.Q_hat, c.E_src, 1_000, DEV, exclude=ds.self_rows)
                qcos = cos(c.Q_hat, c.Q_src)
                rows = []
                for q, rel in enumerate(ds.qrels):
                    q_toks = " ".join(_tokens(ds.queries[q]))
                    q_idf = [idf(w) for w in _tokens(ds.queries[q])] or [0.0]
                    in_q = sum(" ".join(_tokens(ds.titles[d])) in q_toks and bool(ds.titles[d]) for d in rel)
                    nat, brg, ora = (dict(zip(r[q].tolist(), range(1, 1001))) for r in (c.native, ranked, c.oracle))
                    for d in rel:
                        toks = _tokens(ds.titles[d]) or _tokens(ds.texts[d])[:4]
                        shared = len(set(_tokens(ds.texts[d] + " " + ds.titles[d])) & set(_tokens(ds.queries[q])))
                        rows.append((q, nat.get(d, 10**6), brg.get(d, 10**6), ora.get(d, 10**6), bool(ds.titles[d]) and " ".join(_tokens(ds.titles[d])) in q_toks,
                                     in_q, max([idf(w) for w in toks] or [0.0]), len(ds.texts[d].split()), shared, len(_tokens(ds.queries[q])), max(q_idf), qcos[q]))
                a = np.array(rows, dtype=np.float64)
                nat, brg, ora = a[:, 1], a[:, 2], a[:, 3]
                top300 = nat <= 300
                lost = top300 & (brg > 300)

                def group(mask) -> dict:
                    m = mask & top300
                    return {"n": int(mask.sum()), "share": float(mask.mean()), "native_gt300": float((nat[mask] > 300).mean()) if mask.any() else None,
                            "lost300": float(lost[m].mean()) if m.any() else None, "lost_top10": float((brg[mask & (nat <= 10)] > 300).mean()) if (mask & (nat <= 10)).any() else None,
                            "median_native": float(np.median(nat[mask])) if mask.any() else None, "median_bridged": float(np.median(brg[mask])) if mask.any() else None}

                terc = lambda col: np.quantile(col, [1 / 3, 2 / 3])
                res = {"all": group(np.ones(len(a), bool)), "title_in_query": group(a[:, 4] == 1), "title_not_in_query": group(a[:, 4] == 0),
                       "query_has_0_titles": group(a[:, 5] == 0), "query_has_1_title": group(a[:, 5] == 1), "query_has_2_titles": group(a[:, 5] >= 2)}
                for name, col in (("title_rarity", a[:, 6]), ("doc_words", a[:, 7]), ("shared_words", a[:, 8]), ("query_words", a[:, 9]), ("query_max_idf", a[:, 10]), ("query_cos", a[:, 11])):
                    lo, hi = terc(col)
                    res[f"{name}_low"], res[f"{name}_mid"], res[f"{name}_high"] = group(col <= lo), group((col > lo) & (col <= hi)), group(col > hi)
                    res[f"{name}_cuts"] = [float(lo), float(hi)]
                # Is the loss a disagreement between the two models? Re-index rank of what the map lost and kept.
                res["reindex"] = {"lost_and_reindex_gt300": float((ora[lost] > 300).mean()) if lost.any() else None,
                                  "kept_and_reindex_gt300": float((ora[top300 & ~lost] > 300).mean()),
                                  "reindex_gt300_all": float((ora > 300).mean()), "reindex_lost_from_native300": float((ora[top300] > 300).mean())}
                by_q = {}
                for row in a:
                    by_q.setdefault(int(row[0]), []).append(row[1] <= 300 and row[2] > 300)
                any_lost = np.array([any(v) for v in by_q.values()])
                res["queries"] = {"with_any_lost": float(any_lost.mean()), "cos_lost": float(qcos[[q for q, v in by_q.items() if any(v)]].mean()),
                                  "cos_kept": float(qcos[[q for q, v in by_q.items() if not any(v)]].mean()), "n_entities": float(np.mean([len(entities(q)) for q in ds.queries]))}
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, groups=res)


def exp_cover(out: Path, tiers, seeds) -> None:
    log = Log(out, "cover", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                tr = c.tr
                Y_fit = np.vstack([c.Y_q[tr] - c.Y_q[tr].mean(0), c.Y_d[tr] - c.Y_d[tr].mean(0)])
                evals, vecs = np.linalg.eigh(np.cov(Y_fit.T))
                evals, vecs = evals[::-1], vecs[:, ::-1]

                def energy(A: np.ndarray, r: int) -> float:
                    return float(((A @ vecs[:, :r]) ** 2).sum() / (A**2).sum())

                def participation(A: np.ndarray) -> float:
                    w = np.linalg.eigvalsh(np.cov((A - A.mean(0)).T))
                    return float(w.sum() ** 2 / (w**2).sum())

                Yq = c.Q_tgt - c.mu_y
                Yv = c.Y_q[c.va] - c.mu_y
                rec = {f"energy_{name}": {r: energy(A, r) for r in (64, 128, 256, 512)} for name, A in (("queries", Yq), ("val_proxies", Yv))}
                rec.update(pr_src_anchor_docs=participation(c.X_d), pr_tgt_anchor_docs=participation(c.Y_d), pr_src_proxies=participation(c.X_q),
                           pr_src_queries=participation(c.Q_src), pr_tgt_queries=participation(c.Q_tgt),
                           query_to_proxy_mean_cos=float(cos(c.Q_tgt.mean(0, keepdims=True), c.Y_q.mean(0, keepdims=True))[0]),
                           centered_norm_queries=float(np.linalg.norm(Yq, axis=1).mean()), centered_norm_proxies=float(np.linalg.norm(Yv, axis=1).mean()),
                           val_proxy_cos=float(cos(c.project(c.Y_q[c.va]), c.X_q[c.va]).mean()), query_cos=float(cos(c.Q_hat, c.Q_src).mean()))
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **rec)


def norm_matched(W: np.ndarray, Yc: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """The engine's ``match_residual_norm``: predictions as long as the targets. Shrinkage raises cosine and hurts retrieval."""
    return W * (np.linalg.norm(Xc, axis=1).mean() / np.linalg.norm(Yc @ W, axis=1).mean())


class Agreement:
    """Top-10 agreement with native on the dev queries over a sample of the index: what the engine selects maps by."""

    def __init__(self, c: Ctx, X_dev: np.ndarray, n: int = 50_000):
        ids = np.sort(np.random.default_rng(0).choice(len(c.E_src), n, replace=False))
        self.pool = torch.as_tensor(np.asarray(c.E_src[ids], dtype=np.float32), device=DEV)
        self.teacher = self._top(X_dev)

    def _top(self, Q: np.ndarray) -> torch.Tensor:
        return (torch.as_tensor(np.asarray(Q, dtype=np.float32), device=DEV) @ self.pool.T).topk(10).indices

    def __call__(self, Q: np.ndarray) -> float:
        top = self._top(Q)
        return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(top.cpu(), self.teacher.cpu())]))


def ridge_by_agreement(Yc, Xc, Y_dev, agree: Agreement, mu_y, mu_x, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    Yt, Xt = torch.as_tensor(Yc, device=DEV).double(), torch.as_tensor(Xc, device=DEV).double()
    best = None
    for lam, W in ridge_path(Yt, Xt, grid):
        W = norm_matched(W.float().cpu().numpy(), Yc, Xc)
        q = (Y_dev - mu_y) @ W + mu_x
        score = agree(q / np.linalg.norm(q, axis=1, keepdims=True))
        if best is None or score > best[0]:
            best = (score, lam, W)
    return best[2], best[1], best[0]


def exp_fits2(out: Path, tiers, seeds) -> None:
    """``fits``, done as the engine does: norm-matched, and selected by agreement with native rather than by cosine."""
    log = Log(out, "fits2", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                real = real_queries(corpus)
                dev_at = len(real) - min(N_DEV, len(real) // 4)
                Y_real, X_real = encode_queries(c.tgt, real, c.ds.task), encode_queries(c.src, real, c.ds.task)
                Y_dev, X_dev = Y_real[dev_at:], X_real[dev_at:]
                agree = Agreement(c, X_dev)
                tr = c.tr
                yq, xq, yd, xd = c.Y_q[tr], c.X_q[tr], c.Y_d[tr], c.X_d[tr]
                Y0, X0 = np.vstack([yq - yq.mean(0), yd - yd.mean(0)]), np.vstack([xq - xq.mean(0), xd - xd.mean(0)])
                rec = {"baseline": c.evaluate(c.Q_hat), "n_log": dev_at}
                W, lam, sc = ridge_by_agreement(Y0, X0, Y_dev, agree, yq.mean(0), xq.mean(0))
                rec["stage1_proxy_doc"] = {"lam": lam, "dev_agree": sc, **c.evaluate(c.project(c.Q_tgt, W=W, mu_y=yq.mean(0), mu_x=xq.mean(0)))}
                for n in (1000, 3000, 10000, 30000, 84000):
                    if n > dev_at:
                        continue
                    y, x = Y_real[:n], X_real[:n]
                    my, mx = y.mean(0), x.mean(0)
                    W_real, lam, sc = ridge_by_agreement(y - my, x - mx, Y_dev, agree, my, mx)
                    rec[f"real_{n}"] = {"lam": lam, "dev_agree": sc, **c.evaluate(c.project(c.Q_tgt, W=W_real, mu_y=my, mu_x=mx))}
                    W_mix, lam, sc = ridge_by_agreement(np.vstack([Y0, y - my]), np.vstack([X0, x - mx]), Y_dev, agree, my, mx)
                    rec[f"mix_{n}"] = {"lam": lam, "dev_agree": sc, **c.evaluate(c.project(c.Q_tgt, W=W_mix, mu_y=my, mu_x=mx))}
                if dev_at >= 84000:
                    y, x = Y_real[:84000], X_real[:84000]
                    my, mx = y.mean(0), x.mean(0)
                    W_real, _, _ = ridge_by_agreement(y - my, x - mx, Y_dev, agree, my, mx)
                    apply, _ = fit_mlp(y, x, Y_dev, X_dev, W_real, my, mx)
                    Q_mlp = apply(c.Q_tgt)
                    rec["mlp_84000"] = {"dev_agree": agree(apply(Y_dev)), **c.evaluate(Q_mlp)}
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **rec)


def exp_scale(out: Path, tiers, seeds) -> None:
    """The balance between the query-specific part and the restored source mean, ``s * (y - mu_y) W + t * mu_x``."""
    log = Log(out, "scale", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                part = (c.Q_tgt - c.mu_y) @ c.W
                res = {}
                for s, t in [(s, 1.0) for s in (0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)] + [(1.0, 0.5), (1.0, 0.0)]:
                    q = s * part + t * c.mu_x
                    res[f"s{s}_t{t}"] = c.evaluate(q / np.linalg.norm(q, axis=1, keepdims=True))
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, grid=res)


def exp_tta(out: Path, tiers, seeds) -> None:
    """Averaging the map's output over variants of a query: its noise may be partly independent across them."""
    log = Log(out, "tta", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                qs = c.ds.queries
                variants = {"lower": [q.rstrip("?").lower() for q in qs],
                            "first": [" ".join(q.split()[: max(2, len(q.split()) // 2)]) for q in qs],
                            "second": [" ".join(q.split()[max(1, len(q.split()) // 2):]) or q for q in qs]}
                mapped = {"orig": c.Q_hat, **{k: c.project(encode_queries(c.tgt, v, c.ds.task)) for k, v in variants.items()}}
                res = {"baseline": c.evaluate(c.Q_hat)}
                for names in (("orig", "lower"), ("orig", "first", "second"), ("orig", "lower", "first", "second")):
                    q = sum(mapped[n] for n in names)
                    res["+".join(names)] = c.evaluate(q / np.linalg.norm(q, axis=1, keepdims=True))
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **res)


def exp_local(out: Path, tiers, seeds) -> None:
    """A mount-feasible nonlinear step: add the residuals of the calibration pairs nearest the query, on the query side."""
    log = Log(out, "local", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                tr = c.tr
                yq, xq, yd, xd = c.Y_q[tr], c.X_q[tr], c.Y_d[tr], c.X_d[tr]
                pred_q = (yq - c.mu_y) @ c.W + c.mu_x
                pred_d = (yd - yd.mean(0)) @ c.W + xd.mean(0)
                Yall = torch.as_tensor(np.vstack([yq, yd]), device=DEV)
                resid = torch.as_tensor(np.vstack([xq - pred_q, xd - pred_d]), device=DEV)
                Q = torch.as_tensor(c.Q_tgt, device=DEV)
                base = torch.as_tensor(c.Q_hat, device=DEV)
                sims = Q @ Yall.T
                res = {"baseline": c.evaluate(c.Q_hat)}
                for k in (10, 50, 200):
                    top, idx = sims.topk(k)
                    for tau in (20.0, 50.0):
                        w = (tau * top).softmax(1)
                        corr = (w[..., None] * resid[idx]).sum(1)
                        for g in (0.5, 1.0, 2.0):
                            q = F.normalize(base + g * corr, dim=1).cpu().numpy()
                            res[f"k{k}_tau{int(tau)}_g{g}"] = c.evaluate(q)
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **res)

def exp_margin(out: Path, tiers, seeds) -> None:
    """Does one curve explain every corpus? Loss by each positive's native margin over rank 300, in units of the map's noise."""
    edges = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 100)
    log = Log(out, "margin", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                ranked = search(c.Q_hat, c.E_src, 1_000, DEV, exclude=c.ds.self_rows)
                rows = []
                for i in np.random.default_rng(0).choice(len(c.Q_src), 800, replace=False).tolist():
                    ids = c.native[i]
                    D = np.asarray(c.E_src[np.sort(ids)], dtype=np.float32)[np.argsort(np.argsort(ids))]
                    s, sh = D @ c.Q_src[i], D @ c.Q_hat[i]
                    noise = (sh - s).std()
                    where = {d: r for r, d in enumerate(ids[:300].tolist())}
                    bridged = set(ranked[i, :300].tolist())
                    for d in c.ds.qrels[i]:
                        if d in where:
                            r = where[d]
                            rows.append(((s[r] - s[299]) / noise, s[r] - s[299], noise, s[r], d not in bridged, r + 1))
                a = np.array(rows, dtype=np.float64)
                z, lost = a[:, 0], a[:, 4]
                curve = []
                for lo, hi in zip(edges, edges[1:]):
                    m = (z >= lo) & (z < hi)
                    curve.append({"z": [lo, hi], "share": float(m.mean()), "lost": float(lost[m].mean()) if m.any() else None, "n": int(m.sum())})
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, curve=curve, lost=float(lost.mean()), n=len(a),
                          z_median=float(np.median(z)), z_p10=float(np.percentile(z, 10)), z_p25=float(np.percentile(z, 25)),
                          margin_median=float(np.median(a[:, 1])), margin_p10=float(np.percentile(a[:, 1], 10)), noise=float(np.median(a[:, 2])),
                          pos_score=float(np.median(a[:, 3])))

def exp_signal(out: Path, tiers, seeds) -> None:
    """How much of native's score differences survive the map? Among native's top 1,000 for a query, regress bridged scores on
    native scores: the slope is the signal carried over, the residual the noise added. Uses cached rankings, no GPU."""
    log = Log(out, "signal", len(CORPORA) * len(tiers) * len(seeds))
    for corpus in CORPORA:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if log.has(key):
                    continue
                t0 = time.time()
                c = Ctx(out, corpus, tier, seed)
                slopes, resid, pos_ratio, pos_excess, spreads, top_ratio = [], [], [], [], [], []
                for i in np.random.default_rng(0).choice(len(c.Q_src), 800, replace=False).tolist():
                    ids = c.native[i]
                    D = np.asarray(c.E_src[np.sort(ids)], dtype=np.float32)[np.argsort(np.argsort(ids))]
                    s, sh = D @ c.Q_src[i], D @ c.Q_hat[i]
                    sc, shc = s - s.mean(), sh - sh.mean()
                    b = float(sc @ shc / (sc @ sc))
                    slopes.append(b)
                    resid.append(float((shc - b * sc).std()))
                    spreads.append(float(s.std()))
                    top_ratio.append(float(shc[:10].mean() / sc[:10].mean()))
                    where = {d: r for r, d in enumerate(ids.tolist())}
                    for d in c.ds.qrels[i]:
                        if d in where and where[d] < 300:
                            r = where[d]
                            pos_ratio.append(float(shc[r] / sc[r]))
                            pos_excess.append(float((shc[r] - b * sc[r]) / resid[-1]))
                log.write(key, t0, corpus=corpus, tier=tier, seed=seed, slope=float(np.mean(slopes)), resid=float(np.mean(resid)), spread=float(np.mean(spreads)),
                          resid_over_signal=float(np.mean(resid) / (np.mean(slopes) * np.mean(spreads))), top10_ratio=float(np.median(top_ratio)),
                          positive_ratio=float(np.median(pos_ratio)), positive_excess_z=float(np.median(pos_excess)), share_below_trend=float(np.mean(np.array(pos_excess) < 0)))


EXPERIMENTS = {"dose": exp_dose, "offset": exp_offset, "fits": exp_fits, "cross": exp_cross, "form": exp_form, "features": exp_features, "cover": exp_cover,
               "margin": exp_margin, "signal": exp_signal, "fits2": exp_fits2, "scale": exp_scale, "tta": exp_tta, "local": exp_local}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=list(EXPERIMENTS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tiers", nargs="+", default=["T1", "T3"], choices=list(PAIRS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[args.exp](args.out, args.tiers, args.seeds)


if __name__ == "__main__":
    main()
