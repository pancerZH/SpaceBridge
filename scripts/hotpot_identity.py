"""Does the map lose exact identity, or is it HotpotQA's tight margins? First-stage tests, with their rules fixed in advance in
docs/reports/hotpotqa-followup.md. Repairs use the native query, so they're diagnostics and can't ship.

Each experiment appends one synced JSON line per unit to --out/<exp>.jsonl and skips units already there, so a rerun continues.
Maps come from the study's fits, so they aren't refitted.

    band     replace one principal band of the mapped query with the native query's, and where a positive's lead lives
    entity   a span's contribution to the query, natively and through the map: titles against other words, rare against common
    rarity   per-positive loss by margin and by rarity, on bins fixed across corpora
    fit      logistic regressions on rarity's rows: how much of HotpotQA's gap does each explanation account for?
    tables   all of the above as markdown, in --out/tables.md
    joint    repair one title, the other, or both; and the reverse: keep the title's mapped part and make the rest native
    boost    amplify identity spans through the map alone, with no native query, so this one could ship
    fame     title transfer by how often the corpus mentions the title outside its own article
    tables2  these three as markdown, in --out/tables2.md

    uv run python -u scripts/hotpot_identity.py band --out outputs/phase1/hp_identity
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from hotpot_study import CORPORA, DEV, Ctx, Log, _tokens, dataset, encode_queries, entities
from spacebridge.lexical import _STOP
from spacebridge_bench import durable
from spacebridge_bench.metrics import per_query
from spacebridge_bench.search import search

FITS = Path("outputs/phase1/hp_study")
HELD_OUT = ("fiqa", "trec-covid")
N_SAMPLE = 100_000
BANDS = {"1-16": (0, 16), "17-32": (16, 32), "33-64": (32, 64), "65-128": (64, 128), "129-256": (128, 256), "257-512": (256, 512), "513-1024": (512, 1024)}
GROUPS = {"head": (0, 64), "mid": (64, 256), "tail": (256, 1024)}
IDF_EDGES = (7, 8, 9, 10, 11)  # upper bounds; log(100000 / (1 + df)) tops out at 11.51 for a word the sample never saw
SHARED_EDGES = (0, 5, 7, 9, 11)
Z_EDGES = (0.5, 1, 1.5, 2, 3, 4, 6)
PAIRS_OF_SPANS = (("title", "control"), ("rare", "common"))
GAINS = (0.25, 0.5, 1.0, 2.0)
SHIPPABLE = ("rare1", "rare_all", "caps")
FAME_EDGES = (0, 2, 9, 49)
FAME_BINS = ("0", "1-2", "3-9", "10-49", "50+")
MIN_BIN = 30
_WORD = re.compile(r"[A-Za-z0-9]+")
_PARENS = re.compile(r"\s*\([^)]*\)\s*$")
_BASIS: dict = {}
_DF: dict = {}
_SAMPLED: dict = {}


def units(exp: str, out: Path, corpora, tiers, seeds):
    log = Log(out, exp, len(corpora) * len(tiers) * len(seeds))
    for corpus in corpora:
        for tier in tiers:
            for seed in seeds:
                key = f"{corpus}|{tier}|s{seed}"
                if not log.has(key):
                    yield log, key, corpus, tier, seed, time.time()


def unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def rows(E, ids) -> np.ndarray:
    """Rows of a store in the order asked for; the store reads sorted rows."""
    uniq, inv = np.unique(np.asarray(ids), return_inverse=True)
    return np.asarray(E[uniq], dtype=np.float32)[inv]


def ratio_ci(na, da, nb, db, B: int = 1000, paired: bool = True) -> dict:
    """The pooled ratios a and b, and their difference with a bootstrap 95% CI over queries: paired, or over two separate sets."""
    rng = np.random.default_rng(0)
    ia = rng.integers(0, len(na), (B, len(na)))
    ib = ia if paired else rng.integers(0, len(nb), (B, len(nb)))
    diff = na[ia].sum(1) / da[ia].sum(1) - nb[ib].sum(1) / db[ib].sum(1)
    a, b = float(na.sum() / da.sum()), float(nb.sum() / db.sum())
    return {"a": a, "b": b, "diff": a - b, "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}


def doc_basis(c: Ctx) -> np.ndarray:
    """Principal directions of a sample of the index's documents, largest first, one per column."""
    key = (c.corpus, c.src)
    if key not in _BASIS:
        picks = np.sort(np.random.default_rng(0).choice(c.ds.N, N_SAMPLE, replace=False))
        D = torch.as_tensor(np.asarray(c.E_src[picks], dtype=np.float32), device=DEV)
        D = (D - D.mean(0)).double()
        _, vecs = torch.linalg.eigh(D.T @ D / (len(D) - 1))
        _BASIS[key] = vecs.flip(1).float().cpu().numpy()
    return _BASIS[key]


def lead_by_band(c: Ctx, V: np.ndarray) -> dict:
    """Each band's part of a positive's lead over native's rank 300, natively and through the map."""
    ranked = search(c.Q_hat, c.E_src, 300, DEV, exclude=c.ds.self_rows)
    picks = []
    for i in np.random.default_rng(0).choice(len(c.Q_src), 800, replace=False).tolist():
        top = set(c.native[i, :300].tolist())
        bridged = set(ranked[i].tolist())
        picks.extend((i, d, c.native[i, 299], d not in bridged) for d in c.ds.qrels[i] if d in top)
    a = np.array(picks)
    lost = a[:, 3] == 1
    v = (rows(c.E_src, a[:, 1]) - rows(c.E_src, a[:, 2])) @ V
    nat, mapped = (c.Q_src[a[:, 0]] @ V) * v, (c.Q_hat[a[:, 0]] @ V) * v
    total_n, total_m = nat.sum(1), mapped.sum(1)
    out = {"n": len(a), "n_lost": int(lost.sum()), "kept": float(total_m.mean() / total_n.mean())}
    for name, (lo, hi) in {**BANDS, **GROUPS}.items():
        n, m = nat[:, lo:hi].sum(1), mapped[:, lo:hi].sum(1)
        out[name] = {"share": float(n.mean() / total_n.mean()), "kept": float(m.mean() / n.mean()), "deficit_share": float((n - m).mean() / (total_n - total_m).mean()),
                     "deficit_share_lost": float((n - m)[lost].mean() / (total_n - total_m)[lost].mean())}
    return out


def exp_band(out: Path, corpora, tiers, seeds) -> None:
    for log, key, corpus, tier, seed, t0 in units("band", out, corpora, tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed)
        V = doc_basis(c)
        base = c.evaluate(c.Q_hat)
        e = c.Q_src - c.Q_hat
        e2 = (e**2).sum(1)
        res = {"baseline": base, "native": c.evaluate(c.Q_src)}
        for name, (lo, hi) in {**BANDS, **GROUPS}.items():
            U = V[:, lo:hi]
            coef = e @ U
            m = c.evaluate(unit(c.Q_hat + coef @ U.T))
            res[name] = {"lost300": m["lost300"], "ret@300": m["ret@300"], "recovered": 1 - m["lost300"] / base["lost300"], "error_energy": float(((coef**2).sum(1) / e2).mean())}
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, dim=V.shape[1], lead=lead_by_band(c, V), **res)


def spans(q: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD.finditer(q)]


def content(w: str) -> bool:
    return w not in _STOP and len(w) > 1


def cut(q: str, ranges) -> str:
    """The query without these character ranges."""
    parts, at = [], 0
    for a, b in sorted(ranges):
        parts.append(q[at:max(a, at)])
        at = max(at, b)
    parts.append(q[at:])
    return " ".join("".join(parts).split())


def keeps_content(text: str) -> bool:
    return any(content(w) for w, _, _ in spans(text))


def corpus_df(corpus: str) -> Counter:
    """Document frequency over a sample of the corpus, as the study's ``features``."""
    if corpus not in _DF:
        ds = dataset(corpus)[1]
        df = Counter()
        _SAMPLED[corpus] = min(ds.N, N_SAMPLE)
        for i in np.random.default_rng(0).choice(ds.N, _SAMPLED[corpus], replace=False).tolist():
            df.update(set(_tokens(ds.titles[i] + " " + ds.texts[i])))
        _DF[corpus] = df
    return _DF[corpus]


def idf(df: dict, w: str) -> float:
    return float(np.log(N_SAMPLE / (1 + df.get(w, 0))))


def title_key(title: str | None) -> tuple[str, ...]:
    return tuple(w for w, _, _ in spans(_PARENS.sub("", title or "")))


def find_title(words: list[str], title: str | None) -> tuple[int, list[int]]:
    """The title's length in words and where it starts in the query, empty for a title with no content word."""
    t = list(title_key(title))
    if not t or not any(content(w) for w in t):
        return 0, []
    return len(t), [s for s in range(len(words) - len(t) + 1) if words[s:s + len(t)] == t]


def variants(ds, corpus: str) -> tuple[dict, dict]:
    """Per span kind, query row to the query without that span. Fixed by the query, so every map reads the same texts, and
    their embeddings are cached. A control avoids positives' titles and capitalized names."""
    df = corpus_df(corpus)
    out: dict = {"title": {}, "control": {}, "rare": {}, "common": {}}
    title_pos = {}
    for i, (q, rel) in enumerate(zip(ds.queries, ds.qrels)):
        sp = spans(q)
        words = [s[0] for s in sp]
        forbid, best = set(), None
        for d in rel:
            n, hits = find_title(words, ds.titles[d])
            for s in hits:
                forbid.update(range(s, s + n))
            if hits and (best is None or n > best[1]):
                best = (d, n, hits)
        for name in entities(q):
            t = [w for w, _, _ in spans(name)]
            for s in range(len(words) - len(t) + 1):
                if t and words[s:s + len(t)] == t:
                    forbid.update(range(s, s + len(t)))
        if best is not None:
            d, n, hits = best
            title = cut(q, [(sp[s][1], sp[s + n - 1][2]) for s in hits])
            starts = [s for s in range(len(words) - n + 1) if not forbid & set(range(s, s + n)) and any(content(w) for w in words[s:s + n])]
            if starts and keeps_content(title):
                s = int(np.random.default_rng(i).choice(starts))
                control = cut(q, [(sp[s][1], sp[s + n - 1][2])])
                if keeps_content(control):
                    out["title"][i], out["control"][i], title_pos[i] = title, control, d
        kept = list(dict.fromkeys(w for w in words if content(w)))
        if len(kept) >= 2:
            score = {w: idf(df, w) for w in kept}
            rare, common = max(kept, key=score.get), min(kept, key=score.get)
            if score[rare] > score[common]:
                a, b = cut(q, [(x, y) for w, x, y in sp if w == rare]), cut(q, [(x, y) for w, x, y in sp if w == common])
                if keeps_content(a) and keeps_content(b):
                    out["rare"][i], out["common"][i] = a, b
    return out, title_pos


def lost_by_query(c: Ctx, ranked: np.ndarray, only: dict | None = None) -> np.ndarray:
    """Per query, positives native had in its top 300 that this ranking doesn't; with ``only``, just that query's one positive."""
    out = np.zeros(len(c.ds.qrels))
    for i, rel in enumerate(c.ds.qrels):
        nat, br = set(c.native[i, :300].tolist()), set(ranked[i, :300].tolist())
        out[i] = sum(d in nat and d not in br and (only is None or only.get(i) == d) for d in rel)
    return out


def exp_entity(out: Path, corpora, tiers, seeds) -> None:
    for log, key, corpus, tier, seed, t0 in units("entity", out, corpora, tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed)
        ds = c.ds
        texts, title_pos = variants(ds, corpus)
        d_x, d_q, res = {}, {}, {}
        for name, by_query in texts.items():
            if by_query:
                idx = sorted(by_query)
                t = [by_query[i] for i in idx]
                dx = c.Q_src[idx] - encode_queries(c.src, t, ds.task)
                dq = c.Q_hat[idx] - c.project(encode_queries(c.tgt, t, ds.task))
                d_x[name], d_q[name] = dict(zip(idx, dx)), dict(zip(idx, dq))
        base_rank = search(c.Q_hat, c.E_src, 300, DEV, exclude=ds.self_rows)
        base_lost = lost_by_query(c, base_rank)
        base_lost_title = lost_by_query(c, base_rank, title_pos)
        e2 = ((c.Q_src - c.Q_hat) ** 2).sum(1)
        for a, b in PAIRS_OF_SPANS:
            common = sorted(set(d_x.get(a, {})) & set(d_x.get(b, {})))
            if not common:
                continue
            sample = np.random.default_rng(0).permutation(common)[:800].tolist()
            stats = {a: [], b: []}
            for i in sample:
                ids = c.native[i]
                D = rows(c.E_src, ids)
                where = {d: r for r, d in enumerate(ids[:300].tolist())}
                targets = [title_pos[i]] if a == "title" else ds.qrels[i]
                at = [where[d] for d in targets if d in where]
                for name in (a, b):
                    dx, dq = d_x[name][i], d_q[name][i]
                    s, sh = D @ dx, D @ dq
                    sc, shc = s - s.mean(), sh - sh.mean()
                    stats[name].append((shc @ sc, sc @ sc, shc @ shc, sum(sh[r] - sh[299] for r in at), sum(s[r] - s[299] for r in at), dq @ dx, dx @ dx, dq @ dq))
            A, B = np.array(stats[a], dtype=np.float64), np.array(stats[b], dtype=np.float64)
            rec = {"n_queries": len(common), "n_sample": len(sample), "transfer": ratio_ci(A[:, 0], A[:, 1], B[:, 0], B[:, 1]),
                   "transfer_median": [float(np.median(A[:, 0] / A[:, 1])), float(np.median(B[:, 0] / B[:, 1]))],
                   "corr_median": [float(np.median(A[:, 0] / np.sqrt(A[:, 1] * A[:, 2]))), float(np.median(B[:, 0] / np.sqrt(B[:, 1] * B[:, 2])))],
                   "lead_ratio": [float(A[:, 3].sum() / A[:, 4].sum()), float(B[:, 3].sum() / B[:, 4].sum())],
                   "gain": [float(A[:, 5].sum() / A[:, 6].sum()), float(B[:, 5].sum() / B[:, 6].sum())],
                   "cos": [float(np.mean(A[:, 5] / np.sqrt(A[:, 6] * A[:, 7]))), float(np.mean(B[:, 5] / np.sqrt(B[:, 6] * B[:, 7])))],
                   "delta_norm": [float(np.sqrt(A[:, 6]).mean()), float(np.sqrt(B[:, 6]).mean())]}
            lost, lost_title, energy = {}, {}, {}
            for name in (a, b):
                Q = c.Q_hat.copy()
                Q[common] = unit(c.Q_hat[common] + np.stack([d_x[name][i] - d_q[name][i] for i in common]))
                ranked = search(Q, c.E_src, 300, DEV, exclude=ds.self_rows)
                lost[name], lost_title[name] = lost_by_query(c, ranked), lost_by_query(c, ranked, title_pos)
                energy[name] = float(1 - ((c.Q_src[common] - Q[common]) ** 2).sum(1).mean() / e2[common].mean())
            base = base_lost[common]
            rec["repair"] = {"lost_base": float(base.sum()), "error_energy": [energy[a], energy[b]],
                             **ratio_ci(base - lost[a][common], base, base - lost[b][common], base)}
            if a == "title":
                base = base_lost_title[common]
                rec["repair_title_positive"] = {"lost_base": float(base.sum()), **ratio_ci(base - lost_title[a][common], base, base - lost_title[b][common], base)}
            res[f"{a}_vs_{b}"] = rec
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **res)


def binned(x: np.ndarray, edges) -> np.ndarray:
    return np.digitize(x, edges, right=True)


def rows_path(out: Path, corpus: str, tier: str, seed: int) -> Path:
    return out / "rarity_rows" / f"{corpus}_{tier}_s{seed}.npz"


def exp_rarity(out: Path, corpora, tiers, seeds) -> None:
    for corpus in corpora:
        corpus_df(corpus)
    pooled = Counter()
    for corpus in corpora:
        pooled.update(corpus_df(corpus))
    pooled = {w: n / len(corpora) for w, n in pooled.items()}
    for log, key, corpus, tier, seed, t0 in units("rarity", out, corpora, tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed)
        ds, df = c.ds, corpus_df(corpus)
        ranked = search(c.Q_hat, c.E_src, 1_000, DEV, exclude=ds.self_rows)
        table = []
        for i, (q, rel) in enumerate(zip(ds.queries, ds.qrels)):
            ids = c.native[i]
            where = {d: r for r, d in enumerate(ids[:300].tolist())}
            at = [d for d in rel if d in where]
            if not at:
                continue
            D = rows(c.E_src, ids)
            s, sh = D @ c.Q_src[i], D @ c.Q_hat[i]
            noise = (sh - s).std()
            bridged = {d: r for r, d in enumerate(ranked[i].tolist())}
            words = {w for w in _tokens(q) if content(w)}
            q_max = max((idf(df, w) for w in words), default=0.0)
            q_max_pooled = max((idf(pooled, w) for w in words), default=0.0)
            q_toks = " ".join(_tokens(q))
            for d in at:
                r = where[d]
                shared = words & set(_tokens(ds.titles[d] + " " + ds.texts[d]))
                title = " ".join(_tokens(ds.titles[d]))
                table.append((i, r + 1, bridged.get(d, 10**6) + 1, (s[r] - s[299]) / noise, q_max, q_max_pooled, max((idf(df, w) for w in shared), default=0.0),
                              max((idf(pooled, w) for w in shared), default=0.0), len(shared), len(_tokens(q)), bool(title) and title in q_toks))
        a = np.array(table, dtype=np.float64)
        path = rows_path(out, corpus, tier, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with durable.atomic_file(path) as f:
            np.savez(f, query=a[:, 0], native=a[:, 1], bridged=a[:, 2], z=a[:, 3], q_max=a[:, 4], q_max_pooled=a[:, 5], shared_max=a[:, 6], shared_max_pooled=a[:, 7],
                     n_shared=a[:, 8], words=a[:, 9], title_in_query=a[:, 10])
        lost, top10 = a[:, 2] > 300, a[:, 1] <= 10
        by_bin = []
        for column, edges, name in ((a[:, 4], IDF_EDGES, "q_max"), (a[:, 5], IDF_EDGES, "q_max_pooled"), (a[:, 6], SHARED_EDGES, "shared_max")):
            b = binned(column, edges)
            for k in range(len(edges) + 1):
                m = b == k
                by_bin.append({"column": name, "bin": k, "n": int(m.sum()), "lost300": float(lost[m].mean()) if m.any() else None,
                               "lost_top10": float(lost[m & top10].mean()) if (m & top10).any() else None, "n_top10": int((m & top10).sum()), "z_median": float(np.median(a[m, 3])) if m.any() else None})
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, n=len(a), lost300=float(lost.mean()), lost_top10=float(lost[top10].mean()), z_median=float(np.median(a[:, 3])), by_bin=by_bin)


def one_hot(b: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[b][:, 1:]


def gaps(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """HotpotQA's log-odds of loss over NQ and over MS MARCO. Columns 0 and 1 are HotpotQA and NQ, with MS MARCO the reference."""
    coef = LogisticRegression(C=1e6, max_iter=3000).fit(X, y).coef_[0]
    return float(coef[0] - coef[1]), float(coef[0])


def exp_fit(out: Path, corpora, tiers, seeds) -> None:
    """How much of HotpotQA's gap each explanation accounts for, with a bootstrap CI over queries within each corpus."""
    log = Log(out, "fit", len(tiers) * len(seeds))
    for tier in tiers:
        for seed in seeds:
            key = f"{tier}|s{seed}"
            if log.has(key):
                continue
            t0 = time.time()
            parts = [np.load(rows_path(out, corpus, tier, seed)) for corpus in corpora]
            get = lambda f: np.concatenate([p[f] for p in parts])
            which = np.concatenate([np.full(len(p["z"]), k) for k, p in enumerate(parts)])
            groups = [[np.where((which == k) & (get("query") == q))[0] for q in np.unique(parts[k]["query"])] for k in range(len(parts))]
            y = (get("bridged") > 300).astype(int)
            base = np.stack([which == corpora.index("hotpotqa-1m"), which == corpora.index("nq-1m")], axis=1).astype(float)
            X_z = one_hot(binned(get("z"), Z_EDGES), len(Z_EDGES) + 1)
            X_r = np.hstack([one_hot(binned(get("q_max"), IDF_EDGES), len(IDF_EDGES) + 1), one_hot(binned(get("shared_max"), SHARED_EDGES), len(SHARED_EDGES) + 1)])
            X_p = np.hstack([one_hot(binned(get("q_max_pooled"), IDF_EDGES), len(IDF_EDGES) + 1), one_hot(binned(get("shared_max_pooled"), SHARED_EDGES), len(SHARED_EDGES) + 1)])
            X_w = np.log(get("words"))[:, None]
            models = {"corpus": base, "z": np.hstack([base, X_z]), "rarity": np.hstack([base, X_r]), "both": np.hstack([base, X_z, X_r]), "rarity_pooled": np.hstack([base, X_p]),
                      "both_pooled": np.hstack([base, X_z, X_p]), "both_words": np.hstack([base, X_z, X_r, X_w]), "words": np.hstack([base, X_w])}
            point = {name: gaps(X, y) for name, X in models.items()}
            rng = np.random.default_rng(0)
            draws = {name: [] for name in models}
            for _ in range(200):
                idx = np.concatenate([groups[k][q] for k in range(len(parts)) for q in rng.integers(0, len(groups[k]), len(groups[k]))])
                for name, X in models.items():
                    draws[name].append(gaps(X[idx], y[idx]))
            ref = np.array(draws["corpus"])
            res = {}
            for name in models:
                d = np.array(draws[name])
                removed = 1 - d / ref
                res[name] = {"gap_nq": point[name][0], "gap_ms": point[name][1], "removed_nq": 1 - point[name][0] / point["corpus"][0], "removed_ms": 1 - point[name][1] / point["corpus"][1],
                             "removed_nq_ci": [float(np.percentile(removed[:, 0], 2.5)), float(np.percentile(removed[:, 0], 97.5))],
                             "removed_ms_ci": [float(np.percentile(removed[:, 1], 2.5)), float(np.percentile(removed[:, 1], 97.5))]}
            log.write(key, t0, tier=tier, seed=seed, n=len(y), models=res)


def read(out: Path, exp: str) -> list[dict]:
    path = out / f"{exp}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def exp_tables(out: Path, corpora, tiers, seeds) -> None:
    t = ["# HotpotQA follow-up: tables", "", "From `scripts/hotpot_identity.py tables`. Rules in `docs/reports/hotpotqa-followup.md`.", "", "## Test 1: band repair", "",
         "Recovered is the share of lost positives that replacing the band brings back. Energy is the band's share of the map's squared error.", "",
         "| Corpus | Tier | Seed | Lost | Native repair, lost | Head, recovered (energy) | Mid | Tail | Lead kept |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    band = read(out, "band")
    for r in band:
        cells = [f"{r[g]['recovered']:.3f} ({r[g]['error_energy']:.3f})" for g in GROUPS]
        t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | {r['baseline']['lost300']:.4f} | {r['native']['lost300']:.4f} | {' | '.join(cells)} | {r['lead']['kept']:.3f} |")
    t += ["", "Each band: recovered, with its share of the error energy, its share of the native lead, the share of that part kept, and its share of the lead lost, over all positives and over lost ones.", "",
          "| Corpus | Tier | Seed | Band | Recovered | Energy | Lead share | Part kept | Share of lead lost | Same, lost positives |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in band:
        for name in list(BANDS) + list(GROUPS):
            b, l = r[name], r["lead"][name]
            t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | {name} | {b['recovered']:.3f} | {b['error_energy']:.3f} | {l['share']:.3f} | {l['kept']:.3f} | {l['deficit_share']:.3f} | {l['deficit_share_lost']:.3f} |")
    t += ["", "## Test 2: entity transfer", "", "Transfer is pooled over queries. Each cell is a, b; the difference is a - b with a bootstrap 95% CI over queries.", "",
          "| Corpus | Tier | Seed | a, b | Queries | Transfer | Difference | Repair, recovered | Difference | Repair, energy | Positive only, recovered | Lead ratio | Gain | Cos | Size of span's change |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in read(out, "entity"):
        for a, b in PAIRS_OF_SPANS:
            p = r.get(f"{a}_vs_{b}")
            if p is None:
                continue
            tr, rp = p["transfer"], p["repair"]
            pos = p.get("repair_title_positive")
            pair = lambda v: f"{v[0]:.3f}, {v[1]:.3f}"
            t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | {a}, {b} | {p['n_queries']} | {tr['a']:.3f}, {tr['b']:.3f} | {tr['diff']:+.3f} [{tr['ci'][0]:+.3f}, {tr['ci'][1]:+.3f}] | {rp['a']:.3f}, {rp['b']:.3f} | "
                     f"{rp['diff']:+.3f} [{rp['ci'][0]:+.3f}, {rp['ci'][1]:+.3f}] | {pair(rp['error_energy'])} | {f'{pos['a']:.3f}, {pos['b']:.3f}' if pos else ''} | {pair(p['lead_ratio'])} | {pair(p['gain'])} | {pair(p['cos'])} | {pair(p['delta_norm'])} |")
    t += ["", "## Test 3: rarity on fixed bins", "", "Lost from native's top 300, and among positives in native's top 10. IDF on each corpus's own sample unless the row says pooled.", ""]
    rarity = read(out, "rarity")
    labels = {"q_max": ("Rarest query term", IDF_EDGES), "q_max_pooled": ("Rarest query term, pooled IDF", IDF_EDGES), "shared_max": ("Rarest term shared with the positive", SHARED_EDGES)}
    for column, (label, edges) in labels.items():
        bounds = [f"<= {edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f"> {edges[-1]}"]
        t += [f"**{label}.** Each cell: all positives, top 10 only, (positives in the top 10).", "", "| Corpus | Tier | Seed | " + " | ".join(bounds) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in bounds) + " |"]
        for r in rarity:
            cells = [(f"{100 * b['lost300']:.1f}, {100 * b['lost_top10']:.1f} ({b['n_top10']})" if b["lost_top10"] is not None else "-") for b in r["by_bin"] if b["column"] == column]
            t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | " + " | ".join(cells) + " |")
        t.append("")
    t += ["**How much of HotpotQA's log-odds of loss each explanation removes.** Observational. Share removed [bootstrap 95% CI over queries].", "",
          "| Tier | Seed | Model | Gap over NQ | Removed | Gap over MS MARCO | Removed |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in read(out, "fit"):
        for name, m in r["models"].items():
            t.append(f"| {r['tier']} | {r['seed']} | {name} | {m['gap_nq']:.3f} | {m['removed_nq']:.2f} [{m['removed_nq_ci'][0]:.2f}, {m['removed_nq_ci'][1]:.2f}] | {m['gap_ms']:.3f} | {m['removed_ms']:.2f} [{m['removed_ms_ci'][0]:.2f}, {m['removed_ms_ci'][1]:.2f}] |")
    with durable.atomic_file(out / "tables.md") as f:
        f.write("\n".join(t).encode() + b"\n")


def span_deltas(c: Ctx, idx: list[int], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """What the spans missing from these texts add to each query, natively and through the map."""
    return c.Q_src[idx] - encode_queries(c.src, texts, c.ds.task), c.Q_hat[idx] - c.project(encode_queries(c.tgt, texts, c.ds.task))


def both_titles(ds) -> dict:
    """Queries naming two positives' titles in separate spans: query row to (A, B, query without A, without B, without both)."""
    out = {}
    for i, (q, rel) in enumerate(zip(ds.queries, ds.qrels)):
        sp = spans(q)
        words = [s[0] for s in sp]
        found = sorted(((d, *find_title(words, ds.titles[d])) for d in rel), key=lambda f: -f[1])
        found = [f for f in found if f[2]]
        if len(found) < 2:
            continue
        (a, na, ha), (b, nb, hb) = found[:2]
        if {s + j for s in ha for j in range(na)} & {s + j for s in hb for j in range(nb)}:
            continue
        ra, rb = [(sp[s][1], sp[s + na - 1][2]) for s in ha], [(sp[s][1], sp[s + nb - 1][2]) for s in hb]
        ta, tb, tab = cut(q, ra), cut(q, rb), cut(q, ra + rb)
        if keeps_content(ta) and keeps_content(tb) and keeps_content(tab):
            out[i] = (a, b, ta, tb, tab)
    return out


def exp_joint(out: Path, corpora, tiers, seeds) -> None:
    for log, key, corpus, tier, seed, t0 in units("joint", out, [c for c in corpora if c != "msmarco-1m"], tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed)
        ds = c.ds
        base_rank = search(c.Q_hat, c.E_src, 300, DEV, exclude=ds.self_rows)
        res = {}

        def after(idx, change) -> np.ndarray:
            Q = c.Q_hat.copy()
            Q[idx] = unit(c.Q_hat[idx] + change)
            return search(Q, c.E_src, 300, DEV, exclude=ds.self_rows)

        # The reverse repair: every title query, title native and the rest mapped, then title mapped and the rest native.
        texts, title_pos = variants(ds, corpus)
        idx = sorted(texts["title"])
        dx, dq = span_deltas(c, idx, [texts["title"][i] for i in idx])
        title_only = {i: title_pos[i] for i in idx}
        base_all, base_title = lost_by_query(c, base_rank)[idx], lost_by_query(c, base_rank, title_only)[idx]
        e = c.Q_src[idx] - c.Q_hat[idx]
        rec = {"n_queries": len(idx), "lost_title": float(base_title.sum()), "lost_other": float((base_all - base_title).sum())}
        for name, change in (("title_native", dx - dq), ("rest_native", e - (dx - dq))):
            ranked = after(idx, change)
            lost_all, lost_title = lost_by_query(c, ranked)[idx], lost_by_query(c, ranked, title_only)[idx]
            rec[name] = {"lost_title": float(lost_title.sum()), "lost_other": float((lost_all - lost_title).sum()), "recovered_title": 1 - float(lost_title.sum() / base_title.sum()),
                         "recovered_other": 1 - float((lost_all - lost_title).sum() / max((base_all - base_title).sum(), 1)), "recovered_all": 1 - float(lost_all.sum() / base_all.sum())}
        res["reverse"] = rec
        if corpus == "hotpotqa-1m":
            pairs = both_titles(ds)
            idx = sorted(pairs)
            d = {name: span_deltas(c, idx, [pairs[i][k] for i in idx]) for name, k in (("a", 2), ("b", 3), ("both", 4))}
            only_a, only_b = {i: pairs[i][0] for i in idx}, {i: pairs[i][1] for i in idx}
            base, base_a, base_b = lost_by_query(c, base_rank)[idx], lost_by_query(c, base_rank, only_a)[idx], lost_by_query(c, base_rank, only_b)[idx]
            changes = {"a": d["a"][0] - d["a"][1], "b": d["b"][0] - d["b"][1], "both": d["both"][0] - d["both"][1]}
            changes["sum"] = changes["a"] + changes["b"]
            lost = {}
            rec = {"n_queries": len(idx), "lost_all": float(base.sum()), "lost_a": float(base_a.sum()), "lost_b": float(base_b.sum())}
            for name, change in changes.items():
                ranked = after(idx, change)
                lost[name] = lost_by_query(c, ranked)[idx]
                la, lb = lost_by_query(c, ranked, only_a)[idx], lost_by_query(c, ranked, only_b)[idx]
                rec[name] = {"recovered_all": 1 - float(lost[name].sum() / base.sum()), "lost_a": float(la.sum()), "lost_b": float(lb.sum())}
            best = max(("a", "b"), key=lambda k: rec[k]["recovered_all"])
            rec["best_single"] = best
            rec["both_vs_best"] = ratio_ci(base - lost["both"], base, base - lost[best], base)
            res["two_titles"] = rec
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **res)


def boost_texts(ds, corpus: str) -> dict:
    """Per kind of span, query row to the query without those spans, removed together. Only ``title`` reads the labels."""
    df = corpus_df(corpus)
    n = _SAMPLED[corpus]
    out = {"rare1": dict(variants(ds, corpus)[0]["rare"]), "rare_all": {}, "title": {}}
    if corpus == "hotpotqa-1m":
        out["caps"] = {}
    for i, (q, rel) in enumerate(zip(ds.queries, ds.qrels)):
        sp = spans(q)
        words = [s[0] for s in sp]
        rare = [(a, b) for w, a, b in sp if content(w) and 1 + df.get(w, 0) <= n / 1e4]
        titled = [(sp[s][1], sp[s + m - 1][2]) for m, hits in (find_title(words, ds.titles[d]) for d in rel) for s in hits]
        kinds = {"rare_all": rare, "title": titled}
        if "caps" in out:
            caps = []
            for name in entities(q):
                t = [w for w, _, _ in spans(name)]
                caps += [(sp[s][1], sp[s + len(t) - 1][2]) for s in range(len(words) - len(t) + 1) if t and words[s:s + len(t)] == t]
            kinds["caps"] = caps
        for kind, ranges in kinds.items():
            if ranges and keeps_content(cut(q, ranges)):
                out[kind][i] = cut(q, ranges)
    return out


def exp_boost(out: Path, corpora, tiers, seeds) -> None:
    for log, key, corpus, tier, seed, t0 in units("boost", out, corpora, tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed, K=8192 if corpus in CORPORA else None)
        ds = c.ds
        order = np.random.default_rng(0).permutation(len(ds.qrels))
        dev, test = (order[: len(order) // 2], order[len(order) // 2:]) if corpus in CORPORA else (order[:0], order)
        native, oracle = per_query(c.native, ds.qrels), per_query(c.oracle, ds.qrels)
        base = per_query(search(c.Q_hat, c.E_src, 1_000, DEV, exclude=ds.self_rows), ds.qrels)
        metrics = [f"recall@{k}" for k in (30, 100, 300, 1000)]

        def ceiling(m: str, part) -> float:
            return min(float(native[m][part].mean()), float(oracle[m][part].mean()))

        def read(part, m) -> dict:
            return {k: float(m[k][part].mean()) / ceiling(k, part) for k in metrics} if len(part) else {}

        res = {"baseline": {"dev": read(dev, base), "test": read(test, base)}, "n_test": len(test)}
        boot = np.random.default_rng(0).integers(0, len(test), (1000, len(test)))
        for kind, by_query in boost_texts(ds, corpus).items():
            idx = sorted(by_query)
            if not idx:
                continue
            without = c.project(encode_queries(c.tgt, [by_query[i] for i in idx], ds.task))
            for g in GAINS:
                Q = c.Q_hat.copy()
                Q[idx] = unit(c.Q_hat[idx] + g * (c.Q_hat[idx] - without))
                m = per_query(search(Q, c.E_src, 1_000, DEV, exclude=ds.self_rows), ds.qrels)
                gain = {}
                for k in metrics:
                    diff = (m[k] - base[k])[test] / ceiling(k, test)
                    gain[k] = {"points": 100 * float(diff.mean()), "ci": [100 * float(np.percentile(diff[boot].mean(1), q)) for q in (2.5, 97.5)]}
                res[f"{kind}@{g}"] = {"n_boosted": len(idx), "dev": read(dev, m), "test": read(test, m), "gain": gain}
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, **res)


def title_mentions(out: Path, corpus: str) -> dict:
    """Per matched title, documents that mention it, leaving out passages of its own article. Written once per corpus."""
    path = out / f"fame_{corpus}.json"
    if not path.exists():
        ds = dataset(corpus)[1]
        wanted = {title_key(ds.titles[d]) for d in variants(ds, corpus)[1].values()}
        by_first = defaultdict(list)
        for t in wanted:
            by_first[t[0]].append(t)
        counts = Counter()
        for i in range(ds.N):
            toks = _tokens(ds.texts[i])
            seen = {t for j, w in enumerate(toks) if w in by_first for t in by_first[w] if tuple(toks[j:j + len(t)]) == t}
            if seen:
                seen.discard(title_key(ds.titles[i]))
                counts.update(seen)
            if i % 200_000 == 0:
                durable.progress(f"fame {corpus}: {i:,} of {ds.N:,} documents")
        with durable.atomic_file(path) as f:
            f.write(json.dumps({" ".join(t): counts.get(t, 0) for t in wanted}).encode())
    return json.loads(path.read_text())


def fame_path(out: Path, corpus: str, tier: str, seed: int) -> Path:
    return out / "fame_rows" / f"{corpus}_{tier}_s{seed}.npz"


def exp_fame(out: Path, corpora, tiers, seeds) -> None:
    for log, key, corpus, tier, seed, t0 in units("fame", out, [c for c in corpora if c != "msmarco-1m"], tiers, seeds):
        c = Ctx(FITS, corpus, tier, seed)
        ds = c.ds
        mentions = title_mentions(out, corpus)
        texts, title_pos = variants(ds, corpus)
        idx = sorted(texts["title"])
        dx_t, dq_t = span_deltas(c, idx, [texts["title"][i] for i in idx])
        dx_c, dq_c = span_deltas(c, idx, [texts["control"][i] for i in idx])
        ranked = search(c.Q_hat, c.E_src, 1_000, DEV, exclude=ds.self_rows)
        table = []
        for j, i in enumerate(idx):
            ids = c.native[i]
            D = rows(c.E_src, ids)
            s_t, sh_t, s_c, sh_c = D @ dx_t[j], D @ dq_t[j], D @ dx_c[j], D @ dq_c[j]
            s_t, sh_t, s_c, sh_c = s_t - s_t.mean(), sh_t - sh_t.mean(), s_c - s_c.mean(), sh_c - sh_c.mean()
            d = title_pos[i]
            nat = {x: r for r, x in enumerate(ids.tolist())}.get(d, 10**6) + 1
            br = {x: r for r, x in enumerate(ranked[i].tolist())}.get(d, 10**6) + 1
            table.append((i, mentions[" ".join(title_key(ds.titles[d]))], sh_t @ s_t, s_t @ s_t, sh_c @ s_c, s_c @ s_c, nat, br))
        a = np.array(table, dtype=np.float64)
        path = fame_path(out, corpus, tier, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with durable.atomic_file(path) as f:
            np.savez(f, query=a[:, 0], mentions=a[:, 1], num_t=a[:, 2], den_t=a[:, 3], num_c=a[:, 4], den_c=a[:, 5], native=a[:, 6], bridged=a[:, 7])
        log.write(key, t0, corpus=corpus, tier=tier, seed=seed, n=len(a), mentions_median=float(np.median(a[:, 1])))


def fame_summary(out: Path, tier: str, seed: int) -> dict:
    """Title transfer by bin in each corpus, the top occupied bin against the bottom one, and HotpotQA reweighted to NQ's mix."""
    z = {c: np.load(fame_path(out, c, tier, seed)) for c in ("hotpotqa-1m", "nq-1m")}
    bins = {c: binned(z[c]["mentions"], FAME_EDGES) for c in z}
    res = {}
    for c in z:
        rows_ = []
        for b, label in enumerate(FAME_BINS):
            m = bins[c] == b
            native_top = m & (z[c]["native"] <= 300)
            rows_.append({"bin": label, "n": int(m.sum()), "title": float(z[c]["num_t"][m].sum() / z[c]["den_t"][m].sum()) if m.any() else None,
                          "control": float(z[c]["num_c"][m].sum() / z[c]["den_c"][m].sum()) if m.any() else None,
                          "lost300": float((z[c]["bridged"][native_top] > 300).mean()) if native_top.any() else None})
        occupied = [b for b, r in enumerate(rows_) if r["n"] >= MIN_BIN]
        lo, hi = bins[c] == occupied[0], bins[c] == occupied[-1]
        res[c] = {"bins": rows_, "top_vs_bottom": {"bottom": FAME_BINS[occupied[0]], "top": FAME_BINS[occupied[-1]],
                                                   **ratio_ci(z[c]["num_t"][hi], z[c]["den_t"][hi], z[c]["num_t"][lo], z[c]["den_t"][lo], paired=False)}}

    def gap(pick: dict) -> tuple[float, float]:
        h, n = pick["hotpotqa-1m"], pick["nq-1m"]
        hp, nq = z["hotpotqa-1m"], z["nq-1m"]
        t_h = hp["num_t"][h].sum() / hp["den_t"][h].sum()
        t_n = nq["num_t"][n].sum() / nq["den_t"][n].sum()
        total, adjusted, used = 0.0, 0.0, 0.0
        for b in range(len(FAME_BINS)):
            mh, mn = h[bins["hotpotqa-1m"][h] == b], n[bins["nq-1m"][n] == b]
            if len(mh) and len(mn):
                w = nq["den_t"][mn].sum()
                adjusted += w * hp["num_t"][mh].sum() / hp["den_t"][mh].sum()
                total += w * nq["num_t"][mn].sum() / nq["den_t"][mn].sum()
                used += w
        return float(t_n - t_h), float((total - adjusted) / used)

    all_rows = {c: np.arange(len(z[c]["mentions"])) for c in z}
    raw, adjusted = gap(all_rows)
    rng = np.random.default_rng(0)
    draws = [gap({c: rng.choice(all_rows[c], len(all_rows[c])) for c in z}) for _ in range(500)]
    removed = [1 - a / r for r, a in draws]
    res["gap"] = {"raw": raw, "reweighted": adjusted, "removed": 1 - adjusted / raw, "removed_ci": [float(np.percentile(removed, 2.5)), float(np.percentile(removed, 97.5))]}
    return res


def exp_tables2(out: Path, corpora, tiers, seeds) -> None:
    t = ["# HotpotQA follow-up, round 2: tables", "", "From `scripts/hotpot_identity.py tables2`. Rules in `docs/reports/hotpotqa-followup.md`.", "", "## Test 4: joint repair", "",
         "Queries naming both positives' titles. Lost A and lost B are positives lost, before and after each repair. Recovered is over all positives on these queries.", "",
         "| Tier | Seed | Queries | Lost A, B | Repair A: recovered, lost A, B | Repair B | Both | Sum, sanity | Both - best single |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    joint = read(out, "joint")
    for r in joint:
        p = r.get("two_titles")
        if p:
            cell = lambda n: f"{p[n]['recovered_all']:.3f}, {p[n]['lost_a']:.0f}, {p[n]['lost_b']:.0f}"
            v = p["both_vs_best"]
            t.append(f"| {r['tier']} | {r['seed']} | {p['n_queries']} | {p['lost_a']:.0f}, {p['lost_b']:.0f} | {cell('a')} | {cell('b')} | {cell('both')} | {cell('sum')} | {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |")
    t += ["", "**The reverse repair, all title queries.** Title native means the title's native part with the rest mapped; rest native the reverse.", "",
          "| Corpus | Tier | Seed | Queries | Lost: title positive, others | Title native: recovered title, others, all | Rest native: recovered title, others, all |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in joint:
        p = r["reverse"]
        cell = lambda n: f"{p[n]['recovered_title']:.3f}, {p[n]['recovered_other']:.3f}, {p[n]['recovered_all']:.3f}"
        t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | {p['n_queries']} | {p['lost_title']:.0f}, {p['lost_other']:.0f} | {cell('title_native')} | {cell('rest_native')} |")
    boost = read(out, "boost")
    settings = sorted({k for r in boost for k in r if "@" in k})
    pick_cells = [r for r in boost if r["corpus"] in CORPORA and r["seed"] == 0]
    t += ["", "## Test 5: the boost", "", "**Choosing, on the dev half, seed 0.** Mean `Recall@300` retention over the six 1M cells; caps is HotpotQA only, so it's shown but can't be chosen for everything.", "",
          "| Setting | Mean over the 1M cells on the dev half | Against baseline, points |", "| --- | --- | --- |"]
    dev_base = float(np.mean([r["baseline"]["dev"]["recall@300"] for r in pick_cells])) if pick_cells else 0.0
    scored = {}
    for k in settings:
        cells = [r[k]["dev"]["recall@300"] for r in pick_cells if k in r]
        if cells:
            base = float(np.mean([r["baseline"]["dev"]["recall@300"] for r in pick_cells if k in r]))
            scored[k] = float(np.mean(cells))
            t.append(f"| {k} | {scored[k]:.4f} ({len(cells)} cells) | {100 * (scored[k] - base):+.2f} |")
    full = {k: v for k, v in scored.items() if k.split("@")[0] in ("rare1", "rare_all") and all(k in r for r in pick_cells)}
    pick = max(full, key=full.get) if full else None
    t += ["", f"Baseline on the dev half: {dev_base:.4f}. **Chosen: {pick}.** Label-free settings for all six cells only.", "", "**On queries not used to choose.** Points of retention [bootstrap 95% CI over queries].", "",
          "| Corpus | Tier | Seed | Queries | Boosted | Baseline `Recall@300` | `Recall@30` | `Recall@100` | `Recall@300` | `Recall@1000` | Title ceiling at the same g, `Recall@300` |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    if pick:
        for r in boost:
            if pick in r:
                g = r[pick]["gain"]
                cell = lambda m: f"{g[m]['points']:+.2f} [{g[m]['ci'][0]:+.2f}, {g[m]['ci'][1]:+.2f}]"
                ceiling = r.get("title@" + pick.split("@")[1])
                t.append(f"| {r['corpus']} | {r['tier']} | {r['seed']} | {r['n_test']} | {r[pick]['n_boosted']} | {r['baseline']['test']['recall@300']:.3f} | {cell('recall@30')} | {cell('recall@100')} | {cell('recall@300')} | {cell('recall@1000')} | "
                         f"{(format(ceiling['gain']['recall@300']['points'], '+.2f') if ceiling else '')} |")
    t += ["", "**Every setting, `Recall@300` points on queries not used to choose, seed 0.**", "", "| Corpus | Tier | " + " | ".join(settings) + " |", "| --- | --- | " + " | ".join("---" for _ in settings) + " |"]
    for r in boost:
        if r["seed"] == 0:
            t.append(f"| {r['corpus']} | {r['tier']} | " + " | ".join((f"{r[k]['gain']['recall@300']['points']:+.2f}" if k in r else "") for k in settings) + " |")
    t += ["", "## Test 6: how well known a name is", "", f"Mentions outside the title's own article, over the whole corpus. Bins under {MIN_BIN} queries don't count as occupied.", "",
          "| Corpus | Tier | Seed | " + " | ".join(FAME_BINS) + " | Top - bottom occupied bin |", "| --- | --- | --- | " + " | ".join("---" for _ in FAME_BINS) + " | --- |"]
    gaps_ = []
    for tier in tiers:
        for seed in seeds:
            if not fame_path(out, "hotpotqa-1m", tier, seed).exists():
                continue
            s = fame_summary(out, tier, seed)
            for c in ("hotpotqa-1m", "nq-1m"):
                v = s[c]["top_vs_bottom"]
                cells = [(f"{b['title']:.3f} / {b['control']:.3f} / {100 * b['lost300']:.1f}% (n {b['n']})" if b["n"] and b["lost300"] is not None else "-") for b in s[c]["bins"]]
                t.append(f"| {c} | {tier} | {seed} | " + " | ".join(cells) + f" | {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], {v['bottom']} to {v['top']} |")
            g = s["gap"]
            gaps_.append(f"| {tier} | {seed} | {g['raw']:.3f} | {g['reweighted']:.3f} | {g['removed']:.2f} [{g['removed_ci'][0]:.2f}, {g['removed_ci'][1]:.2f}] |")
    t += ["", "Each cell above: title transfer / control transfer / title positives lost from native's top 300.", "", "**HotpotQA's gap to NQ in title transfer, reweighted to NQ's mix of bins.**", "", "| Tier | Seed | Gap | Reweighted | Removed |", "| --- | --- | --- | --- | --- |", *gaps_]
    with durable.atomic_file(out / "tables2.md") as f:
        f.write("\n".join(t).encode() + b"\n")


EXPERIMENTS = {"band": exp_band, "entity": exp_entity, "rarity": exp_rarity, "fit": exp_fit, "tables": exp_tables, "joint": exp_joint, "boost": exp_boost, "fame": exp_fame, "tables2": exp_tables2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=list(EXPERIMENTS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpora", nargs="+", default=list(CORPORA), choices=list(CORPORA) + list(HELD_OUT))
    p.add_argument("--tiers", nargs="+", default=["T1", "T3"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[args.exp](args.out, args.corpora, args.tiers, args.seeds)


if __name__ == "__main__":
    main()
