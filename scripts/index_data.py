"""Writes data/catalog.csv: what every local, git-ignored data and result file holds. Explained in data/README.md.

Cache files are named by a hash of their contents' inputs, and that name is how the code finds them, so they can't be
renamed. This identifies them instead, by computing the same keys the code computes and looking for files with those
names. A file named here as ``computed`` is therefore known, not inferred. What no replay reaches is listed as
``unidentified``, with its shape.

Reads the caches only: the catalog is the one file it writes. Needs no GPU and no network. About 20 minutes, on 6 processes and up to 60 GB of RAM.

    uv run python -u scripts/index_data.py
"""
import csv
import hashlib
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np  # noqa: E402

from spacebridge.anchors import random_anchors  # noqa: E402
from spacebridge_bench import durable  # noqa: E402
from spacebridge_bench.datasets import _LOADER_VERSION, DATASETS, _load_beir  # noqa: E402
from spacebridge_bench.models import MODELS  # noqa: E402
from spacebridge_bench.run import anchor_candidates, make_proxies  # noqa: E402
from spacebridge_bench.search import RANKING_K  # noqa: E402

DATA, CACHE, OUTPUTS = Path("data"), Path("data/cache"), Path("outputs")
CATALOG = DATA / "catalog.csv"
COLUMNS = ("area", "path", "bytes", "modified", "rows", "dim", "dtype", "model", "dataset", "role", "detail", "identified_by")

PROXY_STYLES = ("title", "lead", "title-any", "short")
DOC_TOKENS = (None, 64, 128, 256)  # None: the model's full length
RANDOM_K = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)
RANDOM_SEEDS = range(3)
SUFFIX_FROM = 1_000_000  # encode_suffix.py encodes everything after the first 1M rows as its own file
ANCHOR_NAME = re.compile(r"^(?P<ds>[^_]+)_(?P<src>[^_]+)_N(?P<N>\d+)_n(?P<n>\d+)_K(?P<K>\d+)_s(?P<s>\d+)(?P<tkm>_tkm)?\.npy$")
RERANK_NAME = re.compile(r"^(?P<key>.+)_L(?P<L>\d+)(?:\.(?P<pid>\d+)\.scores|\.npz)$")
RESULT_SUFFIXES = {".jsonl", ".npz", ".json", ".md", ".txt"}
LOG_QUERY_CORPORA = ("hotpotqa-1m", "msmarco-1m", "nq-1m")
N_TEST, N_LOG, N_DEV = 2_000, 85_000, 500  # as in scripts/hotpot_study.py
SKIP_RESULT_DIRS = {"clone", ".venv", "__pycache__"}  # m3_quickstart/clone is a full checkout with its own environment

HAVE: dict[str, set[str]] = {}  # embedding model directory -> file stems
ROWS: dict[str, set[int]] = {}  # embedding model directory -> row counts of those files
ANCHORS_BY_BASE: dict[str, list[tuple[Path, re.Match]]] = {}
SIZES_BY_BASE: dict[str, set[int]] = {}
LEXICAL_BY_NAME: dict[str, str] = {}
TRANSLATION_KEYS: dict[str, list[str]] = {}


def npy_shape(path: Path) -> tuple[tuple[int, ...], str]:
    a = np.load(path, mmap_mode="r")
    return a.shape, str(a.dtype)


def _key_start(model: str, max_len: int):
    return hashlib.sha256(json.dumps({"hf_id": MODELS[model].hf_id, "max_seq_length": max_len}).encode())


def find_files(model: str, max_len: int, inputs: Iterable[str], prefixes: bool = False) -> list[tuple[str, int]]:
    """(file stem, rows) of the cached files holding ``inputs`` (a list), or with ``prefixes`` any first rows of them
    (any iterable: it is read only as far as the model's longest file)."""
    rows = ROWS[model]
    if not rows or (not prefixes and len(inputs) not in rows):
        return []
    last = max(rows) if prefixes else len(inputs)
    h, found = _key_start(model, max_len), []
    for i, text in enumerate(inputs, 1):
        h.update(text.encode())
        h.update(b"\x00")
        if (prefixes or i == last) and i in rows:
            stem = h.copy().hexdigest()[:24]
            if stem in HAVE[model]:
                found.append((stem, i))
        if i == last:
            break
    return found


def load_parsed(name: str):
    """A dataset's parsed pickle, read directly: ``load_beir`` would write one if it were missing."""
    pickles = sorted((CACHE / "datasets").glob(f"{name}_*.pkl"))
    if pickles:
        with pickles[0].open("rb") as f:
            return pickle.load(f)
    return _load_beir(name, None, 0)  # nothing written: a dataset with embeddings but no pickle, from the HF cache


def identify_dataset(name: str) -> dict:
    """Every embedding file reachable from one dataset, in one process: the largest load in tens of GB."""
    ds = load_parsed(name)
    out: list[tuple[str, str, str, str, str]] = []
    covered: set[str] = set()

    for model in HAVE:
        spec = MODELS[model]
        max_len = spec.max_seq_length
        for stem, n in find_files(model, max_len, (spec.format_doc(a, b) for a, b in zip(ds.titles, ds.texts)), prefixes=True):
            role = "corpus-documents" if n == ds.N else "corpus-documents-first-rows"
            out.append((model, stem, role, name, f"all {n:,} rows" if n == ds.N else f"first {n:,} of {ds.N:,} rows"))
            covered.add(model)
        if ds.N > SUFFIX_FROM:
            for stem, n in find_files(model, max_len, [spec.format_doc(a, b) for a, b in zip(ds.titles[SUFFIX_FROM:], ds.texts[SUFFIX_FROM:])]):
                out.append((model, stem, "corpus-documents-after-first-1m", name, f"rows {SUFFIX_FROM:,} to {ds.N:,}"))
                covered.add(model)
        for label, queries in (("test queries", ds.queries), ("test queries in the corpus language", ds.src_queries)):
            if queries:
                for stem, n in find_files(model, max_len, [spec.format_query(q, ds.task) for q in queries], prefixes=True):
                    role = "test-queries" if n == len(queries) else "test-queries-first-rows"
                    out.append((model, stem, role, name, f"{label}: {n:,} of {len(queries):,}"))
                    covered.add(model)

    for stem_model, stem, n in _log_query_files(name, ds):
        out.append((stem_model, stem, "log-queries", name, f"non-test queries for scripts/hotpot_study.py: first {n:,}"))

    anchor_sets: list[tuple[str, str, np.ndarray]] = []
    for path, m in ANCHORS_BY_BASE.get(name, []):
        ids = np.load(path)
        select = "k-means" if m["tkm"] else "k-means (first version, before the torch one)"
        anchor_sets.append((m["ds"], f"{select}, source {m['src']}, K={m['K']}, seed {m['s']}", ids))
    sizes = {ds.N} | {n for n in SIZES_BY_BASE.get(name, ()) if n < ds.N}
    for n in sorted(sizes):
        view = ds if n == ds.N else replace(ds, name=f"{name}@{n}", doc_ids=ds.doc_ids[:n])
        # What anchors may come from: the benchmark's rule, the studies' (hotpot_study.py keeps positives out), the library's.
        pools = {
            "benchmark rows": anchor_candidates(view),
            "no positives": np.setdiff1d(np.arange(n), [d for rel in ds.qrels for d in rel]),
            "any row": np.arange(n),
        }
        for rule, allowed in pools.items():
            for K in RANDOM_K:
                if K <= len(allowed) and any(K in ROWS[m] for m in covered):
                    for seed in RANDOM_SEEDS:
                        anchor_sets.append((view.name, f"random from {rule}, K={K}, seed {seed}", allowed[random_anchors(len(allowed), K, seed)]))

    seen_doc: set[tuple[str, int, bytes]] = set()
    for set_name, describe, ids in anchor_sets:
        ids = ids[ids >= 0]
        K = len(ids)
        if not any(K in ROWS[m] for m in covered):
            continue
        titles, texts = [ds.titles[i] for i in ids], [ds.texts[i] for i in ids]
        id_bytes = ids.tobytes()
        for style in PROXY_STYLES:
            proxies = make_proxies(titles, texts, style)
            for model in covered:
                spec = MODELS[model]
                for stem, _ in find_files(model, spec.max_seq_length, [spec.format_query(p, ds.task) for p in proxies]):
                    out.append((model, stem, "anchor-query-proxies", set_name, f"{describe}, proxy style {style}"))
        for model in covered:
            spec = MODELS[model]
            inputs = None
            for tokens in DOC_TOKENS:
                max_len = min(tokens or spec.max_seq_length, spec.max_seq_length)
                if (model, max_len, id_bytes) in seen_doc:
                    continue
                seen_doc.add((model, max_len, id_bytes))
                inputs = inputs or [spec.format_doc(a, b) for a, b in zip(titles, texts)]
                for stem, _ in find_files(model, max_len, inputs):
                    out.append((model, stem, "anchor-documents", set_name, f"{describe}, cut to {max_len} tokens"))

    translations = {}
    key = hashlib.sha256("\n".join(ds.queries).encode()).hexdigest()[:16]
    for file_name in TRANSLATION_KEYS.get(key, []):
        translator = file_name.split("_")[0]
        done = {r["i"]: r["text"] for r in durable.read_jsonl(CACHE / "translations" / file_name)[0]}
        translations[file_name] = f"{translator}: {name} queries translated to English, {len(done):,} of {len(ds.queries):,}"
        if len(done) == len(ds.queries):
            for model in HAVE:
                spec = MODELS[model]
                for stem, _ in find_files(model, spec.max_seq_length, [spec.format_query(done[i], ds.task) for i in range(len(ds.queries))]):
                    out.append((model, stem, "translated-queries", name, f"{translator} translation"))

    lexical = {}
    h = hashlib.blake2b(digest_size=12)
    cuts = {int(k.split("@")[1]): k for k in LEXICAL_BY_NAME if "@" in k and k.split("@")[0] == name}
    for i, doc_id in enumerate(ds.doc_ids, 1):
        h.update(str(doc_id).encode() + b"\x00")
        if i in cuts and h.hexdigest() == LEXICAL_BY_NAME[cuts[i]]:
            lexical[cuts[i]] = True
    if name in LEXICAL_BY_NAME and h.hexdigest() == LEXICAL_BY_NAME[name]:
        lexical[name] = True

    return {
        "name": name, "matches": out, "N": ds.N, "n_queries": len(ds.queries), "lang": ds.lang, "query_lang": ds.query_lang,
        "self_rows": ds.self_rows, "translations": translations, "lexical": lexical,
    }


def _log_query_files(name: str, ds) -> list[tuple[str, str, int]]:
    """The 85,000 + 500 log queries of ``hotpot_study.real_queries``: other queries of the same corpus, none a test query."""
    if name not in LOG_QUERY_CORPORA:
        return []
    if name == "nq-1m":
        texts = list(ds.queries[N_TEST:])
    else:
        from datasets import load_dataset

        spec = DATASETS[name]
        ids = sorted({str(q) for q in load_dataset(spec.repo, "default", split="train")["query-id"]})
        np.random.default_rng(0).shuffle(ids)
        rows = load_dataset(spec.repo, "queries", split=spec.queries_split)
        text = dict(zip(map(str, rows["_id"]), rows["text"]))
        seen = set(ds.queries[:N_TEST])
        texts = [text[i] for i in ids[: N_LOG + N_DEV + 1000] if i in text and text[i] not in seen][: N_LOG + N_DEV]
    found = []
    for model in HAVE:
        spec = MODELS[model]
        found += [(model, stem, n) for stem, n in find_files(model, spec.max_seq_length, [spec.format_query(t, ds.task) for t in texts], prefixes=True)]
    return found


def identify_banks() -> list[tuple[str, str, str, str, str]]:
    """Language banks: FLORES sentences (from data/flores) and the Tatoeba fallback, in the prompts they were encoded with."""
    tasks = sorted({spec.task for spec in DATASETS.values()})
    banks: list[tuple[str, list[str]]] = []
    for path in sorted((DATA / "flores").glob("flores_*.json")):
        lang = path.name.split("_")[1].split("-")[1]
        variant = path.name.split("_")[2]
        parsed = json.loads(path.read_text())
        for side in ("en", lang):
            for split in ("dev", "devtest"):
                banks.append((f"FLORES {variant}, {split}, en-{lang}, {side} side", parsed[split][side]))
            banks.append((f"FLORES {variant}, dev+devtest, en-{lang}, {side} side", parsed["dev"][side] + parsed["devtest"][side]))
    try:
        from spacebridge.language import load_shift_bank

        en, zh, _ = load_shift_bank("zh", bank="tatoeba")
        banks += [("Tatoeba, en-zh, en side", en), ("Tatoeba, en-zh, zh side", zh)]
    except Exception as e:  # not in the HF cache
        durable.progress(f"Tatoeba bank not checked: {type(e).__name__}")
    out = []
    for label, sentences in banks:
        for model in HAVE:
            spec = MODELS[model]
            templates = {spec.query_template.format(task=t, text="{}") for t in tasks}
            for template in templates:
                for stem, n in find_files(model, spec.max_seq_length, [template.replace("{}", s, 1) for s in sentences], prefixes=True):
                    out.append((model, stem, "language-bank-queries", "", f"{label}, first {n:,}"))
            for tokens in DOC_TOKENS:
                max_len = min(tokens or spec.max_seq_length, spec.max_seq_length)
                for stem, n in find_files(model, max_len, [spec.format_doc("", s) for s in sentences], prefixes=True):
                    out.append((model, stem, "language-bank-documents", "", f"{label}, first {n:,}, cut to {max_len} tokens"))
    return out


def _unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def identify_rankings(by_stem: dict, infos: dict) -> dict[str, tuple[str, str, str]]:
    """Exact-search rankings: their key is the document file's name, dimension and length, then the query vectors."""
    targets = {p.stem.rsplit("_k", 1)[0] for p in (CACHE / "rankings").glob("*.npy")}
    found: dict[str, tuple[str, str, str]] = {}
    per_pair: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"docs": [], "queries": []})
    for (model, stem), labels in by_stem.items():
        for role, dataset, _ in labels:
            if role == "corpus-documents":
                per_pair[model, dataset]["docs"].append(stem)
            elif role in ("test-queries", "test-queries-first-rows", "translated-queries"):
                per_pair[model, dataset]["queries"].append(stem)
    for (model, dataset), files in per_pair.items():
        spec = MODELS[model]
        dims = {spec.dim} | {m.truncate_dim for m in MODELS.values() if m.cache_key == model and m.truncate_dim}
        info = infos[dataset]
        for q_stem in files["queries"]:
            Q_full = _unit(np.load(CACHE / "embeddings" / model / f"{q_stem}.npy").astype(np.float32))
            for dim in dims:
                Q = Q_full if dim == spec.dim else _unit(Q_full[:, :dim])
                excludes = [("excluding self-matches", info["self_rows"][: len(Q)]), ("", None)]
                for doc_stem in files["docs"]:
                    for size in {info["N"]} | {n for n in SIZES_BY_BASE.get(dataset, ()) if n < info["N"]}:
                        name = dataset if size == info["N"] else f"{dataset}@{size}"
                        for note, exclude in excludes:
                            h = hashlib.sha256(f"{doc_stem}|{dim}|{size}".encode())
                            h.update(np.asarray(Q, dtype=np.float32).tobytes())
                            if exclude is not None:
                                h.update(np.asarray(exclude, dtype=np.int64).tobytes())
                            key = h.hexdigest()[:24]
                            if key in targets:
                                found[key] = (model, name, f"{len(Q):,} queries over {size:,} rows, {dim} dimensions, top {RANKING_K:,} {note}".strip())
    return found


def scan_embeddings(by_stem: dict) -> list[dict]:
    rows = []
    for model in sorted(HAVE):
        for path in sorted((CACHE / "embeddings" / model).iterdir()):
            if path.suffix != ".npy":
                rows.append(dict(area="embeddings", path=str(path), bytes=_size(path), model=model, role="unfinished-encode", detail="parts of an encode that did not finish", identified_by="name"))
                continue
            shape, dtype = npy_shape(path)
            labels = by_stem.get((model, path.stem), [])
            rows.append(dict(
                area="embeddings", path=str(path), bytes=path.stat().st_size, rows=shape[0], dim=shape[1], dtype=dtype, model=model,
                dataset=";".join(sorted({d for _, d, _ in labels if d})), role=";".join(sorted({r for r, _, _ in labels})) or "unidentified",
                detail=_cap(sorted({f"{d + ': ' if d else ''}{x}" for _, d, x in labels})), identified_by="computed" if labels else "",
            ))
    return rows


def _cap(items: list[str], keep: int = 4) -> str:
    return " | ".join(items[:keep]) + (f" | +{len(items) - keep} more" if len(items) > keep else "")


def _size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def scan_other_caches(infos: dict, rankings: dict) -> list[dict]:
    rows = []
    for path in sorted((CACHE / "datasets").glob("*.pkl")):
        name, stem = path.stem.rsplit("_", 1)
        spec = DATASETS.get(name)
        same = spec is not None and hashlib.sha256(repr((spec, None, 0, _LOADER_VERSION)).encode()).hexdigest()[:12] == stem
        info = infos.get(name, {})
        rows.append(dict(
            area="datasets", path=str(path), bytes=path.stat().st_size, rows=info.get("N", ""), dataset=name, role="parsed-dataset",
            detail=f"{info.get('N', 0):,} documents, {info.get('n_queries', 0):,} queries, {spec.repo if spec else 'not in the registry'}",
            identified_by="computed" if same else "name, key differs from the current code",
        ))
    for path in sorted((CACHE / "anchors").glob("*.npy")):
        m = ANCHOR_NAME.match(path.name)
        shape, dtype = npy_shape(path)
        rows.append(dict(
            area="anchors", path=str(path), bytes=path.stat().st_size, rows=shape[0], dtype=dtype, model=m["src"], dataset=m["ds"], role="anchor-rows",
            detail=f"{'k-means' if m['tkm'] else 'k-means (first version)'}, K={m['K']}, seed {m['s']}, from {int(m['n']):,} of {int(m['N']):,} rows", identified_by="name",
        ))
    for path in sorted((CACHE / "rankings").glob("*.npy")):
        shape, dtype = npy_shape(path)
        model, dataset, detail = rankings.get(path.stem.rsplit("_k", 1)[0], ("", "", ""))
        rows.append(dict(
            area="rankings", path=str(path), bytes=path.stat().st_size, rows=shape[0], dim=shape[1], dtype=dtype, model=model, dataset=dataset,
            role="exact-search-ranking" if model else "unidentified", detail=detail, identified_by="computed" if model else "",
        ))
    for path in sorted((CACHE / "hnsw").glob("*.faiss")):
        dataset, stem, dim, *build = path.stem.split("_")
        model = next((m for m in HAVE if stem in HAVE[m]), "")
        rows.append(dict(
            area="hnsw", path=str(path), bytes=path.stat().st_size, dim=dim.rstrip("d"), model=model, dataset=dataset, role="ann-index",
            detail=f"{' '.join(build)}, over data/cache/embeddings/{model}/{stem}.npy", identified_by="name",
        ))
    for path in sorted(p for p in (CACHE / "lexical").iterdir() if p.is_dir()):
        name, _ = path.name.rsplit("_", 1)
        verified = infos.get(name.split("@")[0], {}).get("lexical", {}).get(name)
        shape = json.loads((path / "meta.json").read_text()).get("shape", ["", ""])
        rows.append(dict(
            area="lexical", path=str(path) + "/", bytes=_size(path), rows=shape[0], dataset=name, role="bm25-index",
            detail=f"{shape[0]:,} documents, {shape[1]:,} terms", identified_by="computed" if verified else "name",
        ))
    for path in sorted((CACHE / "rerank").iterdir()):
        m = RERANK_NAME.match(path.name)
        if m is None:
            rows.append(dict(area="rerank", path=str(path), bytes=path.stat().st_size, role="unidentified", detail="not a name the reranker cache writes"))
            continue
        kind = f"append-only log of process {m['pid']}" if m["pid"] else "merged scores"
        rows.append(dict(
            area="rerank", path=str(path), bytes=path.stat().st_size, rows=path.stat().st_size // 12 if m["pid"] else "",
            model=m["key"], role="reranker-scores", detail=f"{kind}, max length {m['L']}, keyed by hash of (prompt, query, document)", identified_by="name",
        ))
    translations = {k: v for info in infos.values() for k, v in info.get("translations", {}).items()}
    for path in sorted((CACHE / "translations").glob("*.jsonl")):
        rows.append(dict(
            area="translations", path=str(path), bytes=path.stat().st_size, role="translated-queries", model=path.name.split("_")[0],
            detail=translations.get(path.name, ""), identified_by="computed" if path.name in translations else "name",
        ))
    for path in sorted(p for p in (CACHE / "onnx").iterdir() if p.is_dir()):
        rows.append(dict(
            area="onnx", path=str(path) + "/", bytes=_size(path), model=path.name, role="onnx-model",
            detail="model.onnx with its external tensors, and model_int8.onnx (quantized, for the CPU); made by scripts/export_reranker_onnx.py", identified_by="name",
        ))
    for path in sorted((DATA / "flores").glob("*.json")):
        _, pair, variant, _ = path.stem.split("_")
        counts = json.loads(path.read_text()).get("counts", {})
        rows.append(dict(
            area="flores", path=str(path), bytes=path.stat().st_size, role="parallel-sentences",
            detail=f"{pair}, {variant}: {json.dumps(counts)}"[:300], identified_by="name",
        ))
    return rows


def scan_results() -> list[dict]:
    """Result files, with what each ran over. No measured values: those stay in the files."""
    rows = []
    for root, dirs, files in os.walk(OUTPUTS):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_RESULT_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix not in RESULT_SUFFIXES:
                continue
            row = dict(area="results", path=str(path), bytes=path.stat().st_size, role=f"result-{path.suffix[1:]}", identified_by="name")
            if path.suffix == ".jsonl":
                records, _ = durable.read_jsonl(path)
                seen = defaultdict(set)
                for r in records:
                    for field in ("dataset", "corpus", "src", "tgt", "K", "seed"):
                        if isinstance(r, dict) and isinstance(r.get(field), (str, int)):
                            seen[field].add(str(r[field]))
                row["rows"] = len(records)
                row["dataset"] = _cap(sorted(seen["dataset"] | seen["corpus"]), 6)
                pairs = [f"{f}={_cap(sorted(seen[f]), 6)}" for f in ("src", "tgt", "K", "seed") if seen[f]]
                row["detail"] = "; ".join(pairs)
            rows.append(row)
    return rows


def main() -> None:
    for model_dir in sorted((CACHE / "embeddings").iterdir()):
        model = model_dir.name
        HAVE[model], ROWS[model] = set(), set()
        for path in model_dir.glob("*.npy"):
            HAVE[model].add(path.stem)
            ROWS[model].add(npy_shape(path)[0][0])
    for path in sorted((CACHE / "anchors").glob("*.npy")):
        m = ANCHOR_NAME.match(path.name)
        if m is None:
            raise ValueError(f"an anchor file with an unexpected name: {path}")
        base, _, cut = m["ds"].partition("@")
        ANCHORS_BY_BASE.setdefault(base, []).append((path, m))
        if cut:
            SIZES_BY_BASE.setdefault(base, set()).add(int(cut))
    for path in sorted(p for p in (CACHE / "lexical").iterdir() if p.is_dir()):
        name, digest = path.name.rsplit("_", 1)
        LEXICAL_BY_NAME[name] = digest
        base, _, cut = name.partition("@")
        if cut:
            SIZES_BY_BASE.setdefault(base, set()).add(int(cut))
    for path in sorted((CACHE / "translations").glob("*.jsonl")):
        TRANSLATION_KEYS.setdefault(path.stem.split("_")[1], []).append(path.name)
    # The 1M-row nested corpora are what the full ones hold in their first rows, so their cuts apply there too.
    for base in list(SIZES_BY_BASE):
        if base.endswith("-1m-nested"):
            SIZES_BY_BASE.setdefault(base.replace("-1m-nested", "-full"), set()).update(SIZES_BY_BASE[base])

    pickled = sorted(p.stem.rsplit("_", 1)[0] for p in (CACHE / "datasets").glob("*.pkl"))
    names = pickled + [n for n in ("nfcorpus",) if n not in pickled]
    by_stem: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    infos: dict[str, dict] = {}
    with get_context("fork").Pool(6) as pool:
        for n, result in enumerate(pool.imap_unordered(_identify_safely, names), 1):
            infos[result["name"]] = result
            for model, stem, role, dataset, detail in result.get("matches", []):
                by_stem[model, stem].append((role, dataset, detail))
            durable.progress(f"{n}/{len(names)} {result['name']}: {len(result.get('matches', [])):,} matches{', ' + result['error'] if 'error' in result else ''}")
    for model, stem, role, dataset, detail in identify_banks():
        by_stem[model, stem].append((role, dataset, detail))
    durable.progress(f"language banks done; {len(by_stem):,} embedding files named")
    rankings = identify_rankings(by_stem, {k: v for k, v in infos.items() if "self_rows" in v})
    durable.progress(f"rankings: {len(rankings):,} named")

    rows = scan_embeddings(by_stem) + scan_other_caches(infos, rankings) + scan_results()
    for r in rows:
        r["modified"] = time.strftime("%Y-%m-%d", time.localtime(Path(r["path"]).stat().st_mtime))
    rows.sort(key=lambda r: r["path"])
    tmp = CATALOG.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(CATALOG)

    print(f"\n{CATALOG}: {len(rows):,} rows\n")
    print(f"{'area':<14}{'files':>8}{'GB':>9}{'unident.':>10}{'unident. GB':>13}")
    areas = sorted({r["area"] for r in rows})
    for area in areas:
        part = [r for r in rows if r["area"] == area]
        lost = [r for r in part if r["role"] == "unidentified"]
        print(f"{area:<14}{len(part):>8,}{sum(r['bytes'] for r in part) / 1e9:>9.2f}{len(lost):>10,}{sum(r['bytes'] for r in lost) / 1e9:>13.3f}")
    print()
    for (area, role), count in sorted(Counter((r["area"], r["role"]) for r in rows).items()):
        print(f"{area:<14}{role:<70}{count:>7,}")


def _identify_safely(name: str) -> dict:
    try:
        return identify_dataset(name)
    except Exception as e:  # one dataset that can't be read leaves its files unidentified, and says so
        return {"name": name, "matches": [], "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    main()
