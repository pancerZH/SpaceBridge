"""What a run was: its full configuration and what it ran on. A resume must never count an old experiment as the one being asked for."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

from . import durable

HF_HUB = Path.home() / ".cache/huggingface/hub"


def config_id(config: dict[str, Any]) -> str:
    """A hash of a run's normalized configuration: anything that could change a number belongs in it."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _revision(kind: str, repo: str) -> Optional[str]:
    ref = HF_HUB / f"{kind}--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(model_ids: Iterable[str], dataset_repos: Iterable[str]) -> dict[str, Any]:
    """Code, dependencies and the revisions the local Hugging Face cache points to. A cache hit may still hold older vectors
    (data/README.md), so this says what the run was given, not what produced every cached file."""
    import torch

    lock = Path("uv.lock")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(_git("status", "--porcelain", "--", "src", "scripts")),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": {m: _revision("models", m) for m in sorted(set(model_ids))},
        "datasets": {d: _revision("datasets", d) for d in sorted(set(dataset_repos))},
    }


def implementation_digest() -> str:
    """A hash of the library's and the harness's source, and the shipped calibration: every default and every code path a number
    depends on. Any edit changes it, so results never resume across code changes."""
    import spacebridge

    h = hashlib.sha256()
    for root in (Path(spacebridge.__file__).parent, Path(__file__).parent):
        for path in sorted(root.rglob("*")):
            if path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
                h.update(str(path.relative_to(root.parent)).encode() + b"\x00" + path.read_bytes() + b"\x01")
    return h.hexdigest()[:16]


def revisions(model_ids: Iterable[str], dataset_repos: Iterable[str]) -> dict:
    return {"models": {m: _revision("models", m) for m in sorted(set(model_ids))}, "datasets": {d: _revision("datasets", d) for d in sorted(set(dataset_repos))}}


def corpus_digest(doc_ids: list, titles: list[str], texts: list[str]) -> str:
    """Identifies a corpus's rows, in order, by their IDs and text."""
    h = hashlib.sha256()
    for i, a, b in zip(doc_ids, titles, texts):
        h.update(f"{i}\x00{a}\x00{b}\x01".encode())
    return h.hexdigest()


def qrels_digest(qrels: list[dict[int, int]]) -> str:
    h = hashlib.sha256()
    for rel in qrels:
        h.update(json.dumps(sorted(rel.items())).encode() + b"\x00")
    return h.hexdigest()


def artifact_digest(path: Path) -> Optional[str]:
    """Content identity of a saved artifact: a directory by its manifest of checksums, a large file by its size and the hash of its first
    and last 4 MB (a sampled digest: it catches a rebuilt or replaced file, not an edit in its middle)."""
    path = Path(path)
    if path.is_dir():
        manifest = path / "manifest.json"
        return hashlib.sha256(manifest.read_bytes()).hexdigest()[:16] if manifest.exists() else None
    if not path.exists():
        return None
    size, h = path.stat().st_size, hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(4 << 20))
        if size > 8 << 20:
            f.seek(size - (4 << 20))
            h.update(f.read())
    return f"{size}:{h.hexdigest()[:16]}"


def query_digest(queries: list[str]) -> str:
    """Identifies the exact, ordered query list."""
    h = hashlib.sha256()
    for q in queries:
        h.update(q.encode() + b"\x00")
    return h.hexdigest()


def check_or_write_config(out: Path, config: dict[str, Any], extra: Optional[dict] = None) -> None:
    """For scripts that resume by key: the output file is tied to one configuration, recorded beside it. Records from any other, or
    with none recorded, raise instead of being counted as this run's."""
    sidecar = out.with_name(f"{out.name}.config.json")
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if saved["config_id"] != config_id(config):
            differ = sorted(k for k in saved["config"].keys() | config.keys() if saved["config"].get(k) != config.get(k))
            raise ValueError(f"{out} holds records of another configuration (differs in {differ}): use another output file")
        return
    if out.exists():
        raise ValueError(f"{out} has records and no {sidecar.name} to say what they ran: use another output file")
    with durable.atomic_file(sidecar) as f:
        f.write(json.dumps({"config_id": config_id(config), "config": config, **(extra or {})}, indent=1, default=str).encode())


def unique_records(records: list[dict], key, protocol=(), allow_duplicates: bool = False) -> list[dict]:
    """One record per ``key(record)``. Two records with one key are an ambiguity, not a choice: this raises, naming them, unless
    ``allow_duplicates`` keeps the last and says how many it dropped. Fields in ``protocol`` must agree across the whole list."""
    for field in protocol:
        values = {json.dumps(r.get(field), sort_keys=True, default=str) for r in records}
        if len(values) > 1:
            raise ValueError(f"records mix different {field}: {sorted(values)[:5]}")
    out: dict = {}
    dropped = 0
    for r in records:
        k = key(r)
        if k in out:
            if not allow_duplicates:
                raise ValueError(f"two records for {k}: pass the allow-duplicates option to keep the last, or remove one")
            dropped += 1
            durable.progress(f"unique_records: {k}: kept the later record (config {r.get('config_id')}), replaced one (config {out[k].get('config_id')})")
        out[k] = r
    if dropped:
        durable.progress(f"unique_records: kept the last of {dropped} duplicate records")
    return list(out.values())
