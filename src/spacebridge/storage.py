"""On-disk format of a bundle or a fitted bridge: a directory with arrays, metadata and a manifest (design §6).

The manifest is written last and holds a version and checksums, so a write that a crash cut short is never mistaken
for a finished one: it fails to load, and saving again repairs it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import durable

FORMAT_VERSION = 1
ARRAYS, META, MANIFEST = "arrays.npz", "meta.json", "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_dir(path: str | Path, kind: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path = Path(path)
    (path / MANIFEST).unlink(missing_ok=True)
    with durable.atomic_file(path / ARRAYS) as f:
        np.savez(f, **arrays)
    with durable.atomic_file(path / META) as f:
        f.write(json.dumps(meta, ensure_ascii=False, default=_json_default).encode())
    manifest = {"kind": kind, "format_version": FORMAT_VERSION, "sha256": {n: _sha256(path / n) for n in (ARRAYS, META)}}
    with durable.atomic_file(path / MANIFEST) as f:
        f.write(json.dumps(manifest, indent=1).encode())


def load_dir(path: str | Path, kind: str) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    if not (path / MANIFEST).exists():
        raise ValueError(f"{path} has no {MANIFEST}: not a {kind}, or its save did not finish")
    manifest = json.loads((path / MANIFEST).read_text())
    if manifest["kind"] != kind:
        raise ValueError(f"{path} holds a {manifest['kind']}, not a {kind}")
    if manifest["format_version"] > FORMAT_VERSION:
        raise ValueError(f"{path} has format {manifest['format_version']}; this version reads up to {FORMAT_VERSION}")
    for name, digest in manifest["sha256"].items():
        if _sha256(path / name) != digest:
            raise ValueError(f"{path / name} does not match its checksum")
    with np.load(path / ARRAYS) as z:
        arrays = {name: z[name] for name in z.files}
    return arrays, json.loads((path / META).read_text())
