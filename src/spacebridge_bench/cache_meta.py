"""Provenance of cache entries written from now on, and a record of the ones that have none.

Cache files are named by a hash of their inputs, and the hash leaves out the model revision, the device and the precision. Renaming the
existing 234 GB of caches would orphan them, so they stay where they are, as legacy entries of unknown provenance. New entries get:

- **a namespace by device class:** CPU runs read and write ``<name>@cpu-fp32``, so they never reuse or extend what a GPU run in ``bf16``
  wrote, and the GPU keeps the original names, which every historical file has;
- **a sidecar** ``<file>.meta.json`` with the model, its revision, the device class and the library versions. A file whose sidecar
  disagrees with the current run raises instead of being reused. The caches write it once the model is loaded, before any entry, so a
  first download records the revision it fetched.

A revision can still be unknown: in a sidecar, if a load left no local reference, or for the current run, if the model isn't in the local
Hugging Face cache. Such a file is never counted as validated: it is reused as ``unverified``, with a warning, and, like a legacy file, it
can still be extended.

The revision is the local reference read just after the load: transformers 5 doesn't record the commit on a loaded model. Another
process moving that reference in the instant between the two would go unnoticed.

``summary()`` says how many legacy, validated and unverified entries a run read, and how many it wrote, for its manifest.
"""

import json
import time
import warnings
from pathlib import Path
from typing import Optional

from . import durable

CPU_SUFFIX = "@cpu-fp32"
FIELDS = ("model", "revision", "device_class")
_USE = {"legacy": set(), "validated": set(), "unverified": set(), "written": set()}


def device_class(device: str, precision: Optional[str] = None) -> str:
    """``cuda-bf16`` or ``cpu-fp32`` by default; ``precision`` overrides the second part (an fp32 GPU reranker, an int8 export)."""
    on_gpu = str(device).startswith("cuda")
    return f"{'cuda' if on_gpu else 'cpu'}-{precision or ('bf16' if on_gpu else 'fp32')}"


def namespace(name: str, device: str) -> str:
    """The cache name for this device: unchanged on the GPU, where every historical entry was written, and ``name@cpu-fp32`` on a CPU."""
    return name if str(device).startswith("cuda") else f"{name}{CPU_SUFFIX}"


def sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.meta.json")


def describe(model: str, revision: Optional[str], device_cls: str, **extra) -> dict:
    import torch

    return {"model": model, "revision": revision, "device_class": device_cls, "torch": torch.__version__, **extra}


def write(path: Path, meta: dict) -> None:
    """Writes ``path``'s sidecar. Callers load the model first, so ``meta`` names the revision that will make the entries."""
    with durable.atomic_file(sidecar(path)) as f:
        f.write(json.dumps({**meta, "written": time.strftime("%F %T")}, indent=1).encode())
    _USE["written"].add(str(path))
    unknown = [k for k in FIELDS if meta.get(k) is None]
    if unknown:
        warnings.warn(f"{path}: its {', '.join(unknown)} is unknown even with the model loaded, so later runs can't verify it. Its entries count as unverified")
        _mark(path, "unverified")


def _mark(path: Path, status: str) -> None:
    for s in ("validated", "unverified"):
        _USE[s].discard(str(path))
    _USE[status].add(str(path))


def check(path: Path, expected: dict) -> str:
    """Compares ``path``'s sidecar with ``expected``, the current run's, and returns what it found:

    - ``"legacy"``: no sidecar. Allowed, and counted.
    - ``"validated"``: the model, revision and device class are known on both sides and match.
    - ``"unverified"``: what is known on both sides matches, but a revision is unknown on one side, so the two can't be compared. Allowed,
      counted apart, and warned about. It is never counted as validated.

    Raises if a field known on both sides differs."""
    meta_path = sidecar(path)
    if not meta_path.exists():
        _USE["legacy"].add(str(path))
        return "legacy"
    meta = json.loads(meta_path.read_text())
    differ = [k for k in FIELDS if meta.get(k) is not None and expected.get(k) is not None and meta[k] != expected[k]]
    if differ:
        detail = "; ".join(f"{k}: {meta[k]!r} in the cache, {expected[k]!r} now" for k in differ)
        raise ValueError(f"{path} was made differently ({detail}). Move it aside to recompute it, or use the matching model and device")
    unknown = [f"{k} {'in its sidecar' if meta.get(k) is None else 'for this run'}" for k in FIELDS if meta.get(k) is None or expected.get(k) is None]
    if unknown:
        if str(path) not in _USE["unverified"]:
            warnings.warn(f"{path}: {', '.join(unknown)} is unknown, so it can't be compared. Reused unverified")
        _mark(path, "unverified")
        return "unverified"
    _mark(path, "validated")
    return "validated"


def summary() -> dict:
    return {k: len(v) for k, v in _USE.items()}
