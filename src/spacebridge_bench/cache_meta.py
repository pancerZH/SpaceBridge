"""Provenance of cache entries written from now on, and a record of the ones that have none.

Cache files are named by a hash of their inputs, and the hash leaves out the model revision, the device and the precision. Renaming the
existing 234 GB of caches would orphan them, so they stay where they are, as legacy entries of unknown provenance. New entries get:

- **a namespace by device class:** CPU runs read and write ``<name>@cpu-fp32``, so they never reuse or extend what a GPU run in ``bf16``
  wrote, and the GPU keeps the original names, which every historical file has;
- **a sidecar** ``<file>.meta.json`` with the model, its revision, the device class and the library versions. A file whose sidecar
  disagrees with the current run raises instead of being reused.

``summary()`` says how many legacy and validated entries a run read, for its manifest.
"""

import json
import time
from pathlib import Path
from typing import Optional

from . import durable

CPU_SUFFIX = "@cpu-fp32"
_USE = {"legacy": set(), "validated": set(), "written": set()}


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
    with durable.atomic_file(sidecar(path)) as f:
        f.write(json.dumps({**meta, "written": time.strftime("%F %T")}, indent=1).encode())
    _USE["written"].add(str(path))


def check(path: Path, expected: dict) -> None:
    """Raises if ``path``'s sidecar says it was made by another model, revision or device class than ``expected``. A file without one
    is a legacy entry: allowed, and counted."""
    meta_path = sidecar(path)
    if not meta_path.exists():
        _USE["legacy"].add(str(path))
        return
    meta = json.loads(meta_path.read_text())
    differ = [k for k in ("model", "revision", "device_class") if meta.get(k) is not None and expected.get(k) is not None and meta[k] != expected[k]]
    if differ:
        detail = "; ".join(f"{k}: {meta[k]!r} in the cache, {expected[k]!r} now" for k in differ)
        raise ValueError(f"{path} was made differently ({detail}). Move it aside to recompute it, or use the matching model and device")
    _USE["validated"].add(str(path))


def summary() -> dict:
    return {k: len(v) for k, v in _USE.items()}
