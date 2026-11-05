"""Crash-safe files for long runs: a finished write is synced to disk before it replaces the old one, and a partial write is never
mistaken for a finished one. After any crash, rerunning the same command continues the work. Tested with real process kills; a real
power loss was not tested and rests on ``fsync``. This is crash recovery, not concurrent publication: each file is replaced atomically,
but a set of files is not, so use one writer per artifact."""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROGRESS_ENV = "SPACEBRIDGE_PROGRESS"  # path of a line-per-event progress log that survives a power loss


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def temp_name(path: Path) -> Path:
    """A temporary name beside ``path``, unique to this write: two threads saving one destination don't share a file."""
    return path.with_name(f"{path.name}.tmp{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")


@contextmanager
def atomic_file(path: str | Path):
    """Writes to a temporary file, then renames it over ``path`` only after the data is on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_name(path)
    try:
        with tmp.open("w+b") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def save_npy(path: str | Path, array: np.ndarray) -> None:
    with atomic_file(path) as f:
        np.save(f, array)


def append_bytes(path: str | Path, data: bytes) -> None:
    """Appends and syncs; a torn tail after a power loss is the reader's to discard (fixed-size records, JSON lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if new:
        fsync_dir(path.parent)


def append_lines(path: str | Path, lines: list[str]) -> None:
    append_bytes(path, "".join(line + "\n" for line in lines).encode())


def read_jsonl(path: str | Path) -> tuple[list[dict], bool]:
    """The records of a JSON-lines file and whether every line parsed (a crash can tear the last one)."""
    records, clean = [], True
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                clean = False
    return records, clean


def progress(message: str) -> None:
    """Prints a progress event and, if ``SPACEBRIDGE_PROGRESS`` is set, appends it to that file on disk."""
    line = f"{time.strftime('%F %T')} {message}"
    print(line, flush=True)
    path = os.environ.get(PROGRESS_ENV)
    if path:
        append_lines(path, [line])
