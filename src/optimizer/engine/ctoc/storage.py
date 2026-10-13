"""Checksummed, atomic JSON storage. Only the coordinator owns shared indices."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def atomic_json(path, value, backup=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    if backup and path.exists():
        # Preserve a known parseable previous snapshot, never a corrupt one.
        try:
            previous = read_json(path)
            atomic_json(path.with_suffix(path.suffix + ".bak"), previous)
        except (ValueError, OSError):
            pass
    # Windows readers (status monitors, virus scanners) may briefly deny delete
    # sharing. Keep the complete temporary file and retry the atomic replacement.
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.5, 0.005 * 2**attempt))
    if os.name == "posix":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def checkpoint(path, value):
    atomic_json(
        path,
        {"sha256": hashlib.sha256(canonical_bytes(value)).hexdigest(), "payload": value},
        backup=True,
    )


def restore(path):
    path = Path(path)
    errors = []
    for candidate in (path, path.with_suffix(path.suffix + ".bak")):
        try:
            obj = read_json(candidate)
            if hashlib.sha256(canonical_bytes(obj["payload"])).hexdigest() != obj["sha256"]:
                raise ValueError("Checkpoint digest mismatch")
            return obj["payload"]
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
    raise ValueError("No intact checkpoint: " + "; ".join(errors))


def append_event(path, event):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(canonical_bytes(event).decode("utf-8") + "\n")
        f.flush()


def source_digest():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()
