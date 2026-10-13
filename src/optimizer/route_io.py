"""Convert independent stage-one route batches into CToc trajectory seeds.

No stage-one Python module is imported.  The exchange boundary is JSON plus
the original CTOC14B scene checksum.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import math
from pathlib import Path
from typing import Iterator


KNOWN_V4_NPZ_SHA256 = "f76b133b41f0e5d9ded832c25b0bf5bdcd828b47e14597afc5a4716538e09350"
KNOWN_V5_NPZ_SHA256 = KNOWN_V4_NPZ_SHA256
KNOWN_STATE_SHA256 = 'cbb256397a2bbb3af7cee89fb173ce1a505918c51790f48e47ea087155861054'


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap(item: dict) -> dict:
    return item.get("route", item)


def _items(path: Path) -> Iterator[tuple[str, dict]]:
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield path.name, _unwrap(json.loads(line))
        return
    data = _load(path)
    if path.name.startswith("generation_"):
        for island in data.get("populations", []):
            for item in island:
                yield path.name, _unwrap(item)
    else:
        for item in data.get("routes", []):
            yield path.name, _unwrap(item)


def discover(batch: Path) -> Iterator[tuple[str, dict]]:
    """Accept v4/v5 batches, committed completed v6 runs, or one seed file."""
    if batch.is_file():
        if batch.name.endswith(".jsonl.gz"):
            yield from _items(batch)
            return
        data = _load(batch)
        if "node_ids" in data or "trajectory" in data:
            yield batch.name, data.get("trajectory", data)
        else:
            yield from _items(batch)
        return
    if (batch / "routes").is_dir():
        # v6 saves ALL candidates in committed shards, not just best_routes.json.
        receipts = sorted((batch / "routes").glob("*.receipt.json"))
        if not receipts:
            raise ValueError("v6 routes directory has no committed shard receipts")
        validated = []
        for receipt in receipts:
            committed = _load(receipt)
            name = committed.get("file", "")
            if (not isinstance(name, str) or "/" in name or "\\" in name
                    or not name.startswith("run_") or not name.endswith(".jsonl.gz")):
                raise ValueError(f"Invalid v6 shard name: {receipt}")
            path = batch / "routes" / name
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != committed.get("sha256"):
                raise ValueError(f"v6 shard checksum mismatch: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                count = 0
                for line in handle:
                    if line.strip():
                        if not isinstance(json.loads(line), dict):
                            raise ValueError(f"Invalid v6 route record: {path}")
                        count += 1
            if type(committed.get("count")) is not int or count != committed["count"]:
                raise ValueError(f"v6 shard count mismatch: {path}")
            validated.append(path)
        # Validate the entire batch before yielding: a bad later shard must not
        # leave half-imported IDs in the coordinator's in-memory deduplication set.
        for path in validated:
            yield from _items(path)
        return
    if (batch / "routes_under15").is_dir():
        for path in sorted((batch / "routes_under15").glob("run_*.jsonl.gz")):
            yield from _items(path)
        # v5 intentionally saves only a small representative sample above 15.
        above = batch / "over15_sample.json"
        if above.is_file():
            yield from _items(above)
        return
    sources = [
        batch / "population" / "beam_seed_routes.json",
        batch / "archives" / "global_archive.json",
        batch / "best_routes.json",
        batch / "integrated" / "candidate_pool_200.json",
    ]
    sources += sorted((batch / "checkpoints").glob("generation_*.json"))
    for path in sources:
        if path.is_file():
            yield from _items(path)


def check_source(batch: Path, problem) -> None:
    if batch.is_file():
        return
    metadata = batch / "run_metadata.json"
    beam_batch = (batch / "routes_under15").is_dir() or (batch / "routes").is_dir()
    if beam_batch and not metadata.is_file():
        raise ValueError("Beam run is missing run_metadata.json")
    if metadata.exists():
        run_meta = _load(metadata)
        if beam_batch and run_meta.get("status") not in ("completed", "stopped"):
            raise ValueError("Beam run is still writing or failed; wait for completed/stopped metadata")
        if (batch / "routes").is_dir() and run_meta.get("schema") != "ctoc14b.stage1.v6":
            raise ValueError("Unrecognized routes-directory schema")
        value = run_meta.get("data_sha256")
        source = run_meta.get("configuration", {}).get("TARGET_COUNT")
        if source is not None and source != 35:
            raise ValueError("Stage-one batch does not contain 35 targets")
        accepted = (KNOWN_V4_NPZ_SHA256, problem.manifest['files']['states.npy'])
        if not isinstance(value,str) or value.lower() not in accepted:
            raise ValueError('Search target data do not match the optimizer ephemeris')



def convert(item: dict, problem) -> tuple[str, dict, float]:
    """Preserve order, epochs, initial state and every Lambert branch exactly."""
    from ctoc.kernels.physics import target_state
    from ctoc.trajectory import normalize

    if item.get("schema") == "ctoc14b.route.v6":
        expected = dict(units="km,s,km/s", frame="ICRF",
                        epoch_utc="2035-01-01T12:00:00Z", mission_end_s=864000)
        if item.get("data_sha256") not in (KNOWN_V5_NPZ_SHA256, KNOWN_STATE_SHA256):
            raise ValueError("Unknown target ephemeris fingerprint")
        for field, value in expected.items():
            if item.get(field) != value:
                raise ValueError(f"v6 exchange mismatch: {field}")

    if "node_ids" in item:
        d = normalize(item)
        if d["model"] not in ("two_body", "j2"):
            raise ValueError("Stage-two intake expects a two-body or J2 seed")
        source_cost = float(item.get("cost", math.nan))
    else:
        order = [int(x) for x in item["order"]]
        times = [float(x) for x in item["times_s"]]
        branches = [[int(x) for x in row] for row in item["branches"]]
        initial = [float(x) for x in item["initial_state_km_km_s"]]
        source_cost = float(item["cost_km_s"])
        if (len(order) != 35 or set(order) != set(range(35)) or len(times) != 35
                or len(branches) != 35 or len(initial) != 6):
            raise ValueError("Incomplete or duplicated target route")
        if (not all(math.isfinite(x) for x in times + initial)
                or not math.isfinite(source_cost) or times[0] <= 0
                or times[-1] > 864000
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("Invalid epochs or initial state")
        positions = [initial[:3]] + [
            target_state(problem.states, k, t)[0].tolist()
            for k, t in zip(order, times)
        ]
        d = normalize(dict(
            initial=initial, positions=positions, times=[0.0, *times],
            node_ids=[-1, *order], branches=branches, model="two_body", frame="ICRF",
        ))
    identity_fields = ("initial", "positions", "times", "node_ids", "branches")
    # Keep every existing two-body seed ID stable. J2 dynamics must not collide
    # with the same geometry interpreted as a two-body trajectory.
    if d["model"] == "j2":
        identity_fields += ("model",)
    key = canonical_hash({name: d[name] for name in identity_fields})
    return key, d, source_cost

