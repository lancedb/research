"""Durable, validated checkpoints for the local benchmark (no network dependencies)."""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def bind_manifest(path, config):
    """Refuse to mix candidates, revisions or settings from different experiments."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Checkpoint configuration changed: {path}; use a new run directory")
    else:
        atomic_json(path, config)
    return digest(config)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process owns this run: {path}") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def packages():
    names = ("lancedb", "torch", "sentence-transformers", "transformers", "datasets",
             "typesafe-sdk", "voyageai", "pylate", "numpy", "pyarrow", "scipy")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def read_journal(path, repair=False):
    """Only a torn final write may be repaired; malformed completed records fail loudly."""
    if not path.exists():
        return []
    rows = []
    with open(path, "rb+" if repair else "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair:
                    raise ValueError(f"Incomplete checkpoint write: {path}")
                f.seek(offset)
                f.truncate()
                break
            rows.append(json.loads(line))
    return rows


def append_record(f, record):
    f.write(json.dumps(record, allow_nan=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


def validate_record(rows, record):
    i = record.get("i")
    if type(i) is not int or not 0 <= i < len(rows):
        raise ValueError("Score query index is out of range")
    scores = record.get("scores", [])
    if len(scores) != len(rows[i]["docs"]) or not all(math.isfinite(s) for s in scores):
        raise ValueError(f"Invalid scores for query {i}")
    for key in ("seconds", "elapsed_seconds"):
        if key in record and (not math.isfinite(record[key]) or record[key] < 0):
            raise ValueError(f"Invalid {key} for query {i}")


def score_records(path, rows, repair=False):
    records = read_journal(path, repair)
    seen = set()
    for r in records:
        validate_record(rows, r)
        if r["i"] in seen:
            raise ValueError(f"Duplicate query {r['i']} in {path}")
        seen.add(r["i"])
    if repair:
        read_journal(path.with_suffix(".sessions.jsonl"), repair=True)
    return sorted(records, key=lambda r: r["i"])


def execute_scores(out, rows, one, workers=1):
    """Bound requests in flight; persist successes even when another request fails.

    A killed process can have billed requests without a saved response. Tokens here
    are the sum of persisted responses, not a replacement for the provider invoice.
    """
    records = score_records(out, rows, repair=True)
    done = {r["i"] for r in records}
    pending = iter(i for i in range(len(rows)) if i not in done)
    if len(done) == len(rows):
        return records
    out.parent.mkdir(parents=True, exist_ok=True)
    started, session, error = time.perf_counter(), uuid.uuid4().hex, None
    sessions = out.with_suffix(".sessions.jsonl")
    count = 0
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        def submit():
            i = next(pending, None)
            if i is not None:
                futures[pool.submit(one, i)] = i
        for _ in range(workers):
            submit()
        try:
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    i = futures.pop(future)
                    try:
                        record = future.result()
                        if record.get("i") != i:
                            raise ValueError("Worker returned the wrong query")
                        validate_record(rows, record)
                        record.update(session=session, session_elapsed=time.perf_counter() - started)
                        append_record(f, record)
                        records.append(record)
                        count += 1
                    except Exception as exc:
                        error = error or exc
                if error is None:
                    for _ in finished:
                        submit()
        finally:
            with open(sessions, "a") as sf:
                append_record(sf, {"session": session, "seconds": time.perf_counter() - started,
                                   "completed": count, "failed": error is not None})
    if error is not None:
        raise error
    return sorted(records, key=lambda r: r["i"])


def timing_summary(out, records):
    # The latest saved response supplies a lower bound after a hard process kill.
    elapsed = {}
    for r in records:
        if "session" in r:
            elapsed[r["session"]] = max(elapsed.get(r["session"], 0), r["session_elapsed"])
    finished = read_journal(out.with_suffix(".sessions.jsonl"), repair=False)
    for s in finished:
        elapsed[s["session"]] = s["seconds"]
    return {"wall_seconds": sum(elapsed.values()),
            "interrupted_sessions": len(set(elapsed) - {s["session"] for s in finished}),
            "retries": sum(r.get("retries", 0) for r in records),
            "tokens": sum(r.get("tokens", 0) for r in records)}
