"""Small FiQA run through the real benchmark; results stay under runs/.

    python smoke.py --check
    python smoke.py
    python smoke.py --models minilm-l6
    python smoke.py --run-dir runs/my-smoke --models jev jev-relevance voyage-rerank-3
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import uuid

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env", override=False)
MODELS = ["minilm-l6", "jev", "jev-relevance", "voyage-rerank-3"]
KEYS = {"jev": "TYPESAFE_API_KEY", "jev-relevance": "TYPESAFE_API_KEY", "voyage-rerank-3": "VOYAGE_API_KEY"}


def preflight(models):
    """Check dependencies and credentials without printing any secret values."""
    from lancedb.rerankers import CrossEncoderReranker, TypeSafeReranker, VoyageAIReranker

    if "batch_size" not in inspect.signature(CrossEncoderReranker).parameters:
        raise RuntimeError("Install requirements.txt: this LanceDB build lacks the benchmark's cross-encoder options")
    if "voyage-rerank-3" in models:
        import voyageai  # noqa: F401
    missing = []
    for key in ("VOYAGE_API_KEY", "TYPESAFE_API_KEY", "NVIDIA_API_KEY"):
        present = bool(os.environ.get(key, "").strip())
        print(f"{key}: {'set' if present else 'missing'}")
        if key in {KEYS.get(m) for m in models} and not present:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Add {', '.join(missing)} to {HERE / '.env'}")


def run_directory(path, queries, documents):
    """Only resume directories created by this smoke runner with the same sample."""
    config = {"kind": "reranker-smoke-v1", "dataset": "fiqa", "queries": queries, "documents": documents, "seed": 0}
    marker = path / "smoke.json"
    if path.exists():
        if not marker.exists() or json.loads(marker.read_text()) != config:
            raise ValueError("Run directory is not a matching smoke run; choose a new directory")
    else:
        path.mkdir(parents=True)
        marker.write_text(json.dumps(config, indent=2) + "\n")
    return path


def validate_scores(rows, records):
    if len(records) != len(rows):
        raise ValueError(f"Incomplete scores: {len(records)}/{len(rows)} queries")
    for i, (row, record) in enumerate(zip(rows, records)):
        if record["i"] != i or len(record["scores"]) != len(row["docs"]):
            raise ValueError(f"Query {i}: score alignment mismatch")
        if not all(math.isfinite(s) for s in record["scores"]):
            raise ValueError(f"Query {i}: non-finite scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Check installation and key presence; no API requests")
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless an NVIDIA GPU is available")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--documents", type=int, default=500)
    parser.add_argument("--run-dir", type=Path, help="New directory or matching smoke run to resume")
    args = parser.parse_args()
    if not 1 <= args.queries <= 20 or not 50 <= args.documents <= 5000:
        parser.error("Smoke runs allow 1–20 queries and 50–5,000 documents")
    preflight(args.models)
    import torch
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This run requires an NVIDIA GPU; execute it on the Brev instance")
    if args.check:
        print("Benchmark dependencies and selected model keys are ready. No live requests made.")
        return

    import bench

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.run_dir or HERE / "runs" / f"smoke-{stamp}-{uuid.uuid4().hex[:6]}"
    if not path.is_absolute():
        path = HERE / path
    path = run_directory(path.resolve(), args.queries, args.documents)
    bench.ROOT = path / "cache"
    bench.RESULTS = path / "results"
    root = bench.ROOT / "fiqa"
    print(f"Smoke run: {path}", flush=True)
    print("Small sampled corpus: metrics validate the pipeline, not benchmark quality.", flush=True)
    if not (root / "candidates.jsonl").exists():
        bench.prepare("fiqa", query_limit=args.queries, corpus_limit=args.documents)
    rows = bench.read_jsonl(root / "candidates.jsonl")
    versions = {name: importlib.metadata.version(name) for name in
                ("lancedb", "torch", "sentence-transformers", "transformers", "typesafe-sdk")}
    hardware = {"cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (path / "environment.json").write_text(json.dumps({"packages": versions, "hardware": hardware}, indent=2) + "\n")
    failures = []
    for model in dict.fromkeys(args.models):
        print(f"Scoring {model}: {len(rows)} queries", flush=True)
        try:
            bench.score("fiqa", model)
            validate_scores(rows, bench.score_records(root / "scores" / f"{model}.jsonl", rows))
            print(f"PASS: {model}", flush=True)
        except Exception as exc:
            failures.append(model)
            print(f"FAIL: {model}: {safe_error(exc)}", flush=True)
    bench.report()
    report = bench.RESULTS / "README.md"
    report.write_text("# Smoke test — sampled FiQA corpus\n\n"
                      "Pipeline check only; these metrics are not comparable to the full benchmark.\n\n"
                      + report.read_text())
    status = {"requested_models": args.models, "failed_models": failures,
              "queries": len(rows), "candidate_pairs_per_model": sum(len(r["docs"]) for r in rows),
              "hardware": hardware}
    (path / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Results: {bench.RESULTS / 'README.md'}", flush=True)
    if failures:
        raise RuntimeError(f"Failed models: {', '.join(failures)}")


def safe_error(exc):
    message = f"{type(exc).__name__}: {exc}"
    for key in ("VOYAGE_API_KEY", "TYPESAFE_API_KEY", "NVIDIA_API_KEY", "BREV_TOKEN", "BREV_API_KEY"):
        secret = os.environ.get(key)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
