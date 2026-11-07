"""Run a resumable pilot against full corpora, isolated from published results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import bench
from run_state import atomic_json, bind_manifest, packages, score_records
from smoke import safe_error

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = [*bench.CROSS_ENCODERS, *bench.LISTWISE, *bench.COLBERTS,
                  *bench.JEV_PROMPTS, "voyage-rerank-3"]


def selected_models(models, dataset):
    return [m for m in models if m != "jev-multihop" or dataset == "hotpotqa"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", choices=bench.DATASETS, default=bench.DATASETS)
    parser.add_argument("--models", nargs="+", choices=[*DEFAULT_MODELS, "voyage-rerank-3-lite"], default=DEFAULT_MODELS)
    parser.add_argument("--stage", choices=["all", "prepare", "gpu", "api", "report"], default="all")
    parser.add_argument("--colbert-python", type=Path, default=HERE / ".venv-colbert/bin/python")
    args = parser.parse_args()
    if not 1 <= args.queries <= 100:
        parser.error("Pilot runs allow 1–100 queries per dataset")
    if args.stage in {"all", "prepare", "gpu"} and not bench.torch.cuda.is_available():
        parser.error("Preparation and open models require an NVIDIA GPU")
    run = args.run_dir.resolve()
    config = {"kind": "full-corpus-pilot-v2", "queries": args.queries, "seed": 0,
              "datasets": args.datasets, "models": args.models, "revisions": bench.REVISIONS}
    if run.exists() and not (run / "pilot.json").exists():
        parser.error("Choose a new pilot directory, or resume one with pilot.json")
    bind_manifest(run / "pilot.json", config)
    env = dict(os.environ, BENCH_ROOT=str(run / "cache"), BENCH_RESULTS=str(run / "results"))
    bench.ROOT, bench.RESULTS = run / "cache", run / "results"
    atomic_json(run / f"environment-{args.stage}.json", {
        "packages": packages(), "gpu": bench.torch.cuda.get_device_name(0) if bench.torch.cuda.is_available() else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version, "api_workers": bench.API_WORKERS, "voyage_tpm": bench.VOYAGE_TPM})
    failures = []
    tasks = []
    if args.stage == "all":
        # Exercise every architecture on a small full corpus before the expensive corpora.
        for d in (d for d in ["fiqa", "scidocs", "gooaq", "nq", "hotpotqa"] if d in args.datasets):
            tasks.append((sys.executable, ["prepare", "--dataset", d, "--queries", str(args.queries)]))
            for model in selected_models(args.models, d):
                python = str(args.colbert_python.resolve()) if model in bench.COLBERTS else sys.executable
                tasks.append((python, ["score", "--dataset", d, "--model", model]))
    if args.stage == "prepare":
        tasks += [(sys.executable, ["prepare", "--dataset", d, "--queries", str(args.queries)]) for d in args.datasets]
    if args.stage in {"gpu", "api"}:
        for model in args.models:
            api = model in bench.JEV_PROMPTS or model in bench.VOYAGE
            if args.stage == "gpu" and api or args.stage == "api" and not api:
                continue
            python = str(args.colbert_python.resolve()) if model in bench.COLBERTS else sys.executable
            for d in args.datasets:
                if model in selected_models(args.models, d):
                    tasks.append((python, ["score", "--dataset", d, "--model", model]))
    for python, command in tasks:
        if command[0] == "prepare" and failures:
            break
        label = " ".join(command)
        print(label, flush=True)
        started = time.time()
        try:
            result = subprocess.run([python, str(HERE / "bench.py"), *command], env=env)
            if result.returncode:
                raise RuntimeError(f"exit {result.returncode}")
        except Exception as exc:
            failures.append({"task": label, "error": safe_error(exc)})
            print(f"FAIL: {label}: {safe_error(exc)}", flush=True)
        atomic_json(run / "progress.json", {"last_task": label, "seconds": time.time() - started, "failures": failures})
        if command[0] == "prepare" and failures:
            break
    if args.stage in {"all", "report"}:
        bench.report()
        report = bench.RESULTS / "README.md"
        report.write_text(f"# Full-corpus pilot — {args.queries} queries per dataset\n\n"
                          "Functional and runtime validation; this sample is too small to rank model quality.\n\n"
                          + report.read_text())
        coverage = {}
        for d in args.datasets:
            root = bench.ROOT / d
            rows = bench.read_jsonl(root / "candidates.jsonl") if (root / "candidates.jsonl").exists() else []
            coverage[d] = {}
            for model in selected_models(args.models, d):
                variants = [f"{model}-pool{f}" for f in bench.POOL_FACTORS] if model in bench.COLBERTS else [model]
                for variant in variants:
                    records = score_records(root / "scores" / f"{variant}.jsonl", rows)
                    coverage[d][variant] = len(records)
                    if len(rows) != args.queries or len(records) != len(rows):
                        failures.append({"task": f"{d}/{variant}", "error": "incomplete"})
        atomic_json(run / "status.json", {"queries_per_dataset": args.queries, "coverage": coverage, "failures": failures})
    if failures:
        raise RuntimeError(f"{len(failures)} pilot tasks failed; see {run}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
