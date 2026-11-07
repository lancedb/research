"""Run the complete benchmark with independent jobs on eight H100s.

Each model still uses one GPU. Jobs share frozen candidates, never score files.
API providers each have one sequential job queue, preserving their rate limits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import bench
from pilot import DEFAULT_MODELS, python_executable, selected_models
from run_state import append_record, atomic_json, bind_manifest, exclusive, packages, score_records
from smoke import safe_error

HERE = Path(__file__).resolve().parent
QUERY_COUNTS = {"gooaq": 20000, "nq": 3452, "hotpotqa": 7405, "fiqa": 648, "scidocs": 1000}
GPU_MODELS = [*bench.CROSS_ENCODERS, *bench.LISTWISE, *bench.COLBERTS]


def cpu_assignments(cpus, gpu_count, per_gpu, api_cpus=2, gpu_sets=None):
    required = gpu_count * per_gpu + 2 * api_cpus
    if len(cpus) < required:
        raise ValueError(f"Need {required} available CPUs, found {len(cpus)}")
    gpu = gpu_sets or [cpus[i * per_gpu:(i + 1) * per_gpu] for i in range(gpu_count)]
    assigned = [c for group in gpu for c in group]
    if (len(gpu) != gpu_count or any(len(group) != per_gpu for group in gpu)
            or len(set(assigned)) != len(assigned) or not set(assigned).issubset(cpus)):
        raise ValueError("GPU CPU sets must be available, disjoint, and match --cpus-per-gpu")
    remaining = [c for c in cpus if c not in assigned]
    api = [remaining[i * api_cpus:(i + 1) * api_cpus] for i in range(2)]
    return gpu, api


def parse_cpu_set(value):
    cpus = []
    for part in value.split(","):
        if "-" in part:
            first, last = map(int, part.split("-"))
            if first > last:
                raise ValueError("CPU range must be ascending")
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(part))
    return cpus


def estimated_seconds(dataset, model):
    """Start long jobs first; these pilot estimates affect scheduling, not scores."""
    pilot = json.loads((HERE / "pilot-results/2026-11-06/summary.json").read_text())
    variants = [f"{model}-pool{p}" for p in bench.POOL_FACTORS] if model in bench.COLBERTS else [model]
    return QUERY_COUNTS[dataset] * sum(
        pilot[dataset][v]["timing"]["wall_seconds"] / 10 for v in variants)


def gpu_jobs():
    jobs = [(d, m) for d in bench.DATASETS for m in GPU_MODELS]
    return sorted(jobs, key=lambda job: estimated_seconds(*job), reverse=True)


def coverage(run):
    """Require every full query set and every original configuration plus Voyage."""
    counts, incomplete, candidates = {}, [], {}
    for dataset, expected in QUERY_COUNTS.items():
        root = run / "cache" / dataset
        rows = bench.read_jsonl(root / "candidates.jsonl")
        candidates[dataset] = bench.candidate_fingerprint(root)
        if len(rows) != expected or len({r["qid"] for r in rows}) != expected:
            raise ValueError(f"Wrong full query set for {dataset}: expected {expected} unique queries")
        if json.loads((root / "prepare.json").read_text())["query_limit"] is not None:
            raise ValueError(f"{dataset} was prepared as a limited sample")
        counts[dataset] = {}
        for model in selected_models(DEFAULT_MODELS, dataset):
            variants = [f"{model}-pool{p}" for p in bench.POOL_FACTORS] if model in bench.COLBERTS else [model]
            for variant in variants:
                path = root / "scores" / f"{variant}.jsonl"
                records = score_records(path, rows)
                counts[dataset][variant] = len(records)
                if len(records) != expected:
                    incomplete.append(f"{dataset}/{variant}")
                elif json.loads(path.with_suffix(".meta.json").read_text())["candidates"] != candidates[dataset]:
                    raise ValueError(f"Candidate mismatch for {dataset}/{variant}")
    status = {"query_counts": QUERY_COUNTS, "coverage": counts, "incomplete": incomplete,
              "candidates": candidates, "score_records": sum(sum(c.values()) for c in counts.values())}
    atomic_json(run / "status.json", status)
    if incomplete:
        raise ValueError(f"{len(incomplete)} configurations are incomplete; see status.json")
    return status


class Runner:
    def __init__(self, run, colbert_python):
        self.run, self.colbert_python = run, colbert_python
        self.lock = threading.Lock()
        self.failures = []

    def job(self, dataset, model, gpu, cpus):
        stage = "prepare" if model is None else "score"
        label = f"{stage}-{dataset}" + (f"-{model}" if model else "")
        command = [stage, "--dataset", dataset]
        if model:
            command += ["--model", model]
        python = python_executable(self.colbert_python) if model in bench.COLBERTS else sys.executable
        env = dict(os.environ, BENCH_ROOT=str(self.run / "cache"), BENCH_RESULTS=str(self.run / "results"),
                   CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else "",
                   OMP_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)),
                   RAYON_NUM_THREADS=str(len(cpus)), TOKENIZERS_PARALLELISM="false")
        started, error = time.time(), None
        record = {"task": label, "gpu": gpu, "cpus": cpus, "started": started, "command": command}
        atomic_json(self.run / "jobs" / f"{label}.json", dict(record, state="running"))
        print(f"START {label} GPU={gpu} CPUs={cpus}", flush=True)
        try:
            with open(self.run / "logs" / f"{label}.log", "a") as log:
                result = subprocess.run(["taskset", "-c", ",".join(map(str, cpus)), python,
                                         str(HERE / "bench.py"), *command], env=env, stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError(f"{label} exited {result.returncode}; see its log")
        except Exception as exc:
            error = safe_error(exc)
        record.update(seconds=time.time() - started, error=error, state="failed" if error else "completed")
        atomic_json(self.run / "jobs" / f"{label}.json", record)
        with self.lock:
            with open(self.run / "stages.jsonl", "a") as f:
                append_record(f, record)
            if error:
                self.failures.append(record)
        print(f"{record['state'].upper()} {label}: {record['seconds']:.1f}s", flush=True)
        return error is None

    def queue(self, jobs, gpu, cpus):
        previous_model = None
        while True:
            try:
                dataset, model = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                # Each subprocess has a fresh token window. Drain the previous
                # provider minute before starting the next Voyage dataset.
                if previous_model in bench.VOYAGE and model in bench.VOYAGE:
                    time.sleep(61)
                self.job(dataset, model, gpu, cpus)
                previous_model = model
            finally:
                jobs.task_done()


def work_queue(jobs):
    result = queue.Queue()
    for job in jobs:
        result.put(job)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "gpu", "api", "report"], default="all")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--cpus-per-gpu", type=int, default=12)
    parser.add_argument("--cpu-sets", nargs=8, type=parse_cpu_set,
                        help="Optional explicit CPU ranges per GPU, chosen from nvidia-smi topo -m")
    parser.add_argument("--colbert-python", type=Path, default=HERE / ".venv-colbert/bin/python")
    args = parser.parse_args()
    if len(args.gpus) != 8 or len(set(args.gpus)) != 8 or min(args.gpus) < 0:
        parser.error("Select eight distinct nonnegative GPU IDs")
    if args.cpus_per_gpu < 1:
        parser.error("--cpus-per-gpu must be positive")
    run = args.run_dir.resolve()
    if run.exists() and not (run / "full.json").exists():
        parser.error("Choose a new directory, or resume one containing full.json")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [HERE / n for n in ("bench.py", "run_state.py", "full.py", "revisions.json",
                                         "requirements-gpu.lock", "requirements-colbert-gpu.lock")]}
    config = {"kind": "full-eight-h100-v1", "query_counts": QUERY_COUNTS, "models": DEFAULT_MODELS,
              "revisions": bench.REVISIONS, "source_sha256": source_hashes,
              "api_workers": bench.API_WORKERS, "voyage_tpm": bench.VOYAGE_TPM}
    bind_manifest(run / "full.json", config)
    for folder in ("logs", "jobs"):
        (run / folder).mkdir(exist_ok=True)
    bench.ROOT, bench.RESULTS = run / "cache", run / "results"
    with exclusive(run / ".full.lock"):
        if args.stage == "report":
            coverage(run)
            bench.report()
            return
        cpus = sorted(os.sched_getaffinity(0))
        gpu_cpus, api_cpus = cpu_assignments(cpus, 8, args.cpus_per_gpu, gpu_sets=args.cpu_sets)
        if args.stage != "api":
            if not bench.torch.cuda.is_available() or bench.torch.cuda.device_count() <= max(args.gpus):
                parser.error("All eight GPUs must be visible")
            if any("H100" not in bench.torch.cuda.get_device_name(g) for g in args.gpus):
                parser.error("This run requires eight H100 GPUs")
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True)
        bind_manifest(run / "execution.json", {"gpu_cpus": gpu_cpus, "api_cpus": api_cpus,
                                               "gpu_ids": args.gpus, "gpu_inventory": inventory})
        atomic_json(run / f"environment-{args.stage}.json", {
            "packages": packages(), "python": sys.version, "cuda_runtime": bench.torch.version.cuda,
            "gpu_cpus": gpu_cpus, "api_cpus": api_cpus, "cpu_affinity": cpus,
            "gpus": inventory})
        runner = Runner(run, args.colbert_python)
        if args.stage in {"all", "prepare"}:
            jobs = work_queue((d, None) for d in bench.DATASETS)
            with ThreadPoolExecutor(8) as pool:
                futures = [pool.submit(runner.queue, jobs, gpu, cores) for gpu, cores in zip(args.gpus, gpu_cpus)]
                for future in futures:
                    future.result()
            if runner.failures:
                raise RuntimeError("Preparation failed; scoring was not started")
        with ThreadPoolExecutor(10) as pool:
            futures = []
            if args.stage in {"all", "gpu"}:
                jobs = work_queue(gpu_jobs())
                futures += [pool.submit(runner.queue, jobs, gpu, cores) for gpu, cores in zip(args.gpus, gpu_cpus)]
            if args.stage in {"all", "api"}:
                for cores, models in zip(api_cpus, ([*bench.JEV_PROMPTS], ["voyage-rerank-3"])):
                    jobs = work_queue((d, m) for m in models for d in bench.DATASETS
                                      if m in selected_models(models, d))
                    futures.append(pool.submit(runner.queue, jobs, None, cores))
            for future in futures:
                future.result()
        if runner.failures:
            raise RuntimeError(f"{len(runner.failures)} jobs failed; rerun this stage to resume checkpoints")
        if args.stage == "all":
            coverage(run)
            bench.report()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(safe_error(exc), file=sys.stderr)
        sys.exit(1)
