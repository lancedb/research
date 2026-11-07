from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

import bench
import full
from run_state import atomic_json


def test_eight_workers_use_disjoint_cpus_and_execute_each_job_once(tmp_path, monkeypatch):
    gpu_cpus, api_cpus = full.cpu_assignments(list(range(112)), 8, 12)
    allocations = gpu_cpus + api_cpus
    assert len(set(sum(allocations, []))) == sum(map(len, allocations))
    with pytest.raises(ValueError, match="Need 100"):
        full.cpu_assignments(list(range(99)), 8, 12)
    for folder in ("logs", "jobs"):
        (tmp_path / folder).mkdir()
    runner = full.Runner(tmp_path, Path("/test/.venv-colbert/bin/python"))
    observed, lock = [], threading.Lock()
    def run(command, *, env, stdout, stderr):
        assert command[:2] == ["taskset", "-c"]
        assert env["OMP_NUM_THREADS"] == "12"
        assert command[2] == ",".join(map(str, gpu_cpus[int(env["CUDA_VISIBLE_DEVICES"])]))
        with lock:
            observed.append(tuple(command[-3:]))
        return SimpleNamespace(returncode=1 if command[-1] == "zerank-2" else 0)
    monkeypatch.setattr(full.subprocess, "run", run)
    jobs = full.gpu_jobs()
    pending = full.work_queue(jobs)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(runner.queue, pending, i, gpu_cpus[i]) for i in range(8)]
        for future in futures:
            future.result()
    assert len(observed) == len(set(observed)) == 60
    assert len(runner.failures) == 5
    assert len(list((tmp_path / "jobs").glob("*.json"))) == 60
    assert json.loads((tmp_path / "jobs/score-nq-zerank-2.json").read_text())["state"] == "failed"


def test_full_coverage_rejects_missing_duplicate_and_sampled_queries(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "QUERY_COUNTS", {"fiqa": 2})
    monkeypatch.setattr(full, "DEFAULT_MODELS", ["jev", "voyage-rerank-3"])
    monkeypatch.setattr(bench, "candidate_fingerprint", lambda _: "frozen")
    root = tmp_path / "cache/fiqa"
    (root / "scores").mkdir(parents=True)
    rows = [{"qid": str(i), "docs": ["a", "b"]} for i in range(2)]
    bench.write_jsonl(root / "candidates.jsonl", rows)
    atomic_json(root / "prepare.json", {"query_limit": None})
    for model in full.DEFAULT_MODELS:
        path = root / "scores" / f"{model}.jsonl"
        bench.write_jsonl(path, [{"i": i, "scores": [0.1, 0.2]} for i in range(2)])
        atomic_json(path.with_suffix(".meta.json"), {"candidates": "frozen"})
    assert full.coverage(tmp_path)["score_records"] == 4
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        full.coverage(tmp_path)
    bench.write_jsonl(path, [{"i": 0, "scores": [0.1, 0.2]}] * 2)
    with pytest.raises(ValueError, match="Duplicate query"):
        full.coverage(tmp_path)
    atomic_json(root / "prepare.json", {"query_limit": 2})
    with pytest.raises(ValueError, match="limited sample"):
        full.coverage(tmp_path)


def test_api_jobs_hide_gpus_and_use_colbert_venv_only_for_colbert(tmp_path, monkeypatch):
    for folder in ("logs", "jobs"):
        (tmp_path / folder).mkdir()
    seen = []
    def run(command, *, env, stdout, stderr):
        seen.append((command, env))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(full.subprocess, "run", run)
    interpreter = tmp_path / ".venv-colbert/bin/python"
    runner = full.Runner(tmp_path, interpreter)
    runner.job("fiqa", "voyage-rerank-3", None, [96, 97])
    runner.job("fiqa", "jina-colbert-v2", 7, list(range(84, 96)))
    assert seen[0][1]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen[1][0][3] == str(interpreter)
    assert seen[1][1]["CUDA_VISIBLE_DEVICES"] == "7"


def test_explicit_numa_cpu_sets_reserve_unused_cores_for_apis():
    groups = [full.parse_cpu_set(f"{start}-{start+17}") for start in range(0, 224, 28)]
    gpu, api = full.cpu_assignments(list(range(224)), 8, 18, gpu_sets=groups)
    assert gpu == groups
    assert api == [[18, 19], [20, 21]]
    groups[-1] = groups[0]
    with pytest.raises(ValueError, match="disjoint"):
        full.cpu_assignments(list(range(224)), 8, 18, gpu_sets=groups)


def test_voyage_dataset_transitions_do_not_reset_the_active_provider_minute(tmp_path, monkeypatch):
    events = []
    runner = full.Runner(tmp_path, Path("python"))
    monkeypatch.setattr(runner, "job", lambda d, m, g, c: events.append(d))
    monkeypatch.setattr(full.time, "sleep", lambda seconds: events.append(seconds))
    runner.queue(full.work_queue([("gooaq", "voyage-rerank-3"), ("nq", "voyage-rerank-3")]), None, [96, 97])
    assert events == ["gooaq", 61, "nq"]
