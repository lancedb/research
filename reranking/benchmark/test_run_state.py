import json
import threading
import time

import pytest

from run_state import (bind_manifest, execute_scores, read_journal, score_records, timing_summary)


ROWS = [{"docs": ["a", "b"]} for _ in range(5)]


def record(i, **extra):
    return {"i": i, "scores": [0.2, 0.8], "seconds": 0.01, **extra}


def test_resume_does_not_repeat_paid_requests_or_lose_tokens(tmp_path):
    path = tmp_path / "scores.jsonl"
    calls = []
    def interrupted(i):
        calls.append(i)
        if i == 2:
            raise RuntimeError("provider down")
        return record(i, tokens=100)
    with pytest.raises(RuntimeError):
        execute_scores(path, ROWS, interrupted)
    assert calls == [0, 1, 2]
    calls.clear()
    def resumed(i):
        calls.append(i)
        return record(i, tokens=100)
    records = execute_scores(path, ROWS, resumed)
    assert calls == [2, 3, 4]
    assert timing_summary(path, records)["tokens"] == 500
    calls.clear()
    execute_scores(path, ROWS, resumed)
    assert calls == []


def test_parallel_failure_drains_and_checkpoints_only_inflight_work(tmp_path):
    path = tmp_path / "scores.jsonl"
    barrier = threading.Barrier(2)
    calls = []
    def one(i):
        calls.append(i)
        barrier.wait(timeout=3)
        if i == 0:
            raise RuntimeError("failure")
        time.sleep(0.05)
        return record(i)
    with pytest.raises(RuntimeError):
        execute_scores(path, ROWS, one, workers=2)
    assert sorted(calls) == [0, 1]
    assert [r["i"] for r in score_records(path, ROWS)] == [1]


@pytest.mark.parametrize("records", [
    [record(0), record(0)], [record(8)], [record(0, scores=[0.2])],
    [record(0, scores=[float("nan"), 1])], [record(0, seconds=-1)],
])
def test_corrupt_completed_scores_fail_instead_of_silently_resuming(tmp_path, records):
    path = tmp_path / "scores.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError):
        score_records(path, ROWS, repair=True)


def test_torn_final_line_is_repaired_but_completed_invalid_json_is_not(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(record(0)) + '\n{"i":')
    assert len(score_records(path, ROWS, repair=True)) == 1
    path.write_text('broken\n')
    with pytest.raises(ValueError):
        score_records(path, ROWS, repair=True)


def test_fingerprint_refuses_changed_model_or_candidate_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    bind_manifest(path, {"candidates": "a", "revision": "1"})
    bind_manifest(path, {"candidates": "a", "revision": "1"})
    with pytest.raises(ValueError, match="configuration changed"):
        bind_manifest(path, {"candidates": "b", "revision": "1"})


def test_wall_clock_throughput_uses_elapsed_session_time(tmp_path):
    path = tmp_path / "scores.jsonl"
    records = [record(0, session="a", session_elapsed=1), record(1, session="a", session_elapsed=1.1)]
    path.with_suffix(".sessions.jsonl").write_text('{"session":"a","seconds":1.5}\n')
    assert timing_summary(path, records)["wall_seconds"] == 1.5
    assert timing_summary(path, records)["interrupted_sessions"] == 0
    assert read_journal(tmp_path / "absent") == []
