import math

import pytest

from bench import small_sample
from smoke import run_directory, validate_scores


def test_sample_is_repeatable_and_retains_all_selected_positives():
    docs = {str(i): f"document {i}" for i in range(100)}
    queries = [(str(i), f"query {i}") for i in range(10)]
    qrels = {str(i): {str(i), str(i + 10)} for i in range(10)}
    sampled = small_sample(docs, queries, qrels, 3, 50)
    assert sampled == small_sample(docs, queries, qrels, 3, 50)
    kept_docs, kept_queries, kept_qrels = sampled
    assert len(kept_docs) == 50 and len(kept_queries) == 3
    assert all(kept_qrels[qid] <= set(kept_docs) for qid, _ in kept_queries)
    assert len(docs) == 100 and len(queries) == 10


def test_smoke_cannot_reuse_full_run_or_change_sample(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    with pytest.raises(ValueError, match="not a matching smoke run"):
        run_directory(full, 5, 500)
    smoke = run_directory(tmp_path / "smoke", 5, 500)
    assert run_directory(smoke, 5, 500) == smoke
    with pytest.raises(ValueError, match="not a matching smoke run"):
        run_directory(smoke, 6, 500)


@pytest.mark.parametrize("records", [
    [], [{"i": 1, "scores": [0.2, 0.4]}], [{"i": 0, "scores": [0.2]}],
    [{"i": 0, "scores": [0.2, math.nan]}],
])
def test_smoke_rejects_incomplete_or_misaligned_scores(records):
    with pytest.raises(ValueError):
        validate_scores([{"docs": ["a", "b"]}], records)


def test_smoke_accepts_complete_scores():
    validate_scores([{"docs": ["a", "b"]}], [{"i": 0, "scores": [0.2, 0.4]}])
