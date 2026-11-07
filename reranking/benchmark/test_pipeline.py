import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import bench


class Encoder:
    calls = 0
    fail_at = None

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        Encoder.calls += 1
        if Encoder.calls == Encoder.fail_at:
            raise RuntimeError("interrupted embedding")
        result = np.array([[1, len(t) % 7, sum(map(ord, t)) % 11, 2] for t in texts], dtype=np.float32)
        return result / np.linalg.norm(result, axis=1, keepdims=True)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ROOT", tmp_path / "cache")
    monkeypatch.setattr(bench, "RESULTS", tmp_path / "results")
    docs = {str(i): f"finance money savings document number {i}" for i in range(120)}
    queries = [(str(i), f"money savings {i}") for i in range(15)]
    monkeypatch.setattr(bench, "load", lambda _: (docs, queries, {str(i): {str(i)} for i in range(15)}))
    monkeypatch.setattr(bench, "SentenceTransformer", Encoder)
    monkeypatch.setattr(bench, "chunks", lambda items, size=32: original_chunks(items, size))
    Encoder.calls, Encoder.fail_at = 0, None
    return bench.ROOT / "fiqa"


original_chunks = bench.chunks


def test_prepare_resumes_chunks_keeps_full_corpus_and_skips_unused_vector_index(corpus, monkeypatch):
    def unused(*args, **kwargs):
        raise AssertionError("Exact search should not build a vector index")
    monkeypatch.setattr(bench.lancedb.table.LanceTable, "create_index", unused)
    Encoder.fail_at = 2
    with pytest.raises(RuntimeError, match="interrupted embedding"):
        bench.prepare("fiqa", query_limit=3)
    table = bench.lancedb.connect(str(corpus / "db")).open_table("docs")
    assert table.count_rows() == 32
    Encoder.fail_at = None
    bench.prepare("fiqa", query_limit=3)
    table = bench.lancedb.connect(str(corpus / "db")).open_table("docs")
    assert table.count_rows() == 120
    assert len(bench.read_jsonl(corpus / "candidates.jsonl")) == 3
    calls = Encoder.calls
    bench.prepare("fiqa", query_limit=3)
    assert Encoder.calls == calls
    with pytest.raises(ValueError, match="configuration changed"):
        bench.prepare("fiqa", query_limit=4)


def test_scores_and_report_validate_frozen_candidates(corpus, monkeypatch):
    bench.prepare("fiqa", query_limit=3)
    monkeypatch.setattr(bench, "make_reranker", lambda _: object())
    monkeypatch.setattr(bench, "rerank_scores", lambda r, q, docs: [0.3] * len(docs))
    bench.score("fiqa", "jev")
    bench.report()
    summary = json.loads((bench.RESULTS / "summary.json").read_text())
    assert set(summary["fiqa"]) == {"none", "jev"}
    assert summary["fiqa"]["jev"]["pairs_per_second"] > 0
    path = corpus / "candidates.jsonl"
    rows = bench.read_jsonl(path)
    rows[0]["query"] = "different query"
    bench.write_jsonl(path, rows)
    with pytest.raises(ValueError, match="Frozen candidates changed"):
        bench.score("fiqa", "jev")


def test_colbert_resumes_partial_scores_and_reuses_document_vectors(corpus, monkeypatch):
    bench.prepare("fiqa", query_limit=3)
    class Colbert:
        document_batches = 0
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, is_query=False, **kwargs):
            if not is_query and kwargs.get("batch_size") == 256:
                Colbert.document_batches += 1
            return [np.eye(4, dtype=np.float16) for _ in texts]
    monkeypatch.setitem(sys.modules, "pylate", types.SimpleNamespace(
        models=types.SimpleNamespace(ColBERT=Colbert), rank=types.SimpleNamespace(rerank=lambda *a, **k: None)))
    monkeypatch.setattr(bench, "synchronize", lambda: None)
    execute = bench.execute_scores
    def interrupted(out, rows, one, **kwargs):
        def fail(i):
            if i == 1:
                raise RuntimeError("interrupted scoring")
            return one(i)
        return execute(out, rows, fail, **kwargs)
    monkeypatch.setattr(bench, "execute_scores", interrupted)
    with pytest.raises(RuntimeError, match="interrupted scoring"):
        bench.score("fiqa", "answerai-colbert-small")
    count = Colbert.document_batches
    monkeypatch.setattr(bench, "execute_scores", execute)
    bench.score("fiqa", "answerai-colbert-small")
    assert Colbert.document_batches == count
    rows = bench.read_jsonl(corpus / "candidates.jsonl")
    for factor in bench.POOL_FACTORS:
        assert len(bench.score_records(corpus / "scores" / f"answerai-colbert-small-pool{factor}.jsonl", rows)) == 3


def test_pilot_limits_multihop_to_hotpotqa():
    from pilot import selected_models
    assert selected_models(["jev", "jev-multihop", "voyage-rerank-3"], "fiqa") == ["jev", "voyage-rerank-3"]
    assert "jev-multihop" in selected_models(["jev-multihop"], "hotpotqa")
