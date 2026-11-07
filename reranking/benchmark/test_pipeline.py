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


def test_prepare_reuses_completed_ann_index_after_query_encoding_interruption(corpus, monkeypatch):
    docs = {str(i): f"finance money document {i}" for i in range(512)}
    queries = [(str(i), f"money {i}") for i in range(3)]
    monkeypatch.setattr(bench, "load", lambda _: (docs, queries, {str(i): {str(i)} for i in range(3)}))

    class AnnEncoder(Encoder):
        def encode(self, texts, **kwargs):
            super().encode(texts, **kwargs)
            vectors = np.array([np.random.default_rng(int(t.split()[-1])).normal(size=4)
                                for t in texts], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(bench, "SentenceTransformer", AnnEncoder)
    ivfpq = bench.IvfPq
    monkeypatch.setattr(bench, "IvfPq", lambda **kw: ivfpq(num_partitions=1, num_sub_vectors=2, **kw))
    Encoder.fail_at = 17  # sixteen corpus chunks, then query encoding after all indexes exist
    with pytest.raises(RuntimeError, match="interrupted embedding"):
        bench.prepare("nq", query_limit=3)
    root = bench.ROOT / "nq"
    table = bench.lancedb.connect(str(root / "db")).open_table("docs")
    assert any(i.index_type == "IvfPq" for i in table.list_indices())
    assert not (root / "candidates.meta.json").exists()

    def already_built(*args, **kwargs):
        raise AssertionError("Resuming should reuse the completed ANN index")

    monkeypatch.setattr(bench.lancedb.table.LanceTable, "create_index", already_built)
    Encoder.fail_at = None
    bench.prepare("nq", query_limit=3)
    assert len(bench.read_jsonl(root / "candidates.jsonl")) == 3


def test_new_query_sample_reuses_frozen_full_corpus_without_reembedding(corpus, monkeypatch):
    bench.prepare("fiqa", query_limit=3)
    original_root = bench.ROOT
    monkeypatch.setenv("BENCH_CORPUS_CACHE", str(original_root))
    monkeypatch.setattr(bench, "ROOT", original_root.parent / "larger-run")
    calls = Encoder.calls
    bench.prepare("fiqa", query_limit=5)
    assert Encoder.calls == calls + 1  # only the five queries, no document embeddings
    assert len(bench.read_jsonl(bench.ROOT / "fiqa/candidates.jsonl")) == 5
    assert not (bench.ROOT / "fiqa/db/docs.lance").exists()
    source = bench.lancedb.connect(str(corpus / "db")).open_table("docs")
    assert source.count_rows() == 120


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


def test_pilot_keeps_virtual_environment_python_symlink(tmp_path):
    from pilot import python_executable
    target = tmp_path / "system-python"
    target.touch()
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(target)
    assert python_executable(interpreter) == str(interpreter)
    assert python_executable(interpreter) != str(interpreter.resolve())


def test_retryable_recognizes_typesafe_status_but_does_not_retry_bad_credentials():
    from typesafe_sdk import TypeSafeInternalServerError, TypeSafeAuthenticationError
    assert bench.retryable(TypeSafeInternalServerError(status=503, body={}, headers={}))
    assert not bench.retryable(TypeSafeAuthenticationError(status=401, body={}, headers={}))


def test_provider_retry_status_and_requested_waits(monkeypatch):
    from voyageai.error import APIError, RateLimitError, Timeout
    from typesafe_sdk import TypeSafeRateLimitError
    monkeypatch.setattr(bench.random, "random", lambda: 0)
    assert bench.retryable(APIError(http_status=503))
    assert bench.retryable(Timeout())
    assert not bench.retryable(APIError(http_status=401))
    assert bench.retry_delay(RateLimitError(http_status=429, headers={"Retry-After": "45"}), 0) == 45
    assert bench.retry_delay(TypeSafeRateLimitError(status=429, body={}, headers={"retry-after": "30"}), 0) == 30
    assert bench.retry_delay(RateLimitError(http_status=429), 2) == 60


def test_full_artifact_verifier_recomputes_metrics_and_checks_prompts(corpus, monkeypatch):
    import full
    import verify_full
    from run_state import atomic_json
    for module in (full, verify_full):
        monkeypatch.setattr(module, "QUERY_COUNTS", {"fiqa": 15})
        monkeypatch.setattr(module, "DEFAULT_MODELS", ["jev"])
    bench.prepare("fiqa")
    monkeypatch.setattr(bench, "make_reranker", lambda _: object())
    monkeypatch.setattr(bench, "rerank_scores", lambda r, q, docs: [0.3] * len(docs))
    bench.score("fiqa", "jev")
    bench.report()
    run = bench.ROOT.parent
    assert verify_full.verify(run)["score_records"] == 15
    report_path = bench.RESULTS / "summary.json"
    summary = json.loads(report_path.read_text())
    summary["fiqa"]["jev"]["hybrid@50"]["hit@10"] += 1
    atomic_json(report_path, summary)
    with pytest.raises(ValueError, match="metrics do not match"):
        verify_full.verify(run)
    bench.report()
    meta_path = corpus / "scores/jev.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["prompt"] = {"instructions": "wrong prompt"}
    atomic_json(meta_path, meta)
    with pytest.raises(ValueError, match="Wrong model or prompt"):
        verify_full.verify(run)
