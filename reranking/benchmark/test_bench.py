import numpy as np

from bench import complete_records, metrics, pool, pool_tokens, rank

ROW = {"relevant": ["b"], "vector": ["a", "b", "c"], "fts": ["c", "d"], "rrf": ["c", "a", "b"],
       "docs": ["a", "b", "c", "d"]}


def test_pool_is_deduplicated_union():
    assert pool(ROW, "hybrid", 2) == ["a", "b", "c", "d"]
    assert pool(ROW, "vector", 2) == ["a", "b"]


def test_rank_without_scores_keeps_retrieval_order():
    assert rank(ROW, "vector", 3, None) == ["a", "b", "c"]
    assert rank(ROW, "hybrid", 3, None) == ["c", "a", "b"]


def test_rank_sorts_by_score_within_pool():
    scores = [0.1, 0.9, 0.5, 0.7]
    assert rank(ROW, "vector", 3, scores) == ["b", "c", "a"]
    assert rank(ROW, "fts", 1, scores) == ["c"]


def test_metrics_hit_rates():
    records = [{"scores": [0.1, 0.9, 0.5, 0.7]}]
    result = metrics([ROW], records, "hybrid", 3)
    assert result["hit@1"] == 100 and result["ndcg@10"] == 100
    baseline = metrics([ROW], [None], "fts", 3)
    assert baseline["hit@10"] == 0 and baseline["candidate_hit"] == 0


def test_pool_tokens_halves_and_keeps_first_token():
    tokens = np.vstack([np.eye(4)[:1], np.eye(4)[[1, 1, 2, 2]]]).astype(np.float16)
    pooled = pool_tokens(tokens, 2)
    assert pooled.shape == (3, 4)
    assert (pooled[0] == tokens[0]).all()
    assert pool_tokens(tokens, 1) is tokens


def test_complete_records_drops_a_cut_off_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('{"i": 0}\n{"i": 1}\n{"i": 2, "sco')
    assert complete_records(path) == 2
    assert path.read_text() == '{"i": 0}\n{"i": 1}\n'
    assert complete_records(tmp_path / "missing.jsonl") == 0
