from types import SimpleNamespace
from unittest.mock import Mock
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_reranker import JevReranker
from compare_jev import summarize, candidates


def client(scores):
    def call(**kwargs):
        answers = {key: SimpleNamespace(noul=scores[q['instructions']['candidate']])
                   for key, q in kwargs['questions'].items()}
        return SimpleNamespace(model='jev-test', answers=answers)
    return SimpleNamespace(system_one=Mock(side_effect=call))


def test_stable_order_scores_and_ids():
    c = client({'a': .2, 'b': .9, 'c': .9})
    r = JevReranker(client=c)
    table = pa.table({'answer': ['a', 'b', 'c'], '_rowid': [7, 8, 9], '_distance': [1., 2., 3.]})
    result = r.rerank_vector('query', table)
    assert result['_rowid'].to_pylist() == [8, 9, 7]
    assert '_distance' not in result.column_names
    assert result['_relevance_score'].to_pylist() == pytest.approx([.9, .9, .2])
    assert c.system_one.call_count == 1
    assert r.resolved_models == {'jev-test'}


def test_empty_no_api_call():
    c = client({})
    r = JevReranker(client=c)
    result = r.rerank_fts('query', pa.table({'answer': pa.array([], type=pa.string())}))
    assert len(result) == 0
    assert '_relevance_score' in result.column_names
    c.system_one.assert_not_called()


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1, True, '0.9'])
def test_invalid_scores_fail(score):
    with pytest.raises(ValueError):
        JevReranker(client=client({'a': score})).score_documents('q', ['a'])


def test_errors_do_not_become_zero_scores():
    c = SimpleNamespace(system_one=Mock(side_effect=RuntimeError('API failed')))
    with pytest.raises(RuntimeError, match='API failed'):
        JevReranker(client=c).score_documents('q', ['a'])


def test_hybrid_deduplicates_row_ids():
    r = JevReranker(client=client({'a': .2, 'b': .9}))
    v = pa.table({'answer': ['a'], '_rowid': pa.array([1], type=pa.uint64()), '_distance': [1.]})
    f = pa.table({'answer': ['a', 'b'], '_rowid': pa.array([1, 2], type=pa.uint64()), '_score': [2., 1.]})
    assert r.rerank_hybrid('q', v, f)['answer'].to_pylist() == ['b', 'a']


def test_metric_and_candidate_depth():
    row = {'vector': list(range(40)), 'fts': list(range(30, 70)),
           'documents': {str(i): str(i) for i in range(70)}, 'answer': '25'}
    scores = {str(i): float(i == 25) for i in range(70)}
    assert len(candidates(row, 'hybrid', 5)) == 40
    metrics = summarize([row], [scores])
    vector = [m for m in metrics if m['method'] == 'vector']
    assert vector[0]['hits'] == 0  # Outside 4*5 candidates: never inject a positive.
    assert vector[1]['hits'] == 1
    assert all(m['queries'] == 1 for m in metrics)


def test_key_file(tmp_path, monkeypatch):
    import typesafe_sdk
    key = tmp_path / 'key'
    key.write_text('test-only-key\n')
    constructor = Mock()
    monkeypatch.setattr(typesafe_sdk, 'TypeSafeClient', constructor)
    JevReranker(api_key_file=str(key))
    constructor.assert_called_once_with(api_key='test-only-key')


def test_batches_preserve_order_and_isolate_candidates():
    c = client({str(i): i/10 for i in range(7)})
    r = JevReranker(client=c, batch_size=3, workers=2)
    assert r.score_documents('q', [str(i) for i in range(7)]) == pytest.approx([i/10 for i in range(7)])
    assert c.system_one.call_count == 3
    for call in c.system_one.call_args_list:
        assert call.kwargs['state'] == {'query': 'q'}
        assert len(call.kwargs['questions']) <= 3


def test_lancedb_query_integration(tmp_path):
    import lancedb
    db = lancedb.connect(str(tmp_path / 'db'))
    table = db.create_table('docs', data=[
        {'answer': 'alpha one', 'vector': [1., 0.]},
        {'answer': 'alpha two', 'vector': [0., 1.]},
    ])
    table.create_fts_index('answer')
    r = JevReranker(client=client({'alpha one': .1, 'alpha two': .9}))
    vector = table.search([1., 0.]).limit(2).rerank(r, 'alpha').to_list()
    fts = table.search('alpha', query_type='fts').limit(2).rerank(r).to_list()
    hybrid = table.search(query_type='hybrid').vector([1., 0.]).text('alpha').limit(2).rerank(r).to_list()
    for results in (vector, fts, hybrid):
        assert [row['answer'] for row in results] == ['alpha two', 'alpha one']
        assert results[0]['_relevance_score'] == pytest.approx(.9)
