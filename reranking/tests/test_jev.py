from types import SimpleNamespace
from unittest.mock import Mock
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from lancedb.rerankers import TypeSafeReranker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_jev import summarize, candidates, score_jev, run, save
from jev_config import QUESTION, read_api_key


@pytest.fixture
def sdk(monkeypatch):
    import typesafe_sdk

    def call(**kwargs):
        score = {'a': .2, 'b': .9, 'alpha one': .1, 'alpha two': .9}[kwargs['state']['document']]
        return SimpleNamespace(answers={'relevance': SimpleNamespace(noul=score)})

    client = SimpleNamespace(system_one=Mock(side_effect=call))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(typesafe_sdk, 'TypeSafeClient', constructor)
    return client, constructor


def test_benchmark_restores_document_order_and_isolates_candidates(sdk):
    model = TypeSafeReranker(column='answer', api_key='test-key',
                             instructions=QUESTION['instructions'], criteria=QUESTION['criteria'])
    assert score_jev(model, 'q', ['a', 'b', 'a']) == pytest.approx([.2, .9, .2])
    client, constructor = sdk
    constructor.assert_called_once_with(api_key='test-key')
    assert client.system_one.call_count == 3
    for call in client.system_one.call_args_list:
        assert call.kwargs['state']['query'] == 'q'
        assert set(call.kwargs['state']) == {'query', 'document'}
        assert call.kwargs['questions'] == {'relevance': QUESTION}


def test_empty_no_api_call(sdk):
    assert score_jev(TypeSafeReranker(column='answer'), 'q', []) == []
    sdk[0].system_one.assert_not_called()


@pytest.mark.parametrize('score', [float('nan'), float('inf'), -.1, 1.1, None])
def test_invalid_scores_fail(sdk, score):
    sdk[0].system_one.side_effect = None
    sdk[0].system_one.return_value = SimpleNamespace(answers={'relevance': SimpleNamespace(noul=score)})
    with pytest.raises(ValueError, match='invalid relevance probability'):
        score_jev(TypeSafeReranker(column='answer'), 'q', ['a'])


def test_errors_do_not_become_zero_scores(sdk):
    sdk[0].system_one.side_effect = RuntimeError('API failed')
    with pytest.raises(RuntimeError, match='API failed'):
        score_jev(TypeSafeReranker(column='answer'), 'q', ['a'])


def test_hybrid_deduplicates_row_ids(sdk):
    r = TypeSafeReranker(column='answer')
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
    assert vector[0]['hits'] == 0
    assert vector[1]['hits'] == 1
    assert all(m['queries'] == 1 for m in metrics)


def test_key_sources(tmp_path, monkeypatch):
    key = tmp_path / 'key'
    key.write_text('file-key\n')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'env-key')
    monkeypatch.delenv('TYPESAFE_API_KEY_FILE', raising=False)
    assert read_api_key() == 'env-key'
    monkeypatch.setenv('TYPESAFE_API_KEY_FILE', str(key))
    assert read_api_key() == 'file-key'
    explicit = tmp_path / 'explicit'
    explicit.write_text('explicit-key\n')
    assert read_api_key(str(explicit)) == 'explicit-key'
    key.write_text(' \n')
    with pytest.raises(ValueError, match='Set TYPESAFE_API_KEY'):
        read_api_key()
    monkeypatch.delenv('TYPESAFE_API_KEY_FILE')
    monkeypatch.delenv('TYPESAFE_API_KEY')
    with pytest.raises(ValueError, match='Set TYPESAFE_API_KEY'):
        read_api_key()


def test_lancedb_query_integration(tmp_path, sdk):
    import lancedb
    db = lancedb.connect(str(tmp_path / 'db'))
    table = db.create_table('docs', data=[
        {'answer': 'alpha one', 'vector': [1., 0.]},
        {'answer': 'alpha two', 'vector': [0., 1.]},
    ])
    table.create_fts_index('answer')
    r = TypeSafeReranker(column='answer')
    vector = table.search([1., 0.]).limit(2).rerank(r, 'alpha').to_list()
    fts = table.search('alpha', query_type='fts').limit(2).rerank(r).to_list()
    hybrid = table.search(query_type='hybrid').vector([1., 0.]).text('alpha').limit(2).rerank(r).to_list()
    for results in (vector, fts, hybrid):
        assert [row['answer'] for row in results] == ['alpha two', 'alpha one']
        assert results[0]['_relevance_score'] == pytest.approx(.9)


def test_runner_uses_native_protocol_and_resumes(tmp_path, monkeypatch, sdk):
    import hashlib
    import json
    import compare_jev

    rows = [{'query': 'q', 'answer': 'b', 'vector': [7, 8], 'fts': [8, 7],
             'documents': {'7': 'a', '8': 'b'}}]
    save(tmp_path / 'manifest.json', {'config': {}})
    save(tmp_path / 'candidates.json', rows)
    monkeypatch.setattr(compare_jev, 'prepare', lambda args, root: rows)
    monkeypatch.setattr(compare_jev.importlib.metadata, 'version', lambda p: 'test-version')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-key')
    monkeypatch.delenv('TYPESAFE_API_KEY_FILE', raising=False)
    args = SimpleNamespace(cache=str(tmp_path), models=['jev'], workers=2,
                           jev_model='jev-1.13.0', api_key_file=None, output=str(tmp_path / 'result.json'))
    old_identity = {'model': args.jev_model, 'question': QUESTION,
                    'protocol': 'query-state-candidate-per-question-v1', 'batch_size': 40}
    dataset_hash = hashlib.sha256((tmp_path / 'candidates.json').read_bytes()).hexdigest()
    old_key = hashlib.sha256(json.dumps([dataset_hash, old_identity], sort_keys=True).encode()).hexdigest()[:16]
    save(tmp_path / 'scores' / f'jev-{old_key}' / '0.json',
         {'scores': {'7': 1., '8': 0.}, 'seconds': 0, 'models': ['old-model']})
    run(args)
    result = json.loads(Path(args.output).read_text())
    jev = result['results']['jev']
    assert jev['requested_model'] == 'jev-1.13.0'
    assert jev['resolved_models'] == []  # Upstream does not expose this metadata.
    assert jev['scoring_protocol']['protocol'] == 'lancedb-typesafe-query-document-state-v2'
    assert result['jev_batch_size'] == 1
    assert sdk[0].system_one.call_count == 2  # Old adapter cache was not reused.
    new_scores = [p for p in (tmp_path / 'scores').glob('jev-*/0.json') if old_key not in str(p)]
    assert json.loads(new_scores[0].read_text())['scores'] == pytest.approx({'7': .2, '8': .9})
    run(args)
    assert sdk[0].system_one.call_count == 2  # Native cache was reused.
    assert 'test-key' not in Path(args.output).read_text()
