"""Protect existing measurements when adding a model to a published comparison."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_jev import append_result


def measurement(results):
    return {"config": {"queries": 2000}, "dataset_fingerprint": "dataset",
            "dataset_revision": "revision", "candidate_sha256": "candidates",
            "created_at": "original", "results": results}


def test_append_preserves_original_measurements_and_provenance():
    old = measurement({"jev": {"metrics": [92.35], "scoring_seconds_p50": .186}})
    new = measurement({"qwen": {"metrics": [90]}})
    new["created_at"] = "later"
    before = deepcopy(old)
    merged = append_result(old, new)
    assert old == before
    assert merged["results"]["jev"] == before["results"]["jev"]
    assert merged["results"]["qwen"] == new["results"]["qwen"]
    assert merged["created_at"] == "original"
    assert merged["additional_runs"][0]["created_at"] == "later"


@pytest.mark.parametrize("field", ["config", "dataset_fingerprint", "dataset_revision", "candidate_sha256"])
def test_append_rejects_incompatible_run(field):
    old, new = measurement({"jev": {}}), measurement({"qwen": {}})
    new[field] = "different"
    with pytest.raises(ValueError, match=field):
        append_result(old, new)


def test_append_rejects_overwriting_existing_model():
    with pytest.raises(ValueError, match="overwrite"):
        append_result(measurement({"jev": {}}), measurement({"jev": {}}))


def test_qwen_only_run_preserves_baselines_and_resumes(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    import sentence_transformers
    import compare_jev

    row = {"query": "capital?", "answer": "Paris", "vector": [0, 1], "fts": [1, 0],
           "documents": {"0": "Paris", "1": "London"}}
    rows = [row, row]
    cache = tmp_path / "cache"
    cache.mkdir()
    compare_jev.save(cache / "candidates.json", rows)
    manifest = {"config": {"queries": 2}, "dataset_fingerprint": "dataset", "dataset_revision": "revision"}
    compare_jev.save(cache / "manifest.json", manifest)
    original = {**manifest, "candidate_sha256": hashlib.sha256((cache / "candidates.json").read_bytes()).hexdigest(),
                "results": {"jev": {"metrics": [92.35]}}}
    output = tmp_path / "result.json"
    compare_jev.save(output, original)
    original_bytes = output.read_bytes()
    monkeypatch.setattr(compare_jev, "prepare", lambda args, root: rows)
    model = SimpleNamespace(device="cpu", predict=Mock(side_effect=[np.array([1., 0.]), RuntimeError("interrupted")]))
    constructor = Mock(return_value=model)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", constructor)
    args = SimpleNamespace(cache=str(cache), output=str(output), models=["qwen"],
                           append=True, workers=4, qwen_batch_size=4)
    with pytest.raises(RuntimeError, match="interrupted"):
        compare_jev.run(args)
    assert output.read_bytes() == original_bytes
    model.predict = Mock(return_value=np.array([1., 0.]))
    compare_jev.run(args)
    assert model.predict.call_count == 1  # Query zero was checkpointed.
    assert all(call.args == ("Qwen/Qwen3-Reranker-8B",) for call in constructor.call_args_list)
    assert constructor.call_args.kwargs["revision"] == compare_jev.QWEN_REVISION
    merged = json.loads(output.read_text())
    assert merged["results"]["jev"] == original["results"]["jev"]
    assert all(m["hits"] == 2 for m in merged["results"]["qwen"]["metrics"])
