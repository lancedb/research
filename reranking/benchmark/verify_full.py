"""Verify full-run artifacts offline and write compact, auditable evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import bench
from full import QUERY_COUNTS, coverage
from pilot import DEFAULT_MODELS, selected_models
from run_state import atomic_json, file_digest, score_records, timing_summary


def verify(run):
    status = coverage(run)
    summary = json.loads((run / "results/summary.json").read_text())
    if set(summary) != set(QUERY_COUNTS):
        raise ValueError("Report does not contain every dataset")
    evidence = {}
    for dataset in QUERY_COUNTS:
        root = run / "cache" / dataset
        rows = bench.read_jsonl(root / "candidates.jsonl")
        ids = sorted({d for r in rows for d in r["docs"]})
        texts, content = bench.CandidateTexts(root), hashlib.sha256()
        for batch in bench.chunks(ids):
            content.update(json.dumps(list(zip(batch, texts.fetch(batch))), ensure_ascii=False).encode())
        candidate_meta = json.loads((root / "candidates.meta.json").read_text())
        if content.hexdigest() != candidate_meta["text_sha256"]:
            raise ValueError(f"Candidate text content changed: {dataset}")
        models = {}
        for model in selected_models(DEFAULT_MODELS, dataset):
            model_id = {**bench.CROSS_ENCODERS, **bench.LISTWISE, **bench.COLBERTS, **bench.VOYAGE}.get(model, bench.JEV_MODEL)
            variants = [f"{model}-pool{p}" for p in bench.POOL_FACTORS] if model in bench.COLBERTS else [model]
            for variant in variants:
                out = root / "scores" / f"{variant}.jsonl"
                meta = json.loads(out.with_suffix(".meta.json").read_text())
                if (meta["model"] != model or meta["model_id"] != model_id
                        or meta["revision"] != bench.REVISIONS["models"].get(model_id)
                        or meta["prompt"] != bench.JEV_PROMPTS.get(model)):
                    raise ValueError(f"Wrong model or prompt: {dataset}/{variant}")
                if model == "jina-colbert-v2" and meta["code_revision"] != bench.JINA_CODE_REVISION:
                    raise ValueError("Jina external implementation revision changed")
                records = score_records(out, rows)
                timing = timing_summary(out, records)
                if timing["wall_seconds"] <= 0:
                    raise ValueError(f"Missing timing: {dataset}/{variant}")
                if model not in bench.JEV_PROMPTS and model not in bench.VOYAGE:
                    if json.loads(out.with_suffix(".warmup.json").read_text())["queries"] != bench.WARMUP_QUERIES:
                        raise ValueError(f"Missing warmup: {dataset}/{variant}")
                if model in bench.VOYAGE:
                    usage = json.loads(out.with_suffix(".tokens.json").read_text())
                    if usage["tokens"] != timing["tokens"] or usage["queries"] != len(rows):
                        raise ValueError(f"Voyage usage does not match saved responses: {dataset}")
                report = summary[dataset].get(variant)
                if report is None or report["timing"] != timing:
                    raise ValueError(f"Report missing or stale: {dataset}/{variant}")
                # Recompute the principal comparison directly from saved scores.
                recalculated = bench.metrics(rows, records, "hybrid", 50)
                if (set(report["hybrid@50"]) != set(recalculated) or any(
                        not math.isclose(report["hybrid@50"][k], value, rel_tol=0, abs_tol=1e-9)
                        for k, value in recalculated.items())):
                    raise ValueError(f"Report metrics do not match scores: {dataset}/{variant}")
                models[variant] = {"score_sha256": file_digest(out), "queries": len(records),
                                   "configuration": meta, "timing": timing}
        if set(summary[dataset]) != {"none", *models}:
            raise ValueError(f"Unexpected configurations in report: {dataset}")
        evidence[dataset] = {"queries": len(rows), "candidate_documents": len(ids),
                             "candidate_metadata": candidate_meta, "models": models}
        print(f"Verified {dataset}: {len(rows)} queries × {len(models)} configurations", flush=True)
    result = {"validated": True, "score_records": status["score_records"], "datasets": evidence,
              "summary_sha256": file_digest(run / "results/summary.json")}
    atomic_json(run / "artifact-validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    verify(args.run_dir.resolve())
