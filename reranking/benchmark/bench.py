"""Off-the-shelf reranker benchmark: every model reorders the same LanceDB candidates.

    python bench.py prepare --dataset nq          # embed, index, retrieve top-50 vector/FTS/hybrid
    python bench.py score --dataset nq --model qwen3-4b
    python bench.py report                        # results/summary.json + results/README.md
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import torch
from datasets import load_dataset
from scipy.cluster import hierarchy
from lancedb.index import IvfPq
from lancedb.query import MatchQuery
from lancedb.rerankers import CrossEncoderReranker, RRFReranker, TypeSafeReranker
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from typesafe_sdk import TypeSafeRateLimitError

ROOT = Path(os.environ.get("BENCH_ROOT", "cache"))
RESULTS = Path(__file__).parent / "results"
DATASETS = ["gooaq", "nq", "hotpotqa", "fiqa", "scidocs"]
EMBEDDER = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEPTH = 50  # candidates per source; hybrid is the union of both lists
DEPTHS = [10, 20, 30, 50]
NPROBES, REFINE = 64, 10
EXACT_SEARCH = {"gooaq", "fiqa", "scidocs"}  # retrieved before we added the IVF_PQ index; kept so results reproduce
MAX_LENGTH = 512
GOOAQ_QUERIES = 20_000  # fixed random sample of the 100k questions; the corpus keeps all 100k answers

CROSS_ENCODERS = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "gte-modernbert": "Alibaba-NLP/gte-reranker-modernbert-base",
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
    "mxbai-large-v2": "mixedbread-ai/mxbai-rerank-large-v2",
    "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen3-8b": "Qwen/Qwen3-Reranker-8B",
    "zerank-2": "zeroentropy/zerank-2-reranker",
}
LISTWISE = {"jina-v3": "jinaai/jina-reranker-v3"}
COLBERTS = {
    "answerai-colbert-small": "answerdotai/answerai-colbert-small-v1",
    "gte-moderncolbert": "lightonai/GTE-ModernColBERT-v1",
    "jina-colbert-v2": "jinaai/jina-colbert-v2",
}
COLBERT_KWARGS = {  # from the model card
    "jina-colbert-v2": {"query_prefix": "[QueryMarker]", "document_prefix": "[DocumentMarker]",
                        "attend_to_expansion_tokens": True, "trust_remote_code": True},
}
POOL_FACTORS = [1, 2, 4]
JEV_MODEL = "jev-1.13.0"


class JinaV3Reranker(CrossEncoderReranker):
    """jina-reranker-v3 is listwise and ships its own `rerank`, not a CrossEncoder head."""

    @property
    def model(self):
        if not hasattr(self, "_jina"):
            self._jina = AutoModel.from_pretrained(self.model_name, dtype=torch.bfloat16, trust_remote_code=True)
            self._jina.to(self.device).eval()
        return self._jina

    def _rerank(self, result_set, query):
        ranked = self.model.rerank(query, result_set[self.column].to_pylist(), max_doc_length=MAX_LENGTH)
        scores = [0.0] * len(result_set)
        for item in ranked:
            scores[item["index"]] = item["relevance_score"]
        return result_set.append_column("_relevance_score", pa.array(scores, type=pa.float32()))


def load(name):
    """Return {doc_id: text}, [(query_id, query)], {query_id: set(relevant doc_ids)}."""
    if name == "gooaq":
        # Rows 2.0M-2.1M: held out from the fine-tuned models in ../finetune, so results stay reusable there.
        rows = load_dataset("sentence-transformers/gooaq", split="train").select(range(2_000_000, 2_100_000))
        docs = {str(i): answer for i, answer in enumerate(rows["answer"])}
        by_answer = {}
        for doc_id, answer in docs.items():
            by_answer.setdefault(answer, set()).add(doc_id)
        keep = sorted(random.Random(0).sample(range(len(rows)), GOOAQ_QUERIES))
        queries = [(str(i), rows[i]["question"]) for i in keep]
        return docs, queries, {str(i): by_answer[answer] for i, answer in enumerate(rows["answer"])}
    corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus")
    docs = {d: f"{t} {x}".strip() for d, t, x in zip(corpus["_id"], corpus["title"], corpus["text"])}
    qrels = {}
    for row in load_dataset(f"mteb/{name}", "default", split="test"):
        if row["score"] > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    queries = load_dataset(f"mteb/{name}", "queries", split="queries")
    queries = [(q, text) for q, text in zip(queries["_id"], queries["text"]) if q in qrels]
    return docs, queries, qrels


def prepare(name):
    root = ROOT / name
    docs, queries, qrels = load(name)
    encoder = SentenceTransformer(EMBEDDER, model_kwargs={"torch_dtype": torch.float16})
    db = lancedb.connect(str(root / "db"))
    if not (root / "db" / "docs.lance").exists():
        ids, texts = list(docs), list(docs.values())
        vectors = encoder.encode(texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True)
        table = db.create_table("docs", pa.table({
            "id": ids, "text": texts,
            "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel(), pa.float32()), vectors.shape[1]),
        }))
        table.create_fts_index("text")
    table = db.open_table("docs")
    # ANN like a production table; refine_factor re-scores the top candidates with full vectors.
    if not any(i.index_type == "IVF_PQ" for i in table.list_indices()):
        table.create_index("vector", config=IvfPq(distance_type="cosine"))
    embeddings = encoder.encode([QUERY_PREFIX + q for _, q in queries], batch_size=512, normalize_embeddings=True)

    def retrieve(i):
        (qid, query), vector = queries[i], embeddings[i]
        terms = MatchQuery(query, "text")  # bag-of-words BM25; a raw string would parse quotes as phrases
        vec = table.search(vector).distance_type("cosine").nprobes(NPROBES).refine_factor(REFINE)
        if name in EXACT_SEARCH:
            vec = vec.bypass_vector_index()
        vec = vec.limit(DEPTH).with_row_id(True).select(["id"]).to_arrow()
        fts = table.search(terms, query_type="fts").limit(DEPTH).with_row_id(True).select(["id"]).to_arrow()
        # Same result as a hybrid query with RRFReranker, without running the vector search twice.
        rrf = RRFReranker().rerank_hybrid(query, vec, fts)["id"].to_pylist()
        vec, fts = vec["id"].to_pylist(), fts["id"].to_pylist()
        return {"qid": qid, "query": query, "relevant": sorted(qrels[qid]), "vector": vec, "fts": fts,
                "rrf": rrf, "docs": list(dict.fromkeys(vec + fts))}

    with ThreadPoolExecutor(32) as pool:
        rows = list(pool.map(retrieve, range(len(queries)), chunksize=64))
    write_jsonl(root / "candidates.jsonl", rows)
    needed = sorted({d for row in rows for d in row["docs"]})
    lancedb.connect(str(root / "db")).create_table(
        "candidates", pa.table({"id": needed, "text": [docs[d] for d in needed]}), mode="overwrite")
    print(f"{name}: {len(rows)} queries, {len(needed)} candidate docs")


def make_reranker(model):
    if model in CROSS_ENCODERS:
        return CrossEncoderReranker(CROSS_ENCODERS[model], batch_size=128, max_length=MAX_LENGTH,
                                    model_kwargs={"torch_dtype": torch.bfloat16})
    if model in LISTWISE:
        return JinaV3Reranker(LISTWISE[model])
    if model == "jev":
        return TypeSafeReranker(JEV_MODEL, batch_size=40, max_concurrency=4)
    raise ValueError(model)


def rerank_scores(reranker, query, texts):
    """Score through the LanceDB reranker, then restore input order."""
    table = pa.table({"text": texts, "pos": list(range(len(texts))),
                      "_distance": pa.array([0.0] * len(texts), pa.float32())})
    ranked = reranker.rerank_vector(query, table).sort_by("pos")
    return ranked["_relevance_score"].to_pylist()


def score(name, model):
    root = ROOT / name
    rows = read_jsonl(root / "candidates.jsonl")
    texts = dict(zip(*lancedb.connect(str(root / "db")).open_table("candidates").to_arrow()
                     .select(["id", "text"]).to_pydict().values()))
    if model in COLBERTS:
        return score_colbert(root, rows, texts, model)
    out = root / "scores" / f"{model}.jsonl"
    done = len(read_jsonl(out)) if out.exists() else 0
    reranker = make_reranker(model)

    def one(i):
        row = rows[i]
        for attempt in range(1, 6):
            started = time.perf_counter()
            try:
                scores = rerank_scores(reranker, row["query"], [texts[d] for d in row["docs"]])
                return {"i": i, "scores": scores, "seconds": time.perf_counter() - started}
            except TypeSafeRateLimitError:
                time.sleep(15 * attempt)  # the wait is not part of the measured latency
        raise RuntimeError(f"query {i}: still rate limited after 5 attempts")

    # GPU models score one query at a time for honest latency. Jev overlaps 8 queries x 4 requests,
    # which stays under TypeSafe's rate limit.
    workers = 8 if model == "jev" else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f, ThreadPoolExecutor(workers) as pool:
        for n, record in enumerate(pool.map(one, range(done, len(rows))), done + 1):
            f.write(json.dumps(record) + "\n")
            if n % 500 == 0:
                f.flush()
                print(f"{name}/{model}: {n}/{len(rows)}", flush=True)


def score_colbert(root, rows, texts, model):
    """Store pooled doc multivectors in LanceDB, then rerank each query by native MaxSim over its candidates."""
    # PyLate pins sentence-transformers 5.3, too old for Qwen3/mxbai CrossEncoders, so ColBERT runs in its own env.
    from pylate import models as pylate_models, rank as pylate_rank
    colbert = pylate_models.ColBERT(COLBERTS[model], device="cuda", model_kwargs={"torch_dtype": torch.float16},
                                    **COLBERT_KWARGS.get(model, {}))
    ids = sorted(texts)
    db = lancedb.connect(str(root / "db"))
    unpooled = colbert.encode([texts[d] for d in ids], is_query=False, batch_size=256, show_progress_bar=True)
    for factor in POOL_FACTORS:
        out = root / "scores" / f"{model}-pool{factor}.jsonl"
        if out.exists():
            continue
        encoded = [pool_tokens(e, factor) for e in unpooled]
        dim = encoded[0].shape[1]
        flat = pa.array(np.concatenate(encoded).ravel().astype(np.float16), pa.float16())
        tokens = pa.FixedSizeListArray.from_arrays(flat, dim)
        offsets = pa.array(np.cumsum([0] + [len(e) for e in encoded]), pa.int32())
        table = db.create_table(f"{model}-pool{factor}", pa.table({
            "id": ids, "vectors": pa.ListArray.from_arrays(offsets, tokens),
        }), mode="overwrite")
        table.create_scalar_index("id")
        records = []
        for i, row in enumerate(rows):
            started = time.perf_counter()
            query = colbert.encode([row["query"]], is_query=True, show_progress_bar=False)[0]
            where = "id IN (" + ",".join(f"'{d}'" for d in row["docs"]) + ")"
            hits = (table.search(query, vector_column_name="vectors").distance_type("cosine")
                    .where(where, prefilter=True).limit(len(row["docs"])).select(["id"]).to_list())
            by_id = {h["id"]: -h["_distance"] for h in hits}
            record = {"i": i, "scores": [by_id[d] for d in row["docs"]], "seconds": time.perf_counter() - started}
            if factor == 1:  # on the fly: encode this query's candidates at query time, like a cross-encoder
                started = time.perf_counter()
                query = colbert.encode([row["query"]], is_query=True, show_progress_bar=False)
                docs = colbert.encode([texts[d] for d in row["docs"]], is_query=False, batch_size=128,
                                      show_progress_bar=False)
                pylate_rank.rerank([row["docs"]], query, [docs], device="cuda")
                record["seconds_on_the_fly"] = time.perf_counter() - started
            records.append(record)
        write_jsonl(out, records)
        storage = {"tokens_per_doc": len(tokens) / len(ids), "bytes_per_doc_fp16": 2 * len(tokens) * dim / len(ids)}
        out.with_suffix(".storage.json").write_text(json.dumps(storage))
        print(f"{root.name}/{model}-pool{factor}: done", flush=True)


def pool_tokens(tokens, factor):
    """PyLate's hierarchical token pooling (keeps the first token), with distances clipped at zero.

    PyLate's own version passes 1 - cos straight to scipy, and rounding on identical tokens gives -1e-7,
    which ward linkage rejects.
    """
    keep, rest = tokens[:1], tokens[1:].astype(np.float32)
    clusters = max(len(rest) // factor, 1)
    if clusters >= len(rest):
        return tokens
    distance = np.clip(1 - rest @ rest.T, 0, None)[np.triu_indices(len(rest), k=1)]
    labels = hierarchy.fcluster(hierarchy.linkage(distance, "ward"), t=clusters, criterion="maxclust")
    pooled = [rest[labels == c].mean(0) for c in np.unique(labels)]
    return np.vstack([keep, np.stack(pooled).astype(tokens.dtype)])


def pool(row, mode, depth):
    vec, fts = row["vector"][:depth], row["fts"][:depth]
    return {"vector": vec, "fts": fts, "hybrid": list(dict.fromkeys(vec + fts))}[mode]


def rank(row, mode, depth, scores):
    """Top list for one query: retrieval order (RRF for hybrid) without scores, else reranked candidates."""
    if scores is None:
        return row["rrf"] if mode == "hybrid" else pool(row, mode, depth)
    by_doc = dict(zip(row["docs"], scores))
    return sorted(pool(row, mode, depth), key=lambda d: -by_doc[d])


def metrics(rows, records, mode, depth):
    hits, ndcg, ceiling = {1: 0, 5: 0, 10: 0}, 0.0, 0
    for row, record in zip(rows, records):
        relevant = set(row["relevant"])
        top = rank(row, mode, depth, record and record["scores"])
        for k in hits:
            hits[k] += any(d in relevant for d in top[:k])
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(top[:10]) if d in relevant)
        ndcg += dcg / sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
        ceiling += any(d in relevant for d in pool(row, mode, depth))
    n = len(rows)
    return {**{f"hit@{k}": 100 * v / n for k, v in hits.items()}, "ndcg@10": 100 * ndcg / n,
            "candidate_hit": 100 * ceiling / n}


def report():
    summary = {}
    for name in DATASETS:
        root = ROOT / name
        if not (root / "candidates.jsonl").exists():
            continue
        rows = read_jsonl(root / "candidates.jsonl")
        entries = {"none": None, **{p.stem: read_jsonl(p) for p in sorted((root / "scores").glob("*.jsonl"))}}
        for model, records in entries.items():
            if records is not None and len(records) != len(rows):
                print(f"skip {name}/{model}: {len(records)}/{len(rows)} queries")
                continue
            records = records or [None] * len(rows)
            result = {f"{mode}@{depth}": metrics(rows, records, mode, depth)
                      for mode in ("vector", "fts", "hybrid") for depth in (DEPTHS if records[0] else [DEPTH])}
            if records[0]:
                result["latency_ms"] = latency([r["seconds"] for r in records])
                result["pairs_per_second"] = sum(len(r["docs"]) for r in rows) / sum(r["seconds"] for r in records)
                if "seconds_on_the_fly" in records[0]:
                    result["latency_on_the_fly_ms"] = latency([r["seconds_on_the_fly"] for r in records])
                storage = root / "scores" / f"{model}.storage.json"
                if storage.exists():
                    result["storage"] = json.loads(storage.read_text())
            summary.setdefault(name, {})[model] = result
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    hybrid = lambda r: r[f"hybrid@{DEPTH}"]["hit@10"]
    header = ["| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines = []
    for name, models in summary.items():
        lines += [f"## {name}", "", *header]
        lines += [row(m, r) for m, r in sorted(models.items(), key=lambda m: -hybrid(m[1]))] + [""]
    (RESULTS / "README.md").write_text("\n".join(lines))


def latency(seconds):
    return {"p50": 1000 * np.median(seconds), "p95": 1000 * np.percentile(seconds, 95)}


def row(model, r):
    cells = [f"{r[f'{mode}@{DEPTH}'][f'hit@{k}']:.2f}" for mode in ("vector", "fts", "hybrid") for k in (5, 10)]
    latency = f"{r['latency_ms']['p50']:.0f}" if "latency_ms" in r else "-"
    return f"| {model} | " + " | ".join(cells) + f" | {latency} |"


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "score", "report"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=[*CROSS_ENCODERS, *LISTWISE, *COLBERTS, "jev"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.dataset)
    elif args.stage == "score":
        score(args.dataset, args.model)
    else:
        report()
