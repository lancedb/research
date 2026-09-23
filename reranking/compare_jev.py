"""Reproducible GooAQ comparison with shared candidates and resumable model runs."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "modernbert": "ayushexel/reranker-ModernBERT-base-gooaq-1-epoch-1995000",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def candidates(row, method, k):
    vector, fts = row["vector"][:4*k], row["fts"][:4*k]
    if method == "vector":
        return vector
    if method == "fts":
        return fts
    return list(dict.fromkeys(vector + fts))


def summarize(rows, predictions):
    output = []
    for method in ("vector", "fts", "hybrid"):
        for k in (5, 10):
            hits = ceiling = 0
            for row, prediction in zip(rows, predictions):
                pool = candidates(row, method, k)
                if prediction is not None:
                    pool = sorted(pool, key=lambda doc: -prediction[str(doc)])
                hits += any(row["documents"][str(doc)] == row["answer"] for doc in pool[:k])
                ceiling += any(row["documents"][str(doc)] == row["answer"] for doc in pool)
            # Hybrid requires a merge ranker: no-reranker baseline only has vector/FTS.
            if method == "hybrid" and predictions[0] is None:
                continue
            output.append({"method": method, "k": k, "queries": len(rows), "hits": hits,
                           "hit_rate_percent": 100 * hits / len(rows),
                           "candidate_hit_rate_percent": 100 * ceiling / len(rows)})
    return output


def load_corpus(root, offset, size):
    """Read just the required Parquet row groups, not the 2M training rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset
    from huggingface_hub import HfApi, HfFileSystem
    corpus_path = root / f"corpus-{offset}-{size}.parquet"
    revision_path = corpus_path.with_suffix(".revision")
    if not corpus_path.exists():
        revision = HfApi().dataset_info("sentence-transformers/gooaq").sha
        fs = HfFileSystem()
        files = sorted(fs.glob(f"datasets/sentence-transformers/gooaq@{revision}/pair/*.parquet"))
        chunks, position = [], 0
        for filename in files:
            with fs.open(filename, "rb") as source:
                parquet = pq.ParquetFile(source)
                for group in range(parquet.metadata.num_row_groups):
                    count = parquet.metadata.row_group(group).num_rows
                    start, end = max(offset, position), min(offset + size, position + count)
                    if start < end:
                        chunks.append(parquet.read_row_group(group).slice(start-position, end-start))
                    position += count
                    if position >= offset + size:
                        break
            if position >= offset + size:
                break
        if not chunks or sum(len(chunk) for chunk in chunks) != size:
            raise ValueError("Requested corpus slice is outside the dataset")
        temporary = corpus_path.with_suffix(".tmp")
        pq.write_table(pa.concat_tables(chunks), temporary)
        revision_path.write_text(revision)
        temporary.replace(corpus_path)
    return Dataset(pq.read_table(corpus_path)), revision_path.read_text().strip()


def prepare(args, root):
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer
    config = {"dataset": "sentence-transformers/gooaq", "offset": args.offset,
              "corpus_size": args.corpus_size, "queries": args.queries,
              "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
              "retrieval": "exact L2 vector + native LanceDB BM25", "overfetch": 4}
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["config"] != config:
            raise ValueError("Cache configuration differs; choose another --cache directory")
        if (root / "candidates.json").exists():
            return json.loads((root / "candidates.json").read_text())
    print("Loading GooAQ held-out corpus", flush=True)
    data, revision = load_corpus(root, args.offset, args.corpus_size)
    if len(data) != args.corpus_size or any(not x["answer"] or not x["question"] for x in data):
        raise ValueError("Incomplete corpus or empty question/answer")
    encoder = SentenceTransformer(config["embedding_model"])
    db = lancedb.connect(str(root / "db"))
    if manifest.exists():
        table = db.open_table("gooaq")
    else:
        print(f"Embedding {len(data)} answers", flush=True)
        vectors = encoder.encode(data["answer"], batch_size=128, show_progress_bar=True)
        schema = pa.schema([("id", pa.int64()), ("answer", pa.string()),
                            ("vector", pa.list_(pa.float32(), vectors.shape[1]))])
        table = db.create_table("gooaq", schema=schema, mode="overwrite")
        for start in range(0, len(data), 2048):
            table.add([{"id": i, "answer": data[i]["answer"], "vector": vectors[i].tolist()}
                       for i in range(start, min(start + 2048, len(data)))])
        table.create_fts_index("answer")
        save(manifest, {"config": config, "dataset_fingerprint": data._fingerprint, "dataset_revision": revision})
    queries = data.select(range(args.queries))
    embeddings = encoder.encode(queries["question"], batch_size=128)
    rows = []
    for i, (example, embedding) in enumerate(zip(queries, embeddings)):
        v = table.search(embedding).limit(40).select(["id", "answer", "_distance"]).to_list()
        f = table.search(example["question"], query_type="fts").limit(40).select(["id", "answer", "_score"]).to_list()
        rows.append({"query": example["question"], "answer": example["answer"],
                     "vector": [x["id"] for x in v], "fts": [x["id"] for x in f],
                     "documents": {str(x["id"]): x["answer"] for x in v+f}})
        if (i+1) % 100 == 0:
            print(f"Retrieved {i+1}/{len(queries)} queries", flush=True)
    save(root / "candidates.json", rows)
    return rows


def run(args):
    root = Path(args.cache)
    root.mkdir(parents=True, exist_ok=True)
    rows = prepare(args, root)
    manifest = json.loads((root / "manifest.json").read_text())
    result = {"created_at": datetime.now(timezone.utc).isoformat(), **manifest,
              "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("lancedb", "sentence-transformers", "datasets", "typesafe-sdk")},
              "results": {}}
    dataset_hash = hashlib.sha256((root / "candidates.json").read_bytes()).hexdigest()
    result["candidate_sha256"] = dataset_hash
    result["jev_workers"] = args.workers
    result["jev_batch_size"] = 40
    for name in args.models:
        predictions, latencies, resolved = [], [], set()
        device = None
        if name == "none":
            predictions = [None] * len(rows)
        else:
            if name == "jev":
                from jev_reranker import JevReranker, QUESTION
                model = JevReranker(model=args.jev_model, api_key_file=args.api_key_file, workers=args.workers)
                identity = {"model": args.jev_model, "question": QUESTION, "protocol": "query-state-candidate-per-question-v1", "batch_size": 40}
            else:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(MODELS[name])
                device = str(model.device)
                identity = {"model": MODELS[name]}
            cache_key = hashlib.sha256(json.dumps([dataset_hash, identity], sort_keys=True).encode()).hexdigest()[:16]
            for i, row in enumerate(rows):
                path = root / "scores" / f"{name}-{cache_key}" / f"{i}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    ids = list(row["documents"])
                    docs = [row["documents"][doc] for doc in ids]
                    started = time.perf_counter()
                    if name == "jev":
                        scores = model.score_documents(row["query"], docs)
                        versions = sorted(model.resolved_models)
                    else:
                        scores = model.predict([(row["query"], doc) for doc in docs], show_progress_bar=False).tolist()
                        versions = [MODELS[name]]
                    record = {"scores": dict(zip(ids, scores)), "seconds": time.perf_counter()-started,
                              "models": versions}
                    save(path, record)
                predictions.append(record["scores"])
                latencies.append(record["seconds"])
                resolved.update(record["models"])
                if (i+1) % 25 == 0:
                    print(f"{name}: {i+1}/{len(rows)} queries", flush=True)
        result["results"][name] = {"metrics": summarize(rows, predictions), "resolved_models": sorted(resolved), "device": device,
                                   "scoring_seconds_p50": float(np.median(latencies)) if latencies else None,
                                   "scoring_seconds_p95": float(np.percentile(latencies, 95)) if latencies else None}
        if name == "jev":
            model.client.close()
        save(Path(args.output), result)
        print(json.dumps(result["results"][name], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=2_000_000)
    parser.add_argument("--corpus-size", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=2_000)
    parser.add_argument("--models", nargs="+", choices=["none", *MODELS, "jev"], default=["none", "minilm", "modernbert", "jev"])
    parser.add_argument("--jev-model", default="jev-1.13.0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key-file")
    parser.add_argument("--cache", default="reranking/.benchmark-cache")
    parser.add_argument("--output", default="reranking/results/jev-comparison.json")
    args = parser.parse_args()
    if not 0 < args.queries <= args.corpus_size or args.offset < 2_000_000:
        parser.error("Use 0 < queries <= corpus-size and offset >= 2000000 (held out from training)")
    run(args)
