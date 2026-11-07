"""Off-the-shelf reranker benchmark: every model reorders the same LanceDB candidates.

    python bench.py prepare --dataset nq          # embed, index, retrieve top-50 vector/FTS/hybrid
    python bench.py score --dataset nq --model qwen3-4b
    python bench.py report                        # runs/results/summary.json + runs/results/README.md
"""
import argparse
import hashlib
from email.utils import parsedate_to_datetime
from itertools import islice
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_state import (atomic_json, bind_manifest, digest, execute_scores, exclusive, file_digest,
                       packages, score_records, timing_summary)

from dotenv import load_dotenv

# Keep credentials local to this benchmark; explicit environment variables win.
load_dotenv(Path(__file__).with_name(".env"), override=False)

import lancedb
import numpy as np
import pyarrow as pa
import torch
from datasets import load_dataset
from scipy.cluster import hierarchy
from lancedb.index import IvfPq
from lancedb.query import MatchQuery
from lancedb.rerankers import CrossEncoderReranker, RRFReranker, TypeSafeReranker, VoyageAIReranker
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from typesafe_sdk import TypeSafeRateLimitError

try:  # only needed for the Voyage models
    from voyageai import error as voyage_errors
    VoyageRateLimitError = voyage_errors.RateLimitError
    VOYAGE_TRANSIENT_ERRORS = (voyage_errors.Timeout, voyage_errors.APIConnectionError,
                              voyage_errors.TryAgain, voyage_errors.ServerError, voyage_errors.ServiceUnavailableError)
except ImportError:
    class VoyageRateLimitError(Exception):
        pass
    VOYAGE_TRANSIENT_ERRORS = ()

ROOT = Path(os.environ.get("BENCH_ROOT", "cache"))
RESULTS = Path(os.environ.get("BENCH_RESULTS", "runs/results"))
REVISIONS = json.loads(Path(__file__).with_name("revisions.json").read_text())

def revision(name, kind="models"):
    return REVISIONS[kind][name]

CHUNK_SIZE = int(os.environ.get("BENCH_CHUNK_SIZE", "4096"))
EMBED_BATCH_SIZE = int(os.environ.get("BENCH_EMBED_BATCH_SIZE", "512"))
API_WORKERS = int(os.environ.get("BENCH_API_WORKERS", "8"))
WARMUP_QUERIES = 2
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
# jina-colbert-v2's auto_map points to this separate implementation repository.
JINA_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"  # jinaai/xlm-roberta-flash-implementation
COLBERT_KWARGS = {  # from the model card
    "jina-colbert-v2": {"query_prefix": "[QueryMarker]", "document_prefix": "[DocumentMarker]",
                        "attend_to_expansion_tokens": True, "trust_remote_code": True,
                        "config_kwargs": {"code_revision": JINA_CODE_REVISION}},
}
POOL_FACTORS = [1, 2, 4]
VOYAGE = {"voyage-rerank-3": "rerank-3", "voyage-rerank-3-lite": "rerank-3-lite"}
VOYAGE_TPM = int(os.environ.get("VOYAGE_TPM", "1800000"))  # conservative default; set from the actual project limit
JEV_MODEL = "jev-1.13.0"
JEV_PROMPTS = {  # "jev" keeps TypeSafeReranker's default: "does the document answer or directly address the query?"
    "jev": {},
    "jev-relevance": {
        "instructions": "Is the document relevant to the query?",
        "criteria": {"true": "The document has information that helps answer the query, even if only part of it.",
                     "false": "The document is unrelated to the query, or only shares keywords with it."},
    },
    "jev-multihop": {  # domain-specific: HotpotQA questions need a bridge passage plus an answer passage
        "instructions": "Could the document be one of the passages needed to answer the query? Some queries "
                        "need two passages, for example one that identifies an entity and one that describes it.",
        "criteria": {"true": "The document states a fact the query depends on, including a fact about an entity "
                             "the query only refers to indirectly.",
                     "false": "The document is unrelated to the query and to the entities it depends on."},
    },
}


class VoyageReranker(VoyageAIReranker):
    """LanceDB's VoyageAIReranker with a tokens-per-minute throttle and a billed-token counter.

    The throttle keeps the run under the project's rate limit; like the Jev rate-limit waits,
    time spent waiting is not part of the measured latency.
    """

    def __init__(self, model_name):
        super().__init__(model_name)
        self.tokens = 0
        self._usage = threading.local()
        self._lock = threading.Lock()
        self._window = []  # (time, estimated tokens) of requests in the last minute
        self._per_doc = 300.0  # running estimate of billed tokens per document

    def throttle(self, n_docs):
        """Block until a request with n_docs documents fits in the per-minute token budget."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._window = [(t, n) for t, n in self._window if now - t < 60]
                estimate = min(self._per_doc * n_docs, VOYAGE_TPM)
                if VOYAGE_TPM <= 0:
                    raise ValueError("VOYAGE_TPM must be positive")
                if sum(n for _, n in self._window) + estimate <= VOYAGE_TPM:
                    self._window.append((now, estimate))
                    return
                wait = 60 - (now - self._window[0][0]) + 0.1
            time.sleep(min(wait, 5))

    def _rerank(self, result_set, query):
        docs = result_set[self.column].to_pylist()
        response = self._client.rerank(query=query, documents=docs, model=self.model_name, truncation=self.truncation)
        with self._lock:
            self.tokens += response.total_tokens
            self._usage.tokens = response.total_tokens
            self._per_doc = 0.9 * self._per_doc + 0.1 * response.total_tokens / max(len(docs), 1)
        scores = [0.0] * len(docs)
        for r in response.results:
            scores[r.index] = r.relevance_score
        return result_set.append_column("_relevance_score", pa.array(scores, type=pa.float32()))


class JinaV3Reranker(CrossEncoderReranker):
    """jina-reranker-v3 is listwise and ships its own `rerank`, not a CrossEncoder head."""

    @property
    def model(self):
        if not hasattr(self, "_jina"):
            self._jina = AutoModel.from_pretrained(self.model_name, dtype=torch.bfloat16, trust_remote_code=True,
                                                       revision=revision(self.model_name),
                                                       code_revision=revision(self.model_name))
            self._jina.to(self.device).eval()
        return self._jina

    def _rerank(self, result_set, query):
        ranked = self.model.rerank(query, result_set[self.column].to_pylist(), max_doc_length=MAX_LENGTH)
        scores = [0.0] * len(result_set)
        for item in ranked:
            scores[item["index"]] = item["relevance_score"]
        return result_set.append_column("_relevance_score", pa.array(scores, type=pa.float32()))


class Corpus:
    """Keep Hugging Face corpus text on disk instead of building a multi-million-row dict."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def items(self):
        for batch in self.dataset.iter(batch_size=CHUNK_SIZE):
            yield from ((d, f"{t} {x}".strip()) for d, t, x in
                        zip(batch["_id"], batch["title"], batch["text"]))


def load(name):
    """Return {doc_id: text}, [(query_id, query)], {query_id: set(relevant doc_ids)}."""
    if name == "gooaq":
        # Rows 2.0M-2.1M: held out from the fine-tuned models in ../finetune, so results stay reusable there.
        rows = load_dataset("sentence-transformers/gooaq", split="train", revision=revision("sentence-transformers/gooaq", "datasets")).select(range(2_000_000, 2_100_000))
        docs = {str(i): answer for i, answer in enumerate(rows["answer"])}
        by_answer = {}
        for doc_id, answer in docs.items():
            by_answer.setdefault(answer, set()).add(doc_id)
        keep = sorted(random.Random(0).sample(range(len(rows)), GOOAQ_QUERIES))
        queries = [(str(i), rows[i]["question"]) for i in keep]
        return docs, queries, {str(i): by_answer[answer] for i, answer in enumerate(rows["answer"])}
    corpus = load_dataset(f"mteb/{name}", "corpus", split="corpus", revision=revision(f"mteb/{name}", "datasets"))
    docs = Corpus(corpus)
    qrels = {}
    for row in load_dataset(f"mteb/{name}", "default", split="test", revision=revision(f"mteb/{name}", "datasets")):
        if row["score"] > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    queries = load_dataset(f"mteb/{name}", "queries", split="queries", revision=revision(f"mteb/{name}", "datasets"))
    queries = [(q, text) for q, text in zip(queries["_id"], queries["text"]) if q in qrels]
    return docs, queries, qrels


def small_sample(docs, queries, qrels, query_limit, corpus_limit):
    """Deterministic smoke sample, retaining all positives for the selected queries."""
    if query_limit < 1 or corpus_limit < DEPTH:
        raise ValueError(f"Use at least 1 query and {DEPTH} documents")
    queries = random.Random(0).sample(queries, min(query_limit, len(queries)))
    positives = {d for qid, _ in queries for d in qrels[qid]}
    if len(positives) > corpus_limit:
        raise ValueError("Corpus limit is too small to retain the relevant documents")
    negatives = sorted(set(docs) - positives)
    selected = positives | set(random.Random(0).sample(negatives, min(corpus_limit - len(positives), len(negatives))))
    return {d: docs[d] for d in sorted(selected)}, queries, {qid: qrels[qid] for qid, _ in queries}


def chunks(items, size=CHUNK_SIZE):
    if size < 1:
        raise ValueError("Chunk size must be positive")
    it = iter(items)
    while batch := list(islice(it, size)):
        yield batch


def prepare(name, *, query_limit=None, corpus_limit=None):
    root = ROOT / name
    with exclusive(root / ".prepare.lock"):
        return _prepare(name, query_limit=query_limit, corpus_limit=corpus_limit)


def _prepare(name, *, query_limit=None, corpus_limit=None):
    root = ROOT / name
    if query_limit is not None and query_limit < 1:
        raise ValueError("Query limit must be positive")
    if corpus_limit is not None and query_limit is None:
        raise ValueError("Supply a query limit for a smoke run")
    config = {"format": 2, "dataset": name, "revisions": REVISIONS["datasets"],
              "embedder": EMBEDDER, "embedder_revision": revision(EMBEDDER),
              "query_limit": query_limit, "corpus_limit": corpus_limit, "seed": 0,
              "depth": DEPTH, "query_prefix": QUERY_PREFIX, "nprobes": NPROBES,
              "refine": REFINE, "exact": name in EXACT_SEARCH or corpus_limit is not None,
              "precision": "float16", "embedding_batch_size": 32 if corpus_limit else EMBED_BATCH_SIZE,
              "chunk_size": CHUNK_SIZE, "packages": packages()}
    corpus_source = os.environ.get("BENCH_CORPUS_CACHE")
    shared_table = None
    if corpus_source:
        if corpus_limit is not None:
            raise ValueError("A reduced smoke corpus cannot use BENCH_CORPUS_CACHE")
        source = Path(corpus_source) / name
        previous = json.loads((source / "prepare.json").read_text())
        # Sampling queries does not change the document embeddings or indexes.
        comparable = lambda c: {k: v for k, v in c.items() if k not in {"query_limit", "reused_corpus"}}
        if comparable(previous) != comparable(config):
            raise ValueError("Shared corpus configuration differs from this run")
        candidate_fingerprint(source)  # requires a completed, intact source preparation
        shared_table = lancedb.connect(str(source / "db")).open_table("docs")
        config["reused_corpus"] = {"configuration": digest(comparable(previous)),
                                   "version": shared_table.version, "rows": shared_table.count_rows()}
    manifest = root / "prepare.json"
    if not manifest.exists() and (root / "db").exists():
        raise ValueError("Unversioned candidate cache; choose a new run directory")
    bind_manifest(manifest, config)
    if (root / "candidates.meta.json").exists():
        candidate_fingerprint(root)
        print(f"{name}: reusing frozen candidates", flush=True)
        return
    docs, queries, qrels = load(name)
    if corpus_limit is not None:
        docs, queries, qrels = small_sample(dict(docs.items()), queries, qrels, query_limit, corpus_limit)
    elif query_limit is not None:
        queries = random.Random(0).sample(queries, min(query_limit, len(queries)))
    encoder = SentenceTransformer(EMBEDDER, revision=revision(EMBEDDER),
                                  model_kwargs={"torch_dtype": torch.float16})
    db = lancedb.connect(str(root / "db"))
    table = shared_table if shared_table is not None else (
        db.open_table("docs") if (root / "db" / "docs.lance").exists() else None)
    done = table.count_rows() if table is not None else 0
    if shared_table is not None and done != len(docs):
        raise ValueError("Shared corpus is incomplete")
    if done > len(docs):
        raise ValueError("Corpus checkpoint contains too many rows")
    for batch in chunks(islice(docs.items(), done, None)):
        ids, texts = map(list, zip(*batch))
        vectors = encoder.encode(texts, batch_size=32 if corpus_limit else EMBED_BATCH_SIZE,
                                 normalize_embeddings=True, show_progress_bar=False)
        data = pa.table({"id": ids, "text": texts,
                         "vector": pa.FixedSizeListArray.from_arrays(
                             pa.array(vectors.ravel(), pa.float32()), vectors.shape[1])})
        if table is None:
            table = db.create_table("docs", data)
        else:
            table.add(data)
        done += len(batch)
        print(f"{name}: embedded {done}/{len(docs)} documents", flush=True)
    indices = table.list_indices()
    if shared_table is None and not any(i.index_type == "FTS" for i in indices):
        table.create_fts_index("text")
    if shared_table is None and not config["exact"] and not any(i.index_type == "IVF_PQ" for i in indices):
        table.create_index("vector", config=IvfPq(distance_type="cosine"))
    if shared_table is None and not any("id" in i.columns for i in table.list_indices()):
        table.create_scalar_index("id")
    if shared_table is None:
        table.optimize()
    embeddings = encoder.encode([QUERY_PREFIX + q for _, q in queries], batch_size=512, normalize_embeddings=True)

    def retrieve(i):
        (qid, query), vector = queries[i], embeddings[i]
        terms = MatchQuery(query, "text")
        vec = table.search(vector).distance_type("cosine").nprobes(NPROBES).refine_factor(REFINE)
        if config["exact"]:
            vec = vec.bypass_vector_index()
        vec = vec.limit(DEPTH).with_row_id(True).select(["id"]).to_arrow()
        fts = table.search(terms, query_type="fts").limit(DEPTH).with_row_id(True).select(["id"]).to_arrow()
        rrf = RRFReranker().rerank_hybrid(query, vec, fts)["id"].to_pylist()
        vec, fts = vec["id"].to_pylist(), fts["id"].to_pylist()
        return {"qid": qid, "query": query, "relevant": sorted(qrels[qid]), "vector": vec, "fts": fts,
                "rrf": rrf, "docs": list(dict.fromkeys(vec + fts))}

    with ThreadPoolExecutor(32) as pool:
        rows = list(pool.map(retrieve, range(len(queries))))
    needed = sorted({d for row in rows for d in row["docs"]})
    candidate_table = None
    content_hash = hashlib.sha256()
    for ids in chunks(needed):
        hits = table.search().where(id_filter(ids)).select(["id", "text"]).limit(len(ids)).to_list()
        by_id = {h["id"]: h["text"] for h in hits}
        texts = [by_id[d] for d in ids]
        content_hash.update(json.dumps(list(zip(ids, texts)), ensure_ascii=False).encode())
        data = pa.table({"id": ids, "text": texts})
        if candidate_table is None:
            candidate_table = db.create_table("candidates", data, mode="overwrite")
        else:
            candidate_table.add(data)
    candidate_table.create_scalar_index("id")
    candidate_table.optimize()
    write_jsonl(root / "candidates.jsonl", rows)
    atomic_json(root / "candidates.meta.json", {"prepare": digest(config),
                "rows_sha256": file_digest(root / "candidates.jsonl"), "text_sha256": content_hash.hexdigest(),
                "table_version": candidate_table.version, "documents": len(needed), "queries": len(rows)})
    print(f"{name}: {len(rows)} queries, {len(needed)} candidate docs", flush=True)


def id_filter(ids):
    return "id IN (" + ",".join("'" + d.replace("'", "''") + "'" for d in ids) + ")"


def candidate_fingerprint(root):
    meta = json.loads((root / "candidates.meta.json").read_text())
    table = lancedb.connect(str(root / "db")).open_table("candidates")
    if (meta["rows_sha256"] != file_digest(root / "candidates.jsonl") or
            meta["table_version"] != table.version or meta["documents"] != table.count_rows() or
            meta["prepare"] != digest(json.loads((root / "prepare.json").read_text()))):
        raise ValueError("Frozen candidates changed; use a new run directory")
    return digest(meta)


class CandidateTexts:
    def __init__(self, root):
        self.table = lancedb.connect(str(root / "db")).open_table("candidates")

    def fetch(self, ids):
        hits = self.table.search().where(id_filter(ids)).select(["id", "text"]).limit(len(ids)).to_list()
        by_id = {h["id"]: h["text"] for h in hits}
        return [by_id[d] for d in ids]


def make_reranker(model):
    if model in CROSS_ENCODERS:
        return CrossEncoderReranker(CROSS_ENCODERS[model], batch_size=128, max_length=MAX_LENGTH,
                                    revision=revision(CROSS_ENCODERS[model]),
                                    model_kwargs={"torch_dtype": torch.bfloat16})
    if model in LISTWISE:
        return JinaV3Reranker(LISTWISE[model])
    if model in VOYAGE:
        return VoyageReranker(VOYAGE[model])
    if model in JEV_PROMPTS:
        return TypeSafeReranker(JEV_MODEL, batch_size=40, max_concurrency=4, **JEV_PROMPTS[model])
    raise ValueError(model)


def rerank_scores(reranker, query, texts):
    """Score through the LanceDB reranker, then restore input order."""
    table = pa.table({"text": texts, "pos": list(range(len(texts))),
                      "_distance": pa.array([0.0] * len(texts), pa.float32())})
    ranked = reranker.rerank_vector(query, table).sort_by("pos")
    return ranked["_relevance_score"].to_pylist()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def score_config(root, model):
    model_id = {**CROSS_ENCODERS, **LISTWISE, **COLBERTS, **VOYAGE}.get(model, JEV_MODEL)
    config = {"format": 2, "candidates": candidate_fingerprint(root), "model": model,
            "model_id": model_id, "revision": REVISIONS["models"].get(model_id),
            "prompt": JEV_PROMPTS.get(model), "max_length": MAX_LENGTH,
            "precision": None if model in JEV_PROMPTS or model in VOYAGE else
                         "float16" if model in COLBERTS else "bfloat16",
            "warmup_queries": WARMUP_QUERIES, "packages": packages(),
            "api_workers": API_WORKERS if model in JEV_PROMPTS or model in VOYAGE else None,
            "voyage_tpm": VOYAGE_TPM if model in VOYAGE else None,
            "batch_size": 40 if model in JEV_PROMPTS else 128,
            "jev_concurrency": 4 if model in JEV_PROMPTS else None}
    if model == "jina-colbert-v2":
        config["code_revision"] = JINA_CODE_REVISION
    return config


def bind_scores(out, config):
    meta = out.with_suffix(".meta.json")
    if out.exists() and not meta.exists():
        raise ValueError(f"Unversioned scores: {out}; use a new run directory")
    bind_manifest(meta, config)


def response_status(exc):
    return (getattr(exc, "status_code", None) or getattr(exc, "status", None) or
            getattr(exc, "http_status", None) or getattr(getattr(exc, "response", None), "status_code", None))


def retryable(exc):
    status = response_status(exc)
    return (isinstance(exc, (TypeSafeRateLimitError, VoyageRateLimitError, TimeoutError, ConnectionError))
            or isinstance(exc, VOYAGE_TRANSIENT_ERRORS)
            or status == 429 or (isinstance(status, int) and 500 <= status < 600)
            or type(exc).__name__ in {"ConnectError", "ReadTimeout", "ConnectTimeout", "APIConnectionError",
                                      "TypeSafeAPITimeoutError"})


def retry_delay(exc, attempt):
    milliseconds = getattr(exc, "retry_after_ms", None)
    headers = getattr(exc, "headers", {}) or {}
    requested = headers.get("retry-after", headers.get("Retry-After"))
    delay = None
    if milliseconds is not None:
        delay = milliseconds / 1000
    elif requested is not None:
        try:
            delay = float(requested)
        except (ValueError, TypeError):
            try:
                delay = parsedate_to_datetime(requested).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                pass
    if delay is None or not np.isfinite(delay):
        rate_limited = response_status(exc) == 429 or isinstance(exc, (TypeSafeRateLimitError, VoyageRateLimitError))
        delay = min(60, (15 if rate_limited else 1) * 2 ** attempt)
    return max(0, min(delay, 120)) + random.random()


def score(name, model):
    root = ROOT / name
    with exclusive(root / "scores" / f".{model}.lock"):
        rows = read_jsonl(root / "candidates.jsonl")
        texts = CandidateTexts(root)
        config = score_config(root, model)
        if model in COLBERTS:
            return score_colbert(root, rows, texts, model, config)
        out = root / "scores" / f"{model}.jsonl"
        bind_scores(out, config)
        completed = score_records(out, rows, repair=True)
        if len(completed) == len(rows):
            save_usage(out, completed, model)
            return
        reranker = make_reranker(model)
        api = model in JEV_PROMPTS or model in VOYAGE
        if not api:
            sample = rows[0]
            passages = texts.fetch(sample["docs"])
            started = time.perf_counter()
            for _ in range(WARMUP_QUERIES):
                rerank_scores(reranker, sample["query"], passages)
            synchronize()
            atomic_json(out.with_suffix(".warmup.json"), {"seconds": time.perf_counter() - started,
                                                         "queries": WARMUP_QUERIES})

        def one(i):
            row = rows[i]
            passages = texts.fetch(row["docs"])
            overall = time.perf_counter()
            for attempt in range(5):
                if model in VOYAGE:
                    reranker.throttle(len(row["docs"]))
                if not api:
                    synchronize()
                started = time.perf_counter()
                try:
                    scores = rerank_scores(reranker, row["query"], passages)
                    if not api:
                        synchronize()
                    return {"i": i, "scores": scores, "seconds": time.perf_counter() - started,
                            "elapsed_seconds": time.perf_counter() - overall, "retries": attempt,
                            "tokens": getattr(reranker._usage, "tokens", 0) if model in VOYAGE else 0}
                except Exception as exc:
                    if not api or not retryable(exc) or attempt == 4:
                        raise
                    time.sleep(retry_delay(exc, attempt))
            raise AssertionError("unreachable")

        try:
            execute_scores(out, rows, one, workers=API_WORKERS if api else 1)
        finally:
            save_usage(out, score_records(out, rows, repair=True), model)


def save_usage(out, records, model):
    if model in VOYAGE:
        atomic_json(out.with_suffix(".tokens.json"), {"queries": len(records),
                    "tokens": sum(r.get("tokens", 0) for r in records),
                    "scope": "persisted successful responses; reconcile with provider billing after interruptions"})


def multivector_batch(ids, encoded, offset):
    dim = encoded[0].shape[1]
    lengths = [len(e) for e in encoded]
    flat = pa.array(np.concatenate(encoded).ravel().astype(np.float16), pa.float16())
    tokens = pa.FixedSizeListArray.from_arrays(flat, dim)
    offsets = pa.array(np.cumsum([0] + lengths), pa.int32())
    return pa.table({"id": ids, "vectors": pa.ListArray.from_arrays(offsets, tokens),
                     "seq": list(range(offset, offset + len(ids))), "token_count": lengths})


def score_colbert(root, rows, texts, model, config):
    """Chunk document encoding, checkpoint each query, and recover partial pooling writes."""
    from pylate import models as pylate_models, rank as pylate_rank
    paths = {f: root / "scores" / f"{model}-pool{f}.jsonl" for f in POOL_FACTORS}
    for f, out in paths.items():
        bind_scores(out, dict(config, pooling=f))
    todo = [f for f, out in paths.items() if len(score_records(out, rows, repair=True)) < len(rows)]
    if not todo:
        return
    model_kwargs = {"torch_dtype": torch.float16}
    if model == "jina-colbert-v2":
        model_kwargs["code_revision"] = JINA_CODE_REVISION
    colbert = pylate_models.ColBERT(COLBERTS[model], revision=revision(COLBERTS[model]), device="cuda",
                                    model_kwargs=model_kwargs, **COLBERT_KWARGS.get(model, {}))
    ids = sorted({d for r in rows for d in r["docs"]})
    db = lancedb.connect(str(root / "db"))
    bind_manifest(root / f"{model}.encoding.json", dict(config, pooling=POOL_FACTORS))
    tables = {f: db.open_table(f"{model}-pool{f}") if (root / "db" / f"{model}-pool{f}.lance").exists()
              else None for f in POOL_FACTORS}
    counts = {f: t.count_rows() if t is not None else 0 for f, t in tables.items()}
    done = min(counts.values())
    if max(counts.values()) > len(ids):
        raise ValueError("ColBERT checkpoint contains too many documents")
    # A process may die between committing the three pooling tables. Replay just that tail.
    for f, table in tables.items():
        if counts[f] > done:
            table.delete(f"seq >= {done}")
    for batch_ids in chunks(ids[done:], 256):
        unpooled = colbert.encode(texts.fetch(batch_ids), is_query=False, batch_size=256, show_progress_bar=False)
        for factor in POOL_FACTORS:
            encoded = [pool_tokens(e, factor) for e in unpooled]
            data = multivector_batch(batch_ids, encoded, done)
            if tables[factor] is None:
                tables[factor] = db.create_table(f"{model}-pool{factor}", data)
            else:
                tables[factor].add(data)
        done += len(batch_ids)
        print(f"{model}: encoded {done}/{len(ids)} candidate documents", flush=True)
    for factor in todo:
        out, table = paths[factor], tables[factor]
        if not any("id" in i.columns for i in table.list_indices()):
            table.create_scalar_index("id")
        table.optimize()
        token_count = sum(sum(b.column("token_count").to_pylist()) for b in
                          table.search().select(["token_count"]).limit(len(ids)).to_batches(batch_size=4096))
        dim = table.schema.field("vectors").type.value_type.list_size
        atomic_json(out.with_suffix(".storage.json"), {"tokens_per_doc": token_count / len(ids),
                    "bytes_per_doc_fp16": 2 * token_count * dim / len(ids)})

        def one(i):
            row = rows[i]
            synchronize()
            started = time.perf_counter()
            query = colbert.encode([row["query"]], is_query=True, show_progress_bar=False)[0]
            hits = (table.search(query, vector_column_name="vectors").distance_type("cosine")
                    .where(id_filter(row["docs"]), prefilter=True).limit(len(row["docs"])).select(["id"]).to_list())
            synchronize()
            by_id = {h["id"]: -h["_distance"] for h in hits}
            record = {"i": i, "scores": [by_id[d] for d in row["docs"]], "seconds": time.perf_counter() - started}
            if factor == 1:
                passages = texts.fetch(row["docs"])
                synchronize()
                started = time.perf_counter()
                query = colbert.encode([row["query"]], is_query=True, show_progress_bar=False)
                docs = colbert.encode(passages, is_query=False, batch_size=128, show_progress_bar=False)
                pylate_rank.rerank([row["docs"]], query, [docs], device="cuda")
                synchronize()
                record["seconds_on_the_fly"] = time.perf_counter() - started
            return record

        started = time.perf_counter()
        for _ in range(WARMUP_QUERIES):
            one(0)
        atomic_json(out.with_suffix(".warmup.json"), {"seconds": time.perf_counter() - started,
                                                     "queries": WARMUP_QUERIES})
        execute_scores(out, rows, one)
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
        entries = {"none": None}
        for p in sorted((root / "scores").glob("*.jsonl")):
            if p.name.endswith(".sessions.jsonl"):
                continue
            meta = p.with_suffix(".meta.json")
            if meta.exists() and json.loads(meta.read_text())["candidates"] != candidate_fingerprint(root):
                raise ValueError(f"Candidate mismatch: {p}")
            entries[p.stem] = score_records(p, rows)
        for model, records in entries.items():
            if records is not None and len(records) != len(rows):
                print(f"skip {name}/{model}: {len(records)}/{len(rows)} queries")
                continue
            records = records or [None] * len(rows)
            result = {f"{mode}@{depth}": metrics(rows, records, mode, depth)
                      for mode in ("vector", "fts", "hybrid") for depth in (DEPTHS if records[0] else [DEPTH])}
            if records[0]:
                result["latency_ms"] = latency([r["seconds"] for r in records])
                out = root / "scores" / f"{model}.jsonl"
                timing = timing_summary(out, records)
                result["timing"] = timing
                if timing["wall_seconds"]:
                    result["pairs_per_second"] = sum(len(r["docs"]) for r in rows) / timing["wall_seconds"]
                if "elapsed_seconds" in records[0]:
                    result["elapsed_latency_ms"] = latency([r["elapsed_seconds"] for r in records])
                if "seconds_on_the_fly" in records[0]:
                    result["latency_on_the_fly_ms"] = latency([r["seconds_on_the_fly"] for r in records])
                storage = root / "scores" / f"{model}.storage.json"
                if storage.exists():
                    result["storage"] = json.loads(storage.read_text())
            summary.setdefault(name, {})[model] = result
    RESULTS.mkdir(parents=True, exist_ok=True)
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


def complete_records(path):
    """Count finished score lines, dropping a last line cut off when a run was killed."""
    if not path.exists():
        return 0
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
        path.write_text("".join(lines))
    return len(lines)


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as f:
        f.writelines(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "score", "report"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=[*CROSS_ENCODERS, *LISTWISE, *COLBERTS, *JEV_PROMPTS, *VOYAGE])
    parser.add_argument("--queries", type=int, help="Pilot: limit queries while keeping the full corpus")
    parser.add_argument("--corpus-limit", type=int, help="Smoke only: also reduce the corpus")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.dataset, query_limit=args.queries, corpus_limit=args.corpus_limit)
    elif args.stage == "score":
        score(args.dataset, args.model)
    else:
        report()
