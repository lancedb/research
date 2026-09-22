# Jev reranker comparison — measured September 21, 2026

Fresh run on 100,000 GooAQ answers and 2,000 queries. Every model received the same candidates. Values are exact-answer-string hit rates (%); higher is better.

| System | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reranker | 82.10 | 89.05 | 60.15 | 67.20 | — | — |
| MS MARCO MiniLM | 81.30 | 86.40 | 69.60 | 76.60 | 79.90 | 84.75 |
| GooAQ-trained ModernBERT | 86.10 | 89.70 | 71.75 | 77.70 | 86.15 | 90.05 |
| Jev 1.13.0 | 86.95 | 92.20 | 70.80 | 77.20 | 87.25 | 92.35 |

## Scoring latency

| System | Median | p95 |
| --- | ---: | ---: |
| MS MARCO MiniLM | 58 ms | 78 ms |
| GooAQ-trained ModernBERT | 447 ms | 602 ms |
| Jev 1.13.0 | 186 ms | 387 ms |

Times cover scoring the union of up to 80 candidates per query, reused across retrieval modes and k values. They exclude query embedding and retrieval. Local models ran on Apple MPS, while Jev used its hosted API with batches of 40 independent questions. These are observations on different execution environments, not hardware-controlled model speed comparisons. A DNS interruption was resumed at query 1,745; failed-attempt downtime is excluded from these successful-query timings.

## Protocol and reproducibility

- Dataset: `sentence-transformers/gooaq`, rows 2,000,000–2,099,999; queries are the first 2,000 rows within that corpus. The first two million rows used for custom-model training are excluded.
- Retrieval: `all-MiniLM-L6-v2`, exact L2 vector search and native LanceDB BM25. Each source supplies 20 candidates at k=5 and 40 at k=10; hybrid deduplicates their union. No positive is inserted.
- Jev uses one fixed relevance question per candidate, batched with the query as shared state. No prompt tuning on these evaluation outcomes was performed.
- All systems completed every query. Full model identifiers, counts, dataset revision, candidate hash and package versions are in [jev-comparison.json](jev-comparison.json).
- This is not an exact reproduction of the historical article: the checked-in scripts use a different corpus/query offset and apply the overfetch factor twice. This run follows the article’s stated 100k corpus and 4× overfetch, with explicit deterministic query selection. See [the benchmark README](../README.md#compare-jev-with-local-rerankers).

Run from the repository root:

```sh
python reranking/compare_jev.py --api-key-file /path/to/private/key
```

Adapter, metric, batching and real LanceDB query integration checks: 14 tests passed. A live Jev request also verified the API path.
