# Native Jev reranker comparison — measured September 23, 2026

Fresh scoring of 2,000 queries over the same candidates retrieved from 100,000 GooAQ answers.

| System | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reranker | 82.10% | 89.05% | 60.15% | 67.20% | — | — |
| MS MARCO MiniLM | 81.30% | 86.40% | 69.60% | 76.60% | 79.90% | 84.75% |
| GooAQ-trained ModernBERT | 86.10% | 89.70% | 71.75% | 77.70% | 86.15% | 90.05% |
| Jev 1.13.0 | 86.80% | 91.55% | 70.85% | 76.90% | 86.90% | 91.70% |

| Reranker | Median | p95 |
| --- | ---: | ---: |
| MS MARCO MiniLM | 57 ms | 77 ms |
| GooAQ-trained ModernBERT | 372 ms | 493 ms |
| Jev 1.13.0 | 805 ms | 1193 ms |

Timings measure scoring the union of up to 80 candidates per query and exclude embedding and retrieval. MiniLM and ModernBERT ran on Apple MPS. Jev used the native TypeSafeReranker with one candidate per request and 32 concurrent requests.

Candidate pairs: 136,586. Source commit: `385b4f50fbb3da7e95785e7bcbfd62630823d7cb`.

All query records, metrics, candidate coverage, and latency percentiles were checked before this report was written.

Across the same 1,177 queries measured at both concurrency settings, median scoring time was 2.894 seconds with four workers and 0.834 seconds with 32 workers; p95 was 3.752 and 1.262 seconds, respectively. The [matched timing comparison](jev-native-concurrency-comparison.json) records these separate passes.

## Protocol and reproducibility

- Dataset: `sentence-transformers/gooaq`, rows 2,000,000–2,099,999; queries are the first 2,000 rows within that corpus.
- Retrieval: `all-MiniLM-L6-v2`, exact L2 vector search and native LanceDB BM25. Each source supplies 20 candidates at k=5 and 40 at k=10; hybrid deduplicates their union. No expected answer is inserted.
- Jev uses the native `TypeSafeReranker` from LanceDB 0.40.0b5, with query and document together in state, the fixed relevance question in `jev_config.py`, and 32 concurrent requests. The requested model is `jev-1.13.0`.
- All models scored every query afresh. Each query's scores are reused across retrieval modes and cutoffs. Full identifiers, hit counts, dataset revision, candidate hash, package versions, and source commits are in [jev-native-comparison.json](jev-native-comparison.json).
- The 32-worker setting was selected using the [eight-query concurrency probes](jev-native-concurrency-probe.json). Concurrency is included in the score-cache identity so timings from different settings cannot be combined on resume.
- The original evaluator's 26 historical runs remain in the repository. This supplemental protocol holds retrieval candidates constant and applies 4× overfetch once; see [the benchmark README](../README.md#supplemental-jev-comparison-with-local-rerankers).

Run from the repository root with a new cache directory to measure fresh scores:

```sh
python -m pip install -r reranking/requirements-jev.txt
python reranking/compare_jev.py --workers 32 --cache reranking/.benchmark-cache/native-fresh --api-key-file /path/to/private/key --output reranking/results/jev-native-comparison.json
```

Validation: 33 native integration, score validation, credential, cache, and metric checks passed. All 2,000 per-model records were independently checked for complete candidate coverage, valid scores, exact hit counts, and matching latency percentiles.

The September 21 batched-adapter experiment is preserved in [jev-adapter-comparison.md](jev-adapter-comparison.md) and [jev-comparison.json](jev-comparison.json).
