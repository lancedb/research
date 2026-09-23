# Jev and Qwen reranker comparison — updated September 23, 2026

Supplemental comparison of selected baselines; the full original reranker list
remains in `eval.py`, with Jev appended when a TypeSafe key is configured.

The original systems were measured September 21, 2026; Qwen was added September 23 using the same cached candidates, without rerunning the other systems. The comparison covers 100,000 GooAQ answers and 2,000 queries. Every model received the same candidates. Values are exact-answer-string hit rates (%); higher is better.

| System | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reranker | 82.10 | 89.05 | 60.15 | 67.20 | — | — |
| MS MARCO MiniLM | 81.30 | 86.40 | 69.60 | 76.60 | 79.90 | 84.75 |
| GooAQ-trained ModernBERT | 86.10 | 89.70 | 71.75 | 77.70 | 86.15 | 90.05 |
| Jev 1.13.0 | 86.95 | 92.20 | 70.80 | 77.20 | 87.25 | 92.35 |
| Qwen3-Reranker-8B | 86.70 | 92.10 | 71.05 | 77.50 | 86.40 | 92.25 |

## Scoring latency

| System | Median | p95 |
| --- | ---: | ---: |
| MS MARCO MiniLM | 58 ms | 78 ms |
| GooAQ-trained ModernBERT | 447 ms | 602 ms |
| Jev 1.13.0 | 186 ms | 387 ms |
| Qwen3-Reranker-8B | 8,229 ms | 10,214 ms |

Times cover scoring the union of up to 80 candidates per query, reused across retrieval modes and k values. They exclude query embedding and retrieval. Local models ran on Apple MPS, while Jev used its hosted API with batches of 40 independent questions. These are observations on different execution environments, not hardware-controlled model speed comparisons. In the original Jev run, a DNS interruption was resumed at query 1,745; failed-attempt downtime is excluded from these successful-query timings.

Qwen used BF16 weights on Apple MPS with batches of four passages. Its latency includes scoring every candidate in the shared union, not a single query–passage pair; model loading is excluded. The local Qwen run completed without interruption. These timings should not be interpreted as GPU-server or hosted-Qwen performance.

## Protocol and reproducibility

- Dataset: `sentence-transformers/gooaq`, rows 2,000,000–2,099,999; queries are the first 2,000 rows within that corpus. The first two million rows used for custom-model training are excluded.
- Retrieval: `all-MiniLM-L6-v2`, exact L2 vector search and native LanceDB BM25. Each source supplies 20 candidates at k=5 and 40 at k=10; hybrid deduplicates their union. No positive is inserted.
- Jev uses one fixed relevance question per candidate, batched with the query as shared state. No prompt tuning on these evaluation outcomes was performed.
- Qwen uses the official `yes` minus `no` logit score and fixed prompt: “Given a web search query, retrieve relevant passages that answer the query”. Model revision: `77d193c791ed757ca307ee72715aa132723da912`; token limit: 8,192; dtype: BF16; batch size: four. See the [official model card](https://huggingface.co/Qwen/Qwen3-Reranker-8B).
- All systems completed every query. Full model identifiers, counts, dataset revision, candidate hash and package versions are in [jev-comparison.json](jev-comparison.json).
- This is not an exact reproduction of the historical article: the checked-in scripts use a different corpus/query offset and apply the overfetch factor twice. This run follows the article’s stated 100k corpus and 4× overfetch, with explicit deterministic query selection. See [the benchmark README](../README.md#supplemental-jev-comparison-with-local-rerankers).

Run the original comparison from the repository root:

```sh
python reranking/compare_jev.py --api-key-file /path/to/private/key
```

To add Qwen to the original comparison output with the original cache:

```sh
python reranking/compare_jev.py --models qwen --append
```

The checked-in output already includes Qwen, so this append command will refuse to overwrite its row. To independently regenerate only Qwen's summary from its cached scores, use `--models qwen --output /tmp/qwen-comparison.json` instead.

All original model results and metadata were verified unchanged against the prior published output. The candidate SHA-256 is identical, and Qwen's metrics and latency percentiles were independently recomputed from all 2,000 score checkpoints. New-run metadata is recorded under `additional_runs` in the JSON artifact.

Adapter, metric, batching, append preservation, restart and real LanceDB query integration checks: 21 tests passed. The original live Jev request verified the API path; Jev was not rerun for the Qwen addition.
