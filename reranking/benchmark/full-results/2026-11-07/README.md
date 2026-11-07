# Full reranker comparison — November 7, 2026

All **106 dataset/model configurations passed** across **32,505 queries**, producing **690,010 complete query-score records**. This reruns every original open model and ColBERT pooling factor, Jev default and generic relevance prompts, the historical HotpotQA multihop prompt, and Voyage AI **`rerank-3`** on shared frozen candidates.

## Main comparison

Hybrid Hit@10 (%), using the union of top-50 vector and top-50 BM25 candidates. The open-model column shows the highest observed score in this run; small differences are descriptive, without significance testing.

| Dataset | Queries | No reranker | Best open model | Jev default | Jev relevance | Voyage rerank-3 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| GooAQ | 20,000 | 81.94 | zerank-2: 92.00 | 92.19 | 92.64 | 94.04 |
| NQ | 3,452 | 72.16 | jina-v3: 90.50 | 86.99 | 87.63 | 90.76 |
| HotpotQA | 7,405 | 96.27 | mxbai-large-v2: 98.54 | 92.63 | 97.57 | 98.12 |
| FiQA | 648 | 66.51 | qwen3-4b: 80.25 | 78.40 | 78.24 | 82.56 |
| SciDocs | 1,000 | 58.80 | qwen3-4b: 71.10 | 65.10 | 65.70 | 64.00 |

Voyage has the highest observed hybrid Hit@10 on **GooAQ (94.04), NQ (90.76), and FiQA (82.56)**. Its NQ lead over Jina v3 is only **0.26 percentage points (nine queries)**. **mxbai-large-v2 leads HotpotQA (98.54)**, and **Qwen3-4B leads SciDocs (71.10)**. Ranking by nDCG@10 changes the NQ and HotpotQA leader to Jina v3.

Voyage exceeds Jev relevance on four of five datasets; SciDocs favors Jev relevance by 1.70 points of Hit@10. The relevance prompt improves Jev over its default on four datasets, with the largest gain on HotpotQA: **92.63 → 97.57**, above the no-reranker baseline of **96.27**. FiQA favors the default by one query (0.15 points). The historical HotpotQA multihop prompt reaches **97.66**.

[Complete tables for every model](all-models.md) · [Raw metrics at every depth](summary.json)

## Ranking quality and latency

Each cell below is **hybrid nDCG@10 / reranking p50 ms / reranking p95 ms**. nDCG is expressed as a percentage. Latency excludes initial retrieval; API query latency includes all request batches and the network. Throughput and elapsed latency, including provider throttling, are separate fields in `summary.json`.

| Dataset | Jev default | Jev relevance | Voyage rerank-3 |
| --- | ---: | ---: | ---: |
| GooAQ | 80.12 / 215 / 352 | 81.51 / 216 / 346 | 84.30 / 234 / 287 |
| NQ | 65.55 / 234 / 357 | 67.20 / 236 / 344 | 71.89 / 263 / 372 |
| HotpotQA | 70.16 / 232 / 361 | 79.03 / 229 / 351 | 82.03 / 253 / 352 |
| FiQA | 51.52 / 271 / 437 | 51.25 / 262 / 368 | 58.87 / 291 / 482 |
| SciDocs | 22.29 / 256 / 356 | 23.57 / 258 / 361 | 23.65 / 294 / 493 |

## Execution and verification

- Scoring ran from **06:22:42 to 11:42:12 UTC on November 7, 2026**: **5.32 hours** wall time. The sum of the 60 individual GPU-job durations was **42.01 hours**, including model loading, document encoding where required, and CPU work. This is a scheduling measurement, not an isolated GPU speedup benchmark.
- One Brev/SF Compute machine with eight NVIDIA H100 80 GB HBM3 GPUs, driver 580.126.20; 112 physical CPU cores / 224 logical CPUs. Each open-model worker used one GPU and 18 disjoint logical CPUs with NUMA placement. Jev and Voyage used separate queues with eight queries in flight per provider.
- Voyage reported **250,987,734 tokens** across **32,505 saved responses**. The run recorded **0 benchmark-level retries** and **0 interrupted scoring sessions**. SDK-internal retries are not separately counted; saved token totals are not a provider invoice.
- **40 tests passed** across the benchmark and full-run suites before this run. The completed run passed full unique-query coverage, finite-score validation, candidate-content checks, model/prompt/revision identity checks, warmup and timing checks, Voyage token reconciliation, and metric recomputation.
- Frozen candidates, raw scores, timing journals, manifests, logs, and results were backed up locally and verified again before the temporary eight-H100 machine was deleted. Recomputable ColBERT document-vector tables were excluded from the local backup.

## Methodology and limitations

- The five full document corpora and indexes from the pilot were reused after checksum and configuration verification. Query preparation froze new candidate lists for the full query sets. Every reranker used those same candidates. NQ and HotpotQA use IVF_PQ; GooAQ, FiQA, and SciDocs use exact search. Historical results were not overwritten.
- Jev is `jev-1.13.0`. The relevance prompt is the exact generic relevance instruction described in the [original comparison](https://www.lancedb.com/blog/how-jev-compares-to-other-rerankers): it accepts information that helps answer any part of the query. The full prompt is stored in each model configuration in `artifact-validation.json`. Voyage uses `rerank-3`, not `rerank-3-lite`.
- Open models use pinned revisions, the original precision and truncation settings, and two unmeasured warmup queries. Jina ColBERT also pins its external implementation. The main and ColBERT environments have separate dependency lockfiles.
- These are independent single-GPU jobs running concurrently on an eight-GPU host. GPU timings can reflect shared-host contention and differ from the original single H100 PCIe run. API timings include the service and network path. Neither these timings nor the summed job durations establish an 8× speedup.
- Reranking latency covers the final successful attempt. Elapsed latency and session throughput also include retry delays and Voyage token throttling. ColBERT stored-vector latency excludes document encoding; its pool-1 job additionally measures on-the-fly encoding, which contributes to wall time.
- Hit@k measures whether at least one relevant document appears in the top k; it does not require recovering every supporting passage for multihop questions. Some models may have seen benchmark data in training. API scores can vary between calls, and tied scores preserve candidate order. No repeated-run confidence intervals were measured.

## Reproducibility artifacts

[Full configuration and model/data revisions](full.json), [coverage](status.json), [offline artifact validation and raw-score hashes](artifact-validation.json), [software and hardware](environment.json), [job timing and CPU/GPU assignments](jobs.json), [run totals and cleanup verification](run.json), and [executed source checksums](source-sha256.txt).

The executed benchmark source commit was `93263cb7ef25f5929ec151e5145fc37ba554a7a8`. Raw scores and candidate content are preserved in the ignored local directory `runs/brev-8h100-full-20261107/`; the repository contains their hashes and complete aggregate results. See [the benchmark instructions](../../README.md#complete-comparison-on-eight-h100s) to reproduce the run.
