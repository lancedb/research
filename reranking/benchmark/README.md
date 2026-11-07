# Off-the-shelf reranker benchmark

How much does a reranker add on top of LanceDB vector, full-text and hybrid search, and what does it cost in latency? This compares popular open rerankers, Jev, and Voyage AI without any fine-tuning. For fine-tuned models see [`../finetune`](../finetune).

**Latest full rerun — November 7, 2026:** [comparison and verification](full-results/2026-11-07/README.md), [every model](full-results/2026-11-07/all-models.md), and [raw metrics](full-results/2026-11-07/summary.json). All 106 configurations passed across 32,505 queries, including every original open model, Jev default/relevance, and an independently executed Voyage `rerank-3` run on the same frozen candidates. Eight H100s ran independent single-GPU jobs concurrently.

The tables below preserve the original single-H100 comparison. Original full tables: [results/README.md](results/README.md). Original raw numbers for every model, dataset, mode and candidate depth (10/20/30/50), including nDCG@10: [results/summary.json](results/summary.json). The historical Voyage AI contribution is [below](#voyage-rerank-3-contributed-by-voyage-ai).

## What we found

Hit@k per dataset for the main models; bold is the best in each column. Vector and FTS rerank their own top 50, hybrid reranks the union. The no-reranker hybrid row is plain RRF. p50 is per-query latency on one H100 PCIe, except Jev which is over the network. Every model, including all ColBERT pooling factors: [results/README.md](results/README.md).

### GooAQ (20,000 queries, 100k answers, exact kNN)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jev, relevance prompt (API) | **88.14** | **92.60** | 74.98 | 77.36 | **88.01** | **92.59** | 169 |
| jev, default prompt (API) | 87.39 | 92.30 | 74.78 | 77.31 | 87.18 | 92.17 | 173 |
| zerank-2 | 87.43 | 92.11 | **75.08** | **77.50** | 87.34 | 92.00 | 335 |
| qwen3-8b | 86.38 | 91.38 | 74.54 | 77.14 | 86.17 | 91.22 | 750 |
| qwen3-4b | 85.98 | 90.92 | 74.67 | 77.22 | 85.74 | 90.61 | 510 |
| jina-v3 | 85.76 | 91.54 | 74.59 | 77.36 | 85.06 | 90.59 | 100 |
| mxbai-large-v2 | 85.23 | 90.36 | 74.58 | 77.18 | 84.89 | 89.99 | 198 |
| gte-moderncolbert, pool 1 | 84.55 | 89.73 | 74.85 | 77.33 | 84.31 | 89.39 | 39 |
| gte-moderncolbert, pool 2 | 84.19 | 89.56 | 74.53 | 77.17 | 83.97 | 89.27 | 35 |
| bge-v2-m3 | 82.48 | 88.17 | 73.75 | 76.59 | 82.09 | 87.52 | 38 |
| minilm-l6 | 78.93 | 84.80 | 72.33 | 75.75 | 78.28 | 83.70 | 19 |
| no reranker | 84.99 | 90.31 | 59.88 | 66.00 | 73.87 | 81.91 | - |

### NQ (3,452 queries, 2.68M passages, IVF_PQ)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jina-v3 | **83.55** | **87.80** | **67.56** | **69.81** | **85.52** | **90.38** | 242 |
| mxbai-large-v2 | 82.59 | 87.11 | 67.06 | 69.58 | 84.59 | 89.51 | 488 |
| qwen3-4b | 80.91 | 86.96 | 66.19 | 69.47 | 82.82 | 89.37 | 1304 |
| qwen3-8b | 81.11 | 86.62 | 66.48 | 69.24 | 82.88 | 88.73 | 1922 |
| bge-v2-m3 | 80.56 | 86.30 | 66.19 | 69.21 | 82.18 | 88.44 | 149 |
| zerank-2 | 79.40 | 85.86 | 65.67 | 68.74 | 80.74 | 87.86 | 1146 |
| jev, relevance prompt (API) | 78.91 | 85.28 | 65.06 | 68.95 | 80.59 | 87.40 | 186 |
| jev, default prompt (API) | 77.90 | 85.23 | 64.51 | 68.66 | 79.20 | 86.88 | 190 |
| gte-moderncolbert, pool 2 | 75.00 | 82.94 | 63.90 | 68.42 | 76.01 | 84.50 | 37 |
| gte-moderncolbert, pool 1 | 74.94 | 82.94 | 63.85 | 68.37 | 75.75 | 84.41 | 50 |
| minilm-l6 | 70.71 | 80.07 | 61.21 | 66.95 | 71.18 | 80.82 | 91 |
| no reranker | 66.54 | 76.83 | 37.49 | 48.61 | 58.98 | 72.25 | - |

### HotpotQA (7,405 queries, 5.23M passages, IVF_PQ)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mxbai-large-v2 | **94.91** | **95.30** | 94.92 | **95.26** | **98.16** | **98.64** | 402 |
| jina-v3 | 94.84 | 95.23 | **94.99** | 95.25 | 98.11 | 98.57 | 186 |
| bge-v2-m3 | 94.65 | 95.15 | 94.90 | 95.21 | 98.04 | 98.53 | 101 |
| qwen3-4b | 94.80 | 95.07 | 94.80 | 95.21 | 98.00 | 98.42 | 1055 |
| qwen3-8b | 94.41 | 95.14 | 94.58 | 95.14 | 97.53 | 98.38 | 1559 |
| gte-moderncolbert, pool 1 | 93.88 | 94.73 | 94.18 | 94.92 | 96.85 | 98.07 | 55 |
| gte-moderncolbert, pool 2 | 93.72 | 94.58 | 93.98 | 94.81 | 96.76 | 97.96 | 46 |
| jev, relevance prompt (API) | 93.64 | 94.80 | 93.56 | 94.80 | 96.37 | 97.70 | 176 |
| minilm-l6 | 92.99 | 94.30 | 93.40 | 94.44 | 95.92 | 97.39 | 38 |
| no reranker | 91.17 | 93.06 | 85.10 | 89.28 | 93.22 | 96.19 | - |
| zerank-2 | 91.51 | 93.63 | 91.80 | 93.76 | 93.76 | 96.06 | 875 |
| jev, default prompt (API) | 87.72 | 90.87 | 89.45 | 92.68 | 89.18 | 92.67 | 190 |

### FiQA (648 queries, 58k posts, exact kNN)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-4b | 72.38 | **79.48** | 61.27 | **65.74** | 72.38 | **80.09** | 1509 |
| qwen3-8b | **73.92** | 79.01 | **62.81** | **65.74** | **73.92** | 79.94 | 2327 |
| jev, relevance prompt (API) | 71.45 | 77.93 | 62.19 | 64.97 | 71.14 | 78.70 | 224 |
| jev, default prompt (API) | 71.14 | 77.47 | 62.19 | 65.12 | 71.91 | 78.40 | 255 |
| zerank-2 | 70.06 | 77.01 | 60.03 | 64.51 | 69.44 | 77.62 | 1495 |
| mxbai-large-v2 | 71.45 | 75.62 | 61.88 | **65.74** | 71.91 | 76.85 | 536 |
| jina-v3 | 66.67 | 74.69 | 58.64 | 63.89 | 66.05 | 74.07 | 449 |
| gte-moderncolbert, pool 1 | 65.90 | 74.23 | 57.25 | 63.89 | 65.28 | 73.30 | 63 |
| gte-moderncolbert, pool 2 | 65.90 | 73.61 | 56.94 | 63.73 | 65.12 | 73.15 | 49 |
| bge-v2-m3 | 64.81 | 71.30 | 58.64 | 62.35 | 63.58 | 69.75 | 147 |
| no reranker | 60.49 | 69.44 | 40.74 | 49.38 | 57.41 | 66.51 | - |
| minilm-l6 | 59.72 | 67.75 | 53.86 | 58.95 | 57.72 | 66.20 | 50 |

### SciDocs (1,000 queries, 26k abstracts, exact kNN)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-4b | **61.20** | **70.70** | **55.10** | **63.50** | **61.30** | **71.60** | 1444 |
| qwen3-8b | 58.30 | 68.40 | 53.20 | 62.30 | 57.70 | 69.20 | 2182 |
| jina-v3 | 56.70 | 66.50 | 51.60 | 59.60 | 54.70 | 66.70 | 438 |
| jev, relevance prompt (API) | 55.30 | 65.60 | 51.10 | 59.00 | 56.20 | 65.90 | 235 |
| jev, default prompt (API) | 53.30 | 64.30 | 49.40 | 57.90 | 52.90 | 64.70 | 269 |
| zerank-2 | 52.40 | 63.50 | 48.90 | 58.90 | 51.70 | 63.30 | 1412 |
| gte-moderncolbert, pool 1 | 50.00 | 60.50 | 46.70 | 55.90 | 48.90 | 58.80 | 70 |
| no reranker | 51.30 | 62.60 | 39.30 | 49.30 | 49.50 | 58.80 | - |
| gte-moderncolbert, pool 2 | 49.30 | 59.80 | 45.90 | 55.70 | 47.60 | 58.50 | 52 |
| mxbai-large-v2 | 48.70 | 60.60 | 44.50 | 54.30 | 46.00 | 57.90 | 513 |
| bge-v2-m3 | 45.90 | 58.50 | 42.60 | 53.70 | 44.20 | 56.20 | 142 |
| minilm-l6 | 44.70 | 56.80 | 41.60 | 51.20 | 42.40 | 52.80 | 51 |

- There's no single winner. Jev (relevance prompt) and zerank-2 lead on GooAQ, jina-v3 on NQ, mxbai-v2 on HotpotQA, and Qwen3-4B/8B on FiQA and SciDocs.
- Qwen3-4B is the most consistent: best on SciDocs and FiQA hybrid, and within 1.6 points of the leader everywhere else. Qwen3-8B only beats it on GooAQ and is about 1.5x slower.
- jina-v3 is the best accuracy per millisecond among the cross-encoders: top on NQ and close on GooAQ and HotpotQA at 100 to 250 ms. It is weaker on FiQA and SciDocs.
- For Jev the prompt matters more than anything else. With `TypeSafeReranker`'s default prompt ("does the document answer or directly address the query?") Jev drops below no reranker on HotpotQA (hybrid 92.67 vs 96.19), because multi-hop questions need a bridge passage that doesn't answer the question by itself. A generic relevance prompt ("is the document relevant to the query? ... even if only part of it") fixes that: 97.70 hybrid Hit@10, and Hit@1 goes from 72.87 to 85.24. It beats or ties the default in 41 of 45 dataset/mode/Hit@k cells; the 4 exceptions are all on FiQA and at most 0.77 points (5 of 648 queries). So the relevance prompt is the fairer Jev number. A HotpotQA-specific multi-hop prompt adds almost nothing on top (97.76). Even with the better prompt, Jev trails Qwen3-4B and mxbai-v2 on HotpotQA at Hit@1 (85 vs 92 to 93) and on FiQA and SciDocs.
- Jev is fast over the API: p50 170 to 270 ms with 8 queries in flight, network included.
- zerank-2 also drops below no reranker on HotpotQA (96.06 vs 96.19). We used its default instruction.
- Reranking helps BM25 the most. NQ FTS goes from 48.61 to 69.81 Hit@10, FiQA from 49.38 to 65.74.
- On vector results the picture depends on the dataset. On NQ every reranker helps. On GooAQ and SciDocs only 6 of 19 beat plain vector search at Hit@10, so a weak reranker makes a good embedding model worse.
- On SciDocs (citation prediction) hybrid, 11 of 19 rerankers end up below plain RRF.
- ColBERT with stored vectors is the cheap option. Reranking from multivectors kept in LanceDB takes 25 to 75 ms. Encoding the candidates at query time (what a cross-encoder has to do) is up to 3x slower on longer passages: gte-moderncolbert goes from 50 to 139 ms on NQ.
- Token pooling at factor 2 halves storage for 0.1 to 0.5 points. Factor 4 costs 1 to 3 points for answerai-colbert-small and gte-moderncolbert, but jina-colbert-v2 falls apart at factor 4 (below no reranker on four of the five datasets). Token counts match the other models, so this looks like a property of the model, not the budget.

| ColBERT storage (fp16, GooAQ) | pool 1 | pool 2 | pool 4 |
| --- | ---: | ---: | ---: |
| answerai-colbert-small (96d) | 9.8 KB | 4.9 KB | 2.5 KB |
| gte-moderncolbert (128d) | 13.7 KB | 6.9 KB | 3.5 KB |
| jina-colbert-v2 (128d) | 15.1 KB | 7.6 KB | 3.9 KB |

## Voyage rerank-3 (contributed by Voyage AI)

Voyage AI contributed the historical `rerank-3` numbers below using this benchmark and LanceDB's `VoyageAIReranker`, with the same protocol and candidate depth. The [November 7 full comparison](full-results/2026-11-07/README.md) independently reruns Voyage alongside all other models on shared candidates. Historical contributed hybrid Hit@10, with Hit@1 in parentheses:

| Dataset | voyage-rerank-3 | No reranker (published) |
| --- | ---: | ---: |
| GooAQ | 94.08 (73.03) | 81.91 (53.90) |
| NQ | 90.56 (55.19) | 72.25 (26.94) |
| HotpotQA | 98.22 (89.56) | 96.19 (78.87) |
| FiQA | 82.41 (59.88) | 66.51 (34.88) |
| SciDocs | 64.20 (27.90) | 58.80 (22.40) |

Full tables per mode with nDCG@10, latency and billed tokens: [results/voyage_rerank3.md](results/voyage_rerank3.md); raw metrics at every candidate depth: [results/voyage_rerank3.json](results/voyage_rerank3.json).

- The candidate lists were rebuilt with `bench.py prepare`, since the originals are not in the repository. On GooAQ, FiQA and SciDocs (exact search) the no-reranker rows match the published ones to within 0.15 points (a dozen of 20,000 queries on GooAQ, one query on FiQA and SciDocs). On NQ and HotpotQA (IVF_PQ, whose training is random) they are within 0.4 and 0.15 points, slightly above the published ones on NQ. The tables in `voyage_rerank3.md` show both.
- Embeddings for the rebuilt candidates were computed without a GPU: fp16 on Apple MPS for GooAQ, FiQA and SciDocs, fp32 on CPU for NQ and HotpotQA.
- Latency was measured over the network from a laptop in Europe, with 8 queries in flight and one request per query (all candidates in one call), so it is not directly comparable with the H100 or Jev numbers above.
- `rerank-3` scores vary slightly between calls (by up to about 0.004); rescoring 100 SciDocs queries changed one Hit@1 and no Hit@10 result.
- Whether `rerank-3`'s training data includes any of these datasets is not public, the same caveat as for several open models above.

## Caveats

- GooAQ uses a fixed random 20,000 of its 100,000 questions (seed 0) against the full 100,000-answer corpus. The BEIR datasets use every test query.
- Latency for small cross-encoders is mostly overhead: about 20 ms per query goes into sentence-transformers/transformers tokenization (`convert_to_tensors`), not the model. It's included because that's what `CrossEncoderReranker` costs today.
- Ties are common. Jev returns probabilities rounded to 0.01, so 28 to 58% of queries have a tie at the top-10 cut (14 to 28% for Qwen3-4B). Ties keep candidate order (vector, then FTS). Breaking them by RRF order instead moves Jev's (default prompt) hybrid Hit@10 by at most 0.3 points, and even putting relevant documents last in every tie costs at most 2.2 points (SciDocs). HotpotQA stays below no reranker either way.
- Jev isn't fully deterministic: rescoring the same pairs gave identical probabilities 74% of the time and within 0.01 94% of the time. The relevance and multi-hop prompt runs were made from a different (CPU) machine than the other Jev runs, so their latency went over a different network path.
- Jev latency is measured with 8 queries and up to 32 requests in flight. A query retried after a rate limit is timed from its last attempt only.
- PyLate's hierarchical pooling crashes on real embeddings (rounding makes 1 - cos slightly negative and scipy's ward linkage rejects it, even in float64). `bench.py` has a copy that clips distances at zero; otherwise it's the same method.
- Several models are trained on MS MARCO, NQ or HotpotQA, and bge-base and MiniLM saw GooAQ, so absolute numbers on those sets are optimistic. The comparison between rerankers is still fair because they all rerank the same candidates.

## Protocol

- **Datasets.** GooAQ (a fixed 20,000 of the 100,000 questions in rows 2,000,000 to 2,099,999, searched against all 100,000 answers; held out from the fine-tuned models), and BEIR NQ, HotpotQA, FiQA and SciDocs test sets. The smallest has 648 queries.
- **Retrieval, once per dataset.** `BAAI/bge-base-en-v1.5` embeddings, LanceDB BM25 (bag-of-words `MatchQuery`), and hybrid with LanceDB's `RRFReranker`. We keep the top 50 from vector and top 50 from BM25. Every reranker reorders exactly these candidates.
- **Vector index.** `bench.py` builds a default LanceDB `IvfPq` index (cosine) and searches with `nprobes=64, refine_factor=10`. NQ and HotpotQA candidates come from this index. GooAQ, FiQA and SciDocs were retrieved with exact search before we added the index, and we kept those candidates rather than rescore every model; `EXACT_SEARCH` in `bench.py` reproduces them exactly. Within a dataset every model reranks the same candidates, so no table mixes ANN and exact search. On those three the index matches exact search closely: recall@10 99.4 to 99.96%, recall@50 95.0 to 99.3%, and vector Hit@10 within 0.3 points.
- **Reranking.** Vector reranks its 50, FTS reranks its 50, hybrid reranks the union (up to 100). Each query's candidates are scored once and the same scores are used for all three modes. Scores are pointwise, so results at depth 10, 20 and 30 fall out of the same run.
- **Metric.** Hit@k: at least one relevant document in the top k. nDCG@10 is also in `summary.json`.
- **Hardware.** Every open model runs on one H100 PCIe (80 GB) in bf16, max length 512, one query at a time with all its candidates in one batch, 18 dedicated CPU cores. Latency is per query and includes the LanceDB reranker call. Jev runs over the network with `TypeSafeReranker(batch_size=40)`, so its latency is not directly comparable.
- **Late interaction.** ColBERT document vectors are computed once, token-pooled (factor 1, 2, 4) and stored in LanceDB. "Stored" latency is query encoding plus MaxSim over the stored candidate vectors. "On the fly" latency encodes the candidates at query time, like a cross-encoder has to.

## Models

| Name | Model | Type |
| --- | --- | --- |
| minilm-l6 | cross-encoder/ms-marco-MiniLM-L6-v2 | cross-encoder, 22M |
| gte-modernbert | Alibaba-NLP/gte-reranker-modernbert-base | cross-encoder, 149M |
| bge-v2-m3 | BAAI/bge-reranker-v2-m3 | cross-encoder, 568M |
| mxbai-large-v2 | mixedbread-ai/mxbai-rerank-large-v2 | LLM reranker, 1.5B |
| qwen3-0.6b / 4b / 8b | Qwen/Qwen3-Reranker-* | LLM reranker |
| zerank-2 | zeroentropy/zerank-2-reranker | LLM reranker, 4B |
| jina-v3 | jinaai/jina-reranker-v3 | listwise, 0.6B, CC-BY-NC |
| answerai-colbert-small | answerdotai/answerai-colbert-small-v1 | ColBERT, 33M |
| gte-moderncolbert | lightonai/GTE-ModernColBERT-v1 | ColBERT, 149M |
| jina-colbert-v2 | jinaai/jina-colbert-v2 | ColBERT, 560M, CC-BY-NC |
| jev | Jev `jev-1.13.0` via TypeSafe API, `TypeSafeReranker` default prompt | API |
| jev-relevance | same, generic relevance prompt | API |
| jev-multihop | same, HotpotQA-specific prompt (HotpotQA only) | API |
| voyage-rerank-3 | Voyage AI `rerank-3` via the Voyage API, LanceDB `VoyageAIReranker` | API |
| voyage-rerank-3-lite | Voyage AI `rerank-3-lite`, same setup (not run) | API |

Cross-encoders and LLM rerankers go through LanceDB's `CrossEncoderReranker`. jina-v3 is listwise and has its own `rerank` method, so it gets a small subclass in `bench.py`. jina-v3 sees the whole candidate list at once, so its hybrid scores come from the union and are not strictly pointwise.

## Run

### Complete comparison on eight H100s

`full.py` runs all **32,505 queries**: GooAQ 20,000, NQ 3,452, HotpotQA
7,405, FiQA 648, and SciDocs 1,000. It includes every open model above,
all three ColBERT pooling factors, Jev default and relevance, Voyage `rerank-3`,
and the historical HotpotQA-only Jev multi-hop prompt: **106 dataset/model
configurations and 690,010 query/configuration score records**.

On a Linux host with eight H100s, at least 100 available logical CPUs, and
enough disk for the corpus and ColBERT caches, install the pinned environments
with `bash setup_gpu.sh`, then run:

```sh
# Reuse the verified full corpus from the pilot without recomputing embeddings.
export BENCH_CORPUS_CACHE=/path/to/pilot/cache
.venv/bin/python prefetch.py --models
.venv/bin/python full.py --run-dir runs/full-8h100
```

Omit `BENCH_CORPUS_CACHE` to build new full corpus caches. The full runner
prepares fresh full query candidate lists, then freezes them for every model.
It rejects reduced query sets and incomplete results before generating a report.
Resume the same directory with the same command; valid per-query checkpoints
are retained. `--stage prepare`, `gpu`, `api`, or `report` runs only that stage.

Each open-model job uses one GPU and 12 disjoint logical CPUs by default
(`--cpus-per-gpu`), with eight jobs scheduled concurrently, longest estimated
jobs first. `--cpu-sets` accepts eight explicit CPU ranges to align each GPU
with its NUMA node; consult `nvidia-smi topo -m` and keep SMT siblings assigned
to the same worker. Jev and Voyage each have a separate sequential job queue and two
reserved CPUs; their existing eight-query concurrency and rate-limit handling
are preserved. CPU allocations, GPU model/UUID/driver, package versions,
source hashes, job durations and failures are recorded alongside the results.
These concurrent measurements share host memory and disk bandwidth and may
use a different H100 variant and CPU allocation from Ayush's original
single-H100 PCIe run. Compare accuracy on the new frozen candidates; treat
latency as measurements of this recorded setup, not a hardware-controlled
speedup over the original run. Model downloads and preparation are excluded
from per-query latency.

### Keys and small smoke run

Copy `.env.example` to `.env` in this directory and fill in `VOYAGE_API_KEY`,
`TYPESAFE_API_KEY` (Jev), and `NVIDIA_API_KEY`. For the GPU host, put your Brev
CLI login token in `BREV_TOKEN` (Account Settings → CLI in the Brev console).
For unattended cloud management, `BREV_API_KEY` holds a
[Brev personal API key](https://docs.nvidia.com/brev/latest/guides/api-keys).
This avoids depending on a short-lived CLI login when deleting an overnight
run's temporary instance. It is separate from the NVIDIA model API key.
The local `.env` is ignored by Git
and loaded automatically; exported environment variables take precedence.
The `NVIDIA_API_KEY` slot is for NGC / hosted model APIs, which do not authenticate
to Brev. The benchmark's open models run on the Brev GPU machine. Never upload
the Brev login token to the workload machine; only Voyage and Jev keys are needed
there.

After installing the requirements below, from this directory:

```sh
python smoke.py --check
python smoke.py
# On the Brev H100, require GPU execution:
python smoke.py --require-cuda
```

The smoke run samples **5 FiQA queries and 500 documents** with seed 0, retaining
all relevant documents for those queries. It uses the benchmark's BGE embeddings,
exact vector search, BM25, RRF, and top-50 candidates per source. It scores the
same candidates with MiniLM, Jev's default and relevance prompts, and Voyage AI
`rerank-3`. This checks ingestion, retrieval, API calls, score alignment, metrics,
and reporting. The reduced corpus makes its quality metrics unsuitable for
comparison with the published full-corpus results; local MiniLM timings also
do not reproduce the H100 measurements.

Each run writes its configuration, dependency versions, scores, status, and
Markdown/JSON results into a new ignored `runs/smoke-.../` directory. It cannot
overwrite the published `results/` or full-run `cache/`. To resume a printed run
directory, pass `--run-dir runs/smoke-...`. To check the local pipeline before
adding API keys, use `--models minilm-l6`; resume that directory with the default
models after adding keys. `--check` checks presence, not credential validity.

On a temporary Brev H100, copy the benchmark scripts and requirements, create
an `.env` containing **only** the Voyage and TypeSafe keys, and run
`bash run_gpu_smoke.sh`. This installs dependencies, runs the tests, checks the
GPU, and runs the smoke test using up to 18 allowed CPUs. It selects the official
PyTorch `2.11.0+cu128` wheel for the Brev image's CUDA-12.8-compatible driver;
the newest default PyPI wheel requires a newer driver. Copy
`runs/h100-smoke/` back before deleting the temporary instance. The script
does not provision or delete cloud resources itself. Save its installed-package
lockfile and hardware metadata with the results for the later full run.

### Reproducible pilot and resumable runs

The [November 6 H100 pilot](pilot-results/2026-11-06/README.md) passed all 106
dataset/model configurations with ten queries per dataset, including Voyage
`rerank-3`. Its report records coverage, dependency versions, and artifact checks;
the sample is too small for model-quality conclusions.

`revisions.json` pins the Hugging Face model and dataset commits.
Jina ColBERT's separate `xlm-roberta-flash-implementation` code revision is also
pinned in `bench.py`. The two
`requirements-*-gpu.lock` files pin clean Linux/Python 3.12 environments, using
PyTorch 2.11.0 with CUDA 12.8. ColBERT has its own environment because PyLate
requires an older sentence-transformers/transformers pair. Set up the GPU host
with `bash setup_gpu.sh`; it installs both environments and runs the tests.
The scripts do not provision cloud resources or delete instances.

Download datasets and weights before the timed stages:

```sh
.venv/bin/python prefetch.py --models
bash run_pilot.sh --run-dir runs/pilot10 --queries 10
```

The pilot samples **10 queries per dataset, seed 0, against the full corpora**.
It includes all original model configurations plus Voyage `rerank-3`, with the
multi-hop Jev prompt on HotpotQA only. Each ColBERT model runs pooling factors
1, 2, and 4. This sample checks compatibility and runtime; it is too small for
quality rankings. Full-corpus preparation still embeds about eight million
documents, even though only 50 queries are scored across the five datasets.
Use the same command to resume. Published results are never its output target.

Stages can run separately with the same run directory and selection:

```sh
bash run_pilot.sh --run-dir runs/pilot10 --stage prepare
bash run_pilot.sh --run-dir runs/pilot10 --stage gpu
# On a CPU host with the same main environment and copied candidate cache:
.venv/bin/python pilot.py --run-dir runs/pilot10 --stage api
.venv/bin/python pilot.py --run-dir runs/pilot10 --stage report
```

Release the GPU after its stages and copying the artifacts. API-only scoring
does not require CUDA. Record the API host/region alongside results, since its
network path affects latency. The runner records packages, GPU, CPU affinity,
rate/concurrency settings, coverage, and failures in the run directory.

Keep the pilot's full `cache/` directory to reuse its corpus embeddings and
indexes for the full run. For example, set `BENCH_CORPUS_CACHE=runs/pilot10/cache`
and `BENCH_ROOT=runs/full/cache` before running `bench.py prepare` without a query
limit. It checks that the source preparation is complete and its embedding
settings, revisions, and packages match, reads the source corpus without changing
it, and freezes new candidates in the new run. Each model must score those new
candidates; the pilot's scores are not reused for different query samples.

- Corpus embeddings are appended in bounded chunks and resume at the last
  committed chunk. ColBERT token vectors are also chunked; a partial write across
  pooling tables is replayed from their common committed boundary.
- Candidate files, table versions, configuration, and model revisions are bound
  to score manifests. Changed inputs or unversioned caches fail rather than
  silently reusing scores. Use a new run directory when changing the experiment.
- Every completed query is flushed to disk, including successful concurrent
  requests when a neighboring request fails. Only a bounded number of API
  queries are in flight. Resuming validates indices, score counts, and finite
  values, then runs only missing queries. Only one process may write a given
  preparation or model checkpoint at a time.
- Open models get two unmeasured warmup queries. Request latency includes the
  reranker call; `elapsed_latency_ms` additionally includes throttling and retry
  waits. `pairs_per_second` uses session wall time, including waits, retries,
  candidate reads, and checkpoint writes. ColBERT pool-1 wall time also includes
  its separately reported on-the-fly comparison. Warmup is recorded separately.
  A hard-killed session has only a lower-bound wall time through its last saved
  response, identified by `interrupted_sessions` in the report.
- Voyage token totals are derived cumulatively from saved responses, including
  across resumes. A request billed just before a hard interruption may lack a
  saved response; reconcile with the provider invoice for final cost accounting.
- Exact-search datasets skip unused IVF_PQ indexes. Retrieval depth, query
  prefix, scoring precision, truncation, and prompts retain the original protocol.

`BENCH_API_WORKERS` controls concurrent API queries (default 8). `VOYAGE_TPM`
defaults to 1.8M tokens/minute; check the actual project limit before tuning it.
The token throttle uses an adaptive estimate and rate-limit retries, not a hard
billing cap. `BENCH_CHUNK_SIZE` (default 4096) and `BENCH_EMBED_BATCH_SIZE`
(default 512) control preparation memory. Set them before creating the run;
their values are part of its preparation fingerprint.

### Individual full-benchmark stages

`CrossEncoderReranker` needs `batch_size` and model kwargs (bf16, max length).
The requirements now pin the released `lancedb==0.41.0` wheel: its TypeSafe and
cross-encoder adapters are identical to the original
`3d73f75d60573b7b77d61b0b786943f37306199d` source pin. Voyage's adapter only changes
explicit API-key precedence; this benchmark uses the environment variable.
Record the package version with new results, since the database engine has
advanced since the original run. PyLate pins an older sentence-transformers,
so ColBERT runs in its own environment.

```sh
pip install -r requirements.txt            # cross-encoders, jev
pip install -r requirements-colbert.txt    # separate env for ColBERT

python bench.py prepare --dataset fiqa
python bench.py score --dataset fiqa --model qwen3-4b
TYPESAFE_API_KEY=... python bench.py score --dataset fiqa --model jev
VOYAGE_API_KEY=... python bench.py score --dataset fiqa --model voyage-rerank-3
python bench.py report
```

The previous full Voyage run processed about 250M tokens. `BENCH_ROOT` sets the
cache directory (default `cache/`). `BENCH_RESULTS` sets the report directory
(default `runs/results/`, to protect the published `results/`). For a fresh full
run, set both to a named directory under `runs/`; omit `--queries` during
preparation. Never mix these new runs with historical candidate caches.
