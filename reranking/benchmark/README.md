# Off-the-shelf reranker benchmark

How much does a reranker add on top of LanceDB vector, full-text and hybrid search, and what does it cost in latency? This compares popular open rerankers and the Jev API without any fine-tuning. For fine-tuned models see [`../finetune`](../finetune).

Full tables: [results/README.md](results/README.md). Raw numbers for every model, dataset, mode and candidate depth (10/20/30/50), including nDCG@10: [results/summary.json](results/summary.json).

## What we found

Hit@k per dataset for the main models; bold is the best in each column. Vector and FTS rerank their own top 50, hybrid reranks the union. The no-reranker hybrid row is plain RRF. p50 is per-query latency on one H100 PCIe, except Jev which is over the network. Every model, including all ColBERT pooling factors: [results/README.md](results/README.md).

### GooAQ (20,000 queries, 100k answers, exact kNN)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jev (API) | 87.39 | **92.30** | 74.78 | 77.31 | 87.18 | **92.17** | 173 |
| zerank-2 | **87.43** | 92.11 | **75.08** | **77.50** | **87.34** | 92.00 | 335 |
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
| jev (API) | 77.90 | 85.23 | 64.51 | 68.66 | 79.20 | 86.88 | 190 |
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
| minilm-l6 | 92.99 | 94.30 | 93.40 | 94.44 | 95.92 | 97.39 | 38 |
| no reranker | 91.17 | 93.06 | 85.10 | 89.28 | 93.22 | 96.19 | - |
| zerank-2 | 91.51 | 93.63 | 91.80 | 93.76 | 93.76 | 96.06 | 875 |
| jev (API) | 87.72 | 90.87 | 89.45 | 92.68 | 89.18 | 92.67 | 190 |

### FiQA (648 queries, 58k posts, exact kNN)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3-4b | 72.38 | **79.48** | 61.27 | **65.74** | 72.38 | **80.09** | 1509 |
| qwen3-8b | **73.92** | 79.01 | **62.81** | **65.74** | **73.92** | 79.94 | 2327 |
| jev (API) | 71.14 | 77.47 | 62.19 | 65.12 | 71.91 | 78.40 | 255 |
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
| jev (API) | 53.30 | 64.30 | 49.40 | 57.90 | 52.90 | 64.70 | 269 |
| zerank-2 | 52.40 | 63.50 | 48.90 | 58.90 | 51.70 | 63.30 | 1412 |
| gte-moderncolbert, pool 1 | 50.00 | 60.50 | 46.70 | 55.90 | 48.90 | 58.80 | 70 |
| no reranker | 51.30 | 62.60 | 39.30 | 49.30 | 49.50 | 58.80 | - |
| gte-moderncolbert, pool 2 | 49.30 | 59.80 | 45.90 | 55.70 | 47.60 | 58.50 | 52 |
| mxbai-large-v2 | 48.70 | 60.60 | 44.50 | 54.30 | 46.00 | 57.90 | 513 |
| bge-v2-m3 | 45.90 | 58.50 | 42.60 | 53.70 | 44.20 | 56.20 | 142 |
| minilm-l6 | 44.70 | 56.80 | 41.60 | 51.20 | 42.40 | 52.80 | 51 |

- There's no single winner. Jev and zerank-2 lead on GooAQ, jina-v3 on NQ, mxbai-v2 on HotpotQA, and Qwen3-4B/8B on FiQA and SciDocs.
- Qwen3-4B is the most consistent: best on SciDocs and FiQA hybrid, and within 1.6 points of the leader everywhere else. Qwen3-8B only beats it on GooAQ and is about 1.5x slower.
- jina-v3 is the best accuracy per millisecond among the cross-encoders: top on NQ and close on GooAQ and HotpotQA at 100 to 250 ms. It is weaker on FiQA and SciDocs.
- Jev is fast over the API and strong on direct question answering (best on GooAQ, close on FiQA), but on HotpotQA it drops below no reranker (hybrid 92.67 vs 96.19). zerank-2 does the same. Multi-hop questions need a bridge passage that doesn't answer the question by itself, and "does this passage answer the query" scores it low.
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

## Caveats

- GooAQ uses a fixed random 20,000 of its 100,000 questions (seed 0) against the full 100,000-answer corpus. The BEIR datasets use every test query.
- Latency for small cross-encoders is mostly overhead: about 20 ms per query goes into sentence-transformers/transformers tokenization (`convert_to_tensors`), not the model. It's included because that's what `CrossEncoderReranker` costs today.
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
| jev | Jev `jev-1.13.0` via TypeSafe API | API |

Cross-encoders and LLM rerankers go through LanceDB's `CrossEncoderReranker`. jina-v3 is listwise and has its own `rerank` method, so it gets a small subclass in `bench.py`. jina-v3 sees the whole candidate list at once, so its hybrid scores come from the union and are not strictly pointwise.

## Run

`CrossEncoderReranker` needs `batch_size` and model kwargs (bf16, max length). That lives on the LanceDB branch `ayush/cross-encoder-kwargs` until it is released. PyLate pins an older sentence-transformers, so ColBERT runs in its own environment.

```sh
pip install -r requirements.txt            # cross-encoders, jev
pip install -r requirements-colbert.txt    # separate env for ColBERT

python bench.py prepare --dataset fiqa
python bench.py score --dataset fiqa --model qwen3-4b
TYPESAFE_API_KEY=... python bench.py score --dataset fiqa --model jev
python bench.py report
```

Scores are appended per query, so an interrupted run resumes where it stopped. `BENCH_ROOT` sets the cache directory (default `cache/`).
