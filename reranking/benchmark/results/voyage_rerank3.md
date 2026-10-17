# Voyage rerank-3 (contributed by Voyage AI)

Run by Voyage AI with `python bench.py score --model voyage-rerank-3` on candidate lists rebuilt with `bench.py prepare`; not yet rerun by LanceDB. Hit@k at candidate depth 50 (hybrid reranks the union of both lists). The "our candidates" and "published" no-reranker rows show how closely the rebuilt candidates match the ones behind [the published tables](README.md). Latency is p50 per query over the network from a laptop in Europe, 8 queries in flight. More detail: [../README.md](../README.md#voyage-rerank-3-contributed-by-voyage-ai).

### GooAQ (20,000 queries)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | Hybrid @1 | Hybrid nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voyage-rerank-3 (API) | 89.97 | 94.00 | 76.34 | 78.11 | 90.00 | 94.08 | 73.03 | 84.34 | 310 |
| no reranker, our candidates | 84.97 | 90.30 | 59.90 | 66.00 | 73.81 | 81.91 | 53.90 | 67.19 | - |
| no reranker, published | 84.99 | 90.31 | 59.88 | 66.00 | 73.87 | 81.91 | 53.90 | 67.18 | - |

Billed tokens: 106,884,465.

### NQ (3,452 queries)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | Hybrid @1 | Hybrid nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voyage-rerank-3 (API) | 83.81 | 88.33 | 67.64 | 69.70 | 85.66 | 90.56 | 55.19 | 72.02 | 381 |
| no reranker, our candidates | 66.77 | 77.20 | 37.49 | 48.64 | 59.36 | 72.62 | 27.11 | 46.26 | - |
| no reranker, published | 66.54 | 76.83 | 37.49 | 48.61 | 58.98 | 72.25 | 26.94 | 46.00 | - |

Billed tokens: 38,329,156.

### HotpotQA (7,405 queries)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | Hybrid @1 | Hybrid nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voyage-rerank-3 (API) | 94.45 | 95.17 | 94.50 | 95.18 | 97.35 | 98.22 | 89.56 | 82.15 | 360 |
| no reranker, our candidates | 91.21 | 92.92 | 85.10 | 89.28 | 93.27 | 96.23 | 79.04 | 71.06 | - |
| no reranker, published | 91.17 | 93.06 | 85.10 | 89.28 | 93.22 | 96.19 | 78.87 | 70.96 | - |

Billed tokens: 71,987,512.

### FiQA (648 queries)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | Hybrid @1 | Hybrid nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voyage-rerank-3 (API) | 76.39 | 80.86 | 65.12 | 66.98 | 77.78 | 82.41 | 59.88 | 58.84 | 401 |
| no reranker, our candidates | 60.49 | 69.44 | 40.74 | 49.38 | 57.56 | 66.51 | 34.88 | 36.78 | - |
| no reranker, published | 60.49 | 69.44 | 40.74 | 49.38 | 57.41 | 66.51 | 34.88 | 36.77 | - |

Billed tokens: 13,620,674.

### SciDocs (1,000 queries)

| Model | Vector @5 | Vector @10 | FTS @5 | FTS @10 | Hybrid @5 | Hybrid @10 | Hybrid @1 | Hybrid nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| voyage-rerank-3 (API) | 55.10 | 64.60 | 51.20 | 59.00 | 54.70 | 64.20 | 27.90 | 23.68 | 422 |
| no reranker, our candidates | 51.20 | 62.60 | 39.30 | 49.30 | 49.50 | 58.70 | 22.40 | 19.81 | - |
| no reranker, published | 51.30 | 62.60 | 39.30 | 49.30 | 49.50 | 58.80 | 22.40 | 19.81 | - |

Billed tokens: 16,339,001.
