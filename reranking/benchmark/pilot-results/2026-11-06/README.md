# Ten-query H100 pilot — November 6, 2026

All **106 dataset/model configurations passed**, producing **1,060 complete query-score records** across five datasets. This validates the full comparison pipeline, including Voyage AI `rerank-3`; ten queries per dataset are too few to compare model quality or establish stable latency percentiles. The full scoring run has not been started.

The pilot uses seed 0 and the full document corpora, with top-50 vector and BM25 candidates, their union for hybrid, and the original prompts, precision, and truncation settings. It includes all eight cross-encoders, Jina v3, three ColBERT models at pooling factors 1/2/4, Jev default/relevance, Jev multihop on HotpotQA, and Voyage rerank-3. It does not include rerank-3-lite.

| Dataset | Corpus documents | Queries | Distinct candidate documents | Configurations | Voyage tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| fiqa | 57,638 | 10 | 866 | 21 | 225,626 |
| scidocs | 25,657 | 10 | 848 | 21 | 198,240 |
| gooaq | 100,000 | 10 | 756 | 21 | 53,040 |
| nq | 2,681,468 | 10 | 913 | 21 | 118,025 |
| hotpotqa | 5,233,329 | 10 | 937 | 22 | 99,540 |

Voyage returned **694,471 tokens** across the 50 saved responses. All configurations have complete timing journals and zero recorded benchmark-level retries; SDK-internal retries are not separately counted. Token totals cover persisted responses and should be reconciled with the provider invoice after interrupted runs.

## Validation and environment

- Brev/Scaleway NVIDIA H100 PCIe, 81,559 MiB, driver 570.124.06; Python 3.12.3, PyTorch 2.11.0+cu128. Open models used CPU cores 0–17 and 18 Torch/OMP/MKL threads. Jev and Voyage ran over the network from the same host.
- **34 tests passed locally and on the H100**, including interrupted corpus/ColBERT preparation, per-query resume, failed concurrent requests, torn timing journals, candidate mismatch rejection, provider retry delays, credential redaction, and reuse of a completed ANN index after interrupted query encoding.
- Downloaded artifacts were checked for full corpus row counts and indexes, frozen candidate file/content fingerprints, ten finite score records per configuration, consistent Voyage token totals, and explicit Jina implementation pins.
- The first ColBERT launch exposed a virtual-environment symlink bug. The interpreter selection was fixed and the pilot resumed successfully. FiQA's Jina ColBERT sample was rerun after making its already-used external implementation revision explicit. No failed or incomplete configuration remains.
- The final benchmark source revision was `76bdd9f`; the pilot resumed across the compatibility and recovery fixes recorded in this PR. Two unmeasured warmup queries precede each open-model configuration. API throughput uses elapsed session time; ColBERT pool-1 throughput also includes the separately measured on-the-fly pass.

[Coverage/status](status.json), [artifact verification](artifact-validation.json), [main environment](environment.json), [ColBERT environment](environment-colbert.json), [run configuration/revisions](pilot.json), and [verified source hashes](source-sha256.txt) are saved beside the [raw pilot metrics](summary.json). Model/dataset revisions and the two GPU dependency lockfiles are in the benchmark directory. Metrics describe this small sample and should not replace the published full-run tables.

## Reuse for the full comparison

The full corpus caches, candidate tables, individual score records, and logs were saved in the ignored local directory `runs/brev-h100-pilot-20261106/`. Restore that directory to the next GPU host and use the same pinned environment. Set `BENCH_CORPUS_CACHE` to its `cache/` directory and choose new `BENCH_ROOT` and `BENCH_RESULTS` directories, then prepare without `--queries`. This reuses document embeddings and indexes while freezing new full-run candidates; every reranker scores the same newly frozen candidates.

The temporary Brev H100 was deleted after the local backup passed verification.

Corpus and ColBERT encoding now operate in bounded chunks and resume from committed data. Exact-search datasets skip unused IVF_PQ indexes. API-only stages can run after releasing the GPU. Full-corpus preparation was the long part of this pilot; preserving these caches avoids repeating it.
