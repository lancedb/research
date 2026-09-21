## Setup
Get your LanceDB URI and API key from lancedb cloud dashbord. You can replace the `URI = None` and `API_KEY = None` with your own values or set env vars.

** Instal W&B to track your experiments - `pip install wandb`

```
python LANCEDB_URI=your_uri LANCEDB_API_KEY=your_api_key python ingst_eval_gooqa.py
```

If not set, the script will fall back to running locally.

### [Optional] Train your rerankers
The training scripts are desgined to run on modal. You can run them on modal by using this command:
```
modal run --detach train_cross_encoder.py
```
or run it locally using
```
python train_cross_encoder.py
```

* You'll need to set the model type. By default it trains `bert-uncased` model. 
* You'll also need to set `HF_TOKEN` env var if you want to automatically push your models to hub

### Run eval
Running this will run evaluation across many trained cross-encoder and colbert architectures, to reproduct the reranker report

```
python eval.py
```


## Compare Jev with local rerankers

Install `requirements-jev.txt`, then run from the repository root:

```sh
python reranking/compare_jev.py --api-key-file /path/to/private/key
```

The file must contain only the TypeSafe API key; keep it outside the repository
with owner-only permissions (`chmod 600`). Alternatively use `TYPESAFE_API_KEY`
or `TYPESAFE_API_KEY_FILE`. Keys are never included in benchmark artifacts.
Jev uses the official `https://api.typesafe.ai/v1/systemone` endpoint through
`typesafe-sdk`; there is no synthetic/offline fallback. The default model is
pinned to `jev-1.13.0`; change it with `--jev-model`.

The default comparison uses the article's stated **100,000-answer corpus**
(GooAQ rows 2,000,000–2,099,999), and the first 2,000 questions within that corpus.
The first two million rows remain excluded, matching the custom models' training
boundary. Each query uses the same cached vector and BM25 candidates for every
reranker. At k=5 each source supplies 20 candidates; at k=10 it supplies 40.
Hybrid merges and deduplicates the two lists. Exact answer-string hit rate is
reported, matching the original evaluator; candidate hit rate gives the ceiling.
No ground-truth answer is inserted into the candidates.

Compared systems:

- Vector and full-text retrieval without reranking.
- Dedicated MS MARCO MiniLM cross-encoder.
- The article's GooAQ-trained ModernBERT cross-encoder.
- Jev, using the probability that each candidate answers the query, following
  [TypeSafe's reranking pattern](https://docs.typesafe.ai/cookbooks/rerank_typesafe).

This is a **fresh controlled comparison**, not a reproduction of the historical
numbers: the original `ingest_eval_gooqa.py` ingests one million rows and `eval.py`
starts queries at 2,100,000, while the article describes 100,000 answers and
queries sampled within them. This runner makes the protocol explicit, uses exact
L2 vector retrieval (no ANN index), and records package versions, dataset
fingerprint, and a hash of the shared candidates. It does not retrain models.

To run individual stages or a smaller smoke benchmark:

```sh
python reranking/compare_jev.py --models none minilm modernbert --output reranking/results/baselines.json
python reranking/compare_jev.py --models jev --api-key-file /path/to/private/key --output reranking/results/jev.json
python reranking/compare_jev.py --corpus-size 1000 --queries 20 --cache reranking/.benchmark-cache/smoke --output /tmp/jev-smoke.json
```

Keep the same cache and protocol arguments across stages. Completed query scores
are checkpointed and reused on restart; failed requests abort the run rather
than silently becoming misses. Delete the score cache to measure fresh latency.
For Jev, one request scores one query-passage pair, with four concurrent requests
by default (`--workers`). At full size, this can require up to 160,000 requests.
Scores are reused across retrieval modes and k values. Reported p50/p95 timings
cover scoring the union of up to 80 candidates per query, excluding retrieval;
these are not the article's single-k GPU latency numbers. API and local-model
latencies also include different network/hardware costs.

The original evaluator can also use `reranker_type="jev"` and
`reranker_path="jev-1.13.0"`, with a key supplied through the environment.
For auditable results and fail-fast API behavior, prefer `compare_jev.py`.

Run adapter and metric checks with:

```sh
python -m pytest reranking/tests -q
```
