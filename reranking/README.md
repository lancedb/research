# Reranking

Two separate experiments:

- [`benchmark/`](benchmark/): off-the-shelf rerankers (cross-encoders, LLM rerankers, ColBERT, Jev API) on GooAQ and four BEIR datasets. Every model reorders the same LanceDB candidates on the same GPU.
- [`finetune/`](finetune/): training cross-encoders and ColBERT models on GooAQ and comparing them with their base models.
