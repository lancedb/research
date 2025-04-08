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

