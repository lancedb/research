import modal
import os


cache_dir = "/models"
volume = modal.Volume.from_name("cross-encoder", create_if_missing=True)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4", "pillow", "huggingface_hub", "sentence-transformers", "wandb",
    "transformers[torch]", 
).env({"HF_HUB_CACHE": cache_dir,
       "HF_HOME": cache_dir,
       })

app = modal.App(image=image)

@app.function(volumes={
    cache_dir: volume
    }, timeout=60*60*12, cpu=32, memory=64000,
    gpu="A100",
    )
def main(num_epochs=1):
    import logging

    from datasets import load_dataset, concatenate_datasets

    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    from sentence_transformers.evaluation import TripletEvaluator

    from sentence_transformers.util import mine_hard_negatives
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    from huggingface_hub import login

    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    import wandb

    login(token=os.environ.get("HF_TOKEN"))
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    train_batch_size = 256
    num_hard_negatives = 1

    model = SentenceTransformer(
        model_name,
    )
    #DEBUG
    #print("Model max length:", model.max_length)
    #print("Model num labels:", model.num_labels)

    full_dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(200_000_0))
    dataset_dict = full_dataset.train_test_split(test_size=5000, seed=12)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    logging.info(train_dataset)
    logging.info(eval_dataset)

    embedding_model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")
    
    chunk_size = 200_000 
    hard_train_datasets = []
    
    for i in range(0, len(train_dataset), chunk_size):
        chunk = train_dataset.select(range(i, min(i + chunk_size, len(train_dataset))))
        chunk_hard_train = mine_hard_negatives(
            chunk,
            embedding_model,
            num_negatives=num_hard_negatives,
            margin=0,
            range_min=0,
            range_max=100,
            sampling_strategy="top",
            batch_size=512,
            output_format="triplet",
        )
        hard_train_datasets.append(chunk_hard_train)
        
    hard_train_dataset = concatenate_datasets(hard_train_datasets)
    
    logging.info(hard_train_dataset)

    loss = MultipleNegativesRankingLoss(model)



    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],
        num_negatives=1,
        batch_size=512,
        include_positives=True,
        output_format="triplet",
    )
    logging.info(hard_eval_dataset)
    evaluator = TripletEvaluator(
    anchors=hard_eval_dataset["question"],
    positives=hard_eval_dataset["answer"],
    negatives=hard_eval_dataset["negative_1"],
    name="gooqa-dev",
        )

    evaluator(model)

    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"emb-{short_model_name}-gooaq-{num_epochs}-epochs"
    args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/mpnet-base-all-nli-triplet",
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    run_name=run_name
    )

    trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
    )
    trainer.train()

    evaluator(model)

    final_output_dir = f"{cache_dir}/{run_name}/model/final"
    model.save_pretrained(final_output_dir)

    model.push_to_hub(run_name)

@app.local_entrypoint()
def local_main():
    epochs = list(range(2,11))
    # map parallel calls
    results = list(main.map(epochs))
