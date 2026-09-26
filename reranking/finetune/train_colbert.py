import modal
import os


cache_dir = "/models"
volume = modal.Volume.from_name("cross-encoder", create_if_missing=True)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_DATASET_NAME = "bigdata-pw/leonardo"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets", "numpy==1.26.4", "pillow", "huggingface_hub",
    "transformers[torch]", "pylate>=1.1.7", "wandb" 
).pip_install("sentence-transformers==4.0.1").env({"HF_HUB_CACHE": cache_dir,
       "HF_HOME": cache_dir,
       })

app = modal.App(image=image)


@app.function(volumes={
    cache_dir: volume
    }, timeout=60*60*12, cpu=32, memory=150000,
    gpu="H100"
    )
def main():
    import logging
    import traceback

    import torch
    from datasets import load_dataset, concatenate_datasets
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.util import mine_hard_negatives
    from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderNanoBEIREvaluator,
        CrossEncoderRerankingEvaluator,
    )
    from pylate import evaluation, losses, models, utils
    from huggingface_hub import login
    import wandb
    
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    model_name = "answerdotai/answerai-colbert-small-v1"
    train_batch_size = 128
    num_epochs = 1
    num_negatives = 1
    
    model = models.ColBERT(model_name_or_path=model_name)
    # Define the loss function before the trainer
    train_loss = losses.Contrastive(model=model)
    
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
            num_negatives=num_negatives,
            margin=0,
            range_min=0,
            range_max=100,
            sampling_strategy="top",
            batch_size=512,
            output_format="triplet",  # Changed to triplet format
        )
        hard_train_datasets.append(chunk_hard_train)
    
    hard_train_dataset = concatenate_datasets(hard_train_datasets)
    logging.info(hard_train_dataset)

    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],
        num_negatives=1,  # Only one negative
        batch_size=512,
        include_positives=True,
        output_format="triplet",  # Changed to triplet format
    )
    logging.info(hard_eval_dataset)
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=hard_eval_dataset["question"],
        positives=hard_eval_dataset["answer"],
        negatives=hard_eval_dataset["negative_1"]
    )

    evaluator = dev_evaluator

    evaluator(model)
    
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"colbert-{short_model_name}-{num_negatives}-neg-{num_epochs}-epoch-gooaq-{len(train_dataset)}"
    
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{cache_dir}/{run_name}/model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=3e-6,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        dataloader_num_workers=12,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=20000,
        save_strategy="steps",
        save_steps=20000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,
        seed=12,
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        eval_dataset=hard_eval_dataset,
        loss=train_loss,
        evaluator=evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    
    trainer.train()
    
    final_output_dir = f"{cache_dir}/{run_name}/model/final"
    model.save_pretrained(final_output_dir)
    
    if HF_TOKEN:
        try:
            model.push_to_hub(run_name)
        except Exception:
            logging.error(
                f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
                f"`huggingface-cli login`, followed by loading the model using `model = models.ColBERT({final_output_dir!r})` "
                f"and saving it using `model.push_to_hub('{run_name}')`."
            )

if __name__ == "__main__":
    main()