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

HF_TOKEN = os.environ.get("HF_TOKEN", None)
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

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderNanoBEIREvaluator,
        CrossEncoderRerankingEvaluator,
    )
    from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from sentence_transformers.evaluation.SequentialEvaluator import SequentialEvaluator
    from sentence_transformers.util import mine_hard_negatives
    from huggingface_hub import login

    # ST logging is a mess,, --fix this later
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    
    import wandb
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    model_name = "answerdotai/ModernBERT-base"

    train_batch_size = 256
    num_epochs = 1
    num_hard_negatives = 5  

    model = CrossEncoder(
        model_name,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
        ),
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
            output_format="labeled-pair",
        )
        hard_train_datasets.append(chunk_hard_train)
        
    hard_train_dataset = concatenate_datasets(hard_train_datasets)
    
    logging.info(hard_train_dataset)

    loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(num_hard_negatives))

    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=train_batch_size,
    )

    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        corpus=full_dataset["answer"],
        num_negatives=30,
        batch_size=512,
        include_positives=True,
        output_format="n-tuple",
    )
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["question"],
                "positive": [sample["answer"]],
                "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=train_batch_size,
        name="gooaq-dev",
        always_rerank_positives=False,
    )

    evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)

    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-gooaq-{num_epochs}-epoch-{len(train_dataset)}"
    args = CrossEncoderTrainingArguments(
        output_dir=f"{cache_dir}/{run_name}/model",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        dataloader_num_workers=12,
        load_best_model_at_end=True,
        metric_for_best_model="eval_gooaq-dev_ndcg@10",
        eval_strategy="steps",
        eval_steps=20000*4,
        save_strategy="steps",
        save_steps=400000,
        save_total_limit=2,
        logging_steps=200,
        logging_first_step=True,
        run_name=run_name,
        seed=12,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    evaluator(model)

    final_output_dir = f"{cache_dir}/{run_name}/model/final"
    model.save_pretrained(final_output_dir)

    model.push_to_hub(run_name)
