import lancedb
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import wandb
from wandb import init
import pyarrow as pa
import math
from lancedb.rerankers import CrossEncoderReranker
from reranker import AnswerdotaiRerankers, PylateReranker


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TABLE_NAME="gooqa"
LANCEDB_URI = os.environ.get("LANCEDB_URI", None) # Get this from LacnceDB cloud dashboard
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY", None)
DB = None
if LANCEDB_URI and LANCEDB_API_KEY:
    DB = lancedb.connect(uri=LANCEDB_URI, api_key=LANCEDB_API_KEY, region="us-east-1")
    logger.info(f"Connected to LanceDB cloud")
else:
    logger.warning(f"LANCEDB_URI or LANCEDB_API_KEY not set. Ingestion will be performed locally.")
    DB = lancedb.connect("db")


def ingest(
    skip: int = 0,
    select: int = 100_000,
    model: str ="all-MiniLM-L6-v2",
    batch_size: int = 128,
    table_name: str = TABLE_NAME,

) -> Optional[lancedb.table.Table]:
    start_time = time.time()
    embedding_model = SentenceTransformer(model)
    embedding_size = embedding_model.get_sentence_embedding_dimension()
    max_seq_length = embedding_model.get_max_seq_length()

    logger.info(f"Starting ingestion: model='{model}', skip={skip}, select={select}, table='{table_name}'")
    logger.info(f"Embedding size: {embedding_size}, Max sequence length: {max_seq_length}")

    schema = pa.schema([
        pa.field("answer", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), list_size=embedding_size)),
    ])

    db_path: str = "db"
    dataset_name: str = "sentence-transformers/gooaq"
    dataset_split: str = "train"
    mode: str = "overwrite"

    try:
        logger.info("Defined LanceDB schema.")

        logger.info(f"Connecting to LanceDB at: '{db_path}'")
        db = DB

        try:
            logger.info(f"Attempting to create (overwrite) table: '{table_name}'")
            table = db.create_table(table_name, schema=schema, mode="overwrite")
            logger.info(f"Table '{table_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create table '{table_name}': {e}", exc_info=True)
            return None

        slice_str = f"{dataset_split}[{skip}:{skip+select}]"
        logger.info(f"Loading dataset slice: '{dataset_name}', split='{slice_str}'...")
        try:
            dataset_slice = load_dataset(dataset_name, split=slice_str, trust_remote_code=True)
            logger.info("Dataset slice loaded.")
        except Exception as e:
            logger.error(f"Failed to load dataset slice '{slice_str}'. Check skip/select values. Error: {e}", exc_info=True)
            try:
                 db.drop_table(table_name)
                 logger.info(f"Cleaned up empty table '{table_name}' due to dataset loading error.")
            except Exception as drop_e:
                 logger.error(f"Failed to clean up table '{table_name}' after error: {drop_e}")
            return None

        num_examples = len(dataset_slice)
        if num_examples == 0:
            logger.warning(f"No examples found in the dataset slice '{slice_str}'. Ingestion finished.")
            return table
        elif num_examples < select:
             logger.warning(f"Loaded {num_examples} examples, which is less than the requested {select} (likely reached end of dataset).")

        total_batches = math.ceil(num_examples / batch_size)
        logger.info(f"Starting ingestion of {num_examples} examples in {total_batches} batches...")

        processed_count = 0
        for i in tqdm(range(0, num_examples, batch_size), desc="Ingesting Batches"):
            batch_start_index = i
            batch_end_index = min(i + batch_size, num_examples)
            current_batch_data = dataset_slice[batch_start_index:batch_end_index]
            answers = current_batch_data['answer']

            valid_answers = [ans for ans in answers if ans and isinstance(ans, str)]
            if not valid_answers:
                logger.warning(f"Batch {i//batch_size + 1}/{total_batches}: Contained no valid answers to embed. Skipping.")
                continue

            embeddings = embedding_model.encode(
                valid_answers,
                show_progress_bar=False,
                batch_size=batch_size
            ).tolist()

            data_to_add = [
                {"answer": answer, "embedding": embedding}
                for answer, embedding in zip(valid_answers, embeddings)
            ]

            try:
                table.add(data_to_add)
                processed_count += len(data_to_add)
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}/{total_batches} to LanceDB: {e}", exc_info=True)

        end_time = time.time()
        total_time = end_time - start_time
        final_rows = table.count_rows()

        logger.info("=" * 30)
        logger.info("Ingestion Summary:")
        logger.info(f"  Dataset Slice: '{slice_str}'")
        logger.info(f"  Processed {processed_count} valid records.")
        logger.info(f"  Total rows in table '{table_name}': {final_rows}")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info("=" * 30)

        return table

    except Exception as e:
        logger.exception(f"Fatal error during data ingestion: {e}")
        try:
            if 'db' in locals() and 'table_name' in locals() and mode == "overwrite":
                 db.drop_table(table_name)
                 logger.info(f"Cleaned up table '{table_name}' due to fatal error.")
        except Exception as cleanup_e:
             logger.error(f"Failed to clean up table '{table_name}' after fatal error: {cleanup_e}")
        return None



def single_query(
    table,
    query: str,
    query_embedding: Optional[List[float]],
    k: int,
    query_type: str,
    reranker=None,
    overfetch_factor: float = 1.0
) -> List[str]:
    """
    Perform a single search query
    
    Args:
        table: LanceDB table
        query: Text query
        query_embedding: Vector embedding for query
        k: Number of results to return
        query_type: Type of search (vector, fts, hybrid, vector_reranked, fts_reranked)
        reranker: Reranker for reranking results
        overfetch_factor: Factor by which to multiply k for initial retrieval
        
    Returns:
        List of retrieved filenames
    """
    overfetch_k = int(k * overfetch_factor)

    if query_type == "vector":
        results = table.search(query_embedding).limit(k).to_list()

    elif query_type == "vector_reranked":
        if reranker is None:
            raise ValueError("Reranker not provided for vector_reranked search")
        results = table.search(query_embedding).limit(overfetch_k).rerank(reranker, query).to_list()
        if len(results) > k:
            results = results[:k]

    elif query_type == "fts":
        results = table.search(query, query_type="fts").limit(k).to_list()

    elif query_type == "fts_reranked":
        if reranker is None:
            raise ValueError("Reranker not provided for fts_reranked search")
        results = table.search(query, query_type="fts").limit(overfetch_k).rerank(reranker).to_list()
        if len(results) > k:
            results = results[:k]

    elif query_type == "hybrid":
        if reranker is None:
            raise ValueError("Reranker not provided for hybrid search")
        results = table.search(query_type="hybrid").vector(query_embedding).text(query).rerank(reranker).limit(overfetch_k).to_list()
        if len(results) > k:
            results = results[:k]

    else:
        raise ValueError(f"Unknown query type: {query_type}")

    retrieved_filenames = [result['answer'] for result in results]
    return retrieved_filenames


def parallel_batch_retrieve(
    questions: List[str],
    query_embeddings: Optional[List[List[float]]],
    table,
    k: int,
    query_type: str,
    reranker=None,
    max_workers: int = 16,
    overfetch_factor: float = 1.0
) -> List[List[str]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, question in enumerate(questions):
            embedding = None
            if needs_embeddings(query_type):
                if query_embeddings is None or len(query_embeddings) <= i:
                     logger.error(f"Embeddings missing for question index {i} with query type {query_type}")
                     futures.append(executor.submit(lambda: []))
                     continue
                embedding = query_embeddings[i]

            futures.append(
                executor.submit(
                    single_query,
                    table,
                    question,
                    embedding,
                    k,
                    query_type,
                    reranker,
                    overfetch_factor
                )
            )

        all_retrieved_answers = []
        for future in tqdm(futures, desc=f"Processing queries ({query_type})", leave=False, total=len(questions)):
            result = future.result()
            all_retrieved_answers.append(result)


    return all_retrieved_answers

def sequential_batch_retrieve(
   questions: List[str],
   query_embeddings: Optional[List[List[float]]],
   table,
   k: int,
   query_type: str,
   reranker=None,
   overfetch_factor: float = 1.0
) -> List[List[str]]:
    """
    Perform retrieval for multiple questions sequentially (without threading)

    Args:
       questions: List of question strings
       query_embeddings: List of query embeddings (optional)
       table: LanceDB table
       k: Number of results to return per query
       query_type: Type of search
       reranker: Reranker for reranking results
       overfetch_factor: Factor by which to multiply k for initial retrieval

    Returns:
       List of lists of retrieved answers
    """
    all_retrieved_answers = []

    for i, question in tqdm(enumerate(questions), desc=f"Processing queries ({query_type})", total=len(questions)):
       embedding = None
       if needs_embeddings(query_type):
           if query_embeddings is None or len(query_embeddings) <= i:
               logger.error(f"Embeddings missing for question index {i} with query type {query_type}")
               all_retrieved_answers.append([])
               continue
           embedding = query_embeddings[i]

       try:
           result = single_query(
               table,
               question,
               embedding,
               k,
               query_type,
               reranker,
               overfetch_factor
           )
           all_retrieved_answers.append(result)
       except Exception as e:
           logger.error(f"Error processing question {i}: {e}")
           all_retrieved_answers.append([])

    return all_retrieved_answers

def precompute_embeddings(questions: List[str], embedding_model, batch_size: int = 32) -> List[List[float]]:
    logger.info(f"Precomputing embeddings for {len(questions)} questions")

    all_embeddings = embedding_model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    logger.info(f"Finished computing {len(all_embeddings)} embeddings")
    return all_embeddings.tolist()


def evaluate_retrieval_threaded(
    test_data: List[Tuple[str, str]],
    table,
    embedding_model,
    k_values: List[int] = [10],
    query_type: str = "vector",
    reranker=None,
    batch_size: int = 100,
    max_workers: int = 16,
    overfetch_factor: float = 1.0
) -> Dict[int, float]:
    overfetch_str = f", overfetch: {overfetch_factor}x" if overfetch_factor > 1 and needs_reranker(query_type) else ""
    logger.info(f"Evaluating {query_type} retrieval with threaded search (batch size: {batch_size}, threads: {max_workers}{overfetch_str})")
    start_time = time.time()

    questions = [item[0] for item in test_data]
    ground_truth_answers = [item[1] for item in test_data]

    embeddings = None
    if needs_embeddings(query_type):
        embeddings = precompute_embeddings(questions, embedding_model, batch_size=batch_size*2)

    results_per_k = {k: 0 for k in k_values}

    for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating Batches ({query_type})"):
        batch_questions = questions[i:i+batch_size]
        batch_ground_truth = ground_truth_answers[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size] if embeddings else None

        max_k = max(k_values)
        effective_overfetch = overfetch_factor if needs_reranker(query_type) else 1.0
        '''
        batch_retrieved_answers_lists = parallel_batch_retrieve(
            batch_questions,
            batch_embeddings,
            table,
            int(max_k * effective_overfetch) if needs_reranker(query_type) else max_k,
            query_type,
            reranker,
            max_workers=max_workers,
            overfetch_factor=effective_overfetch
        )
        '''
        batch_retrieved_answers_lists = sequential_batch_retrieve(
            batch_questions,
            batch_embeddings,
            table,
            int(max_k * effective_overfetch) if needs_reranker(query_type) else max_k,
            query_type,
            reranker,
            overfetch_factor=effective_overfetch
        )
        

        for retrieved_answers, ground_truth_answer in zip(batch_retrieved_answers_lists, batch_ground_truth):
            for k in k_values:
                if ground_truth_answer in retrieved_answers[:k]:
                    results_per_k[k] += 1

    final_hit_rates = {}
    for k in k_values:
        hit_rate = (results_per_k[k] / len(test_data)) * 100
        final_hit_rates[k] = hit_rate
        logger.info(f"Hit rate for k={k}: {hit_rate:.2f}%")

    elapsed = time.time() - start_time
    logger.info(f"Evaluation for {query_type} completed in {elapsed:.2f} seconds")
    return final_hit_rates


def needs_reranker(query_type: str) -> bool:
    return query_type in ["hybrid", "vector_reranked", "fts_reranked"]


def needs_embeddings(query_type: str) -> bool:
    return query_type in ["vector", "vector_reranked", "hybrid"]


def optimized_search_pipeline(
    skip: int = 100_000,
    select: int = 100_000,
    overfetch_factor: float = 4,
    query_types: Optional[List[str]] = None,
    k_values: List[int] = [5],
    table_name: str = TABLE_NAME,
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    reranker_path: Optional[str] = "cross-encoder/ms-marco-MiniLM-L6-v2",# "models/reranker-ms-marco-MiniLM-L6-v2-gooaq-bce/checkpoint-9038/",
    reranker_type = "cross-encoder",
    wandb_project: str = "gooaq-retrieval-eval",
    wandb_run_name: Optional[str] = None,
    parallel_query = False,
    use_pylate = False
):
    try:
        db = DB
        table = db.open_table(table_name)
        logger.info(f"Opened LanceDB table '{table_name}' from'")
    except Exception as e:
        logger.error(f"Failed to open LanceDB table '{table_name}'. Ensure it exists and is populated correctly. Error: {e}")
        return

    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    reranker = None
    if reranker_path:
        if use_pylate:
            reranker = PylateReranker(reranker_path, column="answer")
        else:
            reranker = AnswerdotaiRerankers(reranker_type, reranker_path, column="answer")
        logger.info(f"Loaded reranker model from: {reranker_path}")


    logger.info("Loading GooAQ test dataset")
    try:
        gooaq_dataset = load_dataset("sentence-transformers/gooaq", split="train", trust_remote_code=True).skip(skip).select(range(select))
    except Exception as e:
        logger.error(f"Failed to load dataset 'sentence-transformers/gooaq'. Error: {e}")
        return
    gooaq_dataset = gooaq_dataset.to_pandas().to_dict(orient="records")
    test_data = []
    for example in gooaq_dataset:
        if example.get('question') and example.get('answer'):
            test_data.append((example['question'], example['answer']))


    if not test_data:
        logger.error("No valid test data loaded or prepared. Exiting.")
        return

    logger.info(f"Prepared {len(test_data)} test examples.")

    run_name = wandb_run_name or f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(project=wandb_project, name=run_name, config={
        "dataset": "sentence-transformers/gooaq",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_path if reranker else "None",
        "k_values": k_values,
        "overfetch_factor": overfetch_factor,
        "table_name": table_name
    })

    if query_types is None:
        query_types = ["vector", "fts", "vector_reranked", "fts_reranked", "hybrid"]
        if not reranker:
            logger.warning("No reranker loaded. Skipping reranker-dependent query types.")
            query_types = [qt for qt in query_types if not needs_reranker(qt)]

    batch_size = 264
    max_workers = 12

    all_run_results = {}

    for query_type in query_types:
        if needs_reranker(query_type) and not reranker:
            logger.warning(f"Skipping query type '{query_type}' because no reranker is available.")
            continue

        try:
            current_reranker = reranker if needs_reranker(query_type) else None
            effective_overfetch = overfetch_factor if needs_reranker(query_type) else 1.0

            results = evaluate_retrieval_threaded(
                test_data=test_data,
                table=table,
                embedding_model=embedding_model,
                k_values=k_values,
                query_type=query_type,
                reranker=current_reranker,
                batch_size=batch_size,
                max_workers=max_workers,
                overfetch_factor=effective_overfetch
            )

            log_data = {}
            for k, hit_rate in results.items():
                 log_data[f"{query_type}_hit_rate_@{k}"] = hit_rate
            run.log(log_data)
            all_run_results[query_type] = results
            logger.info(f"Results for {query_type} retrieval: {results}")

        except Exception as e:
            logger.exception(f"Error evaluating {query_type} retrieval: {str(e)}")

    run.finish()

    logger.info("\n=== SUMMARY OF RESULTS ===")
    for query_type, results in all_run_results.items():
        result_str = ", ".join([f"@{k}={hit_rate:.2f}%" for k, hit_rate in results.items()])
        logger.info(f"{query_type}: {result_str}")


if __name__ == "__main__":
    ingest(200_000_0, 100_0000) # Ingest 1M rows after an offset of 2M (training set)
    tbl = DB[TABLE_NAME]
    tbl.create_fts_index("answer", use_tantivy=False)
    tbl.create_index(index_type="IVF_HNSW_SQ", vector_column_name="embedding")
    