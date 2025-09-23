import os
import sys
import argparse
import time
from typing import Optional, List, Dict
import torch
import lancedb
import pyarrow as pa
from PIL import Image
from utils import (
    iter_documents,
    load_document_ground_truth,
    infer_page_from_filename,
    find_correct_pages,
    print_aggregated_results,
    print_summary_table,
)
from vision_models import (
    load_model_and_processor,
    embed_image,
    embed_text,
    get_colqwen_vectors,
)

DEFAULT_SNAPSHOT_ROOT = "./document-haystack"
DEFAULT_LANCEDB_DIR = "./lancedb_docs"

def ingest_document_table(
    db: lancedb.LanceDBConnection,
    table_name: str,
    doc_path: str,
    model,
    processor,
    model_id: str,
    strategy: str,
) -> Optional[lancedb.table.LanceTable]:
    print(f"  Ingesting data for document: {os.path.basename(doc_path)}")
    rows, embed_dim = [], None
    for variant_name in sorted(os.listdir(doc_path)):
        variant_path = os.path.join(doc_path, variant_name)
        if not os.path.isdir(variant_path):
            continue
        images_dir = next(
            (
                os.path.join(variant_path, d)
                for d in os.listdir(variant_path)
                if d.startswith("Images_")
            ),
            None,
        )
        if not images_dir:
            continue
        print(f"    - Processing variant: {variant_name}")
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img_path = os.path.join(images_dir, fname)
                image_data = Image.open(img_path)
                row = {
                    "variant": variant_name,
                    "page_num": infer_page_from_filename(fname),
                }
                if strategy == "rerank":
                    vec_flat, vec_multi = get_colqwen_vectors(
                        image_data, model, processor, is_image=True
                    )
                    if embed_dim is None:
                        embed_dim = vec_flat.shape[0]
                    row["vector_flat"] = vec_flat.tolist()
                    row["vector_multi"] = vec_multi.tolist()
                elif strategy == "flatten":
                    embedding, _ = get_colqwen_vectors(
                        image_data, model, processor, is_image=True
                    )
                    if embed_dim is None:
                        embed_dim = embedding.shape[0]
                    row["vector"] = embedding.tolist()
                else:  # base strategy
                    embedding = embed_image(image_data, model, processor, model_id)
                    if embed_dim is None:
                        embed_dim = (
                            embedding.shape[1]
                            if len(embedding.shape) > 1
                            else embedding.shape[0]
                        )
                    row["vector"] = embedding.tolist()
                rows.append(row)
            except Exception as e:
                print(f"      Skipping image {fname} due to error: {e}", file=sys.stderr)

    if not rows:
        return None

    if strategy == "rerank":
        schema = pa.schema(
            [
                pa.field("variant", pa.string()),
                pa.field("page_num", pa.int32()),
                pa.field("vector_flat", pa.list_(pa.float32(), embed_dim)),
                pa.field("vector_multi", pa.list_(pa.list_(pa.float32(), embed_dim))),
            ]
        )
    elif "clip" in model_id or strategy == "flatten":
        vector_type = pa.list_(pa.float32(), embed_dim)
        schema = pa.schema(
            [
                pa.field("variant", pa.string()),
                pa.field("page_num", pa.int32()),
                pa.field("vector", vector_type),
            ]
        )
    else:
        vector_type = pa.list_(pa.list_(pa.float32(), embed_dim))
        schema = pa.schema(
            [
                pa.field("variant", pa.string()),
                pa.field("page_num", pa.int32()),
                pa.field("vector", vector_type),
            ]
        )

    tbl = db.create_table(table_name, schema=schema, exist_ok=True)
    tbl.add(rows)
    print(f"  Successfully ingested {len(tbl)} total pages for {os.path.basename(doc_path)}.")
    
    print("  Creating index...")
    if strategy == "rerank":
        tbl.create_index(metric="l2", vector_column_name="vector_flat")
        tbl.create_index(metric="cosine", vector_column_name="vector_multi")
    else:
        tbl.create_index(metric="cosine", vector_column_name="vector")
    print("  Index created.")

    return tbl


def evaluate_document(
    table: lancedb.table.LanceTable,
    doc_ground_truth: Dict,
    model,
    processor,
    k_values: List[int],
    model_id: str,
    strategy: str,
) -> Dict:
    total_hits_at_k, total_questions, successful_searches, total_latency = (
        {k: 0 for k in k_values},
        0,
        0,
        0.0,
    )
    max_k = max(k_values)
    for variant_name, ground_truth in doc_ground_truth.items():
        print(f"    - Evaluating questions for variant: {variant_name}")
        total_questions += len(ground_truth["questions"])
        for question in ground_truth["questions"]:
            correct_pages = find_correct_pages(question, ground_truth["needles"])
            if not correct_pages:
                continue
            
            # --- Latency Measurement START ---
            start_time = time.time()
            try:
                if strategy == "rerank":
                    query_flat, query_multi = get_colqwen_vectors(
                        question, model, processor, is_image=False
                    )
                    candidates = (
                        table.search(query_flat, vector_column_name="vector_flat")
                        .limit(max_k * 4)
                        .with_row_id(True)
                        .to_pandas()
                    )
                    if candidates.empty:
                        continue
                    candidate_row_ids = tuple(candidates["_rowid"].to_list())
                    results = (
                        table.search(query_multi, vector_column_name="vector_multi")
                        .where(f"_rowid IN {candidate_row_ids}")
                        .limit(max_k)
                        .to_list()
                    )
                elif strategy == "flatten":
                    query_vec, _ = get_colqwen_vectors(
                        question, model, processor, is_image=False
                    )
                    results = table.search(query_vec).limit(max_k).to_list()
                else:  # base strategy
                    query_vec = embed_text(question, model, processor, model_id)
                    results = table.search(query_vec).limit(max_k).to_list()
                
                successful_searches += 1
            except Exception as e:
                print(f"      Search failed for '{question}': {e}", file=sys.stderr)
                continue
            finally:
                end_time = time.time()
                total_latency += end_time - start_time
            # --- Latency Measurement END ---

            for k in k_values:
                if any(
                    res["variant"] == variant_name and res["page_num"] in correct_pages
                    for res in results[:k]
                ):
                    total_hits_at_k[k] += 1
    return {
        "total_questions": total_questions,
        "successful_searches": successful_searches,
        "hits_at_k": total_hits_at_k,
        "total_latency": total_latency,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-document evaluation for Document Haystack."
    )
    parser.add_argument("--snapshot_root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--lancedb_dir", default=DEFAULT_LANCEDB_DIR)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    args = parser.parse_args()

    if not os.path.exists(args.lancedb_dir):
        os.makedirs(args.lancedb_dir)

    model_ids = [
        "vidore/colqwen2-v0.1",
        "vidore/colpali-v1.3",
        "vidore/colpali-v1.2",
        "vidore/colSmol-256M",
        "vidore/colqwen2-v1.0",
        "vidore/colqwen2-v0.1",
        "vidore/colqwen2.5-v0.2",
        "openai/clip-vit-base-patch32"
    ]


    db = lancedb.connect(args.lancedb_dir)
    summary_results = []

    for model_id in model_ids:
        strategies = ["base"]
        if "Qwen" in model_id:
            strategies.extend(["flatten", "rerank"])

        for strategy in strategies:
            print(f"\n--- Processing Model: {model_id} with Strategy: {strategy} ---")
            try:
                model, processor = load_model_and_processor(model_id)
            except Exception as e:
                print(f"Failed to load model {model_id}: {e}", file=sys.stderr)
                continue
            
            all_doc_results = []

            for doc_name, doc_path in iter_documents(args.snapshot_root):
                print(f"\n--- Processing Document: {doc_name} ---")
                table_name = f"doc_{doc_name.lower().replace('.', '_').replace('-', '_')}"
                if table_name in db.table_names():
                    db.drop_table(table_name)

                table = ingest_document_table(
                    db, table_name, doc_path, model, processor, model_id, strategy
                )
                if not table:
                    continue

                doc_ground_truth = load_document_ground_truth(doc_path)
                if not doc_ground_truth:
                    db.drop_table(table_name)
                    continue

                doc_results = evaluate_document(
                    table,
                    doc_ground_truth,
                    model,
                    processor,
                    args.k_values,
                    model_id,
                    strategy,
                )
                all_doc_results.append(doc_results)
                db.drop_table(table_name)
                print(f"  Evaluation complete for {doc_name}.")

            if all_doc_results:
                hit_rates, avg_latency = print_aggregated_results(
                    all_doc_results,
                    args.k_values,
                    f"{model_id} ({strategy} strategy)",
                )
                summary_results.append({
                    "model_name": model_id,
                    "strategy": strategy,
                    "hit_rates": hit_rates,
                    "avg_latency": avg_latency,
                })

    if summary_results:
        print_summary_table(summary_results, args.k_values)


if __name__ == "__main__":
    main()