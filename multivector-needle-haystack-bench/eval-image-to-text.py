import os
import sys
import re
import argparse
from typing import Iterator, Optional, Tuple, List, Dict
import torch
import lancedb
import pyarrow as pa
from PIL import Image
import numpy as np

from colpali_engine.models import ColQwen2
from transformers import AutoProcessor

# --- CONFIG ---
DEFAULT_SNAPSHOT_ROOT = "../document-haystack"
LANCEDB_DIR = "./lancedb_docs_image_needles"
MODEL_ID = "vidore/colqwen2-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor():
    dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
    print("Loading ColQwen2 model (device_map='auto')...")
    model = ColQwen2.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=dtype, device_map="auto")
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    print("Loaded model; model device example:", next(model.parameters()).device)
    return model, proc

@torch.no_grad()
def image_to_multivector(pil_image: Image.Image, model, processor) -> np.ndarray:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    batch_inputs = processor(images=[pil_image], return_tensors="pt").to(model.device)
    out = model(**batch_inputs)

    emb_tensor = getattr(out, "image_embeds", getattr(out, "last_hidden_state", out))
    
    if emb_tensor is None:
        raise RuntimeError("Could not extract image embeddings from model outputs.")
    
    emb = emb_tensor.squeeze(0) if emb_tensor.ndim == 3 and emb_tensor.shape[0] == 1 else emb_tensor
    return emb.detach().cpu().to(torch.float32).numpy()

def infer_page_from_filename(fname: str) -> int:
    m = re.search(r"_page_(\d+)", fname)
    if m: return int(m.group(1))
    m2 = re.search(r"page[_\-]?(\d+)", fname)
    if m2: return int(m2.group(1))
    return -1


def iter_documents(root: str) -> Iterator[Tuple[str, str]]:
    root = os.path.abspath(root)
    for doc_name in sorted(os.listdir(root)):
        doc_path = os.path.join(root, doc_name)
        if os.path.isdir(doc_path) and not doc_name.startswith('.'):
            yield doc_name, doc_path

def ingest_document_table(db: lancedb.LanceDBConnection, table_name: str, doc_path: str, model, processor) -> Optional[lancedb.table.LanceTable]:
    print(f"  Ingesting data for document: {os.path.basename(doc_path)}")
    rows, embed_dim = [], None
    for variant_name in sorted(os.listdir(doc_path)):
        variant_path = os.path.join(doc_path, variant_name)
        if not os.path.isdir(variant_path): continue
        images_dir = next((os.path.join(variant_path, d) for d in os.listdir(variant_path) if d.startswith("Images_")), None)
        if not images_dir: continue
        print(f"    - Processing variant: {variant_name}")
        for fname in os.listdir(images_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue
            try:
                img_path = os.path.join(images_dir, fname)
                embedding = image_to_multivector(Image.open(img_path), model, processor)
                if embed_dim is None: embed_dim = embedding.shape[1]
                rows.append({"variant": variant_name, "page_num": infer_page_from_filename(fname), "vector": embedding.tolist()})
            except Exception as e:
                print(f"      Skipping image {fname} due to error: {e}", file=sys.stderr)
    if not rows or embed_dim is None: return None
    vector_type = pa.list_(pa.list_(pa.float32(), embed_dim))
    schema = pa.schema([pa.field("variant", pa.string()), pa.field("page_num", pa.int32()), pa.field("vector", vector_type)])
    tbl = db.create_table(table_name, schema=schema, exist_ok=True)
    tbl.add(rows)
    print(f"  Successfully ingested {len(tbl)} total pages for {os.path.basename(doc_path)}.")
    return tbl

def load_document_image_ground_truth(doc_path: str) -> Dict[str, List[Dict]]:
    doc_ground_truth = {}
    for variant_name in sorted(os.listdir(doc_path)):
        variant_path = os.path.join(doc_path, variant_name)
        if not os.path.isdir(variant_path): continue
        
        needles_info_file = os.path.join(variant_path, "image_needles_info.csv")
        image_needles_dir = os.path.join(variant_path, "Image_Needles")
        
        if not (os.path.exists(needles_info_file) and os.path.exists(image_needles_dir)):
            continue

        variant_queries = []
        with open(needles_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split(',')
                    if len(parts) < 2: continue
                    needle_filename, page_num = parts[0], int(parts[1])
                    needle_path = os.path.join(image_needles_dir, needle_filename)
                    if os.path.exists(needle_path):
                        variant_queries.append({
                            "needle_path": needle_path,
                            "correct_page": page_num
                        })
                except (ValueError, IndexError):
                    continue 
        
        if variant_queries:
            doc_ground_truth[variant_name] = variant_queries
    return doc_ground_truth


def evaluate_document_image_queries(table: lancedb.table.LanceTable, doc_ground_truth: Dict, model, processor, k_values: List[int]) -> Dict:
    total_hits_at_k = {k: 0 for k in k_values}
    total_queries = 0
    successful_searches = 0

    for variant_name, queries in doc_ground_truth.items():
        print(f"    - Evaluating image queries for variant: {variant_name}")
        total_queries += len(queries)
        for query_info in queries:
            correct_page = query_info['correct_page']
            needle_path = query_info['needle_path']
            
            try:
                needle_image = Image.open(needle_path)
                query_vec = image_to_multivector(needle_image, model, processor)
                results = table.search(query_vec).limit(max(k_values)).to_list()
                successful_searches += 1
            except Exception as e:
                print(f"      Search failed for needle '{os.path.basename(needle_path)}': {e}", file=sys.stderr)
                continue
            
            for k in k_values:
                is_hit = any(
                    res['variant'] == variant_name and res['page_num'] == correct_page
                    for res in results[:k]
                )
                if is_hit:
                    total_hits_at_k[k] += 1

    return {
        'total_questions': total_queries, 
        'successful_searches': successful_searches,
        'hits_at_k': total_hits_at_k
    }


def print_aggregated_results(all_results: List[Dict], k_values: List[int]):
    total_q = sum(res['total_questions'] for res in all_results)
    total_searches = sum(res['successful_searches'] for res in all_results)
    total_hits_at_k = {k: sum(res['hits_at_k'][k] for res in all_results) for k in k_values}

    print("\n" + "="*50)
    print("AGGREGATED IMAGE NEEDLE EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total Documents Evaluated: {len(all_results)}")
    print(f"Total Image Queries: {total_q}")
    print(f"Total Successful Searches: {total_searches}")

    print("\nOverall Hit Rate @ K:")
    print("-" * 20)
    for k in k_values:
        hit_rate = total_hits_at_k[k] / total_searches if total_searches > 0 else 0.0
        print(f"Hit@{k:2d}: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
    print("-" * 20)

def main():
    parser = argparse.ArgumentParser(description="Per-document image needle evaluation for Document Haystack.")
    parser.add_argument("--snapshot_root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--lancedb_dir", default=LANCEDB_DIR)
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    args = parser.parse_args()

    if not os.path.exists(args.lancedb_dir):
        os.makedirs(args.lancedb_dir)

    model, processor = load_model_and_processor()
    db = lancedb.connect(args.lancedb_dir)
    all_doc_results = []
    
    for doc_name, doc_path in iter_documents(args.snapshot_root):
        print(f"\n--- Processing Document: {doc_name} ---")
        table_name = f"doc_{doc_name.lower().replace('.', '_').replace('-', '_')}"
        if table_name in db.table_names(): db.drop_table(table_name)
            
        table = ingest_document_table(db, table_name, doc_path, model, processor)
        if not table: continue
        
        doc_ground_truth = load_document_image_ground_truth(doc_path)
        if not doc_ground_truth:
            print("  No image needle ground truth found.")
            db.drop_table(table_name)
            continue
            
        doc_results = evaluate_document_image_queries(table, doc_ground_truth, model, processor, args.k_values)
        all_doc_results.append(doc_results)
        
        db.drop_table(table_name)
        print(f"  Evaluation complete for {doc_name}.")

    if all_doc_results:
        print_aggregated_results(all_doc_results, args.k_values)
    else:
        print("\nNo documents were successfully evaluated.")

if __name__ == "__main__":
    main()