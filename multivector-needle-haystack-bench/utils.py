import os
import re
from typing import Iterator, Tuple, List, Dict
import torch
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster

def iter_documents(root: str) -> Iterator[Tuple[str, str]]:
    root = os.path.abspath(root) 
    for doc_name in sorted(os.listdir(root)):
        doc_path = os.path.join(root, doc_name)
        if os.path.isdir(doc_path):
            yield doc_name, doc_path

def load_document_ground_truth(doc_path: str) -> Dict[str, Dict]:
    """Loads all needles and questions for all variants within a document."""
    doc_ground_truth = {}
    for variant_name in sorted(os.listdir(doc_path)):
        variant_path = os.path.join(doc_path, variant_name)
        if not os.path.isdir(variant_path): continue
        
        needles_file = os.path.join(variant_path, "needles_info.csv")
        questions_file = os.path.join(variant_path, "prompt_questions.txt")
        
        if not (os.path.exists(needles_file) and os.path.exists(questions_file)): continue

        needles = []
        with open(needles_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = re.split(r',(?=\d)', line, 1)
                if len(parts) == 2:
                    needle_text = parts[0].strip('" ')
                    page_num = re.search(r'\d+', parts[1])
                    if page_num:
                        needles.append({'needle': needle_text, 'page': int(page_num.group(0))})
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        if questions and needles:
            doc_ground_truth[variant_name] = {'needles': needles, 'questions': questions}
    return doc_ground_truth

def infer_page_from_filename(fname: str) -> int:
    m = re.search(r"_page_(\d+)", fname)
    if m: return int(m.group(1))
    m2 = re.search(r"page[_\-]?(\d+)", fname)
    if m2: return int(m2.group(1))
    return -1

def extract_key_from_question(q: str):
    m = re.search(r"What is the secret (.+?) in the document\?", q, re.I)
    return m.group(1).strip().lower() if m else None

def extract_key_from_needle(n: str):
    m = re.search(r"The secret (.+?) is .+\.?", n, re.I)
    return m.group(1).strip().lower() if m else None

def find_correct_pages(q: str, needles: List[Dict]) -> List[int]:
    q_key = extract_key_from_question(q)
    return [n['page'] for n in needles if q_key and extract_key_from_needle(n['needle']) == q_key]

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

def print_aggregated_results(all_results: List[Dict], k_values: List[int], model_name: str) -> (Dict[int, float], float, float):
    total_q = sum(res['total_questions'] for res in all_results)
    total_searches = sum(res['successful_searches'] for res in all_results)
    total_hits_at_k = {k: sum(res['hits_at_k'][k] for res in all_results) for k in k_values}
    total_inference_latency = sum(res['total_inference_latency'] for res in all_results)
    total_search_latency = sum(res['total_search_latency'] for res in all_results)
    avg_inference_latency = total_inference_latency / total_searches if total_searches > 0 else 0.0
    avg_search_latency = total_search_latency / total_searches if total_searches > 0 else 0.0

    print("\n" + "="*50)
    print(f"AGGREGATED EVALUATION RESULTS ({model_name})")
    print("="*50)
    
    print(f"Total Documents Evaluated: {len(all_results)}")
    print(f"Total Questions Processed: {total_q}")
    print(f"Total Successful Searches: {total_searches}")
    print(f"Average Inference Latency: {avg_inference_latency:.4f} seconds")
    print(f"Average Search Latency: {avg_search_latency:.4f} seconds")

    print("\nOverall Hit Rate @ K:")
    print("-" * 20)
    hit_rates = {}
    for k in k_values:
        hit_rate = total_hits_at_k[k] / total_searches if total_searches > 0 else 0.0
        hit_rates[k] = hit_rate
        print(f"Hit@{{k:2d}}: {hit_rate:.4f} ({hit_rate*100:.1f}%)")
    print("-" * 20)
    return hit_rates, avg_inference_latency, avg_search_latency

def print_summary_table(summary_results: List[Dict], k_values: List[int]):
    """Prints a summary table comparing model performance."""
    print("\n" + "="*95)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*95)

    # Header
    header = f"{'{'Model':<40} {'Strategy':<10} {'Avg. Latency (s)':<20}"
    for k in k_values:
        header += f" | Hit@{{k:<2d}}"
    print(header)
    print("-" * len(header))

    # Rows
    for result in summary_results:
        row = f"{result['model_name']:<40} {result['strategy']:<10} {result['avg_latency']:<20.4f}"
        for k in k_values:
            hit_rate = result['hit_rates'][k]
            row += f" | {hit_rate:<6.2%}"
        print(row)
    print("="*95)

def pool_embeddings_hierarchical(
    p_embeddings,
    pool_factor: int,
    protected_tokens: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_embeddings = torch.tensor(p_embeddings).to(device)
    
    # Get the embeddings for the current passage
    passage_embeddings = p_embeddings
    token_length = passage_embeddings.shape[0]

    # Remove the tokens at protected_tokens indices
    protected_embeddings = passage_embeddings[:protected_tokens]
    passage_embeddings = passage_embeddings[protected_tokens:]

    # Cosine similarity computation (vector are already normalized)
    similarities = torch.mm(passage_embeddings, passage_embeddings.t())

    # Convert similarities to a distance for better ward compatibility
    similarities = 1 - similarities.cpu().numpy()

    # Create hierarchical clusters using ward's method
    Z = linkage(similarities, metric="euclidean", method="ward")
    # Determine the number of clusters we want in the end based on the pool factor
    max_clusters = (
        (token_length - protected_tokens) // pool_factor 
        if (token_length - protected_tokens) // pool_factor > 0 
        else 1
    )
    cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

    # Pool embeddings within each cluster
    pooled_embeddings = []
    for cluster_id in range(1, max_clusters + 1):
        cluster_indices = torch.where(
            torch.tensor(cluster_labels == cluster_id, device=device)
        )[0]
        if cluster_indices.numel() > 0:
            pooled_embedding = passage_embeddings[cluster_indices].mean(dim=0)
            pooled_embeddings.append(pooled_embedding)

    # Re-add the protected tokens to pooled_embeddings
    pooled_embeddings = list(protected_embeddings.cpu()) + pooled_embeddings
    
    return torch.stack(pooled_embeddings).cpu().numpy()