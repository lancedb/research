import lancedb
import time
import statistics
from reranker import AnswerdotaiRerankers
import torch
torch._dynamo.config.disable = True
torch.compile(mode="disable")  # Disable torch.compile globally

# Initialize rerankers
reranker_colbert_modernbert = AnswerdotaiRerankers("colbert", "Y-J-Ju/ModernBERT-base-ColBERT", "answer")
reranker_colbert_minilim = AnswerdotaiRerankers("colbert", "ayushexel/colbert-MiniLM-L6-H384-uncased-1-neg-1-epoch-gooaq-1995000", "answer")
reranker_colbert_answer_v1 = AnswerdotaiRerankers("colbert", "answerdotai/answerai-colbert-small-v1", "answer")

# Cross-encoder models with compilation disabled
reranker_ce_modernbert = AnswerdotaiRerankers(
    "cross-encoder", 
    "ayushexel/reranker-ModernBERT-base-gooaq-1-epoch-1995000", 
    "answer",
)

reranker_ce_minilim = AnswerdotaiRerankers(
    "cross-encoder", 
    "ayushexel/reranker-MiniLM-L6-H384-uncased-gooaq-5-epoch-1995000", 
    "answer",
)

# Sample embedding vector
emb = list(range(384))
query = "is this a dog?"
# Configure test parameters
NUM_RUNS = 100
LIMITS = [20, 50, 100]  # Multiple limits to test

def rerank_and_measure(tbl, limit, reranker=None, name="No Reranker"):
    """
    Measure the latency of search with or without reranker
    
    Args:
        tbl: LanceDB table
        limit: Number of results to retrieve
        reranker: Reranker to use (None for baseline)
        name: Name of the reranker for reporting
        
    Returns:
        float: Average latency in milliseconds
    """
    latencies = []
    
    for _ in range(NUM_RUNS):
        start_time = time.time()
        
        if reranker:
            tbl.search(emb).limit(limit).rerank(reranker, query).to_list()
        else:
            tbl.search(emb).limit(limit).to_list()
            
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)
    
    return statistics.mean(latencies)

def main():
    # Connect to database - replace with your actual connection code
    db = lancedb.connect("db")
    tbl = db.open_table("gooqa")
    
    # Define the test cases
    test_cases = [
        {"reranker": None, "name": "No Reranker (Baseline)"},
        {"reranker": reranker_colbert_modernbert, "name": "ColBERT ModernBERT"},
        {"reranker": reranker_colbert_minilim, "name": "ColBERT MiniLM"},
        {"reranker": reranker_colbert_answer_v1, "name": "ColBERT Answer V1"},
        {"reranker": reranker_ce_modernbert, "name": "Cross-Encoder ModernBERT"},
        {"reranker": reranker_ce_minilim, "name": "Cross-Encoder MiniLM"}
    ]
    
    # Run tests and collect results
    results = {}
    
    for limit in LIMITS:
        results[limit] = []
        print(f"\nRunning tests with limit={limit}")
        
        for test in test_cases:
            print(f"  Running test for {test['name']}...")
            avg_latency = rerank_and_measure(tbl, limit, test["reranker"], test["name"])
            results[limit].append({"name": test["name"], "avg_latency": avg_latency})
            print(f"  Completed {NUM_RUNS} runs for {test['name']}")
    
    # Print results table with average latency for all limits
    print("\nSearch Latency Benchmark (Avg of {} runs)".format(NUM_RUNS))
    print("=" * 80)
    header = "{:<30}".format("Reranker")
    for limit in LIMITS:
        header += " | {:>10}".format(f"Limit {limit}")
    print(header)
    print("-" * 80)
    
    # Get the reranker names (assuming they're the same across all limits)
    reranker_names = [r["name"] for r in results[LIMITS[0]]]
    
    for i, name in enumerate(reranker_names):
        row = "{:<30}".format(name)
        for limit in LIMITS:
            row += " | {:>10.2f}".format(results[limit][i]["avg_latency"])
        print(row)
    
    # Print relative performance compared to baseline for each limit
    print("\nRelative Slowdown Factor (compared to baseline)")
    print("=" * 80)
    header = "{:<30}".format("Reranker")
    for limit in LIMITS:
        header += " | {:>10}".format(f"Limit {limit}")
    print(header)
    print("-" * 80)
    
    for i, name in enumerate(reranker_names):
        if name != "No Reranker (Baseline)":
            row = "{:<30}".format(name)
            for limit in LIMITS:
                baseline_latency = next(r["avg_latency"] for r in results[limit] if r["name"] == "No Reranker (Baseline)")
                current_latency = results[limit][i]["avg_latency"]
                slowdown = current_latency / baseline_latency
                row += " | {:>10.2f}x".format(slowdown)
            print(row)

if __name__ == "__main__":
    main()