# Multi-Vector Needle-in-a-Haystack Benchmark

This project provides a comprehensive benchmark for evaluating the performance of multi-modal, multi-vector search models on a high-precision, "needle in a haystack" retrieval task. It measures the ability of these models to find the exact page containing a specific piece of information within a large collection of visually complex documents.

The benchmark is built using [LanceDB](https://github.com/lancedb/lancedb) for efficient vector storage and search.

## The Task: Document Haystack Dataset

Our benchmark is built on the **AmazonScience/document-haystack** dataset. This dataset contains 25 visually complex source documents, such as financial reports and academic papers.

The evaluation follows a rigorous per-document methodology:

1.  **Data Ingestion:** For a single source document (e.g., "AIG"), the script ingests all pages from all of its page-length variants (from 5 to 200 pages long). This creates a temporary LanceDB table containing approximately 1,230 pages.
2.  **The Challenge:** The script then queries this table using a set of "needle" questions, where the goal is to retrieve the **exact page number** containing the answer.
3.  **Measurement:** A retrieval is successful if the correct page number is within the top K results. We measure retrieval accuracy (Hit@K) and the average search and inference latency for each query.

## Getting Started

### Prerequisites

- Python 3.8+
- Git LFS (for cloning the dataset)

### 1. Clone the Dataset

You must first clone the `document-haystack` repository, as the evaluation scripts expect it to be in the same parent directory.

**Important:** You must use Git LFS to download the actual document files.

```bash
git clone https://github.com/AmazonScience/document-haystack.git
cd document-haystack
git lfs pull
cd ..
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Evaluation

The main script for running the benchmark is `evaluate_v2.py`. It will iterate through a predefined set of models and retrieval strategies, printing a summary table at the end.

```bash
python evaluate_v2.py
```

### Command-Line Arguments

You can customize the evaluation with the following arguments:

-   `--snapshot_root`: Path to the cloned `document-haystack` directory. Defaults to `./document-haystack`.
-   `--lancedb_dir`: Directory to store temporary LanceDB tables. Defaults to `./lancedb_docs`.
-   `--k_values`: A list of `k` values to calculate hit rates for. Defaults to `1 3 5 10 20`.
-   `--num_workers`: Number of parallel processes to use for evaluation. This can significantly speed up the benchmark. Defaults to `1`.
-   `--no-index`: Disable vector index creation to run a brute-force (exact) search.

**Example with multiple workers:**

```bash
python evaluate_v2.py --num_workers 4 --snapshot_root /path/to/your/document-haystack
```

## Retrieval Strategies Evaluated

The script evaluates several different strategies for embedding and retrieval:

-   **`base`**: The full, late-interaction multi-vector search. This is the most computationally intensive but often the most accurate method.
-   **`flatten`**: Averages the token-level embeddings for a page into a single vector, turning the search into a standard Approximate Nearest Neighbor (ANN) search.
-   **`rerank`**: A two-stage process that first uses a `flattened` vector to find a large set of candidates and then performs a full multi-vector search on only those candidates.
-   **`max_pooling`**: Creates a single vector by taking the element-wise maximum across all token embeddings.
-   **`cls_pooling`**: Uses the embedding of the special `[CLS]` token as the single vector representation for a page.

## Models Evaluated

The benchmark is configured to run on a variety of multi-vector and single-vector models, including:

-   `vidore/colqwen2-v1.0`
-   `vidore/colpali-v1.3`
-   `vidore/colSmol-256M`
-   `vidore/colSmol-500M`
-   `openai/clip-vit-base-patch32` (as a baseline)

