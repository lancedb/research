# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors


import pyarrow as pa
from typing import Optional
from pylate import models, rank
from lancedb.rerankers import Reranker


class PylateReranker(Reranker):
    """
    Reranks results using the Pylate library.
    
    This reranker uses the ColBERT implementation from pylate to rerank query results,
    leveraging pylate's high-level API for simplicity.
    
    Parameters
    ----------
    model_name_or_path : str, default "sentence-transformers/all-MiniLM-L6-v2"
        The name or path of the model to use for encoding.
    column : str, default "text"
        The name of the column to use as input for reranking.
    return_score : str, default "relevance"
        Options are "relevance" or "all". Determines which scores to return.
    device : str, optional
        The device to use for reranking (e.g., "cpu", "cuda").
    batch_size : int, default 32
        Batch size to use when encoding documents and queries.
    **kwargs
        Additional keyword arguments to pass to the Pylate model.
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        column: str = "text",
        return_score: str = "relevance",
        device: Optional[str] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(return_score)
        self.column = column
        self.device = device
        self.batch_size = batch_size
        
        self.pylate_rank = rank
        
        # Initialize the ColBERT model
        self.model = models.ColBERT(
            model_name_or_path=model_name_or_path,
            device=device,
            **kwargs
        )

    def _rerank(self, result_set: pa.Table, query: str):
        """Rerank the results using the pylate library."""
        # Extract documents
        docs = result_set[self.column].to_pylist()
        doc_ids = list(range(len(docs)))
        
        # Encode the query
        query_embeddings = self.model.encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )
        
        # Encode the documents
        doc_embeddings = self.model.encode(
            [docs],  # Wrap in list to match expected format
            is_query=False,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # Use pylate's high-level rerank function
        reranked_results = self.pylate_rank.rerank(
            documents_ids=[doc_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=doc_embeddings,
            device=self.device,
        )
        
        # Extract scores and reordered document indices
        reranked_docs = reranked_results[0]
        
        # Create a mapping from original index to new score
        score_map = {item["id"]: item["score"] for item in reranked_docs}
        
        # Add the scores in the original order
        original_scores = [score_map[i] for i in doc_ids]
        
        # Add the scores to the result set
        result_set = result_set.append_column(
            "_relevance_score", pa.array(original_scores, type=pa.float32())
        )
        
        return result_set

    def _handle_empty_results(self, results: pa.Table):
        """Helper method to handle empty results consistently."""
        if len(results) > 0:
            return results
        return results.append_column(
            "_relevance_score", pa.array([], type=pa.float32())
        )

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ):
        """Rerank both vector and FTS results together."""
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._handle_empty_results(combined_results)
        
        if len(combined_results) > 0:
            combined_results = self._rerank(combined_results, query)
            
        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)
        elif self.score == "all":
            # Keep all scores
            pass
            
        combined_results = combined_results.sort_by(
            [("_relevance_score", "descending")]
        )
        return combined_results

    def rerank_vector(self, query: str, vector_results: pa.Table):
        """Rerank vector search results."""
        vector_results = self._handle_empty_results(vector_results)
        
        if len(vector_results) > 0:
            vector_results = self._rerank(vector_results, query)
            
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])
            
        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results

    def rerank_fts(self, query: str, fts_results: pa.Table):
        """Rerank full-text search results."""
        fts_results = self._handle_empty_results(fts_results)
        
        if len(fts_results) > 0:
            fts_results = self._rerank(fts_results, query)
            
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])
            
        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results