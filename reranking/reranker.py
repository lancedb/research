# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors


import pyarrow as pa
import pylate
from pylate import indexes, models, retrieve, rank
from lancedb.rerankers import Reranker
from lancedb.util import attempt_import_or_raise

class AnswerdotaiRerankers(Reranker):
    """
    Reranks the results using the Answerdotai Rerank API.
    All supported reranker model types can be found here:
    - https://github.com/AnswerDotAI/rerankers


    Parameters
    ----------
    model_type : str, default "colbert"
        The type of the model to use.
    model_name : str, default "rerank-english-v2.0"
        The name of the model to use from the given model type.
    column : str, default "text"
        The name of the column to use as input to the cross encoder model.
    return_score : str, default "relevance"
        options are "relevance" or "all". Only "relevance" is supported for now.
    **kwargs
        Additional keyword arguments to pass to the model. For example, 'device'.
        See AnswerDotAI/rerankers for more information.
    """

    def __init__(
        self,
        model_type="colbert",
        model_name: str = "answerdotai/answerai-colbert-small-v1",
        column: str = "text",
        return_score="relevance",
        **kwargs,
    ):
        super().__init__(return_score)
        self.column = column
        rerankers = attempt_import_or_raise(
            "rerankers"
        )  # import here for faster ops later
        self.reranker = rerankers.Reranker(
            model_name=model_name, model_type=model_type, **kwargs
        )

    def _rerank(self, result_set: pa.Table, query: str):
        docs = result_set[self.column].to_pylist()
        doc_ids = list(range(len(docs)))
        result = self.reranker.rank(query, docs, doc_ids=doc_ids)

        # get the scores of each document in the same order as the input
        scores = [result.get_result_by_docid(i).score for i in doc_ids]

        # add the scores
        result_set = result_set.append_column(
            "_relevance_score", pa.array(scores, type=pa.float32())
        )
        return result_set

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ):
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._handle_empty_results(combined_results)
        if len(combined_results) > 0:
            combined_results = self._rerank(combined_results, query)
        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)
        elif self.score == "all":
            raise NotImplementedError(
                "Answerdotai Reranker does not support score='all' yet"
            )
        combined_results = combined_results.sort_by(
            [("_relevance_score", "descending")]
        )
        return combined_results

    def rerank_vector(self, query: str, vector_results: pa.Table):
        vector_results = self._handle_empty_results(vector_results)
        if len(vector_results) > 0:
            vector_results = self._rerank(vector_results, query)
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])
        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results
    
    def _handle_empty_results(self, results: pa.Table):
        """
        Helper method to handle empty FTS results consistently
        """
        if len(results) > 0:
            return results        
        return results.append_column(
            "_relevance_score", pa.array([], type=pa.float32())
        )

    def rerank_fts(self, query: str, fts_results: pa.Table):
        fts_results = self._handle_empty_results(fts_results)
        if len(fts_results) > 0:
            fts_results = self._rerank(fts_results, query)
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])
        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results

    
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from lancedb.rerankers import Reranker
from lancedb.util import attempt_import_or_raise
from typing import Optional


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
        
        # Import pylate here
        print("pylate - ", pylate.__all__)
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