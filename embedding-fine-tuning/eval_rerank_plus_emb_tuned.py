from ingest_eval_gooqa import optimized_search_pipeline, ingest
import lancedb

if __name__ == "__main__":
    # Use best embedding model and some of the best reranker models
    
    rerankers = [
    "ayushexel/colbert-answerai-colbert-small-v1-1-neg-1-epoch-gooaq-1995000", # Finetuned
    "ayushexel/reranker-ModernBERT-base-gooaq-1-epoch-1995000", # trained cross-encoder
    "ayushexel/colbert-ModernBERT-base-2-neg-4-epoch-gooaq-1995000", # Trained colbert model
    ]
    
    best_tuned_emb_models = [
    "ayushexel/emb-all-MiniLM-L6-v2-gooaq-6-epochs", # best top 5
    "ayushexel/emb-all-MiniLM-L6-v2-gooaq-9-epochs", # best top 10
    ]
    
    for reranker in rerankers:
        for emb in best_tuned_emb_models:
            tbl_name = f"{emb[-15:]}-{reranker[10:30]}"
            db = lancedb.connect("db")
            if tbl_name not in db:
                print("creating new table")
                ingest(200_000_0, 100_0000, table_name=tbl_name, model=emb)
                tbl = db[tbl_name]
                tbl.create_fts_index("answer", use_tantivy=False)
                tbl.create_index(index_type="IVF_HNSW_SQ", vector_column_name="embedding")
                
            optimized_search_pipeline(
                                      210_000_0, 2000,  
                                      embedding_model_name=emb,
                                      wandb_run_name=f"{emb[-15:]}-{reranker[10:30]}-k@5", 
                                      reranker_path=reranker, 
                                      query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                                      k_values = [5],
                                      wandb_project = "gooaq-emb-and-rerank",
                                      use_pylate=True if "colbert" in reranker else False,
                                      table_name=tbl_name
                                     )

            optimized_search_pipeline(
                                      210_000_0, 2000,  
                                      embedding_model_name=emb,
                                      wandb_run_name=f"{emb[-15:]}-{reranker[10:30]}-k@10", 
                                      reranker_path=reranker, 
                                      query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                                      k_values = [10],
                                      wandb_project = "gooaq-emb-and-rerank",
                                      use_pylate=True if "colbert" in reranker else False,
                                      table_name=tbl_name
                                     )

