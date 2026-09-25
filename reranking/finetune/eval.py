from ingest_eval_gooqa import optimized_search_pipeline

if __name__ == "__main__":
    # Eval trained Cross encoders
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_1_epoch_2M_unseen_5", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-1-epoch-1995000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_1epoch_2M_unseen_10", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-1-epoch-1995000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_5_epoch_2M_unseen_5", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-bce-1995000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_5_epoch_2M_unseen_10", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-bce-1995000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="modernbert-untrained-control-5", 
                              reranker_path="ayushexel/ce-modernbert-untrained", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid", "vector", "fts"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="modernbert-untrained-control-10", 
                              reranker_path="ayushexel/ce-modernbert-untrained", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid", "vector", "fts"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="marco-minilm-trained-control-5", 
                              reranker_path="ayushexel/ce-modernbert-untrained", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid", "vector", "fts"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="marco-minilm-trained-control-10", 
                              reranker_path="ayushexel/ce-modernbert-untrained", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid", "vector", "fts"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )

    
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_1_epoch_500k_unseen_5", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-bce-495000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="Modernbert_tuned_1_epoch_500k_unseen_10", 
                              reranker_path="ayushexel/reranker-ModernBERT-base-gooaq-bce-495000", 
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )

    # Eval Colbert
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-MiniLM-L6-H384-uncased-1-neg-1-epoch_5", 
                              reranker_path="ayushexel/colbert-MiniLM-L6-H384-uncased-1-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-MiniLM-L6-H384-uncased-1-neg-1-epoch-10", 
                              reranker_path="ayushexel/colbert-MiniLM-L6-H384-uncased-1-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained"
                             )
    
    
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="ayushexel/colbert-ModernBERT-base-2-neg-1-epoch-gooaq-1995000", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-2-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                                use_pylate=True
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="ayushexel/colbert-ModernBERT-base-2-neg-1-epoch-gooaq-1995000", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-2-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )

    
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="ayushexel/colbert-ModernBERT-base-1-neg-1-epoch-5", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-1-neg-1-epoch-gooaq-1995000-final", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                                use_pylate=True
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="ayushexel/colbert-ModernBERT-base-1-neg-1-epoch-10", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-1-neg-1-epoch-gooaq-1995000-final", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-2-neg-4-epoch-5", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-2-neg-4-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                                use_pylate=True
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-2-neg-4-epoch-10", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-2-neg-4-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
  
   
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-5-neg-1-epoch-5", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-5-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                                use_pylate=True
                             )
  
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-5-neg-5-epoch", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-5-neg-5-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-5-neg-5-epoch", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-5-neg-5-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    
    
    
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-1-neg-5-epoch_10", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-1-neg-5-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="colbert-ModernBERT-base-1-neg-5-epoch_5", 
                              reranker_path="ayushexel/colbert-ModernBERT-base-1-neg-5-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="answerai-colbert-small-v1-1-neg-1-epoch_0", 
                              reranker_path="ayushexel/colbert-answerai-colbert-small-v1-1-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [10],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="answerai-colbert-small-v1-1-neg-1-epoch_5", 
                              reranker_path="ayushexel/colbert-answerai-colbert-small-v1-1-neg-1-epoch-gooaq-1995000", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )
    optimized_search_pipeline(
                              210_000_0, 2000,  
                              wandb_run_name="answerai-colbert-small-v1-control_5", 
                              reranker_path="answerdotai/answerai-colbert-small-v1", 
                              reranker_type="colbert",
                              query_types = ["vector_reranked", "fts_reranked", "hybrid"],
                              k_values = [5],
                              wandb_project = "gooaq-reranker-2M-trained",
                             use_pylate=True
                             )

    