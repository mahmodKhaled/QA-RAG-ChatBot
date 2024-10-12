class CFG:
    # LLMs
    model_name = 'Qwen2.5'
    temperature = 0
    top_p = 0.95
    repetition_penalty = 1.15    

    # splitting
    split_chunk_size = 200
    split_overlap = 50
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'    
