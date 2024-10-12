class CFG:
    # LLMs
    model_name = 'Qwen2.5' # Qwen2.5, llama2-7b-chat, llama2-13b-chat, mistral-7B
    temperature = 0
    top_p = 0.95
    repetition_penalty = 1.15    

    # splitting
    split_chunk_size = 200
    split_overlap = 50
    
    # embeddings
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'    

    # similar passages
    k = 6
    
    # paths
    PDFs_path = '/kaggle/input/harry-potter-books-in-pdf-1-7/HP books/'
    Embeddings_path =  '/kaggle/input/faiss-hp-sentence-transformers'
    Output_folder = './harry-potter-vectordb'
