from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from src.config import CFG
from src.utils import get_path_to_file
from typing import List

def load_pdf_files(
    folder_name: str,
):
    loader = DirectoryLoader(
        get_path_to_file(folder_name=folder_name, file_name=''),
        loader_cls=PyPDFLoader,
        use_multithreading=True
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CFG.split_chunk_size,
        chunk_overlap = CFG.split_overlap
    )
    texts = text_splitter.split_documents(documents)

    return texts

def load_vectordb(
    texts: List[str],
    config: CFG
):
    embedding = HuggingFaceInstructEmbeddings(
        model_name = config.embeddings_model_repo,
    )

    vectordb = Chroma(embedding_function=embedding)
    vectordb.from_documents(documents=texts, embedding=embedding)

    return vectordb
