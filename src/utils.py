import os

def get_path_to_file(
    folder_name: str,
    file_name: str,
) -> str:
    """
    This function returns the path to the file.

    Parameters
    ----------
    folder_name : str
        Folder name where the file is located.
    
    file_name : str
        File name.
    
    Returns
    -------
    path_to_file : str
        Path to the file.
    """
    base_dir = os.path.dirname(os.getcwd())
    if 'QA-RAG-ChatBot' not in base_dir:
        base_dir = os.path.join(base_dir, 'QA-RAG-ChatBot')
    path_to_file = os.path.join(base_dir, folder_name, file_name)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    return path_to_file

def post_process_llm_response(
    llm_response: str
):
    response_sentence = 'Helpful Answer:'
    response_sentence_len = len(response_sentence)
    sentence_index = llm_response.find(response_sentence)
    llm_response = llm_response[sentence_index + response_sentence_len:]
    llm_response = llm_response.strip()

    return llm_response

def preprocess_path(file_path):
    """
    Preprocesses the file path by removing the file name and converting
    the path separators to a platform-independent format.
    
    Args:
        file_path (str): The full path to the file.
    
    Returns:
        str: The processed folder path.
    """
    # Get the directory (i.e., remove the file name) from the file path
    folder_path = os.path.dirname(file_path)
    
    # Convert the path to Unix-style forward slashes (if needed)
    processed_path = folder_path.replace("\\", "/")
    
    return processed_path
