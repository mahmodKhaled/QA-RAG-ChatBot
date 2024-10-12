import gradio as gr
import warnings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.config import CFG
from src.model import load_model, create_llm
from src.db import load_pdf_files, load_vectordb
from src.utils import post_process_llm_response, preprocess_path

warnings.filterwarnings("ignore")

def rag_pipeline(pdfs_folder):
    # Load the tokenizer, model, and maximum length configuration
    tokenizer, model, max_len = load_model(model_name=CFG.model_name)

    # Create the language model (LLM) using the loaded model, tokenizer, and configuration
    llm = create_llm(model, tokenizer, max_len, CFG)

    # Load PDF files from the specified folder
    texts = load_pdf_files(folder_name=pdfs_folder)

    # Create a vector database from the loaded texts using the provided configuration
    vectordb = load_vectordb(texts=texts, config=CFG)

    # Define the prompt template for the QA chain
    template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. Use three sentences maximum.
    Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:
    """
    qa_chain_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Create the RetrievalQA chain using the LLM and the vector database retriever
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )

    # Return the configured QA chain
    return qa_chain

def chatbot(pdfs_folder):
    # Run the RAG pipeline with the uploaded folder
    qa_chain = rag_pipeline(pdfs_folder)
    
    def chat(question):
        result = qa_chain({"query": question})
        response = post_process_llm_response(result["result"])
        return [(question, response)]  # Return as a tuple (question, response) in a list
    
    return chat

def gradio_interface():
    # Define the folder input, status message, question input, and chat interface
    with gr.Blocks() as app:
        folder_input = gr.File(label="Upload PDF Folder", file_count="directory", type="filepath")
        status_message = gr.Markdown("### Status: Waiting for file upload...")
        question_input = gr.Textbox(label="Enter your question here", placeholder="Ask a question based on the uploaded documents")
        chat_box = gr.Chatbot(label="Q/A RAG ChatBot")
        submit_button = gr.Button("Submit Question")

        # Store the chatbot instance
        bot = None

        # Handle the folder upload and initialize the chatbot
        def handle_upload(folder):
            nonlocal bot
            folder_path = preprocess_path(file_path=folder[0])  # Process folder path
            bot = chatbot(folder_path)  # Initialize chatbot with the uploaded folder
            message = "### Status: Files uploaded successfully. Model loaded and ready to use!"  # Update status and enable question input

            return message

        # Handle the question submission and get the answer from the bot
        def handle_question(question, history):
            if bot is None:
                return "Please upload a folder first.", history
            response = bot(question)
            history.append(response[0])  # Append the (question, response) tuple to chat history
            return history

        # Connect the file upload to the folder handling function
        folder_input.change(handle_upload, inputs=folder_input, outputs=status_message)

        # Connect the question input to the chatbot and update chat history
        submit_button.click(handle_question, inputs=[question_input, chat_box], outputs=chat_box)

    return app

# Run the Gradio app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
