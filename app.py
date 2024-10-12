import gradio as gr

HISTORY = []

def chatbot(user_input, history):
    if user_input.lower() == "hello":
        bot_response = "Hi! How can I assist you today?"
    elif user_input.lower() == "bye":
        bot_response = "Goodbye! Take care!"
    else:
        bot_response = "I'm here to help. Feel free to ask anything!"

    HISTORY.append((user_input, bot_response))

    return bot_response

def gradio_interface():
    chat_interface = gr.ChatInterface(
        fn=chatbot,
        title="Q/A RAG ChatBot",
        description="A chatbot using Retrieval-Augmented Generation (RAG) for PDF-based question answering.",
    )

    return chat_interface

# Run the Gradio app
if __name__ == "__main__":
    app = gradio_interface()
    app.launch()
