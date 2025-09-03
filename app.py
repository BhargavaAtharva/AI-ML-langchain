import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
import asyncio

load_dotenv()

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

def get_pdf_docs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            temp_path = tmp_file.name
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())
        os.remove(temp_path)
    return docs

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store, model_name="gemini-pro"):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    if st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
    else:
        st.warning("Please upload and process a PDF file first.")

def main():
    st.set_page_config(page_title="RAG with Gemini ⚡️", page_icon="📄", layout="wide")
    st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(110deg, #010101, #0a1f44, #1e6091);
                color: white;
                font-family: 'Poppins', sans-serif;
            }
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');
            .title-container {
                text-align: center;
                margin-bottom: 10px;
            }
            .title-container h1 {
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
            }
            .title-container p {
                font-family: 'Poppins', sans-serif;
                font-weight: 300;
                font-size: 16px;
            }
            .chat-message {
                display: flex;
                align-items: flex-start;
                gap: 12px;
                margin: 10px 0;
            }
            .chat-bubble {
                border-radius: 12px;
                padding: 12px 15px;
                max-width: 80%;
                line-height: 1.4;
                font-size: 15px;
            }
            .user-bubble {
                background-color: rgba(0, 123, 255, 0.2);
                border-left: 4px solid #00b4d8;
                align-self: flex-end;
            }
            .assistant-bubble {
                background-color: rgba(255, 255, 255, 0.1);
                border-left: 4px solid #ffd166;
            }
            .animated-bar {
                height: 5px;
                width: 100%;
                background: linear-gradient(-45deg, #000000, #0077b6, #48cae4);
                background-size: 400% 400%;
                animation: gradientMove 10s ease infinite;
                margin-bottom: 20px;
                border-radius: 10px;
            }
            @keyframes gradientMove {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
        </style>
    """, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("<div class='animated-bar'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class="title-container">
            <h1>📄 RAG with Google Gemini ⚡️</h1>
            <p>Upload a PDF and chat with it. Your past Q&A will be remembered during this session.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("⚙️ Settings")
        selected_model = st.selectbox(
            "Choose a Gemini model:",
            ["gemini-pro", "gemini-2.5-flash"]
        )
        st.title("📂 Upload your PDF")
        pdf_docs = st.file_uploader("Drag and drop PDF files here and click 'Process'", accept_multiple_files=True, type="pdf")
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDF, please wait..."):
                    docs = get_pdf_docs(pdf_docs)
                    text_chunks = get_text_chunks(docs)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vector_store, selected_model)
                    st.success("PDF processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    st.subheader("💬 Chat with your PDF")
    for message in st.session_state.chat_history:
        role = "user" if message.type == 'human' else "assistant"
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
        with st.chat_message(role):
            st.markdown(f"<div class='chat-bubble {bubble_class}'>{message.content}</div>", unsafe_allow_html=True)

    user_question = st.chat_input("Ask something about the PDF...")
    if user_question:
        handle_user_input(user_question)
        st.rerun()

if __name__ == '__main__':
    main()
