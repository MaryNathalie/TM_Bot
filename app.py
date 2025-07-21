import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import tempfile
import hashlib
import re 

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory

from src.loader import (
    process_pdf_with_docling, 
    split_markdown_text, 
    check_gpu, 
    load_and_split_with_pypdf
)
from src.vectorstore import create_vector_store
from src.prompts import QA_PROMPT, FINANCIAL_QA_PROMPT
from src.chain import create_rag_chain, create_summarization_chain, create_router_agent


# Cache Directory
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

st.set_page_config(page_title="TMBot: PDF Analyzer", page_icon="📄")


# Helper functions
def get_file_hash(file_content):
    return hashlib.sha256(file_content).hexdigest()

def format_chat_history(messages):
    """Formats the chat history into a string for downloading."""
    history_str = ""
    for message in messages:
        history_str += f"{message['role'].capitalize()}: {message['content']}\n"
    return history_str


# Authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Authentication")
    st.write("Please enter your OpenAI API key to proceed.")
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key_input")
    if st.button("Authenticate"):
        if api_key:
            try:
                os.environ["OPENAI_API_KEY"] = api_key
                OpenAIEmbeddings()
                st.session_state.authenticated = True
                st.session_state.api_key = api_key
                st.success("Authentication successful!")
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
        else:
            st.warning("Please enter your API key.")
else:
    # Main Application
    st.title("📄 TMBot: PDF Analyzer")
    st.markdown("""
    Welcome! This application allows you to chat with your PDF documents.
    
    **How to use:**
    1.  **Upload a PDF** using the uploader below.
    2.  Ask a **question** in the chat box, or type **"summarize"** to get a summary.
    3.  Manage your session using the controls in the sidebar.
    """)

    # Check for GPU
    gpu_available = check_gpu()
    st.info(f"GPU available: {gpu_available}. Using {'Docling (GPU accelerated)' if gpu_available else 'PyPDF (CPU fallback)'}.")

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")


    # Sidebar for Controls
    with st.sidebar:
        st.header("Configure Analysis")
        model_choice = st.selectbox("LLM Model", ("gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)

        st.header("Manage Session")
        if st.button("Clear Session & Start Over"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
            
        if st.session_state.get("messages"):
            chat_history_str = format_chat_history(st.session_state.messages)
            st.download_button(
                label="Download Chat History",
                data=chat_history_str,
                file_name="chat_history.txt",
                mime="text/plain"
            )

        st.header("Usage Tracking")
        if 'total_cost' not in st.session_state: st.session_state.total_cost = 0.0
        st.write(f"Total Cost (USD): ${st.session_state.total_cost:.5f}")

    # Session State Initialization
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "router_agent" not in st.session_state:
        st.session_state.router_agent = None

    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)

        if st.session_state.get("last_file_hash") != file_hash:
            st.session_state.router_agent = None
            st.session_state.messages = []
            st.session_state.memory.clear()
            
            docs = None # Initialize docs variable

            # Use appropriate processing method based on GPU availability
            if gpu_available:
                # Docling (GPU) Path with Caching
                markdown_path = os.path.join(CACHE_DIR, f"{file_hash}.md")
                if os.path.exists(markdown_path):
                    st.success("Loading processed markdown from cache...")
                    with open(markdown_path, "r", encoding="utf-8") as f:
                        markdown_content = f.read()
                else:
                    with st.spinner("Performing first-time processing with Docling (GPU)..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(file_content)
                            tmp_file_path = tmp_file.name
                        
                        markdown_content = process_pdf_with_docling(tmp_file_path, enable_ocr=False)
                        
                        with open(markdown_path, "w", encoding="utf-8") as f:
                            f.write(markdown_content)
                        
                        os.remove(tmp_file_path)
                
                docs = split_markdown_text(markdown_content)
            
            else:
                # PyPDF (CPU) Path (no caching for this path)
                with st.spinner("Processing PDF with PyPDF (CPU Fallback)..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(file_content)
                        tmp_file_path = tmp_file.name
                    
                    docs = load_and_split_with_pypdf(tmp_file_path)
                    os.remove(tmp_file_path)
            
            # Agent Initialization
            if docs:
                with st.spinner("Initializing the analysis agent..."):
                    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
                    vector_store = create_vector_store(docs, embeddings)
                    
                    llm = ChatOpenAI(model_name=model_choice, temperature=temperature, openai_api_key=st.session_state.api_key)
                    
                    general_rag_chain = create_rag_chain(vector_store, llm, QA_PROMPT, st.session_state.memory)
                    financial_rag_chain = create_rag_chain(vector_store, llm, FINANCIAL_QA_PROMPT, st.session_state.memory)
                    
                    summarizer_func = create_summarization_chain(llm, docs)

                    llm_router = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=st.session_state.api_key)
                    st.session_state.router_agent = create_router_agent(general_rag_chain, financial_rag_chain, summarizer_func, llm_router)
                    
                    st.session_state.last_file_hash = file_hash
                    st.success("Document processed! The agent is ready.")

    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            safe_content = message["content"].replace('$', '\\$')
            st.markdown(safe_content)

    if prompt := st.chat_input("Ask a question, or type 'summarize'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.router_agent is None:
            st.warning("Please upload a document first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    with get_openai_callback() as cb:
                        response = st.session_state.router_agent.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.memory.chat_memory.messages
                        })
                        st.session_state.total_cost += cb.total_cost
                    
                    answer = response.get("output", "Sorry, I encountered an error.")
                    
                    safe_answer_for_display = answer.replace('$', '\\$')
                    st.markdown(safe_answer_for_display)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
