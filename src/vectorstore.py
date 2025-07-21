# src/vectorstore.py

from langchain_community.vectorstores import FAISS

def create_vector_store(chunked_docs, embeddings_model):
    """
    Creates a FAISS vector store from a list of document chunks and an embedding model.
    """
    try:
        vectorstore = FAISS.from_documents(documents=chunked_docs, embedding=embeddings_model)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

